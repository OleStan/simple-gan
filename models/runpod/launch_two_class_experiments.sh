#!/bin/bash
set -euo pipefail

# Full pipeline: sync → train dual_wgan + improved_wgan_v2 for nz sizes
# then run reports and quality metrics on pod, then download everything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

LATENT_SIZES=(10 20 30)
EPOCHS=500
CONCURRENT=3   # how many per-model jobs run simultaneously
RESULT_TAG="sigmoid_vs_linear_v2"

echo "============================================"
echo "  Sigmoid vs Linear Profile GAN Experiments (v2)"
echo "  Models: dual_wgan + improved_wgan_v2"
echo "  Latent sizes: ${LATENT_SIZES[*]}"
echo "  Epochs: $EPOCHS"
echo "  Result folder: results/${RESULT_TAG}/"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Sync code + data to pod
# ---------------------------------------------------------------------------
echo "=== [1/5] Syncing code and data to pod ==="
# Note: Using --delete to ensure training data generated locally is mirrored to pod
rsync -avz --progress --delete \
    --no-perms --no-owner --no-group \
    --exclude '.git' \
    --exclude 'results/' \
    --exclude 'nets/' \
    --exclude 'img/' \
    --exclude 'data/raw/' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'runpod/.env' \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$LOCAL_PROJECT_DIR/" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/" || true

echo "Training data synced."
ssh_cmd "python3 -c 'import json; d=json.load(open(\"$REMOTE_PROJECT_DIR/data/training/normalization_params.json\")); print(\"  n_classes:\", d[\"n_classes\"], \"| N:\", d[\"N\"], \"| classes:\", d.get(\"class_names\", \"N/A\"))'  2>/dev/null || echo '  WARN: could not read norm params'"

echo ""

# ---------------------------------------------------------------------------
# 2. Install / verify dependencies on pod
# ---------------------------------------------------------------------------
echo "=== [2/5] Installing dependencies on pod === "
ssh_cmd "pip install -q -r $REMOTE_PROJECT_DIR/models/runpod/requirements.txt"
echo ""

# ---------------------------------------------------------------------------
# 3. Train both models for each latent size
# ---------------------------------------------------------------------------
echo "=== [3/5] Training both models for each latent size ==="
echo ""

run_batch() {
    local model="$1"     # "dual_wgan" or "improved_wgan_v2"
    local nz="$2"

    local log="/tmp/train_${model}_nz${nz}_v2.log"
    local script_rel

    if [ "$model" = "dual_wgan" ]; then
        script_rel="models/dual_wgan/train_latent_experiment.py"
        local completed_glob="$REMOTE_PROJECT_DIR/results/${RESULT_TAG}/dual_wgan_nz${nz}_*/models/netG_final.pth"
    else
        script_rel="models/improved_wgan_v2/train_latent_experiment.py"
        local completed_glob="$REMOTE_PROJECT_DIR/results/${RESULT_TAG}/improved_wgan_v2_nz${nz}_*/models/netG_final.pt"
    fi

    # Cleanup any existing logs from previous failures
    ssh_cmd "rm -f ${log}"

    echo "  🚀 Launching ${model} nz=${nz} → log: ${log}"
    ssh_cmd "cd $REMOTE_PROJECT_DIR/${script_rel%/*} && nohup env RESULT_TAG=${RESULT_TAG} python train_latent_experiment.py --nz ${nz} --epochs ${EPOCHS} > ${log} 2>&1 &"
    sleep 2
}

for NZ in "${LATENT_SIZES[@]}"; do
    run_batch "dual_wgan" "$NZ"
    run_batch "improved_wgan_v2" "$NZ"
done

# Wait for all to complete
echo ""
echo "Waiting for all training jobs to complete..."
while true; do
    STILL=0
    for NZ in "${LATENT_SIZES[@]}"; do
        R1=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz ${NZ}' | grep dual_wgan | grep -v grep | wc -l" 2>/dev/null || echo "0")
        R2=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz ${NZ}' | grep improved_wgan_v2 | grep -v grep | wc -l" 2>/dev/null || echo "0")
        STILL=$((STILL + R1 + R2))
    done
    [ "$STILL" -eq 0 ] && break
    echo "  $(date '+%H:%M:%S') — $STILL training process(es) still active..."
    sleep 60
done
echo "✅ All training complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Run reports + quality metrics on pod
# ---------------------------------------------------------------------------
echo "=== [4/5] Running reports and quality metrics on pod ==="

REPORT_CMD=$(cat <<HEREDOC
set -euo pipefail
ROOT=${REMOTE_PROJECT_DIR}
export PYTHONPATH=\$ROOT

for result_dir in \$ROOT/results/${RESULT_TAG}/dual_wgan_nz*/; do
    [ -f "\${result_dir}models/netG_final.pth" ] || continue
    [ -f "\${result_dir}quality_report/quality_summary.json" ] && echo "SKIP \$result_dir" && continue

    echo "--- Report: \$result_dir ---"
    python \$ROOT/scripts/reports/generate_dual_wgan_report.py "\$result_dir" \
        2>&1 | tail -5 || echo "  WARN: report failed for \$result_dir"

    echo "--- Quality: \$result_dir ---"
    python \$ROOT/scripts/reports/run_quality_check.py \
        --model dual_wgan \
        --model_dir "\$result_dir" \
        --training_data \$ROOT/data/training \
        --n_generated 500 \
        --output_dir "\${result_dir}quality_report" \
        2>&1 | tail -10 || echo "  WARN: quality check failed for \$result_dir"
done

for result_dir in \$ROOT/results/${RESULT_TAG}/improved_wgan_v2_nz*/; do
    [ -f "\${result_dir}models/netG_final.pt" ] || continue
    [ -f "\${result_dir}quality_report/quality_summary.json" ] && echo "SKIP \$result_dir" && continue

    echo "--- Report: \$result_dir ---"
    python \$ROOT/scripts/reports/generate_improved_wgan_v2_report.py "\$result_dir" \
        2>&1 | tail -5 || echo "  WARN: report failed for \$result_dir"

    echo "--- Quality: \$result_dir ---"
    python \$ROOT/scripts/reports/run_quality_check.py \
        --model improved_wgan_v2 \
        --model_dir "\$result_dir" \
        --training_data \$ROOT/data/training \
        --n_generated 500 \
        --output_dir "\${result_dir}quality_report" \
        2>&1 | tail -10 || echo "  WARN: quality check failed for \$result_dir"
done

echo "All reports and quality checks done."
HEREDOC
)

ssh_cmd "bash -c '$REPORT_CMD'"
echo ""

# ---------------------------------------------------------------------------
# 5. Download all results
# ---------------------------------------------------------------------------
echo "=== [5/5] Downloading results from pod ==="
mkdir -p "$LOCAL_PROJECT_DIR/results/${RESULT_TAG}"
rsync -avz --progress \
    --no-perms --no-owner --no-group \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/results/${RESULT_TAG}/" \
    "$LOCAL_PROJECT_DIR/results/${RESULT_TAG}/"

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results in: $LOCAL_PROJECT_DIR/results/${RESULT_TAG}/"
echo "============================================"
