#!/bin/bash
set -euo pipefail

# Full pipeline: sync → train dual_wgan + improved_wgan_v2 for nz sizes
# then run reports and quality metrics on pod, then download everything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

LATENT_SIZES=(6 10 20 30)
EPOCHS=500
CONCURRENT=2   # how many per-model jobs run simultaneously
RESULT_TAG="sigmoid_vs_linear"

echo "============================================"
echo "  Sigmoid vs Linear Profile GAN Experiments"
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
rsync -avz --progress \
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

echo "Generating sigmoid vs linear dataset on pod..."
ssh_cmd "cd $REMOTE_PROJECT_DIR && PYTHONPATH=$REMOTE_PROJECT_DIR python scripts/data_generation/generate_training_data.py > /tmp/datagen.log 2>&1"
echo "  Dataset generation complete."
ssh_cmd "python3 -c 'import json; d=json.load(open(\"$REMOTE_PROJECT_DIR/data/training/normalization_params.json\")); print(\"  n_classes:\", d[\"n_classes\"], \"| N:\", d[\"N\"], \"| classes:\", [c[\"name\"] for c in d[\"class_configs\"]])'  2>/dev/null"

echo ""

# ---------------------------------------------------------------------------
# 2. Install / verify dependencies on pod
# ---------------------------------------------------------------------------
echo "=== [2/5] Installing dependencies on pod ==="
ssh_cmd "pip install -q -r $REMOTE_PROJECT_DIR/models/runpod/requirements.txt"
echo ""

# ---------------------------------------------------------------------------
# 3. Train both models for each latent size (CONCURRENT jobs at a time)
# ---------------------------------------------------------------------------
echo "=== [3/5] Training both models for each latent size ==="
echo ""

run_batch() {
    local model="$1"     # "dual_wgan" or "improved_wgan_v2"
    local nz="$2"

    local log="/tmp/train_${model}_nz${nz}_2c.log"
    local script_rel

    if [ "$model" = "dual_wgan" ]; then
        script_rel="models/dual_wgan/train_latent_experiment.py"
        local completed_glob="$REMOTE_PROJECT_DIR/results/${RESULT_TAG}/dual_wgan_nz${nz}_*/models/netG_final.pth"
    else
        script_rel="models/improved_wgan_v2/train_latent_experiment.py"
        local completed_glob="$REMOTE_PROJECT_DIR/results/${RESULT_TAG}/improved_wgan_v2_nz${nz}_*/models/netG_final.pt"
    fi

    local already_done
    already_done=$(ssh_cmd "ls ${completed_glob} 2>/dev/null | wc -l" || echo "0")
    if [ "$already_done" != "0" ]; then
        echo "  ✅ ${model} nz=${nz} already completed, skipping."
        return
    fi

    local already_running
    already_running=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz ${nz}' | grep '${model}' | grep -v grep | wc -l" 2>/dev/null || echo "0")
    if [ "$already_running" != "0" ]; then
        echo "  ⚠️  ${model} nz=${nz} already running, skipping."
        return
    fi

    echo "  🚀 Launching ${model} nz=${nz} → log: ${log}"
    ssh_cmd "cd $REMOTE_PROJECT_DIR/${script_rel%/*} && nohup env RESULT_TAG=${RESULT_TAG} python train_latent_experiment.py --nz ${nz} --epochs ${EPOCHS} > ${log} 2>&1 &"
    sleep 2
}

# Process in batches of CONCURRENT per model
for ((i=0; i<${#LATENT_SIZES[@]}; i+=CONCURRENT)); do
    echo "--- Batch $((i/CONCURRENT + 1)): nz=${LATENT_SIZES[*]:$i:$CONCURRENT} ---"

    for ((j=0; j<CONCURRENT && i+j<${#LATENT_SIZES[@]}; j++)); do
        NZ=${LATENT_SIZES[$((i+j))]}
        run_batch "dual_wgan" "$NZ"
        run_batch "improved_wgan_v2" "$NZ"
    done

    # Wait for current batch to finish before launching next
    if [ $((i + CONCURRENT)) -lt ${#LATENT_SIZES[@]} ]; then
        echo ""
        echo "Waiting for batch to complete... (Ctrl+C safe — jobs run on pod)"
        while true; do
            STILL=0
            for ((j=0; j<CONCURRENT && i+j<${#LATENT_SIZES[@]}; j++)); do
                NZ=${LATENT_SIZES[$((i+j))]}
                R1=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz ${NZ}' | grep dual_wgan | grep -v grep | wc -l" 2>/dev/null || echo "0")
                R2=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz ${NZ}' | grep improved_wgan_v2 | grep -v grep | wc -l" 2>/dev/null || echo "0")
                STILL=$((STILL + R1 + R2))
            done
            [ "$STILL" -eq 0 ] && break
            echo "  $(date '+%H:%M:%S') — $STILL process(es) still running..."
            sleep 60
        done
        echo "  ✅ Batch done."
        echo ""
    fi
done

# Wait for the final batch
echo ""
echo "Waiting for final batch to complete..."
while true; do
    STILL=0
    for NZ in "${LATENT_SIZES[@]}"; do
        R1=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz ${NZ}' | grep dual_wgan | grep -v grep | wc -l" 2>/dev/null || echo "0")
        R2=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz ${NZ}' | grep improved_wgan_v2 | grep -v grep | wc -l" 2>/dev/null || echo "0")
        STILL=$((STILL + R1 + R2))
    done
    [ "$STILL" -eq 0 ] && break
    echo "  $(date '+%H:%M:%S') — $STILL process(es) still running..."
    sleep 60
done
echo "✅ All training complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Run reports + quality metrics on pod for every result dir
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
    cd \$ROOT && python scripts/reports/generate_dual_wgan_report.py "\$result_dir" \
        2>&1 | tail -5 || echo "WARN: report failed for \$result_dir"

    echo "--- Quality: \$result_dir ---"
    cd \$ROOT && python scripts/reports/run_quality_check.py \
        --model dual_wgan \
        --model_dir "\$result_dir" \
        --training_data data/training \
        --n_generated 500 \
        --output_dir "\${result_dir}quality_report" \
        2>&1 | tail -10 || echo "WARN: quality check failed for \$result_dir"
done

for result_dir in \$ROOT/results/${RESULT_TAG}/improved_wgan_v2_nz*/; do
    [ -f "\${result_dir}models/netG_final.pt" ] || continue
    [ -f "\${result_dir}quality_report/quality_summary.json" ] && echo "SKIP \$result_dir" && continue

    echo "--- Report: \$result_dir ---"
    cd \$ROOT && python scripts/reports/generate_improved_wgan_v2_report.py "\$result_dir" \
        2>&1 | tail -5 || echo "WARN: report failed for \$result_dir"

    echo "--- Quality: \$result_dir ---"
    cd \$ROOT && python scripts/reports/run_quality_check.py \
        --model improved_wgan_v2 \
        --model_dir "\$result_dir" \
        --training_data data/training \
        --n_generated 500 \
        --output_dir "\${result_dir}quality_report" \
        2>&1 | tail -10 || echo "WARN: quality check failed for \$result_dir"
done

echo "All reports and quality checks done."
HEREDOC
)

ssh_cmd "REMOTE_PROJECT_DIR=$REMOTE_PROJECT_DIR bash -c '$REPORT_CMD'"
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
echo "  Results in: $LOCAL_PROJECT_DIR/results/"
echo "============================================"
