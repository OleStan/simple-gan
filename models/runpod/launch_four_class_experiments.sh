#!/bin/bash
set -euo pipefail

# Full pipeline: sync → train dual_wgan + improved_wgan_v2 in tmux sessions
# then run reports and quality metrics on pod, then download everything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

LATENT_SIZES=(10 20 30)
EPOCHS=1000
RESULT_TAG="four_classes_v1"

echo "============================================"
echo "  4-Class Profile GAN Experiments (TMUX Mode)"
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
echo ""

# ---------------------------------------------------------------------------
# 2. Install / verify dependencies on pod
# ---------------------------------------------------------------------------
echo "=== [2/5] Installing dependencies on pod === "
ssh_cmd "pip install -q -r $REMOTE_PROJECT_DIR/models/runpod/requirements.txt"
ssh_cmd "apt-get update && apt-get install -y tmux" || true
echo ""

# ---------------------------------------------------------------------------
# 3. Train models in tmux sessions
# ---------------------------------------------------------------------------
echo "=== [3/5] Launching training in tmux sessions ==="

run_tmux_job() {
    local model="$1"     # "dual_wgan" or "improved_wgan_v2"
    local nz="$2"
    local session_name="train_${model}_nz${nz}"
    local log="/tmp/${session_name}.log"
    local script_rel

    if [ "$model" = "dual_wgan" ]; then
        script_rel="models/dual_wgan/train_latent_experiment.py"
    else
        script_rel="models/improved_wgan_v2/train_latent_experiment.py"
    fi

    echo "  🚀 Launching ${model} nz=${nz} in tmux session: ${session_name}"
    
    # Kill existing session if any
    ssh_cmd "tmux kill-session -t ${session_name} 2>/dev/null || true"
    
    # Start new detached session and run training
    ssh_cmd "tmux new-session -d -s ${session_name} \"cd $REMOTE_PROJECT_DIR/${script_rel%/*} && env RESULT_TAG=${RESULT_TAG} python train_latent_experiment.py --nz ${nz} --epochs ${EPOCHS} 2>&1 | tee ${log}\""
}

for NZ in "${LATENT_SIZES[@]}"; do
    run_tmux_job "dual_wgan" "$NZ"
    run_tmux_job "improved_wgan_v2" "$NZ"
done

echo ""
echo "All jobs launched in tmux!"
echo "To monitor a session: ssh -p $RUNPOD_SSH_PORT root@$RUNPOD_SSH_HOST 'tmux attach -t train_improved_wgan_v2_nz20'"
echo "Waiting for completion..."

while true; do
    STILL=0
    for NZ in "${LATENT_SIZES[@]}"; do
        # Count sessions starting with 'train_'
        COUNT=$(ssh_cmd "tmux ls 2>/dev/null | grep 'train_' | wc -l" || echo "0")
        STILL=$COUNT
    done
    [ "$STILL" -eq 0 ] && break
    echo "  $(date '+%H:%M:%S') — $STILL tmux session(s) still active..."
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

# Clean up any partial quality reports to ensure fresh ones
find \$ROOT/results/${RESULT_TAG}/ -name "quality_report" -type d -exec rm -rf {} + || true

for result_dir in \$ROOT/results/${RESULT_TAG}/dual_wgan_nz*/; do
    [ -f "\${result_dir}models/netG_final.pth" ] || continue
    echo "--- Report: \$result_dir ---"
    python \$ROOT/scripts/reports/generate_dual_wgan_report.py "\$result_dir" 2>&1 | tail -5
    echo "--- Quality: \$result_dir ---"
    python \$ROOT/scripts/reports/run_quality_check.py --model dual_wgan --model_dir "\$result_dir" --training_data \$ROOT/data/training --n_generated 500 --output_dir "\${result_dir}quality_report" 2>&1 | tail -10
done

for result_dir in \$ROOT/results/${RESULT_TAG}/improved_wgan_v2_nz*/; do
    [ -f "\${result_dir}models/netG_final.pt" ] || continue
    echo "--- Report: \$result_dir ---"
    python \$ROOT/scripts/reports/generate_improved_wgan_v2_report.py "\$result_dir" 2>&1 | tail -5
    echo "--- Quality: \$result_dir ---"
    python \$ROOT/scripts/reports/run_quality_check.py --model improved_wgan_v2 --model_dir "\$result_dir" --training_data \$ROOT/data/training --n_generated 500 --output_dir "\${result_dir}quality_report" 2>&1 | tail -10
done
HEREDOC
)

ssh_cmd "bash -c '$REPORT_CMD'"
echo ""

# ---------------------------------------------------------------------------
# 5. Download results
# ---------------------------------------------------------------------------
echo "=== [5/5] Downloading results from pod ==="
mkdir -p "$LOCAL_PROJECT_DIR/results/${RESULT_TAG}"
rsync -avz --progress \
    --no-perms --no-owner --no-group \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/results/${RESULT_TAG}/" \
    "$LOCAL_PROJECT_DIR/results/${RESULT_TAG}/"

echo "Pipeline complete!"
