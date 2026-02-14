#!/bin/bash
set -euo pipefail

# Run ALL dual WGAN training experiments in parallel
# This launches all experiments simultaneously for maximum speed
# Usage: ./runpod/run_latent_experiments_parallel.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Latent space sizes to test
LATENT_SIZES=(6 8 10 12 16 24 32 48 64 96 128)
EPOCHS=500

echo "============================================"
echo "  Dual WGAN Latent Space Experiments"
echo "  >>> PARALLEL MODE <<<"
echo "============================================"
echo "  Latent sizes: ${LATENT_SIZES[@]}"
echo "  Epochs per experiment: $EPOCHS"
echo "  All experiments will run simultaneously!"
echo ""

# Upload the experiment script if not already done
echo "[1/2] Uploading experiment script to pod..."
rsync -avz --progress \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$SCRIPT_DIR/../dual_wgan/train_latent_experiment.py" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/models/dual_wgan/"

echo ""
echo "[2/2] Starting ALL experiments in parallel..."
echo ""

LAUNCHED=0
SKIPPED=0

for NZ in "${LATENT_SIZES[@]}"; do
    SESSION_NAME="dual_wgan_nz${NZ}"

    # Check if session already exists
    EXISTING_SESSION=$(ssh_cmd "tmux has-session -t $SESSION_NAME 2>/dev/null && echo 'exists' || echo 'none'" 2>/dev/null || echo 'none')

    if [ "$EXISTING_SESSION" = "exists" ]; then
        echo "⚠️  Session '$SESSION_NAME' already exists, skipping..."
        ((SKIPPED++))
        continue
    fi

    # Start training in tmux (in background, no waiting)
    REMOTE_CMD="cd $REMOTE_PROJECT_DIR/models/dual_wgan && python train_latent_experiment.py --nz $NZ --epochs $EPOCHS"

    ssh_cmd "tmux new-session -d -s $SESSION_NAME '$REMOTE_CMD; echo \"Training nz=$NZ finished. Press enter to close.\"; read'" &

    echo "✓ Launched training for nz=$NZ in session '$SESSION_NAME'"
    ((LAUNCHED++))
done

# Wait for all background SSH commands to complete
wait

echo ""
echo "============================================"
echo "  Launch Complete!"
echo "============================================"
echo "  Launched: $LAUNCHED experiments"
echo "  Skipped: $SKIPPED experiments (already running)"
echo ""

# Give tmux sessions a moment to start
sleep 2

echo "Active sessions:"
ssh_cmd "tmux list-sessions 2>/dev/null || echo '  No sessions running'"
echo ""

echo "GPU Status:"
ssh_cmd "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
echo ""

echo "============================================"
echo "Estimated completion time: ~6 minutes"
echo ""
echo "To monitor progress:"
echo "  ./check_runpod_training.sh"
echo ""
echo "To attach to a specific experiment:"
echo "  ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST"
echo "  tmux attach -t dual_wgan_nz6"
echo ""
echo "To download results when done:"
echo "  ./runpod/sync_from_pod.sh all"
echo "============================================"
