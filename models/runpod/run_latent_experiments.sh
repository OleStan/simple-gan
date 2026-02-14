#!/bin/bash
set -euo pipefail

# Run multiple dual WGAN training experiments with different latent space sizes
# Usage: ./runpod/run_latent_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Latent space sizes to test
LATENT_SIZES=(6 8 10 12 16 24 32 48 64 96 128)
EPOCHS=500

echo "============================================"
echo "  Dual WGAN Latent Space Experiments"
echo "============================================"
echo "  Latent sizes: ${LATENT_SIZES[@]}"
echo "  Epochs per experiment: $EPOCHS"
echo ""

# Upload the experiment script if not already done
echo "[1/2] Uploading experiment script to pod..."
rsync -avz --progress \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$SCRIPT_DIR/../dual_wgan/train_latent_experiment.py" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/models/dual_wgan/"

echo ""
echo "[2/2] Starting experiments sequentially..."
echo ""

for NZ in "${LATENT_SIZES[@]}"; do
    SESSION_NAME="dual_wgan_nz${NZ}"

    echo "============================================"
    echo "  Experiment: nz=$NZ"
    echo "============================================"

    # Check if session already exists
    EXISTING_SESSION=$(ssh_cmd "tmux has-session -t $SESSION_NAME 2>/dev/null && echo 'exists' || echo 'none'")

    if [ "$EXISTING_SESSION" = "exists" ]; then
        echo "⚠️  Session '$SESSION_NAME' already exists, skipping..."
        echo ""
        continue
    fi

    # Start training in tmux
    REMOTE_CMD="cd $REMOTE_PROJECT_DIR/models/dual_wgan && python train_latent_experiment.py --nz $NZ --epochs $EPOCHS"

    ssh_cmd "tmux new-session -d -s $SESSION_NAME '$REMOTE_CMD; echo \"Training nz=$NZ finished. Press enter to close.\"; read'"

    echo "✓ Started training for nz=$NZ in session '$SESSION_NAME'"
    echo "  Monitor: ssh ... \"tmux attach -t $SESSION_NAME\""
    echo ""

    # Wait a bit before starting the next one
    sleep 2
done

echo ""
echo "============================================"
echo "  All Experiments Launched!"
echo "============================================"
echo ""
echo "Active sessions:"
ssh_cmd "tmux list-sessions 2>/dev/null || echo '  No sessions running'"
echo ""
echo "To monitor a specific experiment:"
echo "  ./runpod/monitor_training.sh <session_name> attach"
echo ""
echo "To download all results when done:"
echo "  ./runpod/sync_from_pod.sh all"
echo ""
