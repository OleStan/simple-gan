#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TRAINING_SCRIPT="${1:-train_dual_wgan_small_latent.py}"
SESSION_NAME="${2:-training}"

echo "============================================"
echo "  Starting Training on RunPod"
echo "============================================"
echo "  Script: $TRAINING_SCRIPT"
echo "  tmux session: $SESSION_NAME"
echo ""

REMOTE_CMD="cd $REMOTE_PROJECT_DIR && python $TRAINING_SCRIPT"

EXISTING_SESSION=$(ssh_cmd "tmux has-session -t $SESSION_NAME 2>/dev/null && echo 'exists' || echo 'none'")

if [ "$EXISTING_SESSION" = "exists" ]; then
    echo "WARNING: tmux session '$SESSION_NAME' already exists."
    echo "Options:"
    echo "  1. Attach to it:  ./runpod/monitor_training.sh"
    echo "  2. Kill it first: ssh into pod and run 'tmux kill-session -t $SESSION_NAME'"
    exit 1
fi

echo "Launching training in tmux session '$SESSION_NAME'..."
ssh_cmd "tmux new-session -d -s $SESSION_NAME '$REMOTE_CMD; echo \"Training finished. Press enter to close.\"; read'"

echo ""
echo "Training started!"
echo ""
echo "Useful commands:"
echo "  Monitor:   ./runpod/monitor_training.sh"
echo "  SSH in:    ./runpod/ssh_connect.sh"
echo "  Download:  ./runpod/sync_from_pod.sh"
