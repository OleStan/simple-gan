#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

SESSION_NAME="${1:-training}"
MODE="${2:-tail}"

echo "============================================"
echo "  Monitor Training on RunPod"
echo "============================================"

case "$MODE" in
    tail)
        echo "Showing last 50 lines of tmux session '$SESSION_NAME'..."
        echo "--------------------------------------------"
        ssh_cmd "tmux capture-pane -t $SESSION_NAME -p -S -50" 2>/dev/null || {
            echo "ERROR: tmux session '$SESSION_NAME' not found."
            echo "Available sessions:"
            ssh_cmd "tmux list-sessions 2>/dev/null" || echo "  (none)"
            exit 1
        }
        ;;
    attach)
        echo "Attaching to tmux session '$SESSION_NAME'..."
        echo "(Detach with Ctrl+B, then D)"
        echo ""
        ssh -t -p "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" \
            "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST" \
            "tmux attach -t $SESSION_NAME"
        ;;
    gpu)
        echo "GPU status:"
        ssh_cmd "nvidia-smi"
        ;;
    disk)
        echo "Disk usage on /workspace:"
        ssh_cmd "df -h /workspace && echo '' && du -sh $REMOTE_PROJECT_DIR/results/*/ 2>/dev/null || echo 'No results yet'"
        ;;
    *)
        echo "Usage: $0 [session_name] [tail|attach|gpu|disk]"
        echo "  tail    - Show last 50 lines of output (default)"
        echo "  attach  - Attach to tmux session interactively"
        echo "  gpu     - Show GPU utilization (nvidia-smi)"
        echo "  disk    - Show disk usage"
        exit 1
        ;;
esac
