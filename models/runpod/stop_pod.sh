#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "============================================"
echo "  Pre-stop checklist"
echo "============================================"

echo "[1/3] Checking for running tmux sessions..."
SESSIONS=$(ssh_cmd "tmux list-sessions 2>/dev/null" || echo "none")
echo "  Sessions: $SESSIONS"

if [ "$SESSIONS" != "none" ]; then
    echo ""
    echo "WARNING: Training sessions are still running!"
    echo "Make sure to download results before stopping the pod."
    echo ""
    read -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "[2/3] Checking results directory..."
ssh_cmd "du -sh $REMOTE_PROJECT_DIR/results/*/ 2>/dev/null" || echo "  No results found."

echo "[3/3] Reminder: Download results before stopping!"
echo ""
echo "To download results:  ./runpod/sync_from_pod.sh"
echo ""
echo "============================================"
echo "  To stop the pod, go to RunPod Console:"
echo "  https://www.runpod.io/console/pods"
echo "  and click 'Stop' on your pod."
echo ""
echo "  Or use runpodctl (if installed):"
echo "  runpodctl stop pod <POD_ID>"
echo "============================================"
