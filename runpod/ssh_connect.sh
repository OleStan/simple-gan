#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "Connecting to RunPod pod..."
ssh -t -p "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST" \
    "cd $REMOTE_PROJECT_DIR && bash"
