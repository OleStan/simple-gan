#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "============================================"
echo "  Syncing code to RunPod"
echo "============================================"

SYNC_DATA=false
for arg in "$@"; do
    case $arg in
        --with-data) SYNC_DATA=true ;;
    esac
done

echo "[1/2] Uploading source code..."
rsync -avz --progress \
    --exclude '.git' \
    --exclude 'results/' \
    --exclude 'nets/' \
    --exclude 'img/' \
    --exclude 'training_data/' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'runpod/.env' \
    --exclude 'additional_information/' \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$LOCAL_PROJECT_DIR/" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/"

if [ "$SYNC_DATA" = true ]; then
    echo "[2/2] Uploading training data..."
    rsync -avz --progress \
        -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
        "$LOCAL_PROJECT_DIR/training_data/" \
        "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/training_data/"
else
    echo "[2/2] Skipping training data (use --with-data to include)"
fi

echo ""
echo "Sync complete!"
