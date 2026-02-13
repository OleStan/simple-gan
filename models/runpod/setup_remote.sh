#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "============================================"
echo "  RunPod Remote Setup"
echo "============================================"

echo "[1/4] Creating project directory on pod..."
ssh_cmd "mkdir -p $REMOTE_PROJECT_DIR"

echo "[2/4] Syncing project files to pod..."
rsync -avz --progress \
    --exclude '.git' \
    --exclude 'results/' \
    --exclude 'nets/' \
    --exclude 'img/' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'runpod/.env' \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$LOCAL_PROJECT_DIR/" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/"

echo "[3/4] Uploading training data..."
rsync -avz --progress \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$LOCAL_PROJECT_DIR/training_data/" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/training_data/"

echo "[4/4] Installing dependencies on pod..."
ssh_cmd "cd $REMOTE_PROJECT_DIR && \
    pip install -U pip && \
    pip install -r runpod/requirements.txt"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo "  Remote project: $REMOTE_PROJECT_DIR"
echo "  Next step: ./runpod/start_training.sh"
