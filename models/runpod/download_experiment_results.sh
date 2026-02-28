#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "============================================"
echo "  Downloading Experiment Results from RunPod"
echo "============================================"

# Download Dodd-GAN results
echo "Checking for Dodd-GAN results..."
rsync -avz --progress \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/experiments/inverse_solver/dodd_gan/results/" \
    "$LOCAL_PROJECT_DIR/experiments/inverse_solver/dodd_gan/results/"

# Download GAN results
echo "Checking for GAN Noise results..."
rsync -avz --progress \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/experiments/inverse_solver/gan/results/" \
    "$LOCAL_PROJECT_DIR/experiments/inverse_solver/gan/results/"

echo ""
echo "Download complete!"
