#!/bin/bash
set -euo pipefail

# Launch optimized NZ=20 training for 1000 epochs with matched-start data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

NZ=20
EPOCHS=1000
RESULT_TAG="optimized_v2_nz20"

echo "============================================"
echo "  Optimized Phase 2 GAN Training"
echo "  Model: improved_wgan_v2"
echo "  Latent size: $NZ"
echo "  Epochs: $EPOCHS"
echo "  Results: results/$RESULT_TAG/"
echo "============================================"

# 1. Sync local data (the new V2 data) to pod
echo "=== [1/4] Syncing new V2 data to pod ==="
rsync -avz --progress --delete \
    --no-perms --no-owner --no-group \
    --exclude '.git' \
    --exclude 'results/' \
    --exclude '__pycache__' \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$LOCAL_PROJECT_DIR/" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/"

# 2. Launch training in background using nohup
echo "=== [2/4] Launching training (1000 epochs) ==="
LOG_FILE="/tmp/train_optimized_nz20.log"
ssh_cmd "cd $REMOTE_PROJECT_DIR/models/improved_wgan_v2 && \
         nohup env RESULT_TAG=$RESULT_TAG python train_latent_experiment.py --nz $NZ --epochs $EPOCHS > $LOG_FILE 2>&1 &"

echo "Training launched!"
echo "Log file on pod: $LOG_FILE"
echo ""
echo "To monitor progress:"
echo "  ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST 'tail -f $LOG_FILE'"
echo ""
echo "Next step: Wait for completion, then run quality metrics."
