#!/bin/bash
# Download all dual WGAN experiment results from RunPod

RUNPOD_HOST="213.173.108.11"
RUNPOD_PORT="13572"
SSH_KEY="~/.ssh/id_ed25519"

echo "Downloading dual WGAN results..."

for NZ in 6 8 10 12 16 24 32 48 64 96 128; do
  echo "Downloading nz=$NZ..."

  # Find the directory on RunPod
  REMOTE_DIR=$(ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_HOST \
    "ls -d /workspace/GANs-for-1D-Signal/results/dual_wgan_nz${NZ}_202602* 2>/dev/null | head -1")

  if [ -n "$REMOTE_DIR" ]; then
    LOCAL_DIR="./results/dual_wgan_nz${NZ}"
    mkdir -p "$LOCAL_DIR"

    rsync -az --progress -e "ssh -p $RUNPOD_PORT -i $SSH_KEY" \
      root@$RUNPOD_HOST:$REMOTE_DIR/ \
      $LOCAL_DIR/

    echo "✓ Downloaded nz=$NZ to $LOCAL_DIR"
  else
    echo "✗ No directory found for nz=$NZ"
  fi
done

echo ""
echo "Download complete!"
echo "Downloaded directories:"
ls -d results/dual_wgan_nz*/ 2>/dev/null | wc -l
