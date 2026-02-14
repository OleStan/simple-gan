#!/bin/bash

# Monitor all running latent space experiments
# Usage: ./monitor_experiments.sh

RUNPOD_HOST="213.173.108.11"
RUNPOD_PORT="13572"
RUNPOD_USER="root"
SSH_KEY="~/.ssh/id_ed25519"

clear
echo "============================================"
echo "  Dual WGAN Experiments Monitor"
echo "============================================"
echo ""

# Count running experiments
RUNNING_COUNT=$(ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST \
  "ps aux | grep 'train_latent_experiment.py' | grep -v grep | grep 'python' | wc -l" 2>/dev/null || echo "0")

echo "Running experiments: $RUNNING_COUNT / 11"
echo ""

# GPU utilization
echo "GPU Status:"
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST \
  "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader" 2>/dev/null | \
  awk -F', ' '{printf "  Utilization: %s\n  Memory: %s / %s\n  Temperature: %s\n  Power: %s\n", $1, $2, $3, $4, $5}'
echo ""

# Check progress by looking at latest logs
echo "Latest progress (checking a few logs)..."
echo ""

for NZ in 6 32 128; do
  LOG_FILE="/tmp/train_nz${NZ}.log"
  LAST_LINE=$(ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST \
    "tail -5 $LOG_FILE 2>/dev/null | grep -E 'Epoch \[|Loss_D|Loss_G' | tail -1" || echo "")

  if [ -n "$LAST_LINE" ]; then
    echo "  nz=$NZ: $LAST_LINE"
  fi
done
echo ""

# Check for completed experiments
echo "Checking for completed experiments..."
COMPLETED=$(ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST \
  "ls /workspace/GANs-for-1D-Signal/results/dual_wgan_nz*/models/netG_final.pth 2>/dev/null | wc -l" || echo "0")
echo "  Completed: $COMPLETED / 11"
echo ""

if [ "$RUNNING_COUNT" -eq "0" ] && [ "$COMPLETED" -gt "0" ]; then
  echo "============================================"
  echo "  ✅ ALL EXPERIMENTS COMPLETED!"
  echo "============================================"
  echo ""
  echo "To download results:"
  echo "  ./models/runpod/sync_from_pod.sh all"
  echo ""
  exit 0
fi

if [ "$RUNNING_COUNT" -eq "0" ]; then
  echo "⚠️  No experiments running!"
  echo ""
  exit 1
fi

echo "============================================"
echo "Training in progress..."
echo ""
echo "To monitor a specific experiment log:"
echo "  ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST"
echo "  tail -f /tmp/train_nz32.log"
echo ""
echo "To check again: ./monitor_experiments.sh"
echo "============================================"
