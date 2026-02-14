#!/bin/bash
set -euo pipefail

# Launch all latent experiments in parallel (simpler version without tmux)
# Logs are saved to /tmp/train_nz*.log on RunPod

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Latent space sizes to test
LATENT_SIZES=(6 8 10 12 16 24 32 48 64 96 128)
EPOCHS=500

echo "============================================"
echo "  Launching ALL Latent Experiments"
echo "  >>> PARALLEL MODE <<<"
echo "============================================"
echo "  Latent sizes: ${LATENT_SIZES[@]}"
echo "  Epochs: $EPOCHS"
echo ""

# First, kill the test process if running
echo "Cleaning up test process..."
ssh_cmd "pkill -f 'train_latent_experiment.py --nz 6' || true"
sleep 2

LAUNCHED=0

for NZ in "${LATENT_SIZES[@]}"; do
    # Check if already running
    RUNNING=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz $NZ' | grep -v grep | wc -l" || echo "0")

    if [ "$RUNNING" != "0" ]; then
        echo "⚠️  Experiment nz=$NZ already running, skipping..."
        continue
    fi

    # Launch training in background
    LOG_FILE="/tmp/train_nz${NZ}.log"
    REMOTE_CMD="cd $REMOTE_PROJECT_DIR/models/dual_wgan && nohup python train_latent_experiment.py --nz $NZ --epochs $EPOCHS > $LOG_FILE 2>&1 &"

    ssh_cmd "$REMOTE_CMD"
    echo "✓ Launched nz=$NZ (log: $LOG_FILE)"
    ((LAUNCHED++))

    # Small delay to avoid overwhelming the system
    sleep 0.5
done

echo ""
echo "============================================"
echo "  Launch Complete!"
echo "============================================"
echo "  Launched: $LAUNCHED experiments"
echo ""

# Wait a moment for processes to start
sleep 3

echo "Checking status..."
echo ""

# Count running processes
TOTAL_RUNNING=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py' | grep -v grep | wc -l" || echo "0")
echo "Running training processes: $TOTAL_RUNNING"
echo ""

# Show GPU usage
echo "GPU Status:"
ssh_cmd "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
echo ""

echo "============================================"
echo "Estimated completion: ~6 minutes"
echo ""
echo "To monitor progress:"
echo "  ./check_runpod_training.sh"
echo ""
echo "To view a specific log (on RunPod):"
echo "  ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST"
echo "  tail -f /tmp/train_nz32.log"
echo ""
echo "To stop all experiments:"
echo "  ssh ... 'pkill -f train_latent_experiment.py'"
echo "============================================"
