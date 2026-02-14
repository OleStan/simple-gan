#!/bin/bash
set -euo pipefail

# Launch Improved WGAN v2 experiments - 2 at a time for stability
# This script launches experiments in batches of 2 to prevent resource exhaustion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Latent space sizes to test
LATENT_SIZES=(6 8 10 12 16 24 32 48 64 96 128)
EPOCHS=500
BATCH_AT_A_TIME=2  # Run 2 experiments simultaneously

echo "============================================"
echo "  Improved WGAN v2 Latent Experiments"
echo "  >>> 2 AT A TIME MODE <<<"
echo "============================================"
echo "  Latent sizes: ${LATENT_SIZES[@]}"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_AT_A_TIME concurrent experiments"
echo ""

# Upload the experiment script
echo "[1/2] Uploading experiment script to pod..."
rsync -avz --progress \
    -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
    "$SCRIPT_DIR/../improved_wgan_v2/train_latent_experiment.py" \
    "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/models/improved_wgan_v2/"

echo ""
echo "[2/2] Launching experiments in batches of $BATCH_AT_A_TIME..."
echo ""

TOTAL_LAUNCHED=0
BATCH_NUM=0

# Process experiments in batches
for ((i=0; i<${#LATENT_SIZES[@]}; i+=BATCH_AT_A_TIME)); do
    ((BATCH_NUM++))

    echo "============================================"
    echo "  Batch $BATCH_NUM"
    echo "============================================"

    # Launch current batch
    PIDS=()
    LAUNCHED_IN_BATCH=0

    for ((j=0; j<BATCH_AT_A_TIME && i+j<${#LATENT_SIZES[@]}; j++)); do
        NZ=${LATENT_SIZES[$((i+j))]}

        # Check if already running or completed
        RUNNING=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz $NZ' | grep improved_wgan_v2 | grep -v grep | wc -l" 2>/dev/null || echo "0")
        COMPLETED=$(ssh_cmd "test -f $REMOTE_PROJECT_DIR/results/improved_wgan_v2_nz${NZ}_*/models/netG_final.pt && echo 'yes' || echo 'no'" 2>/dev/null || echo "no")

        if [ "$RUNNING" != "0" ]; then
            echo "⚠️  nz=$NZ already running, skipping..."
            continue
        fi

        if [ "$COMPLETED" = "yes" ]; then
            echo "✅ nz=$NZ already completed, skipping..."
            continue
        fi

        # Launch experiment
        LOG_FILE="/tmp/train_improved_nz${NZ}.log"
        REMOTE_CMD="cd $REMOTE_PROJECT_DIR/models/improved_wgan_v2 && nohup python train_latent_experiment.py --nz $NZ --epochs $EPOCHS > $LOG_FILE 2>&1 &"

        ssh_cmd "$REMOTE_CMD" &
        PIDS+=($!)

        echo "🚀 Launched nz=$NZ (log: $LOG_FILE)"
        ((LAUNCHED_IN_BATCH++))
        ((TOTAL_LAUNCHED++))

        sleep 1
    done

    # Wait for all SSH commands in this batch to complete
    for pid in "${PIDS[@]}"; do
        wait $pid 2>/dev/null || true
    done

    if [ $LAUNCHED_IN_BATCH -eq 0 ]; then
        echo "No new experiments launched in this batch"
        continue
    fi

    # Give processes time to start
    sleep 5

    # Verify they started
    echo ""
    echo "Verifying batch started..."
    for ((j=0; j<BATCH_AT_A_TIME && i+j<${#LATENT_SIZES[@]}; j++)); do
        NZ=${LATENT_SIZES[$((i+j))]}
        RUNNING=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz $NZ' | grep improved_wgan_v2 | grep -v grep | wc -l" 2>/dev/null || echo "0")
        if [ "$RUNNING" != "0" ]; then
            echo "  ✓ nz=$NZ is running"
        fi
    done

    # If not the last batch, wait for current batch to complete
    if [ $((i + BATCH_AT_A_TIME)) -lt ${#LATENT_SIZES[@]} ]; then
        echo ""
        echo "Waiting for batch $BATCH_NUM to complete before starting next batch..."
        echo "Press Ctrl+C to exit (experiments will continue running on RunPod)"

        while true; do
            STILL_RUNNING=0
            for ((j=0; j<BATCH_AT_A_TIME && i+j<${#LATENT_SIZES[@]}; j++)); do
                NZ=${LATENT_SIZES[$((i+j))]}
                RUNNING=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py --nz $NZ' | grep improved_wgan_v2 | grep -v grep | wc -l" 2>/dev/null || echo "0")
                if [ "$RUNNING" != "0" ]; then
                    ((STILL_RUNNING++))
                fi
            done

            if [ $STILL_RUNNING -eq 0 ]; then
                echo "✅ Batch $BATCH_NUM completed!"
                break
            fi

            # Show progress every 30 seconds
            echo "  $(date '+%H:%M:%S') - $STILL_RUNNING experiments still running..."
            sleep 30
        done

        echo ""
    fi
done

echo ""
echo "============================================"
echo "  All Batches Launched!"
echo "============================================"
echo "  Total experiments launched: $TOTAL_LAUNCHED"
echo ""

# Final status
echo "Current GPU status:"
ssh_cmd "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" | \
    awk -F', ' '{printf "  Utilization: %s, Memory: %s / %s, Temp: %s\n", $1, $2, $3, $4}'
echo ""

echo "Running processes:"
RUNNING_COUNT=$(ssh_cmd "ps aux | grep 'train_latent_experiment.py' | grep improved_wgan_v2 | grep -v grep | wc -l" || echo "0")
echo "  Improved WGAN v2: $RUNNING_COUNT experiments"
echo ""

echo "To monitor progress:"
echo "  ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST"
echo "  tail -f /tmp/train_improved_nz32.log"
echo ""
echo "To download results when complete:"
echo "  scp -P $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY -r \\"
echo "    $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/results/improved_wgan_v2_nz* \\"
echo "    ./results/"
echo "============================================"
