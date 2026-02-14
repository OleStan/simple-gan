#!/bin/bash
# Continue launching improved WGAN v2 experiments - 2 at a time
# Waits for previous batch to complete before launching next

RUNPOD_HOST="213.173.108.11"
RUNPOD_PORT="13572"
SSH_KEY="~/.ssh/id_ed25519"

# All latent sizes
ALL_SIZES=(6 8 10 12 16 24 32 48 64 96 128)
BATCH_SIZE=2

echo "============================================"
echo "  Improved WGAN v2 - Continuous Training"
echo "============================================"
echo ""

# Function to check if an experiment is running
is_running() {
    local nz=$1
    ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_HOST \
        "ps aux | grep 'train_latent_experiment.py --nz $nz' | grep improved_wgan_v2 | grep -v grep" >/dev/null 2>&1
    return $?
}

# Function to check if an experiment is completed
is_completed() {
    local nz=$1
    ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_HOST \
        "test -f /workspace/GANs-for-1D-Signal/results/improved_wgan_v2_nz${nz}_*/models/netG_final.pth" >/dev/null 2>&1
    return $?
}

# Function to launch an experiment
launch_experiment() {
    local nz=$1
    echo "🚀 Launching nz=$nz..."
    ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_HOST \
        "cd /workspace/GANs-for-1D-Signal/models/improved_wgan_v2 && nohup python train_latent_experiment.py --nz $nz --epochs 500 > /tmp/train_improved_nz${nz}.log 2>&1 &"
    sleep 2
}

# Process experiments in batches
for ((i=0; i<${#ALL_SIZES[@]}; i+=BATCH_SIZE)); do
    BATCH_NUM=$((i/BATCH_SIZE + 1))
    CURRENT_BATCH=()

    # Collect current batch
    for ((j=0; j<BATCH_SIZE && i+j<${#ALL_SIZES[@]}; j++)); do
        CURRENT_BATCH+=(${ALL_SIZES[$((i+j))]})
    done

    echo ""
    echo "============================================"
    echo "  Batch $BATCH_NUM: nz=${CURRENT_BATCH[@]}"
    echo "============================================"

    # Launch experiments in this batch
    LAUNCHED=0
    for nz in "${CURRENT_BATCH[@]}"; do
        if is_completed $nz; then
            echo "✅ nz=$nz already completed, skipping"
            continue
        fi

        if is_running $nz; then
            echo "⏳ nz=$nz already running"
        else
            launch_experiment $nz
            ((LAUNCHED++))
        fi
    done

    if [ $LAUNCHED -eq 0 ]; then
        echo "No new experiments launched in this batch"
    fi

    # Wait for current batch to complete (unless it's the last batch)
    if [ $((i + BATCH_SIZE)) -lt ${#ALL_SIZES[@]} ]; then
        echo ""
        echo "⏳ Waiting for batch $BATCH_NUM to complete..."
        echo "   Press Ctrl+C to exit (experiments will continue on RunPod)"

        while true; do
            STILL_RUNNING=0

            for nz in "${CURRENT_BATCH[@]}"; do
                if is_running $nz; then
                    ((STILL_RUNNING++))
                fi
            done

            if [ $STILL_RUNNING -eq 0 ]; then
                echo "✅ Batch $BATCH_NUM completed!"
                break
            fi

            # Show progress
            GPU_UTIL=$(ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_HOST \
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null || echo "?")
            echo "   $(date '+%H:%M:%S') - $STILL_RUNNING experiments running | GPU: ${GPU_UTIL}%"

            sleep 30
        done
    else
        echo ""
        echo "✅ Final batch launched!"
    fi
done

echo ""
echo "============================================"
echo "  All Batches Complete!"
echo "============================================"
echo ""

# Final status
echo "Final GPU status:"
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_HOST \
    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"

echo ""
echo "Completed experiments:"
COMPLETED=0
for nz in "${ALL_SIZES[@]}"; do
    if is_completed $nz; then
        echo "  ✅ nz=$nz"
        ((COMPLETED++))
    fi
done

echo ""
echo "Progress: $COMPLETED / ${#ALL_SIZES[@]} experiments completed"
echo ""
echo "To download results:"
echo "  scp -P $RUNPOD_PORT -i $SSH_KEY -r \\"
echo "    root@$RUNPOD_HOST:/workspace/GANs-for-1D-Signal/results/improved_wgan_v2_nz* \\"
echo "    ./results/"
