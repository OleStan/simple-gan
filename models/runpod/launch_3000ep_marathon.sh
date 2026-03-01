#!/bin/bash
set -euo pipefail

# Marathon script: Train both 2-class and 4-class datasets for 3000 epochs
# sequentially to avoid GPU OOM and data conflicts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

EPOCHS=3000
LATENT_SIZES=(10 20 30)

run_experiment() {
    local dataset_script="$1"
    local tag="$2"
    
    echo "############################################################"
    echo "  STARTING MARATHON PHASE: $tag ($EPOCHS epochs)"
    echo "############################################################"
    
    # 1. Generate data locally
    echo "--- [1/4] Generating $tag data locally ---"
    python "$LOCAL_PROJECT_DIR/$dataset_script"
    
    # 2. Sync to pod (using SKIP_SYNC logic in launcher)
    echo "--- [2/4] Syncing to pod ---"
    rsync -avz --progress --delete -O \
        --exclude '.git' --exclude 'results/' --exclude '__pycache__' \
        -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
        "$LOCAL_PROJECT_DIR/" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/" || true

    # 3. Launch training on pod using tmux
    # We'll use a modified version of the launch script logic directly here
    echo "--- [3/4] Launching 6 models in tmux sessions ---"
    
    # Clean up existing tmux sessions
    ssh_cmd "tmux kill-server 2>/dev/null || true"
    
    for nz in "${LATENT_SIZES[@]}"; do
        for model in "dual_wgan" "improved_wgan_v2"; do
            local session="m_${tag}_${model}_nz${nz}"
            local script_rel="models/${model}/train_latent_experiment.py"
            local log="/tmp/${session}.log"
            
            echo "  🚀 Launching ${model} nz=${nz}"
            ssh_cmd "tmux new-session -d -s ${session} \"cd $REMOTE_PROJECT_DIR/${script_rel%/*} && env RESULT_TAG=${tag} python train_latent_experiment.py --nz ${nz} --epochs ${EPOCHS} 2>&1 | tee ${log}\""
        done
    done
    
    echo "Waiting for this phase to complete..."
    while true; do
        STILL=$(ssh_cmd "tmux ls 2>/dev/null | grep 'm_${tag}' | wc -l" || echo "0")
        [ "$STILL" -eq 0 ] && break
        echo "  $(date '+%H:%M:%S') — $STILL training sessions active..."
        sleep 300 # check every 5 mins for the marathon
    done
    
    # 4. Run metrics and download
    echo "--- [4/4] Finalizing Phase: $tag ---"
    # We use the reporter logic
    ssh_cmd "export PYTHONPATH=$REMOTE_PROJECT_DIR; cd $REMOTE_PROJECT_DIR && python scripts/reports/run_marathon_reports.py $tag"
    
    echo "Downloading results for $tag..."
    mkdir -p "$LOCAL_PROJECT_DIR/results/$tag"
    mkdir -p "$LOCAL_PROJECT_DIR/models/registry/$tag"
    
    # Download everything including the promoted best model
    rsync -avz --progress \
        -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
        "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/results/$tag/" \
        "$LOCAL_PROJECT_DIR/results/$tag/"
        
    rsync -avz --progress \
        -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
        "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/models/registry/$tag/" \
        "$LOCAL_PROJECT_DIR/models/registry/$tag/"
}

# --- Execution ---
run_experiment "scripts/data_generation/gen_2class_matched.py" "two_classes_3000ep"
run_experiment "scripts/data_generation/gen_4class_dispersion.py" "four_classes_3000ep"

echo "MARATHON COMPLETE!"
