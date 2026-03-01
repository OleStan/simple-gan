#!/bin/bash
set -euo pipefail

# Full pipeline: sync → train in tmux sessions → reports → download.
# Optimized for 3000 epoch marathon.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

LATENT_SIZES=(10 20 30)
EPOCHS="${EPOCHS:-1000}"
RESULT_TAG="${RESULT_TAG:-four_classes_v1}"
SKIP_SYNC="${SKIP_SYNC:-0}"

echo "============================================"
echo "  Profile GAN Experiments (TMUX Mode)"
echo "  Models: dual_wgan + improved_wgan_v2"
echo "  Latent sizes: ${LATENT_SIZES[*]}"
echo "  Epochs: $EPOCHS"
echo "  Result folder: results/${RESULT_TAG}/"
echo "============================================"

# 1. Sync
if [ "$SKIP_SYNC" != "1" ]; then
    echo "=== [1/5] Syncing code and data to pod ==="
    rsync -avz --progress --delete -O \
        --exclude '.git' --exclude 'results/' --exclude '__pycache__' \
        -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
        "$LOCAL_PROJECT_DIR/" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/" || true
else
    echo "=== [1/5] Skipping sync ==="
fi

# 2. Deps
echo "=== [2/5] Installing dependencies ==="
if [ "$SKIP_SYNC" != "1" ]; then
    ssh_cmd "pip install -q -r $REMOTE_PROJECT_DIR/models/runpod/requirements.txt"
    ssh_cmd "apt-get update && apt-get install -y tmux" || true
else
    pip install -q -r "$REMOTE_PROJECT_DIR/models/runpod/requirements.txt"
fi

# 3. Train
echo "=== [3/5] Launching training in tmux sessions ==="
run_job() {
    local m="$1"
    local nz="$2"
    local s="train_${m}_nz${nz}"
    local log="/tmp/${s}.log"
    local rel="models/${m}/train_latent_experiment.py"
    local cmd="cd $REMOTE_PROJECT_DIR/${rel%/*} && env RESULT_TAG=${RESULT_TAG} python train_latent_experiment.py --nz ${nz} --epochs ${EPOCHS} 2>&1 | tee ${log}"

    echo "  🚀 Launching ${m} nz=${nz} in session ${s}"
    if [ "$SKIP_SYNC" != "1" ]; then
        ssh_cmd "tmux kill-session -t $s 2>/dev/null || true"
        ssh_cmd "tmux new-session -d -s $s \"$cmd\""
    else
        tmux kill-session -t "$s" 2>/dev/null || true
        tmux new-session -d -s "$s" "$cmd"
    fi
}

for nz in "${LATENT_SIZES[@]}"; do
    run_job "dual_wgan" "$nz"
    run_job "improved_wgan_v2" "$nz"
done

echo "Waiting for completion..."
while true; do
    if [ "$SKIP_SYNC" != "1" ]; then
        COUNT=$(ssh_cmd "tmux ls 2>/dev/null | grep 'train_' | wc -l" || echo "0")
    else
        COUNT=$(tmux ls 2>/dev/null | grep 'train_' | wc -l" || echo "0")
    fi
    [ "$COUNT" -eq 0 ] && break
    echo "  $(date '+%H:%M:%S') — $COUNT tmux sessions active..."
    sleep 60
done

# 4. Reports
echo "=== [4/5] Running reports ==="
# Instead of complex string passing, we call the marathon reporter script directly
if [ "$SKIP_SYNC" != "1" ]; then
    ssh_cmd "export PYTHONPATH=$REMOTE_PROJECT_DIR; cd $REMOTE_PROJECT_DIR && python scripts/reports/run_marathon_reports.py $RESULT_TAG"
else
    export PYTHONPATH=$REMOTE_PROJECT_DIR
    python "$REMOTE_PROJECT_DIR/scripts/reports/run_marathon_reports.py" "$RESULT_TAG"
fi

# 5. Download
if [ "$SKIP_SYNC" != "1" ]; then
    echo "=== [5/5] Downloading results ==="
    mkdir -p "$LOCAL_PROJECT_DIR/results/${RESULT_TAG}"
    rsync -avz --progress \
        -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
        "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$REMOTE_PROJECT_DIR/results/${RESULT_TAG}/" \
        "$LOCAL_PROJECT_DIR/results/${RESULT_TAG}/"
fi

echo "DONE."
