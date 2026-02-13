#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="$SCRIPT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found."
    echo "Copy .env.example to .env and fill in your credentials:"
    echo "  cp runpod/.env.example runpod/.env"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

RUNPOD_SSH_HOST="${RUNPOD_SSH_HOST:?'RUNPOD_SSH_HOST not set in .env'}"
RUNPOD_SSH_PORT="${RUNPOD_SSH_PORT:?'RUNPOD_SSH_PORT not set in .env'}"
RUNPOD_SSH_USER="${RUNPOD_SSH_USER:-root}"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-~/.ssh/id_ed25519}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/workspace/GANs-for-1D-Signal}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$PROJECT_ROOT}"

ssh_cmd() {
    ssh -p "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST" "$@"
}

scp_download() {
    local remote_path="$1"
    local local_path="$2"
    scp -P "$RUNPOD_SSH_PORT" -i "$RUNPOD_SSH_KEY" -r \
        "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$remote_path" "$local_path"
}

rsync_download() {
    local remote_path="$1"
    local local_path="$2"
    rsync -avz --progress \
        -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
        "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$remote_path" "$local_path"
}

rsync_upload() {
    local local_path="$1"
    local remote_path="$2"
    rsync -avz --progress \
        -e "ssh -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY" \
        "$local_path" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:$remote_path"
}
