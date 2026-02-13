#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "============================================"
echo "  Downloading results from RunPod"
echo "============================================"

DOWNLOAD_TARGET="${1:-all}"

case "$DOWNLOAD_TARGET" in
    models)
        echo "Downloading trained models only..."
        ssh_cmd "ls $REMOTE_PROJECT_DIR/results/" 2>/dev/null
        LATEST_RUN=$(ssh_cmd "ls -td $REMOTE_PROJECT_DIR/results/dual_wgan_*/ 2>/dev/null | head -1")
        if [ -z "$LATEST_RUN" ]; then
            echo "ERROR: No training results found on pod."
            exit 1
        fi
        echo "Latest run: $LATEST_RUN"
        LOCAL_DEST="$LOCAL_PROJECT_DIR/results/$(basename "$LATEST_RUN")"
        mkdir -p "$LOCAL_DEST/models"
        rsync_download "$LATEST_RUN/models/" "$LOCAL_DEST/models/"
        ;;
    latest)
        echo "Downloading latest training run..."
        LATEST_RUN=$(ssh_cmd "ls -td $REMOTE_PROJECT_DIR/results/dual_wgan_*/ 2>/dev/null | head -1")
        if [ -z "$LATEST_RUN" ]; then
            echo "ERROR: No training results found on pod."
            exit 1
        fi
        echo "Latest run: $LATEST_RUN"
        LOCAL_DEST="$LOCAL_PROJECT_DIR/results/$(basename "$LATEST_RUN")"
        mkdir -p "$LOCAL_DEST"
        rsync_download "$LATEST_RUN/" "$LOCAL_DEST/"
        ;;
    all)
        echo "Downloading all results..."
        mkdir -p "$LOCAL_PROJECT_DIR/results"
        rsync_download "$REMOTE_PROJECT_DIR/results/" "$LOCAL_PROJECT_DIR/results/"
        ;;
    *)
        echo "Usage: $0 [all|latest|models]"
        echo "  all     - Download all results (default)"
        echo "  latest  - Download only the latest training run"
        echo "  models  - Download only models from the latest run"
        exit 1
        ;;
esac

echo ""
echo "Download complete!"
