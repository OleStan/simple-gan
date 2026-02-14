#!/bin/bash

# Quick commands to check RunPod training status

RUNPOD_HOST="213.173.108.11"
RUNPOD_PORT="13572"
RUNPOD_USER="root"
SSH_KEY="~/.ssh/id_ed25519"

echo "============================================"
echo "  RunPod Training Status Check"
echo "============================================"
echo ""

# List all tmux sessions
echo "[1] Active tmux sessions:"
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST "tmux ls 2>/dev/null || echo 'No active sessions'"
echo ""

# Check GPU usage
echo "[2] GPU utilization:"
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
echo ""

# Check running Python processes
echo "[3] Running training processes:"
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST "ps aux | grep 'train_latent_experiment.py' | grep -v grep || echo 'No training processes found'"
echo ""

echo "============================================"
echo "To attach to a specific training session:"
echo "  ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST"
echo "  tmux attach -t dual_wgan_nz6    # or any other session name"
echo ""
echo "To detach from tmux: Ctrl+B, then D"
echo "============================================"
