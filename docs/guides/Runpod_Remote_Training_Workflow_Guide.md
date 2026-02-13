# Runpod Remote Training Workflow Guide

## Goal

Develop locally in your own environment while running GPU training
remotely on Runpod. At the end, you should: - Code locally in your IDE -
Run training on a remote GPU - Persist checkpoints safely - Download
trained models and results back to your local machine

------------------------------------------------------------------------

# Architecture Overview

Local Machine: - Code (IDE, git) - Local experiment tracking - Final
models stored locally

Runpod Pod: - GPU training - Stores checkpoints in persistent storage
(/workspace) - Long-running processes via SSH + tmux

------------------------------------------------------------------------

# Step 1 -- Create Persistent Storage (Network Volume)

Why?\
To prevent losing models when stopping or recreating Pods.

In Runpod Console: 1. Go to **Storage → Network Volumes** 2. Create a
volume 3. Attach it to your Pod

In Secure Cloud, the network volume mounts at:

    /workspace

Everything inside `/workspace` persists across Pod restarts.

------------------------------------------------------------------------

# Step 2 -- Deploy a GPU Pod

1.  Go to **Pods**
2.  Click **Deploy**
3.  Select GPU (A40, A100, etc.)
4.  Attach your Network Volume
5.  Deploy

Wait until the Pod status becomes **Running**.

------------------------------------------------------------------------

# Step 3 -- Connect via SSH

From your local terminal:

    ssh root@<host> -p <port>

If using SSH keys, generate one locally:

    ssh-keygen -t ed25519

Add the public key to your Runpod account settings.

------------------------------------------------------------------------

# Step 4 -- Setup Project on Remote Machine

Navigate to persistent storage:

    cd /workspace

Clone your repository:

    git clone https://github.com/your/repo.git
    cd repo

Create virtual environment:

    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt

------------------------------------------------------------------------

# Step 5 -- Run Training Safely (tmux)

To prevent interruption when SSH disconnects:

    tmux new -s training

Then start training:

    python train.py --config configs/exp1.yaml

Detach safely:

CTRL+B, then D

Reattach later:

    tmux attach -t training

------------------------------------------------------------------------

# Step 6 -- Store Outputs Properly

Always save outputs inside:

    /workspace/repo/outputs/

Example structure:

    outputs/
      checkpoints/
      logs/
      metrics/
      export/

This ensures results survive Pod restarts.

------------------------------------------------------------------------

# Step 7 -- Download Models Back to Local Machine

From your local machine:

    rsync -avz --progress   root@<host>:/workspace/repo/outputs/   ./outputs/

Or use scp:

    scp -r root@<host>:/workspace/repo/outputs ./outputs

------------------------------------------------------------------------

# Optional -- Using runpodctl CLI

Install runpodctl locally to: - Create Pods from CLI - Execute remote
commands - Transfer files programmatically

This is useful for automation and CI/CD pipelines.

------------------------------------------------------------------------

# Recommended Workflow

1.  Code locally
2.  Push to Git
3.  SSH into Pod
4.  Pull latest code
5.  Run training in tmux
6.  Sync results locally
7.  Stop Pod when finished

------------------------------------------------------------------------

# Cost Control Tips

-   Stop Pod when not training
-   Terminate only when you no longer need the disk
-   Monitor GPU usage
-   Keep checkpoints compressed

------------------------------------------------------------------------

# Final Result

You now have:

-   Local development environment
-   Remote GPU execution
-   Persistent checkpoints
-   Local copies of trained models

This workflow scales from single-GPU experiments to production-grade
training pipelines.
