# RunPod Remote GPU Training Guide

Complete guide for running Dual WGAN training on RunPod GPU instances from your local machine.

---

## Prerequisites

### What you need

- **RunPod account** — [runpod.io](https://www.runpod.io)
- **SSH key pair** — ed25519 recommended
- **rsync** — pre-installed on macOS
- **Training data** — `training_data/X_raw.npy` and `training_data/normalization_params.json` present locally

### Generate SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519 -C "runpod"
```

Add the public key (`~/.ssh/id_ed25519.pub`) to your RunPod account:
**RunPod Console → Settings → SSH Public Keys → Add**

---

## Architecture

```text
┌─────────────────────┐          SSH / rsync          ┌─────────────────────┐
│   Local Machine      │ ◄──────────────────────────► │   RunPod Pod (GPU)   │
│                      │                               │                      │
│  • IDE / code edits  │   sync_to_pod.sh ──────────► │  • /workspace/       │
│  • runpod/ scripts   │                               │    └─ GANs-for-1D-  │
│  • final results     │   ◄──────── sync_from_pod.sh │       Signal/        │
│                      │                               │       ├─ training   │
│  results/            │                               │       │   _data/    │
│   └─ dual_wgan_*/    │                               │       └─ results/   │
└─────────────────────┘                               └─────────────────────┘
```

**Key principle**: Code locally, train remotely, download results.

---

## Quick Start (5 steps)

### 1. Create RunPod Pod

1. Go to [RunPod Console → Pods](https://www.runpod.io/console/pods)
2. Click **Deploy**
3. Select a GPU (recommended: **RTX A4000** or better for this project)
4. Select a template with PyTorch (e.g., `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`)
5. **Attach a Network Volume** — this persists data at `/workspace` across pod restarts
6. Deploy and wait for **Running** status

### 2. Configure credentials

```bash
cp runpod/.env.example runpod/.env
```

Edit `runpod/.env` with your pod's SSH details (found in RunPod Console → Pod → Connect → SSH):

```bash
RUNPOD_SSH_HOST=<host-from-runpod>
RUNPOD_SSH_PORT=<port-from-runpod>
RUNPOD_SSH_USER=root
RUNPOD_SSH_KEY=~/.ssh/id_ed25519
```

### 3. Setup remote environment

```bash
./runpod/setup_remote.sh
```

This uploads all source code + training data and installs Python dependencies on the pod.

### 4. Start training

```bash
./runpod/start_training.sh
```

By default, runs `train_dual_wgan_small_latent.py`. To run a different script:

```bash
./runpod/start_training.sh train_improved_wgan_v2.py my_session
```

### 5. Download results

```bash
./runpod/sync_from_pod.sh latest
```

---

## All Available Scripts

| Script | Purpose |
|---|---|
| `runpod/setup_remote.sh` | First-time setup: upload code, data, install deps |
| `runpod/sync_to_pod.sh` | Push code changes (use `--with-data` to include training data) |
| `runpod/start_training.sh [script] [session]` | Launch training in tmux |
| `runpod/monitor_training.sh [session] [mode]` | Check training progress |
| `runpod/sync_from_pod.sh [all\|latest\|models]` | Download results |
| `runpod/ssh_connect.sh` | Open interactive SSH session |
| `runpod/stop_pod.sh` | Pre-stop checklist and instructions |

---

## Typical Workflow

### Day-to-day iteration

```bash
# 1. Edit code locally in your IDE

# 2. Push code changes to pod
./runpod/sync_to_pod.sh

# 3. Start training
./runpod/start_training.sh

# 4. Monitor progress (non-blocking)
./runpod/monitor_training.sh              # last 50 lines
./runpod/monitor_training.sh training gpu  # GPU utilization

# 5. Attach to see live output
./runpod/monitor_training.sh training attach
# (Ctrl+B, then D to detach)

# 6. When done, download results
./runpod/sync_from_pod.sh latest

# 7. Stop pod to save money
./runpod/stop_pod.sh
```

### Running multiple experiments

```bash
./runpod/start_training.sh train_dual_wgan_small_latent.py exp1
./runpod/start_training.sh train_improved_wgan_v2.py exp2

./runpod/monitor_training.sh exp1 tail
./runpod/monitor_training.sh exp2 tail
```

---

## Monitoring Modes

```bash
./runpod/monitor_training.sh [session] tail     # Last 50 lines of output
./runpod/monitor_training.sh [session] attach   # Live interactive view
./runpod/monitor_training.sh [session] gpu      # nvidia-smi
./runpod/monitor_training.sh [session] disk     # Disk usage
```

---

## Download Options

```bash
./runpod/sync_from_pod.sh all      # Everything in results/
./runpod/sync_from_pod.sh latest   # Only the most recent training run
./runpod/sync_from_pod.sh models   # Only model .pth files from latest run
```

---

## Cost Control

- **Stop** the pod when not training (data on Network Volume persists)
- **Terminate** only when you no longer need the volume
- Use `./runpod/monitor_training.sh training gpu` to verify GPU is actually being used
- Use `./runpod/stop_pod.sh` before stopping — it checks for running sessions and reminds you to download results

---

## Troubleshooting

### "Connection refused" on SSH

- Pod may not be running. Check RunPod Console.
- SSH host/port may have changed after a pod restart. Update `runpod/.env`.

### "tmux session already exists"

```bash
# Kill the old session
./runpod/ssh_connect.sh
tmux kill-session -t training
exit
```

### Training data not found on pod

```bash
./runpod/sync_to_pod.sh --with-data
```

### Pod ran out of disk space

```bash
./runpod/monitor_training.sh training disk
./runpod/ssh_connect.sh
# Clean up old results manually
```

---

## File Structure on Pod

```text
/workspace/GANs-for-1D-Signal/
├── training_data/
│   ├── X_raw.npy
│   └── normalization_params.json
├── wgan_dual_profiles.py
├── train_dual_wgan_small_latent.py
├── train_improved_wgan_v2.py
├── ...
└── results/
    └── dual_wgan_<timestamp>/
        ├── models/
        │   ├── netG_epoch_50.pth
        │   ├── netC_epoch_50.pth
        │   ├── netG_final.pth
        │   └── netC_final.pth
        ├── training_images/
        ├── training_history.json
        ├── training_curves.png
        └── config.json
```

Everything under `/workspace/` persists across pod restarts (when using Network Volume).

---

## Security Notes

- `runpod/.env` is gitignored — credentials never enter version control
- SSH keys are used instead of passwords
- The `.env.example` file contains only placeholders
