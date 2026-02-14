#!/bin/bash
# Launch remaining improved WGAN v2 experiments

echo "Launching remaining improved WGAN v2 experiments..."

for NZ in 10 12 16 24 32 48 64 96 128; do
  echo "  🚀 Launching nz=$NZ..."
  ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 \
    "cd /workspace/GANs-for-1D-Signal/models/improved_wgan_v2 && nohup python train_latent_experiment.py --nz $NZ --epochs 500 > /tmp/train_improved_nz${NZ}.log 2>&1 &" &
  sleep 0.5
done

wait

echo ""
echo "✅ All remaining experiments launched!"
echo ""
echo "Checking status..."
sleep 3

ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 \
  "ps aux | grep 'python train_latent_experiment.py' | grep improved_wgan_v2 | grep -v grep | wc -l" | \
  xargs echo "Running improved WGAN v2 processes:"

ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 \
  "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
