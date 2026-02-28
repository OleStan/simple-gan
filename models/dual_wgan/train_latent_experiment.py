#!/usr/bin/env python
"""
Train dual-head WGAN with configurable latent space size.
Usage: python train_latent_experiment.py --nz 32 --epochs 500
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
import time
import argparse

from model import DualHeadGenerator, Critic, weights_init, compute_gradient_penalty


class ProfileDataset(Dataset):
    def __init__(self, data_path, norm_params_path):
        self.data = np.load(data_path)

        with open(norm_params_path, 'r') as f:
            self.norm_params = json.load(f)

        self.K = self.norm_params['K']
        self.sigma_min = self.norm_params['sigma_min']
        self.sigma_max = self.norm_params['sigma_max']
        self.mu_min = self.norm_params['mu_min']
        self.mu_max = self.norm_params['mu_max']

        labels_path = Path(data_path).parent / 'y_labels.npy'
        if labels_path.exists():
            self.labels = np.load(labels_path).astype(np.int64)
        else:
            self.labels = np.zeros(len(self.data), dtype=np.int64)

        self.normalized_data = self._normalize(self.data)

    def _normalize(self, data):
        sigma = data[:, :self.K]
        mu = data[:, self.K:2*self.K]

        sigma_norm = 2 * (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min) - 1
        mu_norm = 2 * (mu - self.mu_min) / (self.mu_max - self.mu_min) - 1

        return np.concatenate([sigma_norm, mu_norm], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.normalized_data[idx]), self.labels[idx]


def denormalize_profiles(data, norm_params, K):
    sigma_normalized = data[:, :K]
    mu_normalized = data[:, K:2*K]

    sigma_min = norm_params['sigma_min']
    sigma_max = norm_params['sigma_max']
    mu_min = norm_params['mu_min']
    mu_max = norm_params['mu_max']

    sigma = (sigma_normalized + 1) / 2 * (sigma_max - sigma_min) + sigma_min
    mu = (mu_normalized + 1) / 2 * (mu_max - mu_min) + mu_min

    return sigma, mu


def main():
    parser = argparse.ArgumentParser(description='Train Dual WGAN with configurable latent space')
    parser.add_argument('--nz', type=int, default=32, help='Latent space dimension')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=50, help='Save model every N epochs')
    parser.add_argument('--image_interval', type=int, default=10, help='Generate images every N epochs')
    args = parser.parse_args()

    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  Dual WGAN Latent Space Experiment - nz={args.nz}")
    print(f"{'='*70}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    with open('../../data/training/normalization_params.json', 'r') as f:
        norm_params = json.load(f)

    K = norm_params['K']
    n_classes = norm_params.get('n_classes', 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_tag = os.environ.get('RESULT_TAG', 'two_classes')
    class_tag = f'nc{n_classes}' if n_classes > 1 else 'nc1'
    output_dir = Path(f'../../results/{result_tag}/dual_wgan_nz{args.nz}_{class_tag}_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'training_images').mkdir(exist_ok=True)

    dataset = ProfileDataset('../../data/training/X_raw.npy',
                             '../../data/training/normalization_params.json')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    print(f"\nDataset configuration:")
    print(f"  K (layers): {K}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Output dim: {2*K}")

    netG = DualHeadGenerator(nz=args.nz, K=K, n_classes=n_classes).to(device)
    netC = Critic(input_dim=2*K, n_classes=n_classes).to(device)

    netG.apply(weights_init)
    netC.apply(weights_init)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=args.lr, betas=(0.5, 0.999))

    scaler = GradScaler()

    print(f"\nModel architecture:")
    print(f"  Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"  Critic parameters: {sum(p.numel() for p in netC.parameters()):,}")
    print(f"\nTraining configuration:")
    print(f"  Latent dimension (nz): {args.nz}")
    print(f"  nz / (2*K) ratio: {args.nz / (2*K):.3f}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Mixed precision: True (AMP)")
    print(f"\nOutput directory: {output_dir}")

    config = {
        'nz': args.nz,
        'K': K,
        'output_dim': 2*K,
        'nz_to_output_ratio': args.nz / (2*K),
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'n_epochs': args.epochs,
        'lr': args.lr,
        'n_critic': 5,
        'lambda_gp': 10,
        'n_classes': n_classes,
        'optimizations': 'AMP + large_batch + multi_worker + pin_memory',
        'experiment': f'latent_space_nz{args.nz}',
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    history = {
        'loss_C': [],
        'loss_G': [],
        'wasserstein_distance': [],
        'gradient_penalty': [],
        'epoch_time': []
    }

    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}\n")

    n_critic = 5
    lambda_gp = 10

    for epoch in range(args.epochs):
        epoch_start = time.time()

        for i, (real_data, real_labels) in enumerate(dataloader):
            real_data = real_data.to(device, non_blocking=True)
            real_labels = real_labels.to(device, non_blocking=True)
            batch_size_current = real_data.size(0)

            gen_labels = None
            if n_classes > 1:
                gen_labels = torch.randint(0, n_classes, (batch_size_current,), device=device)

            # Train Critic
            for _ in range(n_critic):
                netC.zero_grad(set_to_none=True)

                with autocast():
                    noise = torch.randn(batch_size_current, args.nz, device=device)
                    fake_data, _, _ = netG(noise, gen_labels)

                    critic_real = netC(real_data, real_labels if n_classes > 1 else None)
                    critic_fake = netC(fake_data.detach(), gen_labels)

                    gp = compute_gradient_penalty(netC, real_data, fake_data.detach(), device,
                                                  labels=real_labels if n_classes > 1 else None)

                    loss_C = -torch.mean(critic_real) + torch.mean(critic_fake) + lambda_gp * gp

                scaler.scale(loss_C).backward()
                scaler.step(optimizerC)
                scaler.update()

            # Train Generator
            netG.zero_grad(set_to_none=True)

            with autocast():
                noise = torch.randn(batch_size_current, args.nz, device=device)
                fake_data, _, _ = netG(noise, gen_labels)

                critic_fake = netC(fake_data, gen_labels)
                loss_G = -torch.mean(critic_fake)

            scaler.scale(loss_G).backward()
            scaler.step(optimizerG)
            scaler.update()

            wasserstein_distance = torch.mean(critic_real) - torch.mean(critic_fake)

            if i % 5 == 0:
                print(f'[{epoch+1}/{args.epochs}][{i}/{len(dataloader)}] '
                      f'Loss_C: {loss_C.item():.4f} Loss_G: {loss_G.item():.4f} '
                      f'W_dist: {wasserstein_distance.item():.4f} GP: {gp.item():.4f}')

        epoch_time = time.time() - epoch_start

        history['loss_C'].append(loss_C.item())
        history['loss_G'].append(loss_G.item())
        history['wasserstein_distance'].append(wasserstein_distance.item())
        history['gradient_penalty'].append(gp.item())
        history['epoch_time'].append(epoch_time)

        avg_epoch_time = sum(history['epoch_time']) / len(history['epoch_time'])
        remaining_epochs = args.epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60

        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s "
              f"(avg: {avg_epoch_time:.2f}s, ETA: {eta_minutes:.1f} min)\n")

        if (epoch + 1) % args.image_interval == 0:
            with torch.no_grad():
                noise = torch.randn(16, args.nz, device=device)
                vis_labels = None
                if n_classes > 1:
                    vis_labels = torch.arange(16, device=device) % n_classes
                fake_data, fake_sigma, fake_mu = netG(noise, vis_labels)
                fake_data_np = fake_data.cpu().numpy()

                sigma_denorm, mu_denorm = denormalize_profiles(fake_data_np, norm_params, K)

                fig, axes = plt.subplots(4, 4, figsize=(16, 12))
                for idx in range(16):
                    ax = axes[idx // 4, idx % 4]
                    ax2 = ax.twinx()

                    ax.plot(sigma_denorm[idx], 'b-', linewidth=1, alpha=0.7)
                    ax2.plot(mu_denorm[idx], 'r-', linewidth=1, alpha=0.7)

                    ax.set_ylabel('σ (S/m)', color='b', fontsize=8)
                    ax2.set_ylabel('μᵣ', color='r', fontsize=8)
                    ax.tick_params(axis='y', labelcolor='b', labelsize=7)
                    ax2.tick_params(axis='y', labelcolor='r', labelsize=7)
                    ax.tick_params(axis='x', labelsize=7)
                    ax.grid(True, alpha=0.3)

                plt.suptitle(f'Generated Profiles (nz={args.nz}) - Epoch {epoch+1}',
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_dir / 'training_images' / f'epoch_{epoch+1:04d}.png',
                           dpi=150, bbox_inches='tight')
                plt.close()

        if (epoch + 1) % args.save_interval == 0:
            torch.save(netG.state_dict(), output_dir / 'models' / f'netG_epoch_{epoch+1}.pth')
            torch.save(netC.state_dict(), output_dir / 'models' / f'netC_epoch_{epoch+1}.pth')

            with open(output_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)

    # Save final models
    torch.save(netG.state_dict(), output_dir / 'models' / 'netG_final.pth')
    torch.save(netC.state_dict(), output_dir / 'models' / 'netC_final.pth')

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history['loss_C'])
    axes[0, 0].set_title('Critic Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['loss_G'])
    axes[0, 1].set_title('Generator Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['wasserstein_distance'])
    axes[1, 0].set_title('Wasserstein Distance')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['gradient_penalty'])
    axes[1, 1].set_title('Gradient Penalty')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Training Curves (nz={args.nz})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    total_time = sum(history['epoch_time'])
    avg_epoch_time = total_time / len(history['epoch_time'])

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Latent dimension: {args.nz}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    print(f"\nModels saved to: {output_dir / 'models'}/")
    print(f"Training history saved to: {output_dir / 'training_history.json'}")
    print(f"Training curves saved to: {output_dir / 'training_curves.png'}")


if __name__ == '__main__':
    main()
