#!/usr/bin/env python
"""
Train Improved WGAN v2 with configurable latent space dimension.
Optimized for GPU with multi-worker data loading and mixed precision.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler

from model import (
    ConditionalConv1DGenerator, SpectralNormConv1DCritic,
    PhysicsInformedLossV2, ProfileQualityMetrics,
    weights_init, compute_gradient_penalty
)


class ProfileDataset(torch.utils.data.Dataset):
    """Dataset for σ and μ profiles."""

    def __init__(self, data_path, normalize=True):
        self.X = np.load(data_path)

        with open(Path(data_path).parent / 'normalization_params.json', 'r') as f:
            self.norm_params = json.load(f)

        self.K = self.norm_params['K']
        self.normalize = normalize

        labels_path = Path(data_path).parent / 'y_labels.npy'
        if labels_path.exists():
            self.labels = np.load(labels_path).astype(np.int64)
        else:
            self.labels = np.zeros(len(self.X), dtype=np.int64)

        if normalize:
            self.X_normalized = self.X.copy()

            sigma_data = self.X[:, :self.K]
            mu_data = self.X[:, self.K:2*self.K]

            sigma_min = self.norm_params['sigma_min']
            sigma_max = self.norm_params['sigma_max']
            mu_min = self.norm_params['mu_min']
            mu_max = self.norm_params['mu_max']

            self.X_normalized[:, :self.K] = 2 * (sigma_data - sigma_min) / (sigma_max - sigma_min) - 1
            self.X_normalized[:, self.K:2*self.K] = 2 * (mu_data - mu_min) / (mu_max - mu_min) - 1

            self.data = self.X_normalized
        else:
            self.data = self.X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), self.labels[idx]


def main():
    parser = argparse.ArgumentParser(description='Train Improved WGAN v2 with configurable latent space')
    parser.add_argument('--nz', type=int, default=100, help='Latent space dimension')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=5e-5, help='Generator learning rate')
    parser.add_argument('--lr_c', type=float, default=2e-4, help='Critic learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    args = parser.parse_args()

    # Training hyperparameters
    n_critic = 5
    lambda_gp = 10
    nz = args.nz
    epoch_num = args.epochs
    batch_size = args.batch_size
    lr_g = args.lr_g
    lr_c = args.lr_c
    num_workers = args.num_workers

    # Model configuration
    use_spectral_norm = True
    use_gradient_penalty = False
    max_grad_norm = 1.0

    # Physics loss scheduling
    lambda_physics_start = 0.0
    lambda_physics_end = 0.2
    physics_warmup_epochs = 100

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"Using device: {device}")

    # Load normalization parameters
    with open('../../data/training/normalization_params.json', 'r') as f:
        norm_params_temp = json.load(f)
    K = norm_params_temp['K']
    n_classes = norm_params_temp.get('n_classes', 1)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_tag = os.environ.get('RESULT_TAG', 'two_classes')
    class_tag = f'nc{n_classes}' if n_classes > 1 else 'nc1'
    output_dir = f'../../results/{result_tag}/improved_wgan_v2_nz{nz}_{class_tag}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/training_images', exist_ok=True)

    print("\n" + "="*70)
    print(f"  Improved WGAN v2 Latent Space Experiment - nz={nz}")
    print("="*70)

    print(f"\nDataset configuration:")
    print(f"  K (layers): {K}")
    print(f"  Output dim: {2*K}")

    # Load dataset with optimizations
    dataset = ProfileDataset('../../data/training/X_raw.npy', normalize=True)
    print(f"  Samples: {len(dataset)}")

    effective_batch = min(batch_size, len(dataset))
    if effective_batch != batch_size:
        print(f"  NOTE: batch_size capped to dataset size ({effective_batch})")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )

    # Initialize models
    netG = ConditionalConv1DGenerator(nz=nz, K=K, conditional=(n_classes > 1), n_classes=n_classes).to(device)
    netG.apply(weights_init)

    netC = SpectralNormConv1DCritic(K=K, use_spectral_norm=use_spectral_norm, n_classes=n_classes).to(device)
    if not use_spectral_norm:
        netC.apply(weights_init)

    physics_loss_fn = PhysicsInformedLossV2(
        lambda_smooth=0.05,
        lambda_bounds=0.02,
        lambda_monotonic=0.0
    )

    print(f"\nModel architecture:")
    print(f"  Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"  Critic parameters: {sum(p.numel() for p in netC.parameters()):,}")

    print(f"\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epoch_num}")
    print(f"  Learning rate (G): {lr_g}")
    print(f"  Learning rate (C): {lr_c}")
    print(f"  Data workers: {num_workers}")
    print(f"  Spectral norm: {use_spectral_norm}")
    print(f"  Gradient clipping: {max_grad_norm}")
    print(f"  Physics λ: {lambda_physics_start} → {lambda_physics_end} (warmup: {physics_warmup_epochs} epochs)")

    # Fixed noise for visualization
    fixed_noise = torch.randn(16, nz, device=device)

    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optimizerC = optim.Adam(netC.parameters(), lr=lr_c, betas=(0.0, 0.9))

    # Mixed precision scaler
    scaler = GradScaler()

    # Save configuration
    config = {
        'nz': nz,
        'n_critic': n_critic,
        'lambda_gp': lambda_gp,
        'lr_g': lr_g,
        'lr_c': lr_c,
        'epoch_num': epoch_num,
        'batch_size': batch_size,
        'K': K,
        'use_spectral_norm': use_spectral_norm,
        'use_gradient_penalty': use_gradient_penalty,
        'max_grad_norm': max_grad_norm,
        'lambda_physics_start': lambda_physics_start,
        'lambda_physics_end': lambda_physics_end,
        'physics_warmup_epochs': physics_warmup_epochs,
        'device': str(device),
        'num_workers': num_workers,
        'n_classes': n_classes
    }

    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    with open(f'{output_dir}/normalization_params.json', 'w') as f:
        json.dump(dataset.norm_params, f, indent=2)

    # Training history
    training_history = {
        'loss_C': [],
        'loss_G': [],
        'loss_G_adv': [],
        'loss_G_physics': [],
        'wasserstein_distance': [],
        'gradient_penalty': [],
        'physics_weight': []
    }

    print("\nStarting training...\n")

    import time
    start_time = time.time()

    # Training loop
    for epoch in range(epoch_num):
        # Physics loss scheduling
        if epoch < physics_warmup_epochs:
            lambda_physics = lambda_physics_start + (lambda_physics_end - lambda_physics_start) * (epoch / physics_warmup_epochs)
        else:
            lambda_physics = lambda_physics_end

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
                netC.zero_grad()

                with autocast():
                    output_real = netC(real_data, labels=real_labels if n_classes > 1 else None)

                    noise = torch.randn(batch_size_current, nz, device=device)
                    fake_data, _, _ = netG(noise, labels=gen_labels)
                    output_fake = netC(fake_data.detach(), labels=gen_labels)

                    critic_loss = output_fake.mean() - output_real.mean()

                    if use_gradient_penalty:
                        gp = compute_gradient_penalty(netC, real_data, fake_data.detach(), device, lambda_gp,
                                                      labels=real_labels if n_classes > 1 else None)
                        critic_loss += gp
                        gradient_penalty_val = gp.item()
                    else:
                        gradient_penalty_val = 0.0

                scaler.scale(critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(netC.parameters(), max_grad_norm)
                scaler.step(optimizerC)
                scaler.update()

            # Train Generator
            netG.zero_grad()

            with autocast():
                noise = torch.randn(batch_size_current, nz, device=device)
                fake_data, sigma_fake, mu_fake = netG(noise, labels=gen_labels)

                output_fake = netC(fake_data, labels=gen_labels)
                g_loss_adv = -output_fake.mean()

                physics_loss, _ = physics_loss_fn(sigma_fake, mu_fake, epoch=epoch, max_epochs=epoch_num)
                g_loss = g_loss_adv + lambda_physics * physics_loss

            scaler.scale(g_loss).backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_grad_norm)
            scaler.step(optimizerG)
            scaler.update()

            wasserstein_distance = output_real.mean().item() - output_fake.mean().item()

            # Print progress
            if i % 5 == 0:
                print(f'[{epoch}/{epoch_num}][{i}/{len(dataloader)}] '
                      f'Loss_C: {critic_loss.item():.4f} '
                      f'Loss_G: {g_loss.item():.4f} '
                      f'W_dist: {wasserstein_distance:.4f} '
                      f'λ_phys: {lambda_physics:.4f}')

        epoch_time = time.time() - epoch_start

        # Save history (guard against empty dataloader)
        if 'critic_loss' in dir():
            training_history['loss_C'].append(float(critic_loss.item()))
            training_history['loss_G'].append(float(g_loss.item()))
            training_history['loss_G_adv'].append(float(g_loss_adv.item()))
            training_history['loss_G_physics'].append(float(physics_loss.item()))
            training_history['wasserstein_distance'].append(float(wasserstein_distance))
            training_history['gradient_penalty'].append(float(gradient_penalty_val))
            training_history['physics_weight'].append(float(lambda_physics))

        # Periodic visualization
        if epoch % 100 == 0:
            netG.eval()
            with torch.no_grad():
                vis_labels = None
                if n_classes > 1:
                    vis_labels = torch.arange(16, device=device) % n_classes
                test_fake, test_sigma, test_mu = netG(fixed_noise, labels=vis_labels)
                test_sigma_np = test_sigma.cpu().numpy()
                test_mu_np = test_mu.cpu().numpy()

                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                fig.suptitle(f'Epoch {epoch}')

                for j in range(4):
                    ax = axes[j//2, j%2]
                    ax.plot(test_sigma_np[j], label='σ', alpha=0.7)
                    ax.plot(test_mu_np[j], label='μ', alpha=0.7)
                    ax.set_ylim(-1.2, 1.2)
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'{output_dir}/training_images/epoch_{epoch:04d}.png', dpi=100)
                plt.close()
            netG.train()

            print(f'Epoch {epoch} time: {epoch_time:.2f}s')

        # Save checkpoints
        if epoch % 50 == 0 or epoch == epoch_num - 1:
            torch.save(netG.state_dict(), f'{output_dir}/models/netG_epoch_{epoch}.pt')
            torch.save(netC.state_dict(), f'{output_dir}/models/netC_epoch_{epoch}.pt')

            with open(f'{output_dir}/training_history.json', 'w') as f:
                json.dump(training_history, f, indent=2)

    # Save final models
    torch.save(netG.state_dict(), f'{output_dir}/models/netG_final.pt')
    torch.save(netC.state_dict(), f'{output_dir}/models/netC_final.pt')

    with open(f'{output_dir}/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average epoch time: {total_time/epoch_num:.2f}s")
    print(f"Results saved to: {output_dir}")
    print(f"Final Wasserstein distance: {training_history['wasserstein_distance'][-1]:.4f}")


if __name__ == '__main__':
    main()
