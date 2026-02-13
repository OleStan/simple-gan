#!/usr/bin/env python
"""
Train Improved WGAN v2 with enhanced stability features:
- Spectral normalization (optional gradient penalty)
- Gradient clipping
- Physics loss scheduling with warmup
- Separate learning rates for G and C
- Better monitoring and checkpointing
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from pathlib import Path
from wgan_improved_v2 import (
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
        
        if normalize:
            self.X_normalized = self.X.copy()
            
            sigma_data = self.X[:, :self.K]
            mu_data = self.X[:, self.K:2*self.K]
            
            sigma_min = sigma_data.min()
            sigma_max = sigma_data.max()
            mu_min = mu_data.min()
            mu_max = mu_data.max()
            
            self.X_normalized[:, :self.K] = 2 * (sigma_data - sigma_min) / (sigma_max - sigma_min) - 1
            self.X_normalized[:, self.K:2*self.K] = 2 * (mu_data - mu_min) / (mu_max - mu_min) - 1
            
            self.data = self.X_normalized
        else:
            self.data = self.X
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


def denormalize_profiles(data, norm_params, K):
    """Denormalize generated profiles back to physical units."""
    sigma_normalized = data[:, :K]
    mu_normalized = data[:, K:2*K]
    
    sigma_min = norm_params['sigma_min']
    sigma_max = norm_params['sigma_max']
    mu_min = norm_params['mu_min']
    mu_max = norm_params['mu_max']
    
    sigma = (sigma_normalized + 1) / 2 * (sigma_max - sigma_min) + sigma_min
    mu = (mu_normalized + 1) / 2 * (mu_max - mu_min) + mu_min
    
    return sigma, mu


def save_checkpoint(epoch, netG, netC, optimizerG, optimizerC, output_dir):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netC_state_dict': netC.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerC_state_dict': optimizerC.state_dict(),
    }
    torch.save(checkpoint, f'{output_dir}/checkpoints/checkpoint_epoch_{epoch}.pt')


def load_checkpoint(checkpoint_path, netG, netC, optimizerG, optimizerC, device):
    """Load checkpoint and return start epoch."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netC.load_state_dict(checkpoint['netC_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerC.load_state_dict(checkpoint['optimizerC_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return start_epoch


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Improved WGAN v2 (Resumable)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--resume_dir', type=str, default=None, help='Resume from latest checkpoint in this directory')
    args = parser.parse_args()
    
    n_critic = 5
    lambda_gp = 10
    lr_g = 5e-5
    lr_c = 2e-4
    with open('./training_data/normalization_params.json', 'r') as f:
        norm_params_temp = json.load(f)
    
    K = norm_params_temp['K']
    
    epoch_num = 500
    batch_size = 32
    nz = 100
    
    use_spectral_norm = True
    use_gradient_penalty = False
    max_grad_norm = 1.0
    
    lambda_physics_start = 0.0
    lambda_physics_end = 0.2
    physics_warmup_epochs = 100
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    start_epoch = 0
    
    if args.resume_dir:
        output_dir = args.resume_dir
        checkpoint_dir = Path(output_dir) / 'checkpoints'
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if checkpoints:
                args.resume = str(checkpoints[-1])
                print(f"Found latest checkpoint: {args.resume}")
    elif args.resume:
        output_dir = str(Path(args.resume).parent.parent)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'./results/improved_wgan_v2_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{output_dir}/training_images', exist_ok=True)
    os.makedirs(f'{output_dir}/visualizations', exist_ok=True)
    
    print("="*60)
    print("Training Improved WGAN v2 with Enhanced Stability")
    print("="*60)
    print(f"Architecture: Conv1D Generator + Spectral Norm Critic")
    print(f"Stability features:")
    print(f"  - Spectral normalization: {use_spectral_norm}")
    print(f"  - Gradient clipping: max_norm={max_grad_norm}")
    print(f"  - Physics loss warmup: {physics_warmup_epochs} epochs")
    print(f"  - Separate LR: G={lr_g}, C={lr_c}")
    print(f"Output: 2 × {K} = {2*K} values")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epoch_num}")
    print(f"Physics λ: {lambda_physics_start} → {lambda_physics_end}")
    print("="*60)
    
    dataset = ProfileDataset('./training_data/X_raw.npy', normalize=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    print(f"\nDataset loaded: {len(dataset)} samples")
    
    netG = ConditionalConv1DGenerator(nz=nz, K=K, conditional=False).to(device)
    netG.apply(weights_init)
    
    netC = SpectralNormConv1DCritic(K=K, use_spectral_norm=use_spectral_norm).to(device)
    if not use_spectral_norm:
        netC.apply(weights_init)
    
    physics_loss_fn = PhysicsInformedLossV2(
        lambda_smooth=0.05, 
        lambda_bounds=0.02, 
        lambda_monotonic=0.0
    )
    
    print(f"\nGenerator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in netC.parameters()):,}")
    
    fixed_noise = torch.randn(16, nz, device=device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optimizerC = optim.Adam(netC.parameters(), lr=lr_c, betas=(0.0, 0.9))
    
    if args.resume:
        start_epoch = load_checkpoint(args.resume, netG, netC, optimizerG, optimizerC, device)
        print(f"Resuming training from epoch {start_epoch}")
        
        history_path = Path(output_dir) / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                training_history = json.load(f)
            print(f"Loaded training history with {len(training_history['loss_C'])} epochs")
        else:
            print("Warning: No training history found, starting fresh history")
            training_history = None
    else:
        training_history = None
    
    with open(f'{output_dir}/normalization_params.json', 'w') as f:
        json.dump(dataset.norm_params, f, indent=2)
    
    config = {
        'n_critic': n_critic,
        'lambda_gp': lambda_gp,
        'lr_g': lr_g,
        'lr_c': lr_c,
        'epoch_num': epoch_num,
        'batch_size': batch_size,
        'nz': nz,
        'K': K,
        'use_spectral_norm': use_spectral_norm,
        'use_gradient_penalty': use_gradient_penalty,
        'max_grad_norm': max_grad_norm,
        'lambda_physics_start': lambda_physics_start,
        'lambda_physics_end': lambda_physics_end,
        'physics_warmup_epochs': physics_warmup_epochs,
        'device': str(device)
    }
    
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    if training_history is None:
        training_history = {
            'loss_C': [],
            'loss_G': [],
            'loss_G_adv': [],
            'loss_G_physics': [],
            'wasserstein_distance': [],
            'gradient_penalty': [],
            'physics_weight': [],
            'quality_metrics': {
                'sigma_smoothness': [],
                'mu_smoothness': [],
                'sigma_monotonicity': [],
                'mu_monotonicity': [],
                'sigma_diversity': [],
                'mu_diversity': []
            }
        }
    
    if start_epoch > 0:
        print(f"\nResuming training from epoch {start_epoch}...\n")
    else:
        print("\nStarting training from scratch...\n")
    
    for epoch in range(start_epoch, epoch_num):
        lambda_physics = lambda_physics_start
        if epoch < physics_warmup_epochs:
            lambda_physics = lambda_physics_start + (lambda_physics_end - lambda_physics_start) * (epoch / physics_warmup_epochs)
        else:
            lambda_physics = lambda_physics_end
        
        for i, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size_current = real_data.size(0)
            
            for _ in range(n_critic):
                netC.zero_grad()
                
                output_real = netC(real_data)
                
                noise = torch.randn(batch_size_current, nz, device=device)
                fake_data, _, _ = netG(noise)
                output_fake = netC(fake_data.detach())
                
                critic_loss = output_fake.mean() - output_real.mean()
                
                if use_gradient_penalty:
                    gp = compute_gradient_penalty(netC, real_data, fake_data.detach(), device, lambda_gp)
                    critic_loss += gp
                    gradient_penalty_val = gp.item()
                else:
                    gradient_penalty_val = 0.0
                
                critic_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(netC.parameters(), max_grad_norm)
                
                optimizerC.step()
            
            netG.zero_grad()
            
            noise = torch.randn(batch_size_current, nz, device=device)
            fake_data, sigma_fake, mu_fake = netG(noise)
            
            output_fake = netC(fake_data)
            g_loss_adv = -output_fake.mean()
            
            physics_loss, physics_metrics = physics_loss_fn(
                sigma_fake, mu_fake, 
                epoch=epoch, max_epochs=epoch_num
            )
            
            g_loss = g_loss_adv + lambda_physics * physics_loss
            
            g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_grad_norm)
            
            optimizerG.step()
            
            wasserstein_distance = output_real.mean().item() - output_fake.mean().item()
            
            if i % 10 == 0:
                print(f'[{epoch}/{epoch_num}][{i}/{len(dataloader)}] '
                      f'Loss_C: {critic_loss.item():.4f} '
                      f'Loss_G: {g_loss.item():.4f} '
                      f'Loss_G_adv: {g_loss_adv.item():.4f} '
                      f'Loss_G_phys: {physics_loss.item():.4f} '
                      f'W_dist: {wasserstein_distance:.4f} '
                      f'λ_phys: {lambda_physics:.4f}')
        
        training_history['loss_C'].append(float(critic_loss.item()))
        training_history['loss_G'].append(float(g_loss.item()))
        training_history['loss_G_adv'].append(float(g_loss_adv.item()))
        training_history['loss_G_physics'].append(float(physics_loss.item()))
        training_history['wasserstein_distance'].append(float(wasserstein_distance))
        training_history['gradient_penalty'].append(float(gradient_penalty_val))
        training_history['physics_weight'].append(float(lambda_physics))
        
        if epoch % 10 == 0:
            netG.eval()
            with torch.no_grad():
                test_fake, test_sigma, test_mu = netG(fixed_noise)
                
                test_sigma_np = test_sigma.cpu().numpy()
                test_mu_np = test_mu.cpu().numpy()
                
                metrics = {
                    'sigma_smoothness': float(ProfileQualityMetrics.smoothness(test_sigma_np)),
                    'mu_smoothness': float(ProfileQualityMetrics.smoothness(test_mu_np)),
                    'sigma_monotonicity': float(ProfileQualityMetrics.monotonicity(test_sigma_np)),
                    'mu_monotonicity': float(ProfileQualityMetrics.monotonicity(test_mu_np)),
                    'sigma_diversity': float(ProfileQualityMetrics.diversity(test_sigma_np)),
                    'mu_diversity': float(ProfileQualityMetrics.diversity(test_mu_np))
                }
                
                for key, value in metrics.items():
                    training_history['quality_metrics'][key].append(value)
                
                print(f'\nQuality Metrics (epoch {epoch}):')
                print(f'  σ smoothness: {metrics["sigma_smoothness"]:.4f}')
                print(f'  μ smoothness: {metrics["mu_smoothness"]:.4f}')
                print(f'  σ diversity: {metrics["sigma_diversity"]:.4f}')
                print(f'  μ diversity: {metrics["mu_diversity"]:.4f}\n')
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'Epoch {epoch} - Generated Profiles', fontsize=14, fontweight='bold')
                
                for i in range(4):
                    row = i // 2
                    col = i % 2
                    ax = axes[row, col]
                    
                    ax.plot(test_sigma_np[i], label='σ (normalized)', alpha=0.7, linewidth=2)
                    ax.plot(test_mu_np[i], label='μ (normalized)', alpha=0.7, linewidth=2)
                    ax.set_xlabel('Layer Index')
                    ax.set_ylabel('Normalized Value')
                    ax.set_title(f'Sample {i+1}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(-1.2, 1.2)
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/training_images/epoch_{epoch:04d}.png', dpi=100, bbox_inches='tight')
                plt.close()
            
            netG.train()
        
        if epoch % 50 == 0 or epoch == epoch_num - 1:
            torch.save(netG.state_dict(), f'{output_dir}/models/netG_epoch_{epoch}.pt')
            torch.save(netC.state_dict(), f'{output_dir}/models/netC_epoch_{epoch}.pt')
            save_checkpoint(epoch, netG, netC, optimizerG, optimizerC, output_dir)
            
            with open(f'{output_dir}/training_history.json', 'w') as f:
                json.dump(training_history, f, indent=2)
    
    print("\nTraining complete! Saving final models...")
    torch.save(netG.state_dict(), f'{output_dir}/models/netG_final.pt')
    torch.save(netC.state_dict(), f'{output_dir}/models/netC_final.pt')
    
    with open(f'{output_dir}/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, 1)
    plt.plot(training_history['loss_C'])
    plt.title('Critic Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    plt.subplot(3, 3, 2)
    plt.plot(training_history['loss_G'], label='G total')
    plt.plot(training_history['loss_G_adv'], label='G adv', alpha=0.7)
    plt.plot(training_history['loss_G_physics'], label='G physics', alpha=0.7)
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 3)
    plt.plot(training_history['wasserstein_distance'])
    plt.title('Wasserstein Distance')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    plt.subplot(3, 3, 4)
    plt.plot(training_history['physics_weight'])
    plt.title('Physics Loss Weight (Schedule)')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    plt.subplot(3, 3, 5)
    plt.plot(training_history['gradient_penalty'])
    plt.title('Gradient Penalty')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    if training_history['quality_metrics']['sigma_smoothness']:
        # Calculate actual evaluation epochs based on start epoch
        total_epochs = len(training_history['loss_C'])
        epochs_evaluated = list(range(start_epoch + (10 - start_epoch % 10) if start_epoch % 10 != 0 else start_epoch, total_epochs, 10))
        if not epochs_evaluated and start_epoch <= 10:
            epochs_evaluated = list(range(0, total_epochs, 10))
        
        # Ensure we don't exceed the available data
        if epochs_evaluated:
            max_plots = min(len(epochs_evaluated), len(training_history['quality_metrics']['sigma_smoothness']))
            epochs_evaluated = epochs_evaluated[:max_plots]
            
            plt.subplot(3, 3, 6)
            plt.plot(epochs_evaluated, training_history['quality_metrics']['sigma_smoothness'][:max_plots], label='σ')
            plt.plot(epochs_evaluated, training_history['quality_metrics']['mu_smoothness'][:max_plots], label='μ')
            plt.title('Smoothness')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 3, 7)
            plt.plot(epochs_evaluated, training_history['quality_metrics']['sigma_diversity'][:max_plots], label='σ')
            plt.plot(epochs_evaluated, training_history['quality_metrics']['mu_diversity'][:max_plots], label='μ')
            plt.title('Diversity')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=150)
    print(f"Training curves saved to {output_dir}/training_curves.png")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Final Wasserstein distance: {training_history['wasserstein_distance'][-1]:.4f}")


if __name__ == '__main__':
    main()
