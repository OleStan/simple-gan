#!/usr/bin/env python
"""Train improved WGAN with physics-informed constraints for σ and μ profiles."""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from pathlib import Path
from wgan_improved import (
    Conv1DGenerator, Conv1DCritic, PhysicsInformedLoss,
    ProfileQualityMetrics, weights_init, compute_gradient_penalty
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


def main():
    n_critic = 5
    lambda_gp = 10
    lr_g = 1e-4
    lr_c = 4e-4
    epoch_num = 500
    batch_size = 32
    nz = 100
    K = 50
    
    lambda_physics = 0.5
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./results/improved_wgan_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/training_images', exist_ok=True)
    os.makedirs(f'{output_dir}/visualizations', exist_ok=True)
    
    print("="*60)
    print("Training Improved WGAN with Physics-Informed Constraints")
    print("="*60)
    print(f"Architecture: 1D Conv Generator + Conv Critic")
    print(f"Physics loss: Smoothness + Bounds penalties")
    print(f"Output: 2 × {K} = {2*K} values")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epoch_num}")
    print(f"LR Generator: {lr_g}, LR Critic: {lr_c}")
    print(f"Gradient penalty λ: {lambda_gp}")
    print(f"Physics penalty λ: {lambda_physics}")
    print("="*60)
    
    dataset = ProfileDataset('./training_data/X_raw.npy', normalize=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    print(f"\nDataset loaded: {len(dataset)} samples")
    
    netG = Conv1DGenerator(nz=nz, K=K).to(device)
    netG.apply(weights_init)
    
    netC = Conv1DCritic(K=K).to(device)
    netC.apply(weights_init)
    
    physics_loss = PhysicsInformedLoss(lambda_smooth=0.1, lambda_bounds=0.05)
    
    print(f"\nGenerator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in netC.parameters()):,}")
    
    fixed_noise = torch.randn(16, nz, device=device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=lr_c, betas=(0.5, 0.999))
    
    with open(f'{output_dir}/normalization_params.json', 'w') as f:
        json.dump(dataset.norm_params, f, indent=2)
    
    training_history = {
        'loss_C': [],
        'loss_G': [],
        'loss_G_adv': [],
        'loss_G_physics': [],
        'wasserstein_distance': [],
        'gradient_penalty': [],
        'quality_metrics': {
            'sigma_smoothness': [],
            'mu_smoothness': [],
            'sigma_monotonicity': [],
            'mu_monotonicity': [],
            'sigma_diversity': [],
            'mu_diversity': []
        }
    }
    
    print("\nStarting training...\n")
    
    for epoch in range(epoch_num):
        epoch_loss_C = []
        epoch_loss_G = []
        epoch_loss_G_adv = []
        epoch_loss_G_physics = []
        epoch_wd = []
        epoch_gp = []
        
        for step, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            b_size = real_data.size(0)
            
            for _ in range(n_critic):
                netC.zero_grad()
                
                noise = torch.randn(b_size, nz, device=device)
                fake_data, _, _ = netG(noise)
                
                real_score = netC(real_data)
                fake_score = netC(fake_data.detach())
                
                gradient_penalty = compute_gradient_penalty(netC, real_data, fake_data.detach(), device)
                
                loss_C = -torch.mean(real_score) + torch.mean(fake_score) + lambda_gp * gradient_penalty
                loss_C.backward()
                optimizerC.step()
                
                wasserstein_distance = torch.mean(real_score) - torch.mean(fake_score)
                
                epoch_loss_C.append(loss_C.item())
                epoch_wd.append(wasserstein_distance.item())
                epoch_gp.append(gradient_penalty.item())
            
            netG.zero_grad()
            
            noise = torch.randn(b_size, nz, device=device)
            fake_data, fake_sigma, fake_mu = netG(noise)
            
            fake_score = netC(fake_data)
            loss_G_adv = -torch.mean(fake_score)
            
            loss_G_phys, phys_components = physics_loss(fake_sigma, fake_mu)
            
            loss_G = loss_G_adv + lambda_physics * loss_G_phys
            
            loss_G.backward()
            optimizerG.step()
            
            epoch_loss_G.append(loss_G.item())
            epoch_loss_G_adv.append(loss_G_adv.item())
            epoch_loss_G_physics.append(loss_G_phys.item())
            
            if step % 10 == 0:
                print(f'[{epoch+1}/{epoch_num}][{step}/{len(dataloader)}] '
                      f'Loss_C: {loss_C.item():.4f} Loss_G: {loss_G.item():.4f} '
                      f'(Adv: {loss_G_adv.item():.4f} + Phys: {loss_G_phys.item():.4f}) '
                      f'W_dist: {wasserstein_distance.item():.4f}')
        
        avg_loss_C = np.mean(epoch_loss_C)
        avg_loss_G = np.mean(epoch_loss_G)
        avg_loss_G_adv = np.mean(epoch_loss_G_adv)
        avg_loss_G_physics = np.mean(epoch_loss_G_physics)
        avg_wd = np.mean(epoch_wd)
        avg_gp = np.mean(epoch_gp)
        
        training_history['loss_C'].append(avg_loss_C)
        training_history['loss_G'].append(avg_loss_G)
        training_history['loss_G_adv'].append(avg_loss_G_adv)
        training_history['loss_G_physics'].append(avg_loss_G_physics)
        training_history['wasserstein_distance'].append(avg_wd)
        training_history['gradient_penalty'].append(avg_gp)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                fake_data, fake_sigma, fake_mu = netG(fixed_noise)
                
                quality_metrics = ProfileQualityMetrics.evaluate_batch(fake_sigma, fake_mu)
                for key, value in quality_metrics.items():
                    training_history['quality_metrics'][key].append(value)
                
                print(f"\nEpoch {epoch+1} Quality Metrics:")
                print(f"  σ smoothness: {quality_metrics['sigma_smoothness']:.4f}")
                print(f"  μ smoothness: {quality_metrics['mu_smoothness']:.4f}")
                print(f"  σ monotonicity: {quality_metrics['sigma_monotonicity']:.4f}")
                print(f"  μ monotonicity: {quality_metrics['mu_monotonicity']:.4f}\n")
                
                fake_data_np = fake_data.cpu().numpy()
                
                sigma_denorm, mu_denorm = denormalize_profiles(
                    fake_data_np, dataset.norm_params, K
                )
                
                fig, axes = plt.subplots(4, 4, figsize=(16, 12))
                for i in range(4):
                    for j in range(4):
                        idx = i * 4 + j
                        ax = axes[i, j]
                        
                        ax2 = ax.twinx()
                        
                        ax.plot(sigma_denorm[idx], 'b-', linewidth=1.5, label='σ')
                        ax2.plot(mu_denorm[idx], 'r-', linewidth=1.5, label='μ')
                        
                        ax.set_ylabel('σ (S/m)', color='b', fontsize=8)
                        ax2.set_ylabel('μᵣ', color='r', fontsize=8)
                        
                        ax.tick_params(axis='y', labelcolor='b', labelsize=7)
                        ax2.tick_params(axis='y', labelcolor='r', labelsize=7)
                        ax.tick_params(axis='x', labelsize=7)
                        
                        ax.set_xticks([])
                        ax.grid(True, alpha=0.3)
                
                plt.suptitle(f'Epoch {epoch+1}: Generated Profiles (Improved)', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/training_images/epoch_{epoch+1:04d}.png', dpi=150)
                plt.close()
        
        if (epoch + 1) % 50 == 0:
            torch.save(netG.state_dict(), f'{output_dir}/models/netG_epoch_{epoch+1}.pth')
            torch.save(netC.state_dict(), f'{output_dir}/models/netC_epoch_{epoch+1}.pth')
    
    torch.save(netG.state_dict(), f'{output_dir}/models/netG_final.pth')
    torch.save(netC.state_dict(), f'{output_dir}/models/netC_final.pth')
    
    with open(f'{output_dir}/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    axes[0, 0].plot(training_history['loss_C'])
    axes[0, 0].set_title('Critic Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(training_history['loss_G'], label='Total')
    axes[0, 1].plot(training_history['loss_G_adv'], label='Adversarial', alpha=0.7)
    axes[0, 1].plot(training_history['loss_G_physics'], label='Physics', alpha=0.7)
    axes[0, 1].set_title('Generator Loss Components', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(training_history['wasserstein_distance'])
    axes[0, 2].set_title('Wasserstein Distance', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Distance')
    axes[0, 2].grid(True, alpha=0.3)
    
    metric_epochs = range(0, epoch_num, 10)
    axes[1, 0].plot(metric_epochs, training_history['quality_metrics']['sigma_smoothness'], 'b-', label='σ')
    axes[1, 0].plot(metric_epochs, training_history['quality_metrics']['mu_smoothness'], 'r-', label='μ')
    axes[1, 0].set_title('Smoothness Score', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(metric_epochs, training_history['quality_metrics']['sigma_monotonicity'], 'b-', label='σ')
    axes[1, 1].plot(metric_epochs, training_history['quality_metrics']['mu_monotonicity'], 'r-', label='μ')
    axes[1, 1].set_title('Monotonicity Score', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(metric_epochs, training_history['quality_metrics']['sigma_diversity'], 'b-', label='σ')
    axes[1, 2].plot(metric_epochs, training_history['quality_metrics']['mu_diversity'], 'r-', label='μ')
    axes[1, 2].set_title('Diversity Score', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    axes[2, 0].plot(training_history['gradient_penalty'])
    axes[2, 0].set_title('Gradient Penalty', fontweight='bold')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Penalty')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].axis('off')
    info_text = (
        f"Improved WGAN Training Summary\n\n"
        f"Architecture: 1D Convolutional\n"
        f"Physics Loss: Yes (λ={lambda_physics})\n"
        f"Final Wasserstein: {training_history['wasserstein_distance'][-1]:.4f}\n"
        f"Final σ Smoothness: {training_history['quality_metrics']['sigma_smoothness'][-1]:.4f}\n"
        f"Final μ Smoothness: {training_history['quality_metrics']['mu_smoothness'][-1]:.4f}"
    )
    axes[2, 1].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300)
    plt.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Models saved to: {output_dir}/models/")
    print(f"Training images: {output_dir}/training_images/")
    print(f"Training curves: {output_dir}/training_curves.png")
    print(f"History: {output_dir}/training_history.json")
    
    return netG, netC, output_dir


if __name__ == '__main__':
    netG, netC, output_dir = main()
