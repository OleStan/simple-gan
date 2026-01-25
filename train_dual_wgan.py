#!/usr/bin/env python
"""Train dual-head WGAN for generating correlated sigma and mu profiles."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

from wgan_dual_profiles import DualHeadGenerator, Critic, weights_init, compute_gradient_penalty


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
        return torch.FloatTensor(self.normalized_data[idx])


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    K = 50
    nz = 100
    batch_size = 32
    n_epochs = 500
    lr = 1e-4
    n_critic = 5
    lambda_gp = 10
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'./results/dual_wgan_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'training_images').mkdir(exist_ok=True)
    
    dataset = ProfileDataset('./training_data/X_raw.npy', 
                            './training_data/normalization_params.json')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    with open('./training_data/normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    netG = DualHeadGenerator(nz=nz, K=K).to(device)
    netC = Critic(input_dim=2*K).to(device)
    
    netG.apply(weights_init)
    netC.apply(weights_init)
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(0.5, 0.999))
    
    print(f"\nGenerator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in netC.parameters()):,}")
    print(f"\nTraining configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  n_critic: {n_critic}")
    print(f"  lambda_gp: {lambda_gp}")
    print(f"\nOutput directory: {output_dir}")
    
    history = {
        'loss_C': [],
        'loss_G': [],
        'wasserstein_distance': [],
        'gradient_penalty': []
    }
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(n_epochs):
        for i, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size_current = real_data.size(0)
            
            for _ in range(n_critic):
                netC.zero_grad()
                
                noise = torch.randn(batch_size_current, nz, device=device)
                fake_data, _, _ = netG(noise)
                
                critic_real = netC(real_data)
                critic_fake = netC(fake_data.detach())
                
                gp = compute_gradient_penalty(netC, real_data, fake_data.detach(), device)
                
                loss_C = -torch.mean(critic_real) + torch.mean(critic_fake) + lambda_gp * gp
                loss_C.backward()
                optimizerC.step()
            
            netG.zero_grad()
            
            noise = torch.randn(batch_size_current, nz, device=device)
            fake_data, _, _ = netG(noise)
            
            critic_fake = netC(fake_data)
            loss_G = -torch.mean(critic_fake)
            
            loss_G.backward()
            optimizerG.step()
            
            wasserstein_distance = torch.mean(critic_real) - torch.mean(critic_fake)
            
            if i % 10 == 0:
                print(f'[{epoch}/{n_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_C: {loss_C.item():.4f} Loss_G: {loss_G.item():.4f} '
                      f'W_dist: {wasserstein_distance.item():.4f} GP: {gp.item():.4f}')
        
        history['loss_C'].append(loss_C.item())
        history['loss_G'].append(loss_G.item())
        history['wasserstein_distance'].append(wasserstein_distance.item())
        history['gradient_penalty'].append(gp.item())
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                noise = torch.randn(16, nz, device=device)
                fake_data, fake_sigma, fake_mu = netG(noise)
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
                
                plt.suptitle(f'Generated Profiles - Epoch {epoch+1}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_dir / 'training_images' / f'epoch_{epoch+1:04d}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
        
        if (epoch + 1) % 50 == 0:
            torch.save(netG.state_dict(), output_dir / 'models' / f'netG_epoch_{epoch+1}.pth')
            torch.save(netC.state_dict(), output_dir / 'models' / f'netC_epoch_{epoch+1}.pth')
            
            with open(output_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
    
    torch.save(netG.state_dict(), output_dir / 'models' / 'netG_final.pth')
    torch.save(netC.state_dict(), output_dir / 'models' / 'netC_final.pth')
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
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
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModels saved to: {output_dir / 'models'}/")
    print(f"Training history saved to: {output_dir / 'training_history.json'}")
    print(f"Training curves saved to: {output_dir / 'training_curves.png'}")


if __name__ == '__main__':
    main()
