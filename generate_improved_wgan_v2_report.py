#!/usr/bin/env python
"""Generate comprehensive report for Improved WGAN v2 training results."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path
from wgan_improved_v2 import ConditionalConv1DGenerator

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_improved_wgan_v2_report.py <model_directory>")
        sys.exit(1)
    
    model_dir = Path(sys.argv[1])
    
    print("="*60)
    print("Generating Improved WGAN v2 Report")
    print("="*60)
    print(f"Model directory: {model_dir}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    with open(model_dir / 'normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    K = norm_params['K']
    nz = 100
    
    netG = ConditionalConv1DGenerator(nz=nz, K=K, conditional=False).to(device)
    
    model_path = model_dir / 'models' / 'netG_final.pt'
    if not model_path.exists():
        available_models = sorted((model_dir / 'models').glob('netG_epoch_*.pt'))
        if available_models:
            model_path = available_models[-1]
            epoch_num = model_path.stem.split('_')[-1]
        else:
            print("Error: No model found!")
            sys.exit(1)
    else:
        epoch_num = "final"
    
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    print(f"\n✓ Model loaded from epoch {epoch_num} (K={K} layers)")
    
    print("\nGenerating 1000 samples...")
    n_samples = 1000
    generated_samples = []
    
    with torch.no_grad():
        for i in range(0, n_samples, 100):
            batch_size = min(100, n_samples - i)
            noise = torch.randn(batch_size, nz, device=device)
            fake_data, sigma_fake, mu_fake = netG(noise)
            generated_samples.append(fake_data.cpu().numpy())
    
    generated_samples = np.vstack(generated_samples)
    
    sigma_generated = generated_samples[:, :K]
    mu_generated = generated_samples[:, K:2*K]
    
    sigma_min = norm_params['sigma_min']
    sigma_max = norm_params['sigma_max']
    mu_min = norm_params['mu_min']
    mu_max = norm_params['mu_max']
    
    sigma_denorm = (sigma_generated + 1) / 2 * (sigma_max - sigma_min) + sigma_min
    mu_denorm = (mu_generated + 1) / 2 * (mu_max - mu_min) + mu_min
    
    print(f"✓ Generated samples:")
    print(f"  σ: [{sigma_denorm.min():.2e}, {sigma_denorm.max():.2e}] S/m")
    print(f"  μ: [{mu_denorm.min():.2f}, {mu_denorm.max():.2f}]")
    
    output_dir = model_dir / f'report_epoch_{epoch_num}'
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    print("  1/5 Sample profiles...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Improved WGAN v2 - Generated Profile Samples', fontsize=16, fontweight='bold')
    
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        ax.plot(sigma_denorm[i], label='σ', linewidth=2, alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(mu_denorm[i], label='μ', color='orange', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('σ (S/m)', color='blue')
        ax2.set_ylabel('μ (relative)', color='orange')
        ax.set_title(f'Sample {i+1}')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  2/5 Distribution comparison...")
    real_data = np.load('training_data/X_raw.npy')
    sigma_real = real_data[:, :K]
    mu_real = real_data[:, K:2*K]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution Comparison: Real vs Generated', fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(sigma_real.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[0, 0].hist(sigma_denorm.flatten(), bins=50, alpha=0.5, label='Generated', density=True)
    axes[0, 0].set_xlabel('σ (S/m)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('σ Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(mu_real.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[0, 1].hist(mu_denorm.flatten(), bins=50, alpha=0.5, label='Generated', density=True)
    axes[0, 1].set_xlabel('μ (relative)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('μ Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(sigma_real.mean(axis=0), label='Real mean', linewidth=2)
    axes[1, 0].plot(sigma_denorm.mean(axis=0), label='Generated mean', linewidth=2, linestyle='--')
    axes[1, 0].fill_between(range(K), 
                            sigma_real.mean(axis=0) - sigma_real.std(axis=0),
                            sigma_real.mean(axis=0) + sigma_real.std(axis=0),
                            alpha=0.2, label='Real ±σ')
    axes[1, 0].fill_between(range(K),
                            sigma_denorm.mean(axis=0) - sigma_denorm.std(axis=0),
                            sigma_denorm.mean(axis=0) + sigma_denorm.std(axis=0),
                            alpha=0.2, label='Generated ±σ')
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('σ (S/m)')
    axes[1, 0].set_title('σ Profile Statistics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(mu_real.mean(axis=0), label='Real mean', linewidth=2)
    axes[1, 1].plot(mu_denorm.mean(axis=0), label='Generated mean', linewidth=2, linestyle='--')
    axes[1, 1].fill_between(range(K),
                            mu_real.mean(axis=0) - mu_real.std(axis=0),
                            mu_real.mean(axis=0) + mu_real.std(axis=0),
                            alpha=0.2, label='Real ±σ')
    axes[1, 1].fill_between(range(K),
                            mu_denorm.mean(axis=0) - mu_denorm.std(axis=0),
                            mu_denorm.mean(axis=0) + mu_denorm.std(axis=0),
                            alpha=0.2, label='Generated ±σ')
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('μ (relative)')
    axes[1, 1].set_title('μ Profile Statistics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  3/5 Training curves...")
    with open(model_dir / 'training_history.json', 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Improved WGAN v2 - Training History', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(history['loss_C'])
    axes[0, 0].set_title('Critic Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['loss_G'], label='Total', alpha=0.8)
    axes[0, 1].plot(history['loss_G_adv'], label='Adversarial', alpha=0.6)
    axes[0, 1].plot(history['loss_G_physics'], label='Physics', alpha=0.6)
    axes[0, 1].set_title('Generator Loss Components')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(history['wasserstein_distance'])
    axes[0, 2].set_title('Wasserstein Distance')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['physics_weight'])
    axes[1, 0].set_title('Physics Loss Weight (Schedule)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['gradient_penalty'])
    axes[1, 1].set_title('Gradient Penalty')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    if history['quality_metrics']['sigma_smoothness']:
        epochs_eval = list(range(0, len(history['loss_C']), 10))
        axes[1, 2].plot(epochs_eval, history['quality_metrics']['sigma_smoothness'], label='σ', marker='o')
        axes[1, 2].plot(epochs_eval, history['quality_metrics']['mu_smoothness'], label='μ', marker='s')
        axes[1, 2].set_title('Smoothness Score')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        axes[2, 0].plot(epochs_eval, history['quality_metrics']['sigma_diversity'], label='σ', marker='o')
        axes[2, 0].plot(epochs_eval, history['quality_metrics']['mu_diversity'], label='μ', marker='s')
        axes[2, 0].set_title('Diversity Score')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(epochs_eval, history['quality_metrics']['sigma_monotonicity'], label='σ', marker='o')
        axes[2, 1].plot(epochs_eval, history['quality_metrics']['mu_monotonicity'], label='μ', marker='s')
        axes[2, 1].set_title('Monotonicity Score')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    axes[2, 2].plot(history['loss_C'][-100:])
    axes[2, 2].set_title('Critic Loss (Last 100 Epochs)')
    axes[2, 2].set_xlabel('Epoch (relative)')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  4/5 Normalized profiles...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Normalized Profile Samples', fontsize=16, fontweight='bold')
    
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        ax.plot(sigma_generated[i], label='σ (norm)', linewidth=2, alpha=0.8)
        ax.plot(mu_generated[i], label='μ (norm)', linewidth=2, alpha=0.8)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'normalized_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  5/5 Quality analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Generated Profile Quality Analysis', fontsize=16, fontweight='bold')
    
    sigma_gradients = np.diff(sigma_denorm, axis=1)
    mu_gradients = np.diff(mu_denorm, axis=1)
    
    axes[0, 0].hist(sigma_gradients.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Gradient (Δσ)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('σ Gradient Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(mu_gradients.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Gradient (Δμ)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('μ Gradient Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(sigma_denorm.mean(axis=1), sigma_denorm.std(axis=1), alpha=0.3)
    axes[1, 0].set_xlabel('Mean σ')
    axes[1, 0].set_ylabel('Std σ')
    axes[1, 0].set_title('σ Mean vs Std')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(mu_denorm.mean(axis=1), mu_denorm.std(axis=1), alpha=0.3, color='orange')
    axes[1, 1].set_xlabel('Mean μ')
    axes[1, 1].set_ylabel('Std μ')
    axes[1, 1].set_title('μ Mean vs Std')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    np.save(output_dir / 'generated_sigma.npy', sigma_denorm)
    np.save(output_dir / 'generated_mu.npy', mu_denorm)
    
    stats = {
        'n_samples': int(n_samples),
        'sigma': {
            'min': float(sigma_denorm.min()),
            'max': float(sigma_denorm.max()),
            'mean': float(sigma_denorm.mean()),
            'std': float(sigma_denorm.std())
        },
        'mu': {
            'min': float(mu_denorm.min()),
            'max': float(mu_denorm.max()),
            'mean': float(mu_denorm.mean()),
            'std': float(mu_denorm.std())
        },
        'final_losses': {
            'critic': float(history['loss_C'][-1]),
            'generator': float(history['loss_G'][-1]),
            'wasserstein_distance': float(history['wasserstein_distance'][-1])
        }
    }
    
    with open(output_dir / 'generation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("Report Generation Complete!")
    print("="*60)
    print(f"\nReport saved to: {output_dir}/")
    print("  - sample_profiles.png")
    print("  - distribution_comparison.png")
    print("  - training_curves.png")
    print("  - normalized_profiles.png")
    print("  - quality_analysis.png")
    print(f"  - generated_sigma.npy ({sigma_denorm.shape})")
    print(f"  - generated_mu.npy ({mu_denorm.shape})")
    print("  - generation_stats.json")


if __name__ == '__main__':
    main()
