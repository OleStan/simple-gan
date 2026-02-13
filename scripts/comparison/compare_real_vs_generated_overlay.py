#!/usr/bin/env python
"""Compare real and generated profiles overlaid on the same graphs."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from wgan_dual_profiles import DualHeadGenerator


def load_model_and_generate(model_dir, n_samples, device):
    """Load trained model and generate samples."""
    with open(f'{model_dir}/normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    K = norm_params['K']
    nz = 100
    
    netG = DualHeadGenerator(nz=nz, K=K).to(device)
    model_path = f'{model_dir}/models/netG_final.pth'
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    with torch.no_grad():
        noise = torch.randn(n_samples, nz, device=device)
        fake_data, _, _ = netG(noise)
        fake_data_np = fake_data.cpu().numpy()
    
    sigma_normalized = fake_data_np[:, :K]
    mu_normalized = fake_data_np[:, K:2*K]
    
    sigma_min = norm_params['sigma_min']
    sigma_max = norm_params['sigma_max']
    mu_min = norm_params['mu_min']
    mu_max = norm_params['mu_max']
    
    sigma_gen = (sigma_normalized + 1) / 2 * (sigma_max - sigma_min) + sigma_min
    mu_gen = (mu_normalized + 1) / 2 * (mu_max - mu_min) + mu_min
    
    return sigma_gen, mu_gen, K


def plot_overlay_comparison(sigma_real, mu_real, sigma_gen, mu_gen, n_pairs=6, save_path=None):
    """Plot real and generated profiles overlaid on the same graphs."""
    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 3*n_pairs))
    
    for i in range(n_pairs):
        ax_sigma = axes[i, 0]
        ax_mu = axes[i, 1]
        
        ax_sigma.plot(sigma_real[i], 'b-', linewidth=2.5, alpha=0.7, label='Real')
        ax_sigma.plot(sigma_gen[i], 'r--', linewidth=2, alpha=0.8, label='Generated')
        
        ax_mu.plot(mu_real[i], 'b-', linewidth=2.5, alpha=0.7, label='Real')
        ax_mu.plot(mu_gen[i], 'r--', linewidth=2, alpha=0.8, label='Generated')
        
        ax_sigma.set_ylabel('σ (S/m)', fontsize=11)
        ax_mu.set_ylabel('μᵣ', fontsize=11)
        
        ax_sigma.legend(loc='best', fontsize=10)
        ax_mu.legend(loc='best', fontsize=10)
        
        if i == 0:
            ax_sigma.set_title('Sigma (σ) Profiles', fontsize=13, fontweight='bold')
            ax_mu.set_title('Mu (μ) Profiles', fontsize=13, fontweight='bold')
        
        if i == n_pairs - 1:
            ax_sigma.set_xlabel('Layer Index', fontsize=11)
            ax_mu.set_xlabel('Layer Index', fontsize=11)
        
        ax_sigma.grid(True, alpha=0.3)
        ax_mu.grid(True, alpha=0.3)
        
        ax_sigma.text(0.02, 0.98, f'Pair {i+1}', transform=ax_sigma.transAxes,
                     fontsize=10, verticalalignment='top', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    plt.suptitle('Real vs Generated Profile Overlay Comparison', 
                 fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    return fig


def plot_combined_overlay(sigma_real, mu_real, sigma_gen, mu_gen, n_pairs=6, save_path=None):
    """Plot real and generated σ and μ profiles together on same axes."""
    fig, axes = plt.subplots(n_pairs, 1, figsize=(16, 3*n_pairs))
    
    if n_pairs == 1:
        axes = [axes]
    
    for i in range(n_pairs):
        ax = axes[i]
        ax2 = ax.twinx()
        
        ax.plot(sigma_real[i], 'b-', linewidth=2.5, alpha=0.7, label='σ Real')
        ax.plot(sigma_gen[i], 'b--', linewidth=2, alpha=0.8, label='σ Generated')
        
        ax2.plot(mu_real[i], 'r-', linewidth=2.5, alpha=0.7, label='μ Real')
        ax2.plot(mu_gen[i], 'r--', linewidth=2, alpha=0.8, label='μ Generated')
        
        ax.set_ylabel('σ (S/m)', color='b', fontsize=11, fontweight='bold')
        ax2.set_ylabel('μᵣ', color='r', fontsize=11, fontweight='bold')
        
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
                 ncol=4, fontsize=10, framealpha=0.9)
        
        if i == 0:
            ax.set_title('Combined σ and μ Profiles (Real vs Generated)', 
                        fontsize=13, fontweight='bold')
        
        if i == n_pairs - 1:
            ax.set_xlabel('Layer Index', fontsize=11)
        
        ax.grid(True, alpha=0.3)
        
        ax.text(0.02, 0.98, f'Sample {i+1}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle('Real vs Generated: Combined σ-μ Profile Overlay', 
                 fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    return fig


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    results_dirs = sorted(Path('./results').glob('dual_wgan_*'))
    if not results_dirs:
        print("No training results found in ./results/")
        return
    
    model_dir = str(results_dirs[-1])
    
    print("="*70)
    print("Real vs Generated Profile Overlay Comparison")
    print("="*70)
    print(f"Model: {model_dir}")
    print(f"Device: {device}")
    
    print("\nLoading real training data...")
    real_data = np.load('./training_data/X_raw.npy')
    
    with open('./training_data/normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    K = norm_params['K']
    sigma_real = real_data[:, :K]
    mu_real = real_data[:, K:2*K]
    
    print(f"✓ Loaded {len(real_data)} real samples (K={K} layers)")
    
    print("\nGenerating samples from trained model...")
    n_samples = 20
    sigma_gen, mu_gen, _ = load_model_and_generate(model_dir, n_samples, device)
    print(f"✓ Generated {n_samples} samples")
    
    report_dir = f'{model_dir}/overlay_comparison'
    Path(report_dir).mkdir(exist_ok=True)
    
    print("\nCreating overlay visualizations...")
    
    print("  1/2 Separate σ and μ overlays...")
    plot_overlay_comparison(sigma_real, mu_real, sigma_gen, mu_gen, 
                           n_pairs=6, save_path=f'{report_dir}/overlay_separate.png')
    
    print("  2/2 Combined σ-μ overlays...")
    plot_combined_overlay(sigma_real, mu_real, sigma_gen, mu_gen,
                         n_pairs=6, save_path=f'{report_dir}/overlay_combined.png')
    
    print("\n" + "="*70)
    print("Overlay Comparison Complete!")
    print("="*70)
    print(f"\nVisualizations saved to: {report_dir}/")
    print(f"  - overlay_separate.png (σ and μ in separate columns)")
    print(f"  - overlay_combined.png (σ and μ on same axes)")
    print("\nNote: Real profiles shown as solid lines, Generated as dashed lines")


if __name__ == '__main__':
    main()
