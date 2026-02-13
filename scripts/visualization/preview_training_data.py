#!/usr/bin/env python
"""Preview the first 5-10 training samples showing sigma, mu, and their pairs."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def main():
    data_dir = Path('./training_data')
    
    print("="*60)
    print("Preview of Training Data with Fixed Boundaries")
    print("="*60)
    
    sigma_layers = np.load(data_dir / 'sigma_layers.npy')
    mu_layers = np.load(data_dir / 'mu_layers.npy')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    K = metadata['K']
    n_samples_preview = 10
    
    print(f"\nDataset Information:")
    print(f"  Total samples: {len(sigma_layers)}")
    print(f"  Layers per sample: {K}")
    print(f"  Previewing first {n_samples_preview} samples")
    
    print(f"\nFixed Boundaries (from Table 3.5):")
    print(f"  σ₁ = {metadata['fixed_boundaries']['sigma_1']:.3e} S/m")
    sigma_51_center = metadata['fixed_boundaries'].get('sigma_51_center', metadata['fixed_boundaries'].get('sigma_51'))
    print(f"  σ₅₁ ≈ {sigma_51_center:.3e} S/m")
    print(f"  μ₁ = {metadata['fixed_boundaries']['mu_1']:.1f}")
    print(f"  μ₅₁ ≈ {metadata['fixed_boundaries']['mu_51_center']:.1f}")
    
    print(f"\nFirst {n_samples_preview} Training Samples:")
    print("-" * 80)
    print(f"{'Sample':<8} {'σ₁ (S/m)':<15} {'μ₁':<8} {'σ₅₁ (S/m)':<15} {'μ₅₁':<8}")
    print("-" * 80)
    
    for i in range(n_samples_preview):
        sigma_1 = sigma_layers[i, 0]
        sigma_51 = sigma_layers[i, -1]
        mu_1 = mu_layers[i, 0]
        mu_51 = mu_layers[i, -1]
        print(f"{i+1:<8} {sigma_1:<15.3e} {mu_1:<8.2f} {sigma_51:<15.3e} {mu_51:<8.4f}")
    
    fig = plt.figure(figsize=(18, 12))
    
    layers = np.arange(K)
    
    ax1 = plt.subplot(2, 2, 1)
    for i in range(n_samples_preview):
        ax1.plot(layers, sigma_layers[i], linewidth=2, alpha=0.7, label=f'Sample {i+1}')
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('σ (S/m)', fontsize=12)
    ax1.set_title(f'Sigma Profiles (First {n_samples_preview} Samples)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(metadata['fixed_boundaries']['sigma_1'], color='red', linestyle='--', linewidth=1, alpha=0.5, label='σ₁ target')
    sigma_51_target = metadata['fixed_boundaries'].get('sigma_51_center', metadata['fixed_boundaries'].get('sigma_51'))
    ax1.axhline(sigma_51_target, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='σ₅₁ target')
    
    ax2 = plt.subplot(2, 2, 2)
    for i in range(n_samples_preview):
        ax2.plot(layers, mu_layers[i], linewidth=2, alpha=0.7, label=f'Sample {i+1}')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('μ (relative)', fontsize=12)
    ax2.set_title(f'Mu Profiles (First {n_samples_preview} Samples)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(metadata['fixed_boundaries']['mu_1'], color='red', linestyle='--', linewidth=1, alpha=0.5, label='μ₁ target')
    ax2.axhline(metadata['fixed_boundaries']['mu_51_center'], color='blue', linestyle='--', linewidth=1, alpha=0.5, label='μ₅₁ target')
    
    ax3 = plt.subplot(2, 2, 3)
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples_preview))
    for i in range(n_samples_preview):
        ax3_sigma = ax3
        ax3_sigma.plot(layers, sigma_layers[i], linewidth=2, alpha=0.7, color=colors[i], linestyle='-', label=f'σ Sample {i+1}')
    ax3_sigma.set_xlabel('Layer Index', fontsize=12)
    ax3_sigma.set_ylabel('σ (S/m)', fontsize=12, color='blue')
    ax3_sigma.tick_params(axis='y', labelcolor='blue')
    ax3_sigma.grid(True, alpha=0.3)
    ax3_sigma.legend(loc='upper left', fontsize=8)
    
    ax3_mu = ax3_sigma.twinx()
    for i in range(n_samples_preview):
        ax3_mu.plot(layers, mu_layers[i], linewidth=2, alpha=0.7, color=colors[i], linestyle='--', label=f'μ Sample {i+1}')
    ax3_mu.set_ylabel('μ (relative)', fontsize=12, color='red')
    ax3_mu.tick_params(axis='y', labelcolor='red')
    ax3_mu.legend(loc='upper right', fontsize=8)
    ax3.set_title(f'Sigma and Mu Pairs (First {n_samples_preview} Samples)', fontsize=14, fontweight='bold')
    
    ax4 = plt.subplot(2, 2, 4)
    
    sigma_1_vals = sigma_layers[:n_samples_preview, 0]
    sigma_51_vals = sigma_layers[:n_samples_preview, -1]
    mu_1_vals = mu_layers[:n_samples_preview, 0]
    mu_51_vals = mu_layers[:n_samples_preview, -1]
    
    x_pos = np.arange(n_samples_preview)
    width = 0.2
    
    ax4_1 = ax4
    bars1 = ax4_1.bar(x_pos - 1.5*width, sigma_1_vals/1e7, width, label='σ₁ (×10⁷)', alpha=0.8, color='lightblue')
    bars2 = ax4_1.bar(x_pos - 0.5*width, sigma_51_vals/1e7, width, label='σ₅₁ (×10⁷)', alpha=0.8, color='darkblue')
    ax4_1.set_ylabel('σ (×10⁷ S/m)', fontsize=11, color='blue')
    ax4_1.tick_params(axis='y', labelcolor='blue')
    ax4_1.set_xlabel('Sample Number', fontsize=12)
    ax4_1.set_xticks(x_pos)
    ax4_1.set_xticklabels([f'{i+1}' for i in range(n_samples_preview)])
    ax4_1.legend(loc='upper left', fontsize=9)
    ax4_1.grid(True, alpha=0.3, axis='y')
    
    ax4_2 = ax4_1.twinx()
    bars3 = ax4_2.bar(x_pos + 0.5*width, mu_1_vals, width, label='μ₁', alpha=0.8, color='lightcoral')
    bars4 = ax4_2.bar(x_pos + 1.5*width, mu_51_vals, width, label='μ₅₁', alpha=0.8, color='darkred')
    ax4_2.set_ylabel('μ (relative)', fontsize=11, color='red')
    ax4_2.tick_params(axis='y', labelcolor='red')
    ax4_2.legend(loc='upper right', fontsize=9)
    
    ax4.set_title('Boundary Values Comparison', fontsize=14, fontweight='bold')
    
    plt.suptitle('Training Data Preview: Fixed Boundaries with Opposite Profiles', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = data_dir / 'training_data_preview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Preview visualization saved to {output_path}")
    
    print("\nBoundary Statistics (all samples):")
    print("-" * 80)
    print(f"σ₁:  mean={sigma_layers[:, 0].mean():.3e}, std={sigma_layers[:, 0].std():.3e}")
    print(f"σ₅₁: mean={sigma_layers[:, -1].mean():.3e}, std={sigma_layers[:, -1].std():.3e}")
    print(f"μ₁:  mean={mu_layers[:, 0].mean():.4f}, std={mu_layers[:, 0].std():.4f}")
    print(f"μ₅₁: mean={mu_layers[:, -1].mean():.4f}, std={mu_layers[:, -1].std():.4f}")
    
    print("\n" + "="*60)
    print("Preview Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
