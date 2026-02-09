#!/usr/bin/env python
"""Visualize the training data with opposite sigma/mu profiles."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def main():
    data_dir = Path('./training_data')
    
    print("="*60)
    print("Visualizing Training Data with Opposite σ/μ Profiles")
    print("="*60)
    
    X_raw = np.load(data_dir / 'X_raw.npy')
    sigma_layers = np.load(data_dir / 'sigma_layers.npy')
    mu_layers = np.load(data_dir / 'mu_layers.npy')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    K = metadata['K']
    N = metadata['N']
    
    print(f"\nDataset Information:")
    print(f"  Samples: {N}")
    print(f"  Layers: {K}")
    print(f"  Profile Type: {metadata['profile_type']}")
    print(f"  Relationship: {metadata['relationship']}")
    print(f"  σ range: [{sigma_layers.min():.2e}, {sigma_layers.max():.2e}] S/m")
    print(f"  μ range: [{mu_layers.min():.2f}, {mu_layers.max():.2f}]")
    
    fig = plt.figure(figsize=(18, 12))
    
    ax1 = plt.subplot(3, 3, 1)
    sample_indices = np.random.choice(N, size=10, replace=False)
    layers = np.arange(K)
    for idx in sample_indices:
        ax1.plot(layers, sigma_layers[idx], alpha=0.6, linewidth=1.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('σ (S/m)')
    ax1.set_title('σ Profiles (10 random samples)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    for idx in sample_indices:
        ax2.plot(layers, mu_layers[idx], alpha=0.6, linewidth=1.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('μ (relative)')
    ax2.set_title('μ Profiles (10 random samples)')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    for idx in sample_indices[:3]:
        sigma_norm = (sigma_layers[idx] - sigma_layers[idx].min()) / (sigma_layers[idx].max() - sigma_layers[idx].min())
        mu_norm = (mu_layers[idx] - mu_layers[idx].min()) / (mu_layers[idx].max() - mu_layers[idx].min())
        ax3.plot(layers, sigma_norm, 'b-', alpha=0.7, linewidth=2, label=f'σ sample {idx}')
        ax3.plot(layers, mu_norm, 'r--', alpha=0.7, linewidth=2, label=f'μ sample {idx}')
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Normalized Value')
    ax3.set_title('Normalized σ vs μ (3 samples)')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(sigma_layers.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_xlabel('σ (S/m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('σ Distribution (all layers)')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(mu_layers.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    ax5.set_xlabel('μ (relative)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('μ Distribution (all layers)')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 3, 6)
    correlations = []
    for i in range(N):
        corr = np.corrcoef(sigma_layers[i], mu_layers[i])[0, 1]
        correlations.append(corr)
    
    correlations = np.array(correlations)
    mean_corr = np.mean(correlations)
    
    ax6.bar(['Correlation'], [mean_corr], alpha=0.7, color='green', edgecolor='black', width=0.5)
    ax6.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax6.set_ylabel('Correlation Coefficient')
    ax6.set_title(f'σ-μ Correlation: {mean_corr:.4f}')
    ax6.set_ylim([-1.1, 1.1])
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.text(0, mean_corr + 0.05, f'{mean_corr:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax7 = plt.subplot(3, 3, 7)
    sigma_means = sigma_layers.mean(axis=1)
    mu_means = mu_layers.mean(axis=1)
    scatter = ax7.scatter(sigma_means, mu_means, alpha=0.5, s=10, c=np.arange(N), cmap='viridis')
    ax7.set_xlabel('Mean σ (S/m)')
    ax7.set_ylabel('Mean μ (relative)')
    ax7.set_title('Mean σ vs Mean μ')
    ax7.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax7, label='Sample Index')
    
    ax8 = plt.subplot(3, 3, 8)
    layer_means_sigma = sigma_layers.mean(axis=0)
    layer_stds_sigma = sigma_layers.std(axis=0)
    ax8.plot(layers, layer_means_sigma, 'b-', linewidth=2, label='Mean')
    ax8.fill_between(layers, layer_means_sigma - layer_stds_sigma, 
                     layer_means_sigma + layer_stds_sigma, alpha=0.3, color='blue', label='±1 std')
    ax8.set_xlabel('Layer Index')
    ax8.set_ylabel('σ (S/m)')
    ax8.set_title('σ Statistics Across Samples')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax9 = plt.subplot(3, 3, 9)
    layer_means_mu = mu_layers.mean(axis=0)
    layer_stds_mu = mu_layers.std(axis=0)
    ax9.plot(layers, layer_means_mu, 'r-', linewidth=2, label='Mean')
    ax9.fill_between(layers, layer_means_mu - layer_stds_mu, 
                     layer_means_mu + layer_stds_mu, alpha=0.3, color='red', label='±1 std')
    ax9.set_xlabel('Layer Index')
    ax9.set_ylabel('μ (relative)')
    ax9.set_title('μ Statistics Across Samples')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Training Data Visualization: Opposite σ/μ Profiles', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = data_dir / 'training_data_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    
    fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        sample_idx = sample_indices[i] if i < len(sample_indices) else i
        sigma = sigma_layers[sample_idx]
        mu = mu_layers[sample_idx]
        
        ax_sigma = ax
        ax_sigma.plot(layers, sigma, 'b-', linewidth=2, label='σ')
        ax_sigma.set_xlabel('Layer Index')
        ax_sigma.set_ylabel('σ (S/m)', color='b')
        ax_sigma.tick_params(axis='y', labelcolor='b')
        ax_sigma.grid(True, alpha=0.3)
        
        ax_mu = ax_sigma.twinx()
        ax_mu.plot(layers, mu, 'r-', linewidth=2, label='μ')
        ax_mu.set_ylabel('μ (relative)', color='r')
        ax_mu.tick_params(axis='y', labelcolor='r')
        
        corr = np.corrcoef(sigma, mu)[0, 1]
        ax.set_title(f'Sample {sample_idx}: Correlation = {corr:.4f}')
    
    plt.suptitle('Individual Sample Profiles: σ and μ Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path2 = data_dir / 'individual_profiles_comparison.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Individual profiles saved to {output_path2}")
    
    print("\nStatistical Summary:")
    print("-" * 60)
    print(f"σ statistics:")
    print(f"  Mean: {sigma_layers.mean():.2e} S/m")
    print(f"  Std:  {sigma_layers.std():.2e} S/m")
    print(f"  Min:  {sigma_layers.min():.2e} S/m")
    print(f"  Max:  {sigma_layers.max():.2e} S/m")
    print(f"\nμ statistics:")
    print(f"  Mean: {mu_layers.mean():.2f}")
    print(f"  Std:  {mu_layers.std():.2f}")
    print(f"  Min:  {mu_layers.min():.2f}")
    print(f"  Max:  {mu_layers.max():.2f}")
    print(f"\nCorrelation statistics:")
    print(f"  Mean: {np.mean(correlations):.4f}")
    print(f"  Std:  {np.std(correlations):.4f}")
    print(f"  Min:  {np.min(correlations):.4f}")
    print(f"  Max:  {np.max(correlations):.4f}")
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
