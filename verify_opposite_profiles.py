#!/usr/bin/env python
"""Verify that sigma and mu profiles have opposite relationships."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    data_dir = Path('./training_data')
    
    sigma_layers = np.load(data_dir / 'sigma_layers.npy')
    mu_layers = np.load(data_dir / 'mu_layers.npy')
    
    print("="*60)
    print("Verifying Opposite σ/μ Profiles")
    print("="*60)
    
    n_samples_to_plot = 5
    
    fig, axes = plt.subplots(n_samples_to_plot, 2, figsize=(14, 3*n_samples_to_plot))
    
    for i in range(n_samples_to_plot):
        sigma = sigma_layers[i]
        mu = mu_layers[i]
        
        layers = np.arange(len(sigma))
        
        ax1 = axes[i, 0]
        ax1.plot(layers, sigma, 'b-', linewidth=2, label='σ profile')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('σ (S/m)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Sample {i+1}: σ and μ Profiles')
        
        ax2 = ax1.twinx()
        ax2.plot(layers, mu, 'r-', linewidth=2, label='μ profile')
        ax2.set_ylabel('μ (relative)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        sigma_normalized = (sigma - sigma.min()) / (sigma.max() - sigma.min())
        mu_normalized = (mu - mu.min()) / (mu.max() - mu.min())
        
        ax3 = axes[i, 1]
        ax3.plot(layers, sigma_normalized, 'b-', linewidth=2, label='σ (normalized)')
        ax3.plot(layers, mu_normalized, 'r-', linewidth=2, label='μ (normalized)')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Normalized Value')
        ax3.set_title(f'Sample {i+1}: Normalized Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        correlation = np.corrcoef(sigma, mu)[0, 1]
        ax3.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(data_dir / 'opposite_profiles_verification.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {data_dir / 'opposite_profiles_verification.png'}")
    
    print("\nStatistical Analysis:")
    print("-" * 60)
    
    correlations = []
    for i in range(len(sigma_layers)):
        corr = np.corrcoef(sigma_layers[i], mu_layers[i])[0, 1]
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    print(f"Correlation statistics across all {len(sigma_layers)} samples:")
    print(f"  Mean correlation: {correlations.mean():.4f}")
    print(f"  Std correlation: {correlations.std():.4f}")
    print(f"  Min correlation: {correlations.min():.4f}")
    print(f"  Max correlation: {correlations.max():.4f}")
    
    negative_corr_count = np.sum(correlations < 0)
    print(f"\n  Samples with negative correlation: {negative_corr_count}/{len(correlations)} ({100*negative_corr_count/len(correlations):.1f}%)")
    
    if correlations.mean() < -0.5:
        print("\n✓ VERIFIED: Strong opposite relationship (negative correlation)")
    elif correlations.mean() < 0:
        print("\n✓ VERIFIED: Opposite relationship (negative correlation)")
    else:
        print("\n⚠ WARNING: Profiles do not show opposite relationship")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
