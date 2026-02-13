#!/usr/bin/env python
"""Compare original and improved WGAN approaches."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os
from scipy import stats

from wgan_dual_profiles import DualHeadGenerator as OriginalGenerator
from wgan_improved import Conv1DGenerator, ProfileQualityMetrics


def load_model_and_generate(model_class, model_path, nz, K, n_samples=1000, device='cpu'):
    """Load a trained model and generate samples."""
    model = model_class(nz=nz, K=K).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    samples = []
    with torch.no_grad():
        for _ in range(n_samples // 64 + 1):
            noise = torch.randn(64, nz, device=device)
            fake_data, _, _ = model(noise)
            samples.append(fake_data.cpu().numpy())
    
    samples = np.concatenate(samples, axis=0)[:n_samples]
    return samples


def denormalize_profiles(data, norm_params, K):
    """Denormalize profiles back to physical units."""
    sigma_normalized = data[:, :K]
    mu_normalized = data[:, K:2*K]
    
    sigma_min = norm_params['sigma_min']
    sigma_max = norm_params['sigma_max']
    mu_min = norm_params['mu_min']
    mu_max = norm_params['mu_max']
    
    sigma = (sigma_normalized + 1) / 2 * (sigma_max - sigma_min) + sigma_min
    mu = (mu_normalized + 1) / 2 * (mu_max - mu_min) + mu_min
    
    return sigma, mu


def compute_statistical_metrics(generated, real):
    """Compute statistical comparison metrics."""
    metrics = {}
    
    metrics['mean_diff'] = np.abs(generated.mean() - real.mean())
    metrics['std_diff'] = np.abs(generated.std() - real.std())
    
    ks_stat, ks_pvalue = stats.ks_2samp(generated.flatten(), real.flatten())
    metrics['ks_statistic'] = ks_stat
    metrics['ks_pvalue'] = ks_pvalue
    
    metrics['min_diff'] = np.abs(generated.min() - real.min())
    metrics['max_diff'] = np.abs(generated.max() - real.max())
    
    return metrics


def evaluate_physical_plausibility(sigma, mu, K):
    """Evaluate physical plausibility of generated profiles."""
    sigma_tensor = torch.FloatTensor(sigma)
    mu_tensor = torch.FloatTensor(mu)
    
    metrics = ProfileQualityMetrics.evaluate_batch(sigma_tensor, mu_tensor)
    
    smoothness_diff = sigma_tensor[:, 1:] - sigma_tensor[:, :-1]
    metrics['sigma_max_gradient'] = torch.abs(smoothness_diff).max(dim=1)[0].mean().item()
    
    smoothness_diff = mu_tensor[:, 1:] - mu_tensor[:, :-1]
    metrics['mu_max_gradient'] = torch.abs(smoothness_diff).max(dim=1)[0].mean().item()
    
    return metrics


def compare_approaches(original_dir, improved_dir, real_data_path, output_dir):
    """Comprehensive comparison of both approaches."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nz = 100
    K = 50
    n_samples = 500
    
    print("Loading normalization parameters...")
    with open(Path(original_dir) / 'normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    print("Loading real data...")
    real_data = np.load(real_data_path)
    real_normalized = real_data.copy()
    sigma_data = real_data[:, :K]
    mu_data = real_data[:, K:2*K]
    real_normalized[:, :K] = 2 * (sigma_data - sigma_data.min()) / (sigma_data.max() - sigma_data.min()) - 1
    real_normalized[:, K:2*K] = 2 * (mu_data - mu_data.min()) / (mu_data.max() - mu_data.min()) - 1
    
    print("\nGenerating samples from ORIGINAL model...")
    original_model_path = Path(original_dir) / 'models' / 'netG_final.pth'
    original_samples = load_model_and_generate(
        OriginalGenerator, original_model_path, nz, K, n_samples, device
    )
    
    print("Generating samples from IMPROVED model...")
    improved_model_path = Path(improved_dir) / 'models' / 'netG_final.pth'
    improved_samples = load_model_and_generate(
        Conv1DGenerator, improved_model_path, nz, K, n_samples, device
    )
    
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    comparison_results = {
        'original': {},
        'improved': {},
        'real': {}
    }
    
    for name, samples in [('original', original_samples), ('improved', improved_samples)]:
        print(f"\n{name.upper()} Model:")
        
        sigma_norm = samples[:, :K]
        mu_norm = samples[:, K:]
        
        sigma, mu = denormalize_profiles(samples, norm_params, K)
        
        print(f"  σ range: [{sigma.min():.2e}, {sigma.max():.2e}] S/m")
        print(f"  μ range: [{mu.min():.2f}, {mu.max():.2f}]")
        
        sigma_stats = compute_statistical_metrics(sigma, sigma_data)
        mu_stats = compute_statistical_metrics(mu, mu_data)
        
        print(f"\n  σ Statistics:")
        print(f"    Mean diff: {sigma_stats['mean_diff']:.2e}")
        print(f"    Std diff: {sigma_stats['std_diff']:.2e}")
        print(f"    KS statistic: {sigma_stats['ks_statistic']:.4f} (p={sigma_stats['ks_pvalue']:.4f})")
        
        print(f"\n  μ Statistics:")
        print(f"    Mean diff: {mu_stats['mean_diff']:.2f}")
        print(f"    Std diff: {mu_stats['std_diff']:.2f}")
        print(f"    KS statistic: {mu_stats['ks_statistic']:.4f} (p={mu_stats['ks_pvalue']:.4f})")
        
        phys_metrics = evaluate_physical_plausibility(sigma_norm, mu_norm, K)
        
        print(f"\n  Physical Plausibility:")
        print(f"    σ smoothness: {phys_metrics['sigma_smoothness']:.4f}")
        print(f"    μ smoothness: {phys_metrics['mu_smoothness']:.4f}")
        print(f"    σ monotonicity: {phys_metrics['sigma_monotonicity']:.4f}")
        print(f"    μ monotonicity: {phys_metrics['mu_monotonicity']:.4f}")
        print(f"    σ diversity: {phys_metrics['sigma_diversity']:.4f}")
        print(f"    μ diversity: {phys_metrics['mu_diversity']:.4f}")
        
        comparison_results[name] = {
            'sigma_stats': sigma_stats,
            'mu_stats': mu_stats,
            'physical_metrics': phys_metrics,
            'sigma_range': [float(sigma.min()), float(sigma.max())],
            'mu_range': [float(mu.min()), float(mu.max())]
        }
    
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.4)
    
    sigma_real = sigma_data
    mu_real = mu_data
    sigma_orig, mu_orig = denormalize_profiles(original_samples, norm_params, K)
    sigma_imp, mu_imp = denormalize_profiles(improved_samples, norm_params, K)
    
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(min(20, len(sigma_real))):
        ax1.plot(sigma_real[i], alpha=0.3, color='blue')
    ax1.set_title('Real σ Profiles', fontweight='bold')
    ax1.set_ylabel('σ (S/m)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(min(20, len(sigma_orig))):
        ax2.plot(sigma_orig[i], alpha=0.3, color='green')
    ax2.set_title('Original Model σ', fontweight='bold')
    ax2.set_ylabel('σ (S/m)')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(min(20, len(sigma_imp))):
        ax3.plot(sigma_imp[i], alpha=0.3, color='red')
    ax3.set_title('Improved Model σ', fontweight='bold')
    ax3.set_ylabel('σ (S/m)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(min(20, len(mu_real))):
        ax4.plot(mu_real[i], alpha=0.3, color='blue')
    ax4.set_title('Real μ Profiles', fontweight='bold')
    ax4.set_ylabel('μᵣ')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    for i in range(min(20, len(mu_orig))):
        ax5.plot(mu_orig[i], alpha=0.3, color='green')
    ax5.set_title('Original Model μ', fontweight='bold')
    ax5.set_ylabel('μᵣ')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    for i in range(min(20, len(mu_imp))):
        ax6.plot(mu_imp[i], alpha=0.3, color='red')
    ax6.set_title('Improved Model μ', fontweight='bold')
    ax6.set_ylabel('μᵣ')
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[0, 3])
    ax7.hist(sigma_real.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    ax7.hist(sigma_orig.flatten(), bins=50, alpha=0.5, label='Original', density=True)
    ax7.hist(sigma_imp.flatten(), bins=50, alpha=0.5, label='Improved', density=True)
    ax7.set_title('σ Distribution', fontweight='bold')
    ax7.set_xlabel('σ (S/m)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(mu_real.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    ax8.hist(mu_orig.flatten(), bins=50, alpha=0.5, label='Original', density=True)
    ax8.hist(mu_imp.flatten(), bins=50, alpha=0.5, label='Improved', density=True)
    ax8.set_title('μ Distribution', fontweight='bold')
    ax8.set_xlabel('μᵣ')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, :2])
    metrics_to_plot = ['sigma_smoothness', 'mu_smoothness', 'sigma_diversity', 'mu_diversity']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    orig_values = [comparison_results['original']['physical_metrics'][m] for m in metrics_to_plot]
    imp_values = [comparison_results['improved']['physical_metrics'][m] for m in metrics_to_plot]
    
    ax9.bar(x - width/2, orig_values, width, label='Original', alpha=0.8)
    ax9.bar(x + width/2, imp_values, width, label='Improved', alpha=0.8)
    ax9.set_ylabel('Score')
    ax9.set_title('Quality Metrics Comparison', fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels([m.replace('_', '\n') for m in metrics_to_plot], fontsize=9)
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    ax10 = fig.add_subplot(gs[2, 2:])
    summary_text = (
        "COMPARISON SUMMARY\n\n"
        "Original Model:\n"
        f"  KS stat (σ): {comparison_results['original']['sigma_stats']['ks_statistic']:.4f}\n"
        f"  KS stat (μ): {comparison_results['original']['mu_stats']['ks_statistic']:.4f}\n"
        f"  Smoothness (σ): {comparison_results['original']['physical_metrics']['sigma_smoothness']:.4f}\n"
        f"  Smoothness (μ): {comparison_results['original']['physical_metrics']['mu_smoothness']:.4f}\n\n"
        "Improved Model:\n"
        f"  KS stat (σ): {comparison_results['improved']['sigma_stats']['ks_statistic']:.4f}\n"
        f"  KS stat (μ): {comparison_results['improved']['mu_stats']['ks_statistic']:.4f}\n"
        f"  Smoothness (σ): {comparison_results['improved']['physical_metrics']['sigma_smoothness']:.4f}\n"
        f"  Smoothness (μ): {comparison_results['improved']['physical_metrics']['mu_smoothness']:.4f}\n"
    )
    ax10.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
              family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax10.axis('off')
    
    plt.suptitle('WGAN Approaches Comparison: Original vs Improved', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f'{output_dir}/comparison_results.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison visualization saved to: {output_dir}/comparison_results.png")
    
    with open(f'{output_dir}/comparison_metrics.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Metrics saved to: {output_dir}/comparison_metrics.json")
    
    return comparison_results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python compare_wgan_approaches.py <original_results_dir> <improved_results_dir>")
        sys.exit(1)
    
    original_dir = sys.argv[1]
    improved_dir = sys.argv[2]
    
    real_data_path = './training_data/X_raw.npy'
    output_dir = './results/comparison_analysis'
    
    compare_approaches(original_dir, improved_dir, real_data_path, output_dir)
