#!/usr/bin/env python
"""Generate comprehensive comparison report between Dual WGAN and Improved WGAN v2."""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

def load_model_data(model_dir):
    """Load generated samples and training history for a model."""
    model_dir = Path(model_dir)
    
    report_dirs = list(model_dir.glob('report_epoch_*'))
    if not report_dirs:
        raise ValueError(f"No report found in {model_dir}")
    
    report_dir = sorted(report_dirs)[-1]
    
    sigma = np.load(report_dir / 'generated_sigma.npy')
    mu = np.load(report_dir / 'generated_mu.npy')
    
    with open(report_dir / 'generation_stats.json', 'r') as f:
        stats = json.load(f)
    
    with open(model_dir / 'training_history.json', 'r') as f:
        history = json.load(f)
    
    return sigma, mu, stats, history, report_dir


def main():
    print("="*60)
    print("Generating Comparison Report")
    print("="*60)
    
    dual_wgan_dir = 'results/dual_wgan_20260126_220422'
    improved_v2_dir = 'results/improved_wgan_v2_20260126_223129'
    
    print(f"\nLoading Dual WGAN data from: {dual_wgan_dir}")
    sigma_dual, mu_dual, stats_dual, history_dual, _ = load_model_data(dual_wgan_dir)
    
    print(f"Loading Improved WGAN v2 data from: {improved_v2_dir}")
    sigma_v2, mu_v2, stats_v2, history_v2, _ = load_model_data(improved_v2_dir)
    
    real_data = np.load('training_data/X_raw.npy')
    K = 50
    sigma_real = real_data[:, :K]
    mu_real = real_data[:, K:2*K]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'results/comparison_analysis_{timestamp}')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    print("\nGenerating comparison visualizations...")
    
    print("  1/6 Distribution comparison...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribution Comparison: Real vs Dual WGAN vs Improved WGAN v2', 
                 fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(sigma_real.flatten(), bins=50, alpha=0.4, label='Real', density=True)
    axes[0, 0].hist(sigma_dual.flatten(), bins=50, alpha=0.4, label='Dual WGAN', density=True)
    axes[0, 0].hist(sigma_v2.flatten(), bins=50, alpha=0.4, label='Improved v2', density=True)
    axes[0, 0].set_xlabel('σ (S/m)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('σ Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(mu_real.flatten(), bins=50, alpha=0.4, label='Real', density=True)
    axes[0, 1].hist(mu_dual.flatten(), bins=50, alpha=0.4, label='Dual WGAN', density=True)
    axes[0, 1].hist(mu_v2.flatten(), bins=50, alpha=0.4, label='Improved v2', density=True)
    axes[0, 1].set_xlabel('μ (relative)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('μ Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].boxplot([sigma_real.flatten(), sigma_dual.flatten(), sigma_v2.flatten()],
                       labels=['Real', 'Dual', 'Improved v2'])
    axes[0, 2].set_ylabel('σ (S/m)')
    axes[0, 2].set_title('σ Box Plot')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(sigma_real.mean(axis=0), label='Real', linewidth=2.5, alpha=0.8)
    axes[1, 0].plot(sigma_dual.mean(axis=0), label='Dual WGAN', linewidth=2, alpha=0.8, linestyle='--')
    axes[1, 0].plot(sigma_v2.mean(axis=0), label='Improved v2', linewidth=2, alpha=0.8, linestyle=':')
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('σ (S/m)')
    axes[1, 0].set_title('σ Mean Profile')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(mu_real.mean(axis=0), label='Real', linewidth=2.5, alpha=0.8)
    axes[1, 1].plot(mu_dual.mean(axis=0), label='Dual WGAN', linewidth=2, alpha=0.8, linestyle='--')
    axes[1, 1].plot(mu_v2.mean(axis=0), label='Improved v2', linewidth=2, alpha=0.8, linestyle=':')
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('μ (relative)')
    axes[1, 1].set_title('μ Mean Profile')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].boxplot([mu_real.flatten(), mu_dual.flatten(), mu_v2.flatten()],
                       labels=['Real', 'Dual', 'Improved v2'])
    axes[1, 2].set_ylabel('μ (relative)')
    axes[1, 2].set_title('μ Box Plot')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  2/6 Training convergence comparison...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Convergence Comparison', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(history_dual['loss_C'], label='Dual WGAN', alpha=0.7)
    axes[0, 0].plot(history_v2['loss_C'], label='Improved v2', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Critic Loss')
    axes[0, 0].set_title('Critic Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history_dual['loss_G'], label='Dual WGAN', alpha=0.7)
    axes[0, 1].plot(history_v2['loss_G'], label='Improved v2', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Generator Loss')
    axes[0, 1].set_title('Generator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(history_dual['wasserstein_distance'], label='Dual WGAN', alpha=0.7)
    axes[0, 2].plot(history_v2['wasserstein_distance'], label='Improved v2', alpha=0.7)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Wasserstein Distance')
    axes[0, 2].set_title('Wasserstein Distance')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history_dual['loss_C'][-100:], label='Dual WGAN', alpha=0.7)
    axes[1, 0].plot(history_v2['loss_C'][-100:], label='Improved v2', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch (last 100)')
    axes[1, 0].set_ylabel('Critic Loss')
    axes[1, 0].set_title('Critic Loss (Final 100 Epochs)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history_dual['loss_G'][-100:], label='Dual WGAN', alpha=0.7)
    axes[1, 1].plot(history_v2['loss_G'][-100:], label='Improved v2', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch (last 100)')
    axes[1, 1].set_ylabel('Generator Loss')
    axes[1, 1].set_title('Generator Loss (Final 100 Epochs)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(history_dual['wasserstein_distance'][-100:], label='Dual WGAN', alpha=0.7)
    axes[1, 2].plot(history_v2['wasserstein_distance'][-100:], label='Improved v2', alpha=0.7)
    axes[1, 2].set_xlabel('Epoch (last 100)')
    axes[1, 2].set_ylabel('Wasserstein Distance')
    axes[1, 2].set_title('Wasserstein Distance (Final 100 Epochs)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  3/6 Quality metrics comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quality Metrics Comparison', fontsize=16, fontweight='bold')
    
    sigma_grad_real = np.abs(np.diff(sigma_real, axis=1)).mean()
    sigma_grad_dual = np.abs(np.diff(sigma_dual, axis=1)).mean()
    sigma_grad_v2 = np.abs(np.diff(sigma_v2, axis=1)).mean()
    
    mu_grad_real = np.abs(np.diff(mu_real, axis=1)).mean()
    mu_grad_dual = np.abs(np.diff(mu_dual, axis=1)).mean()
    mu_grad_v2 = np.abs(np.diff(mu_v2, axis=1)).mean()
    
    axes[0, 0].bar(['Real', 'Dual WGAN', 'Improved v2'], 
                   [sigma_grad_real, sigma_grad_dual, sigma_grad_v2],
                   color=['blue', 'orange', 'green'], alpha=0.7)
    axes[0, 0].set_ylabel('Mean Absolute Gradient')
    axes[0, 0].set_title('σ Smoothness (Lower = Smoother)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].bar(['Real', 'Dual WGAN', 'Improved v2'],
                   [mu_grad_real, mu_grad_dual, mu_grad_v2],
                   color=['blue', 'orange', 'green'], alpha=0.7)
    axes[0, 1].set_ylabel('Mean Absolute Gradient')
    axes[0, 1].set_title('μ Smoothness (Lower = Smoother)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    sigma_std_real = sigma_real.std(axis=0).mean()
    sigma_std_dual = sigma_dual.std(axis=0).mean()
    sigma_std_v2 = sigma_v2.std(axis=0).mean()
    
    mu_std_real = mu_real.std(axis=0).mean()
    mu_std_dual = mu_dual.std(axis=0).mean()
    mu_std_v2 = mu_v2.std(axis=0).mean()
    
    axes[1, 0].bar(['Real', 'Dual WGAN', 'Improved v2'],
                   [sigma_std_real, sigma_std_dual, sigma_std_v2],
                   color=['blue', 'orange', 'green'], alpha=0.7)
    axes[1, 0].set_ylabel('Mean Std Dev')
    axes[1, 0].set_title('σ Diversity')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].bar(['Real', 'Dual WGAN', 'Improved v2'],
                   [mu_std_real, mu_std_dual, mu_std_v2],
                   color=['blue', 'orange', 'green'], alpha=0.7)
    axes[1, 1].set_ylabel('Mean Std Dev')
    axes[1, 1].set_title('μ Diversity')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  4/6 Sample comparison...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Sample Profile Comparison (Real vs Generated)', fontsize=16, fontweight='bold')
    
    for i in range(3):
        axes[i, 0].plot(sigma_real[i], label='σ', linewidth=2, alpha=0.8, color='blue')
        axes[i, 0].plot(mu_real[i], label='μ', linewidth=2, alpha=0.8, color='orange')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].set_title(f'Real Sample {i+1}', fontweight='bold')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(sigma_dual[i], label='σ', linewidth=2, alpha=0.8, color='blue')
        axes[i, 1].plot(mu_dual[i], label='μ', linewidth=2, alpha=0.8, color='orange')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].set_title(f'Dual WGAN Sample {i+1}')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(sigma_v2[i], label='σ', linewidth=2, alpha=0.8, color='blue')
        axes[i, 2].plot(mu_v2[i], label='μ', linewidth=2, alpha=0.8, color='orange')
        axes[i, 2].set_ylabel('Value')
        axes[i, 2].set_title(f'Improved v2 Sample {i+1}')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
        
        axes[i, 3].plot(sigma_real[i], label='Real σ', linewidth=2, alpha=0.6, color='blue', linestyle='-')
        axes[i, 3].plot(sigma_dual[i], label='Dual σ', linewidth=1.5, alpha=0.6, color='green', linestyle='--')
        axes[i, 3].plot(sigma_v2[i], label='Improved σ', linewidth=1.5, alpha=0.6, color='red', linestyle=':')
        axes[i, 3].set_ylabel('σ (S/m)')
        axes[i, 3].set_title(f'σ Overlay {i+1}')
        axes[i, 3].legend(fontsize=8)
        axes[i, 3].grid(True, alpha=0.3)
    
    for ax in axes[-1, :]:
        ax.set_xlabel('Layer Index')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  5/6 Real data showcase...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Real Training Data Examples', fontsize=16, fontweight='bold')
    
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        ax.plot(sigma_real[i*10], label='σ', linewidth=2.5, alpha=0.8, color='blue')
        ax.plot(mu_real[i*10], label='μ', linewidth=2.5, alpha=0.8, color='orange')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Real Sample {i*10+1}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([sigma_real.min()*0.95, sigma_real.max()*1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_data_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  6/6 Statistical comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Statistical Comparison', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(sigma_real.std(axis=0), label='Real', linewidth=2.5, alpha=0.8)
    axes[0, 0].plot(sigma_dual.std(axis=0), label='Dual WGAN', linewidth=2, alpha=0.8, linestyle='--')
    axes[0, 0].plot(sigma_v2.std(axis=0), label='Improved v2', linewidth=2, alpha=0.8, linestyle=':')
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Std Dev')
    axes[0, 0].set_title('σ Standard Deviation per Layer')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(mu_real.std(axis=0), label='Real', linewidth=2.5, alpha=0.8)
    axes[0, 1].plot(mu_dual.std(axis=0), label='Dual WGAN', linewidth=2, alpha=0.8, linestyle='--')
    axes[0, 1].plot(mu_v2.std(axis=0), label='Improved v2', linewidth=2, alpha=0.8, linestyle=':')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].set_title('μ Standard Deviation per Layer')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    sigma_diff_dual = np.abs(sigma_dual.mean(axis=0) - sigma_real.mean(axis=0))
    sigma_diff_v2 = np.abs(sigma_v2.mean(axis=0) - sigma_real.mean(axis=0))
    
    axes[1, 0].plot(sigma_diff_dual, label='Dual WGAN', linewidth=2, alpha=0.8)
    axes[1, 0].plot(sigma_diff_v2, label='Improved v2', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Absolute Difference')
    axes[1, 0].set_title('σ Mean Difference from Real')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    mu_diff_dual = np.abs(mu_dual.mean(axis=0) - mu_real.mean(axis=0))
    mu_diff_v2 = np.abs(mu_v2.mean(axis=0) - mu_real.mean(axis=0))
    
    axes[1, 1].plot(mu_diff_dual, label='Dual WGAN', linewidth=2, alpha=0.8)
    axes[1, 1].plot(mu_diff_v2, label='Improved v2', linewidth=2, alpha=0.8)
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('Absolute Difference')
    axes[1, 1].set_title('μ Mean Difference from Real')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  7/7 Performance summary...")
    
    from scipy.stats import wasserstein_distance as wd
    
    wd_sigma_dual = wd(sigma_real.flatten(), sigma_dual.flatten())
    wd_sigma_v2 = wd(sigma_real.flatten(), sigma_v2.flatten())
    wd_mu_dual = wd(mu_real.flatten(), mu_dual.flatten())
    wd_mu_v2 = wd(mu_real.flatten(), mu_v2.flatten())
    
    mse_sigma_dual = ((sigma_dual.mean(axis=0) - sigma_real.mean(axis=0)) ** 2).mean()
    mse_sigma_v2 = ((sigma_v2.mean(axis=0) - sigma_real.mean(axis=0)) ** 2).mean()
    mse_mu_dual = ((mu_dual.mean(axis=0) - mu_real.mean(axis=0)) ** 2).mean()
    mse_mu_v2 = ((mu_v2.mean(axis=0) - mu_real.mean(axis=0)) ** 2).mean()
    
    comparison_stats = {
        'dual_wgan': {
            'final_losses': stats_dual.get('final_losses', {
                'critic': float(history_dual['loss_C'][-1]),
                'generator': float(history_dual['loss_G'][-1]),
                'wasserstein_distance': float(history_dual['wasserstein_distance'][-1])
            }),
            'sigma_stats': stats_dual['sigma'],
            'mu_stats': stats_dual['mu'],
            'quality_metrics': {
                'sigma_smoothness': float(sigma_grad_dual),
                'mu_smoothness': float(mu_grad_dual),
                'sigma_diversity': float(sigma_std_dual),
                'mu_diversity': float(mu_std_dual),
                'wasserstein_distance_sigma': float(wd_sigma_dual),
                'wasserstein_distance_mu': float(wd_mu_dual),
                'mse_sigma': float(mse_sigma_dual),
                'mse_mu': float(mse_mu_dual)
            }
        },
        'improved_wgan_v2': {
            'final_losses': stats_v2.get('final_losses', {
                'critic': float(history_v2['loss_C'][-1]),
                'generator': float(history_v2['loss_G'][-1]),
                'wasserstein_distance': float(history_v2['wasserstein_distance'][-1])
            }),
            'sigma_stats': stats_v2['sigma'],
            'mu_stats': stats_v2['mu'],
            'quality_metrics': {
                'sigma_smoothness': float(sigma_grad_v2),
                'mu_smoothness': float(mu_grad_v2),
                'sigma_diversity': float(sigma_std_v2),
                'mu_diversity': float(mu_std_v2),
                'wasserstein_distance_sigma': float(wd_sigma_v2),
                'wasserstein_distance_mu': float(wd_mu_v2),
                'mse_sigma': float(mse_sigma_v2),
                'mse_mu': float(mse_mu_v2)
            }
        },
        'real_data': {
            'sigma_stats': {
                'min': float(sigma_real.min()),
                'max': float(sigma_real.max()),
                'mean': float(sigma_real.mean()),
                'std': float(sigma_real.std())
            },
            'mu_stats': {
                'min': float(mu_real.min()),
                'max': float(mu_real.max()),
                'mean': float(mu_real.mean()),
                'std': float(mu_real.std())
            },
            'quality_metrics': {
                'sigma_smoothness': float(sigma_grad_real),
                'mu_smoothness': float(mu_grad_real),
                'sigma_diversity': float(sigma_std_real),
                'mu_diversity': float(mu_std_real)
            }
        }
    }
    
    with open(output_dir / 'comparison_stats.json', 'w') as f:
        json.dump(comparison_stats, f, indent=2)
    
    print("\n" + "="*60)
    print("Comparison Report Complete!")
    print("="*60)
    print(f"\nReport saved to: {output_dir}/")
    print("  - distribution_comparison.png")
    print("  - training_convergence.png")
    print("  - quality_metrics.png")
    print("  - sample_comparison.png (with real data)")
    print("  - real_data_samples.png (real data showcase)")
    print("  - statistical_comparison.png")
    print("  - comparison_stats.json")
    
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    print("\n📊 Wasserstein Distance (lower = better match to real data):")
    print(f"  σ - Dual WGAN:      {wd_sigma_dual:.2e}")
    print(f"  σ - Improved v2:    {wd_sigma_v2:.2e}")
    print(f"  μ - Dual WGAN:      {wd_mu_dual:.2f}")
    print(f"  μ - Improved v2:    {wd_mu_v2:.2f}")
    
    print("\n📈 Mean Square Error (lower = better):")
    print(f"  σ - Dual WGAN:      {mse_sigma_dual:.2e}")
    print(f"  σ - Improved v2:    {mse_sigma_v2:.2e}")
    print(f"  μ - Dual WGAN:      {mse_mu_dual:.2f}")
    print(f"  μ - Improved v2:    {mse_mu_v2:.2f}")
    
    print("\n🎯 Smoothness (lower = smoother):")
    print(f"  σ - Real:           {sigma_grad_real:.2e}")
    print(f"  σ - Dual WGAN:      {sigma_grad_dual:.2e}")
    print(f"  σ - Improved v2:    {sigma_grad_v2:.2e}")
    print(f"  μ - Real:           {mu_grad_real:.2f}")
    print(f"  μ - Dual WGAN:      {mu_grad_dual:.2f}")
    print(f"  μ - Improved v2:    {mu_grad_v2:.2f}")
    
    print("\n🌈 Diversity (std dev):")
    print(f"  σ - Real:           {sigma_std_real:.2e}")
    print(f"  σ - Dual WGAN:      {sigma_std_dual:.2e}")
    print(f"  σ - Improved v2:    {sigma_std_v2:.2e}")
    print(f"  μ - Real:           {mu_std_real:.2f}")
    print(f"  μ - Dual WGAN:      {mu_std_dual:.2f}")
    print(f"  μ - Improved v2:    {mu_std_v2:.2f}")
    
    print("\n🏆 Winner Analysis:")
    if wd_sigma_dual < wd_sigma_v2:
        print("  σ Distribution Match: Dual WGAN ✓")
    else:
        print("  σ Distribution Match: Improved v2 ✓")
    
    if wd_mu_dual < wd_mu_v2:
        print("  μ Distribution Match: Dual WGAN ✓")
    else:
        print("  μ Distribution Match: Improved v2 ✓")
    
    if abs(sigma_grad_dual - sigma_grad_real) < abs(sigma_grad_v2 - sigma_grad_real):
        print("  σ Smoothness Match: Dual WGAN ✓")
    else:
        print("  σ Smoothness Match: Improved v2 ✓")
    
    if abs(mu_grad_dual - mu_grad_real) < abs(mu_grad_v2 - mu_grad_real):
        print("  μ Smoothness Match: Dual WGAN ✓")
    else:
        print("  μ Smoothness Match: Improved v2 ✓")


if __name__ == '__main__':
    main()
