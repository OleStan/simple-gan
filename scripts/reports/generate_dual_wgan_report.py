#!/usr/bin/env python
"""Generate comprehensive report for dual-head WGAN training results."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
from datetime import datetime

from models.dual_wgan.model import DualHeadGenerator


def load_trained_model(model_dir, device):
    """Load trained generator model."""
    with open(f'{model_dir}/normalization_params.json', 'r') as f:
        norm_params = json.load(f)

    K = norm_params['K']
    n_classes = norm_params.get('n_classes', 1)

    config_path = Path(f'{model_dir}/config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        nz = model_config.get('nz', 100)
    else:
        nz = 100

    netG = DualHeadGenerator(nz=nz, K=K, n_classes=n_classes).to(device)

    final_path = f'{model_dir}/models/netG_final.pth'
    if Path(final_path).exists():
        model_path = final_path
        epoch_label = "final"
    else:
        model_files = list(Path(f'{model_dir}/models').glob('netG_epoch_*.pth'))
        if not model_files:
            raise FileNotFoundError(f"No generator checkpoints found in {model_dir}/models/")
        model_files.sort(key=lambda p: int(p.stem.replace('netG_epoch_', '')))
        model_path = str(model_files[-1])
        epoch_label = model_files[-1].stem.replace('netG_epoch_', '')

    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    return netG, norm_params, epoch_label, nz, n_classes


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


def generate_samples(netG, n_samples, nz, device, norm_params, K, n_classes=1, class_label=None):
    """Generate samples from trained generator, optionally conditioned on class_label."""
    with torch.no_grad():
        noise = torch.randn(n_samples, nz, device=device)
        labels = None
        if n_classes > 1:
            if class_label is not None:
                labels = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
            else:
                labels = torch.arange(n_samples, device=device) % n_classes
        fake_data, _, _ = netG(noise, labels)
        fake_data_np = fake_data.cpu().numpy()
        labels_np = labels.cpu().numpy() if labels is not None else np.zeros(n_samples, dtype=np.int64)
        sigma_denorm, mu_denorm = denormalize_profiles(fake_data_np, norm_params, K)

    return sigma_denorm, mu_denorm, fake_data_np, labels_np


def plot_sample_profiles(sigma_samples, mu_samples, n_display=16, labels=None, save_path=None):
    """Plot sample generated profiles."""
    n_rows = 4
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= n_display:
                break

            ax = axes[i, j]
            ax2 = ax.twinx()

            ax.plot(sigma_samples[idx], 'b-', linewidth=1.5, alpha=0.8)
            ax2.plot(mu_samples[idx], 'r-', linewidth=1.5, alpha=0.8)

            ax.set_ylabel('σ (S/m)', color='b', fontsize=9)
            ax2.set_ylabel('μᵣ', color='r', fontsize=9)

            ax.tick_params(axis='y', labelcolor='b', labelsize=8)
            ax2.tick_params(axis='y', labelcolor='r', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_xlabel('Layer', fontsize=8)
            ax.grid(True, alpha=0.3)

            if labels is not None:
                ax.set_title(f'Class {labels[idx]}', fontsize=7)

    plt.suptitle('Generated σ and μ Profiles', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_paired_comparison(real_data_path, sigma_gen, mu_gen, K, n_pairs=6, save_path=None):
    """Plot side-by-side comparison of real vs generated profile pairs."""
    real_data = np.load(real_data_path)
    sigma_real = real_data[:, :K]
    mu_real = real_data[:, K:2*K]
    
    n_samples_real = min(len(sigma_real), n_pairs)
    n_samples_gen = min(len(sigma_gen), n_pairs)
    
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 3*n_pairs))
    
    for i in range(n_pairs):
        ax_real = axes[i, 0]
        ax_gen = axes[i, 1]
        
        ax_real2 = ax_real.twinx()
        ax_gen2 = ax_gen.twinx()
        
        ax_real.plot(sigma_real[i], 'b-', linewidth=2, label='σ')
        ax_real2.plot(mu_real[i], 'r-', linewidth=2, label='μ')
        
        ax_gen.plot(sigma_gen[i], 'b-', linewidth=2, label='σ')
        ax_gen2.plot(mu_gen[i], 'r-', linewidth=2, label='μ')
        
        ax_real.set_ylabel('σ (S/m)', color='b', fontsize=10)
        ax_real2.set_ylabel('μᵣ', color='r', fontsize=10)
        ax_gen.set_ylabel('σ (S/m)', color='b', fontsize=10)
        ax_gen2.set_ylabel('μᵣ', color='r', fontsize=10)
        
        ax_real.tick_params(axis='y', labelcolor='b')
        ax_real2.tick_params(axis='y', labelcolor='r')
        ax_gen.tick_params(axis='y', labelcolor='b')
        ax_gen2.tick_params(axis='y', labelcolor='r')
        
        if i == 0:
            ax_real.set_title('Real Data', fontsize=12, fontweight='bold')
            ax_gen.set_title('Generated Data', fontsize=12, fontweight='bold')
        
        if i == n_pairs - 1:
            ax_real.set_xlabel('Layer', fontsize=10)
            ax_gen.set_xlabel('Layer', fontsize=10)
        
        ax_real.grid(True, alpha=0.3)
        ax_gen.grid(True, alpha=0.3)
        
        ax_real.text(0.02, 0.98, f'Pair {i+1}', transform=ax_real.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Real vs Generated Profile Pairs Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_distribution_comparison(real_data_path, sigma_gen, mu_gen, K, save_path=None):
    """Compare distributions of real vs generated data."""
    real_data = np.load(real_data_path)
    sigma_real = real_data[:, :K]
    mu_real = real_data[:, K:2*K]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(sigma_real.flatten(), bins=50, alpha=0.7, label='Real', color='blue', density=True)
    axes[0, 0].hist(sigma_gen.flatten(), bins=50, alpha=0.7, label='Generated', color='red', density=True)
    axes[0, 0].set_xlabel('σ (S/m)', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].set_title('σ Distribution Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(mu_real.flatten(), bins=50, alpha=0.7, label='Real', color='blue', density=True)
    axes[0, 1].hist(mu_gen.flatten(), bins=50, alpha=0.7, label='Generated', color='red', density=True)
    axes[0, 1].set_xlabel('μᵣ', fontsize=11)
    axes[0, 1].set_ylabel('Density', fontsize=11)
    axes[0, 1].set_title('μ Distribution Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    sigma_real_mean = sigma_real.mean(axis=0)
    sigma_real_std = sigma_real.std(axis=0)
    sigma_gen_mean = sigma_gen.mean(axis=0)
    sigma_gen_std = sigma_gen.std(axis=0)

    axes[1, 0].plot(sigma_real_mean, 'b-', linewidth=2, label='Real Mean')
    axes[1, 0].fill_between(range(K),
                            sigma_real_mean - sigma_real_std,
                            sigma_real_mean + sigma_real_std,
                            alpha=0.3, color='blue', label='Real ±1σ')
    axes[1, 0].plot(sigma_gen_mean, 'r--', linewidth=2, label='Generated Mean')
    axes[1, 0].fill_between(range(K),
                            sigma_gen_mean - sigma_gen_std,
                            sigma_gen_mean + sigma_gen_std,
                            alpha=0.3, color='red', label='Generated ±1σ')
    axes[1, 0].set_xlabel('Layer Index', fontsize=11)
    axes[1, 0].set_ylabel('σ (S/m)', fontsize=11)
    axes[1, 0].set_title('σ Profile Statistics', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    mu_real_mean = mu_real.mean(axis=0)
    mu_real_std = mu_real.std(axis=0)
    mu_gen_mean = mu_gen.mean(axis=0)
    mu_gen_std = mu_gen.std(axis=0)

    axes[1, 1].plot(mu_real_mean, 'b-', linewidth=2, label='Real Mean')
    axes[1, 1].fill_between(range(K),
                            mu_real_mean - mu_real_std,
                            mu_real_mean + mu_real_std,
                            alpha=0.3, color='blue', label='Real ±1σ')
    axes[1, 1].plot(mu_gen_mean, 'r--', linewidth=2, label='Generated Mean')
    axes[1, 1].fill_between(range(K),
                            mu_gen_mean - mu_gen_std,
                            mu_gen_mean + mu_gen_std,
                            alpha=0.3, color='red', label='Generated ±1σ')
    axes[1, 1].set_xlabel('Layer Index', fontsize=11)
    axes[1, 1].set_ylabel('μᵣ', fontsize=11)
    axes[1, 1].set_title('μ Profile Statistics', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_per_class_distributions(real_data_path, sigma_per_class, mu_per_class, K,
                                 real_labels_path=None, n_classes=2, save_path=None):
    """Overlay per-class σ and μ mean profiles for real vs generated."""
    real_data = np.load(real_data_path)
    sigma_real = real_data[:, :K]
    mu_real = real_data[:, K:2*K]

    if real_labels_path and Path(real_labels_path).exists():
        real_labels = np.load(real_labels_path).astype(np.int64)
    else:
        real_labels = np.zeros(len(real_data), dtype=np.int64)

    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for cls in range(n_classes):
        c = colors[cls % len(colors)]
        mask = real_labels == cls
        if mask.sum() == 0:
            continue

        r_sigma_mean = sigma_real[mask].mean(axis=0)
        r_mu_mean = mu_real[mask].mean(axis=0)
        axes[0, 0].plot(r_sigma_mean, color=c, linewidth=2, label=f'Real Class {cls}')
        axes[0, 1].plot(r_mu_mean, color=c, linewidth=2, label=f'Real Class {cls}')

        g_sigma_mean = sigma_per_class[cls].mean(axis=0)
        g_mu_mean = mu_per_class[cls].mean(axis=0)
        axes[1, 0].plot(g_sigma_mean, color=c, linewidth=2, linestyle='--', label=f'Gen Class {cls}')
        axes[1, 1].plot(g_mu_mean, color=c, linewidth=2, linestyle='--', label=f'Gen Class {cls}')

    for ax, title, ylabel in zip(
        [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
        ['Real σ per Class', 'Real μ per Class', 'Generated σ per Class', 'Generated μ per Class'],
        ['σ (S/m)', 'μᵣ', 'σ (S/m)', 'μᵣ'],
    ):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Per-Class Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_training_curves(model_dir, save_path=None):
    """Plot training curves."""
    history_path = f'{model_dir}/training_history.json'
    if not Path(history_path).exists():
        print(f"    Note: Training history not available yet (training still in progress)")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['loss_C'], linewidth=1.5, color='#2E86AB')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Critic Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['loss_G'], linewidth=1.5, color='#A23B72')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Generator Loss', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['wasserstein_distance'], linewidth=1.5, color='#18A558')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Distance', fontsize=11)
    axes[1, 0].set_title('Wasserstein Distance', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['gradient_penalty'], linewidth=1.5, color='#F18F01')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Penalty', fontsize=11)
    axes[1, 1].set_title('Gradient Penalty', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_report(model_dir, n_samples=1000, training_data_dir='data/training'):
    """Generate comprehensive report."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dir = Path(training_data_dir)
    real_x_path = train_dir / 'X_raw.npy'
    real_y_path = train_dir / 'y_labels.npy'

    print("="*60)
    print("Generating WGAN Dual-Head Report")
    print("="*60)
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")

    netG, norm_params, epoch_label, nz, n_classes = load_trained_model(model_dir, device)
    K = norm_params['K']

    print(f"\n✓ Model loaded from epoch {epoch_label} (K={K} layers, nz={nz}, n_classes={n_classes})")

    print(f"\nGenerating {n_samples} samples...")
    sigma_gen, mu_gen, _, gen_labels = generate_samples(netG, n_samples, nz, device, norm_params, K, n_classes)

    print(f"✓ Generated samples:")
    print(f"  σ: [{sigma_gen.min():.2e}, {sigma_gen.max():.2e}] S/m")
    print(f"  μ: [{mu_gen.min():.2f}, {mu_gen.max():.2f}]")

    report_dir = f'{model_dir}/report_epoch_{epoch_label}'
    Path(report_dir).mkdir(exist_ok=True)

    print("\nGenerating visualizations...")
    n_plots = 6 if n_classes > 1 else 5

    print(f"  1/{n_plots} Sample profiles...")
    plot_sample_profiles(sigma_gen, mu_gen, n_display=16,
                         labels=gen_labels if n_classes > 1 else None,
                         save_path=f'{report_dir}/sample_profiles.png')

    if real_x_path.exists():
        print(f"  2/{n_plots} Paired comparison (Real vs Generated)...")
        plot_paired_comparison(str(real_x_path), sigma_gen, mu_gen, K, n_pairs=6,
                               save_path=f'{report_dir}/paired_comparison.png')

        print(f"  3/{n_plots} Distribution comparison...")
        plot_distribution_comparison(str(real_x_path), sigma_gen, mu_gen, K,
                                     save_path=f'{report_dir}/distribution_comparison.png')
    else:
        print(f"  WARN: Real data not found at {real_x_path}, skipping comparisons.")

    print(f"  4/{n_plots} Training curves...")
    curves_fig = plot_training_curves(model_dir, save_path=f'{report_dir}/training_curves.png')
    if curves_fig is None:
        print("    Skipping training curves (will be available when training completes)")

    print(f"  5/{n_plots} Additional samples...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i in range(6):
        ax = axes[i // 3, i % 3]
        ax2 = ax.twinx()
        idx = i * 20
        ax.plot(sigma_gen[idx], 'b-', linewidth=2, label='σ')
        ax2.plot(mu_gen[idx], 'r-', linewidth=2, label='μ')
        ax.set_ylabel('σ (S/m)', color='b', fontsize=10)
        ax2.set_ylabel('μᵣ', color='r', fontsize=10)
        ax.set_xlabel('Layer', fontsize=10)
        title = f'Sample {i+1}'
        if n_classes > 1:
            title += f' (Class {gen_labels[idx]})'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{report_dir}/detailed_samples.png', dpi=300, bbox_inches='tight')
    plt.close()

    if n_classes > 1 and real_x_path.exists():
        print(f"  6/{n_plots} Per-class distribution comparison...")
        sigma_per_class = {}
        mu_per_class = {}
        for cls in range(n_classes):
            sigma_gen_cls, mu_gen_cls, _, _ = generate_samples(
                netG, n_samples // n_classes, nz, device, norm_params, K, n_classes, class_label=cls
            )
            sigma_per_class[cls] = sigma_gen_cls
            mu_per_class[cls] = mu_gen_cls

        plot_per_class_distributions(
            str(real_x_path),
            sigma_per_class, mu_per_class, K,
            real_labels_path=str(real_y_path) if real_y_path.exists() else None,
            n_classes=n_classes,
            save_path=f'{report_dir}/per_class_distributions.png',
        )

    np.save(f'{report_dir}/generated_sigma.npy', sigma_gen)
    np.save(f'{report_dir}/generated_mu.npy', mu_gen)
    if n_classes > 1:
        np.save(f'{report_dir}/generated_labels.npy', gen_labels)

    stats = {
        'n_samples': int(n_samples),
        'K': int(K),
        'n_classes': n_classes,
        'sigma': {
            'min': float(sigma_gen.min()),
            'max': float(sigma_gen.max()),
            'mean': float(sigma_gen.mean()),
            'std': float(sigma_gen.std())
        },
        'mu': {
            'min': float(mu_gen.min()),
            'max': float(mu_gen.max()),
            'mean': float(mu_gen.mean()),
            'std': float(mu_gen.std())
        },
        'model_dir': model_dir,
        'generation_date': datetime.now().isoformat()
    }

    if n_classes > 1:
        stats['per_class'] = {}
        for cls in range(n_classes):
            mask = gen_labels == cls
            stats['per_class'][str(cls)] = {
                'sigma_mean': float(sigma_gen[mask].mean()) if mask.sum() > 0 else 0.0,
                'mu_mean': float(mu_gen[mask].mean()) if mask.sum() > 0 else 0.0,
                'n_samples': int(mask.sum()),
            }

    with open(f'{report_dir}/generation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("Report Generation Complete!")
    print("="*60)
    print(f"\nReport saved to: {report_dir}/")
    print(f"  - sample_profiles.png")
    print(f"  - paired_comparison.png")
    print(f"  - distribution_comparison.png")
    if n_classes > 1:
        print(f"  - per_class_distributions.png")
    print(f"  - training_curves.png")
    print(f"  - detailed_samples.png")
    print(f"  - generated_sigma.npy / generated_mu.npy")
    print(f"  - generation_stats.json")

    return stats, report_dir


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        results_dirs = sorted(Path('./results').glob('dual_wgan_*'))
        if not results_dirs:
            print("No training results found in ./results/")
            sys.exit(1)
        model_dir = str(results_dirs[-1])
        print(f"Using most recent training: {model_dir}")
    
    stats, report_dir = generate_report(model_dir, n_samples=1000)
