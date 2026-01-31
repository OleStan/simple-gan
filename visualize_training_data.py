#!/usr/bin/env python3
"""Visualization script for training data samples."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add the eddy_current_data_generator to path for visualization functions
sys.path.append(str(Path(__file__).parent / "eddy_current_data_generator"))

from visualization.profile_visualizer import plot_dataset_statistics, plot_multiple_profiles_comparison


def load_training_data():
    """Load training data and metadata."""
    data_dir = Path("training_data")
    
    # Load data
    X_raw = np.load(data_dir / "X_raw.npy")
    sigma_layers = np.load(data_dir / "sigma_layers.npy")
    mu_layers = np.load(data_dir / "mu_layers.npy")
    
    # Load metadata
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return X_raw, sigma_layers, mu_layers, metadata


def visualize_sample_profiles(sigma_layers, mu_layers, metadata, n_samples=30):
    """Visualize sample profiles from training data."""
    print(f"Visualizing {n_samples} sample profiles...")
    
    # Create radial coordinates for plotting
    r = np.linspace(metadata['r_min'], metadata['r_max'], metadata['K'])
    
    # Select random samples
    np.random.seed(42)
    indices = np.random.choice(len(sigma_layers), n_samples, replace=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot different subsets of samples
    sample_groups = [
        ("Low σ", indices[:5]),
        ("Medium σ", indices[5:10]),
        ("High σ", indices[10:15]),
        ("Low μ", indices[15:20]),
        ("Medium μ", indices[20:25]),
        ("High μ", indices[25:30])
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    
    for idx, (title, sample_indices) in enumerate(sample_groups):
        ax = axes[idx]
        
        for i, sample_idx in enumerate(sample_indices):
            sigma_profile = sigma_layers[sample_idx]
            mu_profile = mu_layers[sample_idx]
            
            # Plot sigma
            ax.plot(r, sigma_profile, color=colors[i], alpha=0.7, linewidth=1.5,
                   label=f'Sample {sample_idx}' if idx == 0 else '')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Radial Position (r)')
        ax.set_ylabel('σ (S/m)')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.suptitle(f'Training Data: {n_samples} Sample σ Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/training_data_sigma_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot mu profiles
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (title, sample_indices) in enumerate(sample_groups):
        ax = axes[idx]
        
        for i, sample_idx in enumerate(sample_indices):
            mu_profile = mu_layers[sample_idx]
            
            # Plot mu
            ax.plot(r, mu_profile, color=colors[i], alpha=0.7, linewidth=1.5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Radial Position (r)')
        ax.set_ylabel('μᵣ')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Data: {n_samples} Sample μ Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/training_data_mu_samples.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_dual_profiles_combined(sigma_layers, mu_layers, metadata, n_samples=20):
    """Visualize dual σ and μ profiles together."""
    print(f"Visualizing {n_samples} dual profiles...")
    
    r = np.linspace(metadata['r_min'], metadata['r_max'], metadata['K'])
    
    # Select samples with different characteristics
    np.random.seed(42)
    indices = np.random.choice(len(sigma_layers), n_samples, replace=False)
    
    # Create grid layout
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample_idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        sigma_profile = sigma_layers[sample_idx]
        mu_profile = mu_layers[sample_idx]
        
        # Plot sigma on left axis
        ax.plot(r, sigma_profile, color='#2E86AB', linewidth=2, label='σ')
        ax.set_xlabel('Radial Position (r)')
        ax.set_ylabel('σ (S/m)', color='#2E86AB')
        ax.tick_params(axis='y', labelcolor='#2E86AB')
        ax.grid(True, alpha=0.3)
        
        # Plot mu on right axis
        ax2 = ax.twinx()
        ax2.plot(r, mu_profile, color='#A23B72', linewidth=2, label='μ')
        ax2.set_ylabel('μᵣ', color='#A23B72')
        ax2.tick_params(axis='y', labelcolor='#A23B72')
        
        ax.set_title(f'Sample {sample_idx}', fontsize=10)
    
    # Hide empty subplots
    for i in range(n_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Training Data: {n_samples} Dual σ-μ Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/training_data_dual_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_data_statistics(sigma_layers, mu_layers, metadata):
    """Visualize comprehensive statistics of training data."""
    print("Visualizing training data statistics...")
    
    # Combine sigma and mu for statistics function
    X_combined = np.column_stack([sigma_layers, mu_layers])
    
    fig = plot_dataset_statistics(X_combined, metadata['K'])
    plt.suptitle('Training Data Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/training_data_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_profile_variations(sigma_layers, mu_layers, metadata):
    """Visualize profile variations and patterns."""
    print("Analyzing profile variations...")
    
    r = np.linspace(metadata['r_min'], metadata['r_max'], metadata['K'])
    
    # Calculate statistics across all samples
    sigma_mean = sigma_layers.mean(axis=0)
    sigma_std = sigma_layers.std(axis=0)
    mu_mean = mu_layers.mean(axis=0)
    mu_std = mu_layers.std(axis=0)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sigma mean and std
    axes[0, 0].plot(r, sigma_mean, color='#2E86AB', linewidth=2, label='Mean')
    axes[0, 0].fill_between(r, sigma_mean - sigma_std, sigma_mean + sigma_std,
                           alpha=0.3, color='#2E86AB', label='±1σ')
    axes[0, 0].set_title('σ Profile Statistics', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Radial Position (r)')
    axes[0, 0].set_ylabel('σ (S/m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mu mean and std
    axes[0, 1].plot(r, mu_mean, color='#A23B72', linewidth=2, label='Mean')
    axes[0, 1].fill_between(r, mu_mean - mu_std, mu_mean + mu_std,
                           alpha=0.3, color='#A23B72', label='±1σ')
    axes[0, 1].set_title('μ Profile Statistics', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Radial Position (r)')
    axes[0, 1].set_ylabel('μᵣ')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sigma coefficient of variation
    sigma_cv = sigma_std / sigma_mean
    axes[1, 0].plot(r, sigma_cv, color='#2E86AB', linewidth=2)
    axes[1, 0].set_title('σ Coefficient of Variation', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Radial Position (r)')
    axes[1, 0].set_ylabel('CV = σ/μ')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mu coefficient of variation
    mu_cv = mu_std / mu_mean
    axes[1, 1].plot(r, mu_cv, color='#A23B72', linewidth=2)
    axes[1, 1].set_title('μ Coefficient of Variation', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Radial Position (r)')
    axes[1, 1].set_ylabel('CV = σ/μ')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Data: Profile Variations Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/training_data_variations.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main visualization function."""
    print("Starting training data visualization...")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Load training data
    X_raw, sigma_layers, mu_layers, metadata = load_training_data()
    
    print(f"Loaded training data:")
    print(f"  - Samples: {metadata['N']}")
    print(f"  - Layers: {metadata['K']}")
    print(f"  - Profile type: {metadata['profile_type']}")
    print(f"  - σ range: [{sigma_layers.min():.2e}, {sigma_layers.max():.2e}]")
    print(f"  - μ range: [{mu_layers.min():.2f}, {mu_layers.max():.2f}]")
    print()
    
    # Run visualizations
    visualize_data_statistics(sigma_layers, mu_layers, metadata)
    visualize_sample_profiles(sigma_layers, mu_layers, metadata, n_samples=30)
    visualize_dual_profiles_combined(sigma_layers, mu_layers, metadata, n_samples=20)
    visualize_profile_variations(sigma_layers, mu_layers, metadata)
    
    print("=" * 60)
    print("Training data visualization completed!")
    print("Check the 'results' directory for saved plots.")


if __name__ == "__main__":
    main()
