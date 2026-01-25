"""Visualization functions for material profiles and datasets."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_continuous_profile(r: np.ndarray, 
                            profile: np.ndarray,
                            title: str = "Material Profile",
                            xlabel: str = "Radial Position (r)",
                            ylabel: str = "Parameter Value",
                            save_path: Optional[str] = None):
    """Plot a continuous material profile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(r, profile, linewidth=2, color='#2E86AB')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_discretized_profile(r: np.ndarray,
                             continuous_profile: np.ndarray,
                             discrete_layers: np.ndarray,
                             K: int,
                             title: str = "Discretized Profile",
                             ylabel: str = "Parameter Value",
                             save_path: Optional[str] = None):
    """Plot continuous profile with discretized layers overlaid."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(r, continuous_profile, linewidth=2, label='Continuous', 
            color='#2E86AB', alpha=0.7)
    
    r_min = r[0]
    r_max = r[-1]
    layer_boundaries = np.linspace(r_min, r_max, K + 1)
    
    for k in range(K):
        r_start = layer_boundaries[k]
        r_end = layer_boundaries[k + 1]
        ax.hlines(discrete_layers[k], r_start, r_end, 
                 colors='#A23B72', linewidth=2, alpha=0.8)
        if k < K - 1:
            ax.vlines(r_end, discrete_layers[k], discrete_layers[k+1],
                     colors='#A23B72', linewidth=1, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Radial Position (r)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dual_profiles(r: np.ndarray,
                      sigma_profile: np.ndarray,
                      mu_profile: np.ndarray,
                      sigma_layers: Optional[np.ndarray] = None,
                      mu_layers: Optional[np.ndarray] = None,
                      K: Optional[int] = None,
                      title: str = "Dual Material Profiles",
                      save_path: Optional[str] = None):
    """Plot both σ and μ profiles on dual y-axes."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_sigma = '#2E86AB'
    ax1.set_xlabel('Radial Position (r)', fontsize=12)
    ax1.set_ylabel('σ (S/m)', color=color_sigma, fontsize=12)
    ax1.plot(r, sigma_profile, color=color_sigma, linewidth=2, 
            label='σ continuous', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color_sigma)
    
    if sigma_layers is not None and K is not None:
        r_min = r[0]
        r_max = r[-1]
        layer_boundaries = np.linspace(r_min, r_max, K + 1)
        
        for k in range(K):
            r_start = layer_boundaries[k]
            r_end = layer_boundaries[k + 1]
            ax1.hlines(sigma_layers[k], r_start, r_end,
                      colors=color_sigma, linewidth=2, linestyle='--', alpha=0.9)
    
    ax2 = ax1.twinx()
    color_mu = '#A23B72'
    ax2.set_ylabel('μᵣ', color=color_mu, fontsize=12)
    ax2.plot(r, mu_profile, color=color_mu, linewidth=2,
            label='μ continuous', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color_mu)
    
    if mu_layers is not None and K is not None:
        for k in range(K):
            r_start = layer_boundaries[k]
            r_end = layer_boundaries[k + 1]
            ax2.hlines(mu_layers[k], r_start, r_end,
                      colors=color_mu, linewidth=2, linestyle='--', alpha=0.9)
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_space_coverage(plan: np.ndarray,
                                  param_names: List[str],
                                  save_path: Optional[str] = None):
    """Plot parameter space coverage from R-sequence plan."""
    d = plan.shape[1]
    
    if d == 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(plan[:, 0], plan[:, 1], alpha=0.6, s=20, color='#2E86AB')
        ax.set_xlabel(param_names[0], fontsize=12)
        ax.set_ylabel(param_names[1], fontsize=12)
        ax.set_title('Parameter Space Coverage', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    elif d >= 3:
        n_plots = min(d * (d - 1) // 2, 6)
        n_rows = 2
        n_cols = 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                if plot_idx >= n_plots:
                    break
                
                ax = axes[plot_idx]
                ax.scatter(plan[:, i], plan[:, j], alpha=0.6, s=15, color='#2E86AB')
                ax.set_xlabel(param_names[i], fontsize=10)
                ax.set_ylabel(param_names[j], fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
            
            if plot_idx >= n_plots:
                break
        
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Parameter Space Coverage (R-sequence)', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dataset_statistics(X: np.ndarray,
                           K: int,
                           save_path: Optional[str] = None):
    """Plot statistical summary of generated dataset."""
    sigma_data = X[:, :K]
    mu_data = X[:, K:2*K]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].hist(sigma_data.flatten(), bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('σ (S/m)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('σ Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(mu_data.flatten(), bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('μᵣ', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('μ Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    sigma_mean = sigma_data.mean(axis=0)
    sigma_std = sigma_data.std(axis=0)
    
    axes[0, 2].plot(sigma_mean, color='#2E86AB', linewidth=2, label='Mean')
    axes[0, 2].fill_between(range(K), sigma_mean - sigma_std, sigma_mean + sigma_std,
                           alpha=0.3, color='#2E86AB', label='±1σ')
    axes[0, 2].set_xlabel('Layer Index', fontsize=11)
    axes[0, 2].set_ylabel('σ (S/m)', fontsize=11)
    axes[0, 2].set_title('σ Layer Statistics', fontsize=12, fontweight='bold')
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    
    mu_mean = mu_data.mean(axis=0)
    mu_std = mu_data.std(axis=0)
    
    axes[1, 0].plot(mu_mean, color='#A23B72', linewidth=2, label='Mean')
    axes[1, 0].fill_between(range(K), mu_mean - mu_std, mu_mean + mu_std,
                           alpha=0.3, color='#A23B72', label='±1σ')
    axes[1, 0].set_xlabel('Layer Index', fontsize=11)
    axes[1, 0].set_ylabel('μᵣ', fontsize=11)
    axes[1, 0].set_title('μ Layer Statistics', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    n_samples_to_plot = min(20, X.shape[0])
    for i in range(n_samples_to_plot):
        axes[1, 1].plot(sigma_data[i], alpha=0.3, linewidth=0.5, color='#2E86AB')
    axes[1, 1].set_xlabel('Layer Index', fontsize=11)
    axes[1, 1].set_ylabel('σ (S/m)', fontsize=11)
    axes[1, 1].set_title(f'σ Sample Profiles (n={n_samples_to_plot})', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    for i in range(n_samples_to_plot):
        axes[1, 2].plot(mu_data[i], alpha=0.3, linewidth=0.5, color='#A23B72')
    axes[1, 2].set_xlabel('Layer Index', fontsize=11)
    axes[1, 2].set_ylabel('μᵣ', fontsize=11)
    axes[1, 2].set_title(f'μ Sample Profiles (n={n_samples_to_plot})', 
                        fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multiple_profiles_comparison(r: np.ndarray,
                                     profiles_dict: dict,
                                     title: str = "Profile Comparison",
                                     ylabel: str = "Parameter Value",
                                     save_path: Optional[str] = None):
    """Plot multiple profiles for comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2E86AB', '#A23B72', '#18A558', '#F18F01', '#C73E1D', '#6A4C93']
    
    for idx, (label, profile) in enumerate(profiles_dict.items()):
        color = colors[idx % len(colors)]
        ax.plot(r, profile, linewidth=2, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Radial Position (r)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
