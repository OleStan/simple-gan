#!/usr/bin/env python3
"""Visualization demo for eddy_current_data_generator."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the eddy_current_data_generator to path
sys.path.append(str(Path(__file__).parent.parent))

from core.dataset_builder import DatasetConfig, build_dataset
from core.material_profiles import ProfileType, generate_dual_profiles
from core.discretization import discretize_dual_profiles
from visualization.profile_visualizer import (
    plot_continuous_profile,
    plot_discretized_profile,
    plot_dual_profiles,
    plot_parameter_space_coverage,
    plot_dataset_statistics,
    plot_multiple_profiles_comparison
)


def demo_individual_profiles():
    """Demonstrate individual profile types."""
    print("Generating individual profile examples...")
    
    # Create radial coordinate
    r = np.linspace(0, 1, 1000)
    
    # Generate different profile types
    profiles = {}
    profile_types = [
        (ProfileType.LINEAR, 1.0),
        (ProfileType.EXPONENTIAL, 2.0),
        (ProfileType.POWER, 3.0),
        (ProfileType.SIGMOID, 10.0)
    ]
    
    for profile_type, shape_param in profile_types:
        profile = generate_dual_profiles(
            r, 1e6, 6e7, 1.0, 100.0,
            profile_type, profile_type, shape_param, shape_param
        )[0]  # Get sigma profile
        profiles[profile_type.value] = profile
    
    # Plot comparison
    fig = plot_multiple_profiles_comparison(
        r, profiles, 
        title="Conductivity Profile Types Comparison",
        ylabel="σ (S/m)",
        save_path="results/profile_types_comparison.png"
    )
    plt.show()
    
    print("✓ Individual profiles demo completed")


def demo_dual_profiles():
    """Demonstrate dual σ and μ profiles."""
    print("Generating dual profile examples...")
    
    r = np.linspace(0, 1, 1000)
    
    # Generate dual profiles
    sigma_profile, mu_profile = generate_dual_profiles(
        r, 1e6, 6e7, 1.0, 100.0,
        ProfileType.LINEAR, ProfileType.EXPONENTIAL, 1.5, 2.0
    )
    
    # Plot continuous dual profiles
    fig = plot_dual_profiles(
        r, sigma_profile, mu_profile,
        title="Dual Material Profiles (Continuous)",
        save_path="results/dual_profiles_continuous.png"
    )
    plt.show()
    
    # Discretize and plot
    K = 50
    sigma_layers, mu_layers = discretize_dual_profiles(r, sigma_profile, mu_profile, K)
    
    fig = plot_dual_profiles(
        r, sigma_profile, mu_profile, sigma_layers, mu_layers, K,
        title="Dual Material Profiles (Discretized)",
        save_path="results/dual_profiles_discretized.png"
    )
    plt.show()
    
    print("✓ Dual profiles demo completed")


def demo_dataset_generation():
    """Demonstrate complete dataset generation and visualization."""
    print("Generating dataset...")
    
    # Create configuration
    config = DatasetConfig(
        N=500,
        K=30,
        r_min=0.0,
        r_max=1.0,
        sigma_bounds=(1e6, 6e7),
        mu_bounds=(1.0, 100.0),
        seed=42
    )
    
    # Build dataset
    X, metadata = build_dataset(config)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Metadata: {metadata}")
    
    # Plot dataset statistics
    fig = plot_dataset_statistics(
        X, config.K,
        save_path="results/dataset_statistics.png"
    )
    plt.show()
    
    # Generate a few example profiles for visualization
    r = np.linspace(config.r_min, config.r_max, 1000)
    n_examples = 5
    
    fig, axes = plt.subplots(2, n_examples, figsize=(20, 8))
    
    for i in range(n_examples):
        # Extract discretized values
        sigma_layers = X[i, :config.K]
        mu_layers = X[i, config.K:2*config.K]
        
        # Create continuous profiles for visualization (approximate)
        sigma_continuous = np.interp(r, np.linspace(config.r_min, config.r_max, config.K), sigma_layers)
        mu_continuous = np.interp(r, np.linspace(config.r_min, config.r_max, config.K), mu_layers)
        
        # Plot sigma
        axes[0, i].plot(r, sigma_continuous, color='#2E86AB', linewidth=2)
        axes[0, i].set_title(f'Sample {i+1}: σ')
        axes[0, i].set_xlabel('Radial Position')
        axes[0, i].set_ylabel('σ (S/m)')
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot mu
        axes[1, i].plot(r, mu_continuous, color='#A23B72', linewidth=2)
        axes[1, i].set_title(f'Sample {i+1}: μᵣ')
        axes[1, i].set_xlabel('Radial Position')
        axes[1, i].set_ylabel('μᵣ')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/example_profiles.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Dataset generation demo completed")


def demo_discretization():
    """Demonstrate discretization process."""
    print("Demonstrating discretization...")
    
    r = np.linspace(0, 1, 1000)
    K_values = [10, 25, 50]
    
    # Generate continuous profile
    sigma_profile, mu_profile = generate_dual_profiles(
        r, 1e6, 6e7, 1.0, 100.0,
        ProfileType.SIGMOID, ProfileType.POWER, 15.0, 3.0
    )
    
    fig, axes = plt.subplots(1, len(K_values), figsize=(18, 6))
    
    for idx, K in enumerate(K_values):
        sigma_layers, mu_layers = discretize_dual_profiles(r, sigma_profile, mu_profile, K)
        
        ax = axes[idx]
        ax.plot(r, sigma_profile, color='#2E86AB', linewidth=2, label='Continuous', alpha=0.7)
        
        r_min = r[0]
        r_max = r[-1]
        layer_boundaries = np.linspace(r_min, r_max, K + 1)
        
        for k in range(K):
            r_start = layer_boundaries[k]
            r_end = layer_boundaries[k + 1]
            ax.hlines(sigma_layers[k], r_start, r_end, 
                     colors='#A23B72', linewidth=2, alpha=0.8)
            if k < K - 1:
                ax.vlines(r_end, sigma_layers[k], sigma_layers[k+1],
                         colors='#A23B72', linewidth=1, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Radial Position (r)')
        ax.set_ylabel('σ (S/m)')
        ax.set_title(f'K = {K} layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/discretization_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Discretization demo completed")


def main():
    """Run all visualization demos."""
    print("Starting eddy_current_data_generator visualization demo...")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Run all demos
    demo_individual_profiles()
    demo_dual_profiles()
    demo_dataset_generation()
    demo_discretization()
    
    print("=" * 60)
    print("All visualization demos completed successfully!")
    print("Check the 'results' directory for saved plots.")


if __name__ == "__main__":
    main()
