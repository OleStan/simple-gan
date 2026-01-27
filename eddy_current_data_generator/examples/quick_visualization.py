#!/usr/bin/env python3
"""Quick visualization script for eddy_current_data_generator."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the eddy_current_data_generator to path
sys.path.append(str(Path(__file__).parent.parent))

from core.dataset_builder import DatasetConfig, build_dataset
from visualization.profile_visualizer import plot_dataset_statistics, plot_dual_profiles
from core.material_profiles import ProfileType, generate_dual_profiles
from core.discretization import discretize_dual_profiles


def quick_demo():
    """Quick demonstration of key visualization features."""
    print("Running quick visualization demo...")
    
    # 1. Generate a small dataset
    config = DatasetConfig(N=100, K=20, seed=42)
    X, metadata = build_dataset(config)
    print(f"Generated dataset: {X.shape}")
    
    # 2. Plot dataset statistics
    fig = plot_dataset_statistics(X, config.K)
    plt.suptitle("Dataset Statistics Visualization", fontsize=16)
    plt.show()
    
    # 3. Show example dual profiles
    r = np.linspace(0, 1, 1000)
    sigma_profile, mu_profile = generate_dual_profiles(
        r, 1e6, 6e7, 1.0, 100.0,
        ProfileType.SIGMOID, ProfileType.EXPONENTIAL, 10.0, 2.0
    )
    
    sigma_layers, mu_layers = discretize_dual_profiles(r, sigma_profile, mu_profile, config.K)
    
    fig = plot_dual_profiles(
        r, sigma_profile, mu_profile, sigma_layers, mu_layers, config.K,
        title="Example Dual Material Profiles"
    )
    plt.show()
    
    print("Quick demo completed!")


if __name__ == "__main__":
    quick_demo()
