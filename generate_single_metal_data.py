#!/usr/bin/env python
"""Generate training dataset for single metal type (less variation) - WGAN training."""

import numpy as np
import json
from pathlib import Path
from eddy_current_data_generator.core.material_profiles import ProfileType, generate_dual_profiles
from eddy_current_data_generator.core.discretization import discretize_dual_profiles


def main():
    output_dir = Path('./training_data')
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Generating Single Metal Type Training Dataset")
    print("="*60)
    
    N = 2000
    K = 50
    r_min = 0.0
    r_max = 1.0
    
    sigma_center = 3.5e7
    sigma_variation = 5e6
    sigma_min_range = (sigma_center - sigma_variation, sigma_center - sigma_variation/2)
    sigma_max_range = (sigma_center + sigma_variation/2, sigma_center + sigma_variation)
    
    mu_center = 50.0
    mu_variation = 15.0
    mu_min_range = (mu_center - mu_variation, mu_center - mu_variation/2)
    mu_max_range = (mu_center + mu_variation/2, mu_center + mu_variation)
    
    shape_sigma_range = (1.2, 1.8)
    shape_mu_range = (1.2, 1.8)
    
    print(f"\nConfiguration:")
    print(f"  N (samples): {N}")
    print(f"  K (layers): {K}")
    print(f"  Profile Type: LINEAR (single type)")
    print(f"  σ center: {sigma_center:.2e} S/m ± {sigma_variation:.2e}")
    print(f"    σ_min range: [{sigma_min_range[0]:.2e}, {sigma_min_range[1]:.2e}]")
    print(f"    σ_max range: [{sigma_max_range[0]:.2e}, {sigma_max_range[1]:.2e}]")
    print(f"  μ center: {mu_center:.1f} ± {mu_variation:.1f}")
    print(f"    μ_min range: {mu_min_range}")
    print(f"    μ_max range: {mu_max_range}")
    print(f"  Shape parameters:")
    print(f"    σ shape: {shape_sigma_range}")
    print(f"    μ shape: {shape_mu_range}")
    
    np.random.seed(42)
    
    sigma_min_vals = np.random.uniform(sigma_min_range[0], sigma_min_range[1], N)
    sigma_max_vals = np.random.uniform(sigma_max_range[0], sigma_max_range[1], N)
    mu_min_vals = np.random.uniform(mu_min_range[0], mu_min_range[1], N)
    mu_max_vals = np.random.uniform(mu_max_range[0], mu_max_range[1], N)
    sigma_shape_vals = np.random.uniform(shape_sigma_range[0], shape_sigma_range[1], N)
    mu_shape_vals = np.random.uniform(shape_mu_range[0], shape_mu_range[1], N)
    
    n_points = 1000
    r = np.linspace(r_min, r_max, n_points)
    
    feature_dim = 2 * K
    X = np.zeros((N, feature_dim))
    
    print("\nGenerating profiles...")
    for i in range(N):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{N}")
        
        sigma_profile, mu_profile = generate_dual_profiles(
            r,
            sigma_min_vals[i], sigma_max_vals[i],
            mu_min_vals[i], mu_max_vals[i],
            ProfileType.LINEAR, ProfileType.LINEAR,
            sigma_shape_vals[i], mu_shape_vals[i]
        )
        
        sigma_layers, mu_layers = discretize_dual_profiles(
            r, sigma_profile, mu_profile, K, 'centers'
        )
        
        X[i, :K] = sigma_layers
        X[i, K:2*K] = mu_layers
    
    sigma_layers_all = X[:, :K]
    mu_layers_all = X[:, K:2*K]
    
    print(f"\n✓ Dataset generated:")
    print(f"  Shape: {X.shape}")
    print(f"  σ layers: {sigma_layers_all.shape}")
    print(f"    Range: [{sigma_layers_all.min():.2e}, {sigma_layers_all.max():.2e}]")
    print(f"    Mean: {sigma_layers_all.mean():.2e}")
    print(f"    Std: {sigma_layers_all.std():.2e}")
    print(f"  μ layers: {mu_layers_all.shape}")
    print(f"    Range: [{mu_layers_all.min():.2f}, {mu_layers_all.max():.2f}]")
    print(f"    Mean: {mu_layers_all.mean():.2f}")
    print(f"    Std: {mu_layers_all.std():.2f}")
    
    print("\nSaving dataset...")
    np.save(output_dir / 'X_raw.npy', X)
    np.save(output_dir / 'sigma_layers.npy', sigma_layers_all)
    np.save(output_dir / 'mu_layers.npy', mu_layers_all)
    
    metadata = {
        'N': N,
        'K': K,
        'feature_dim': feature_dim,
        'r_min': r_min,
        'r_max': r_max,
        'profile_type': 'LINEAR',
        'sigma_center': sigma_center,
        'sigma_variation': sigma_variation,
        'mu_center': mu_center,
        'mu_variation': mu_variation,
        'discretization_mode': 'centers',
        'seed': 42,
        'data_type': 'single_metal',
        'shape_params': {
            'sigma_min_range': [float(sigma_min_vals.min()), float(sigma_min_vals.max())],
            'sigma_max_range': [float(sigma_max_vals.min()), float(sigma_max_vals.max())],
            'mu_min_range': [float(mu_min_vals.min()), float(mu_min_vals.max())],
            'mu_max_range': [float(mu_max_vals.min()), float(mu_max_vals.max())],
            'sigma_shape_range': [float(sigma_shape_vals.min()), float(sigma_shape_vals.max())],
            'mu_shape_range': [float(mu_shape_vals.min()), float(mu_shape_vals.max())]
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    sigma_min = float(sigma_layers_all.min())
    sigma_max = float(sigma_layers_all.max())
    mu_min = float(mu_layers_all.min())
    mu_max = float(mu_layers_all.max())
    
    normalization_params = {
        'K': K,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'mu_min': mu_min,
        'mu_max': mu_max,
        'N': N
    }
    
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(normalization_params, f, indent=2)
    
    print(f"\n✓ Files saved to {output_dir}/:")
    print(f"  - X_raw.npy ({X.shape})")
    print(f"  - sigma_layers.npy ({sigma_layers_all.shape})")
    print(f"  - mu_layers.npy ({mu_layers_all.shape})")
    print(f"  - metadata.json")
    print(f"  - normalization_params.json")
    
    print("\n" + "="*60)
    print("Single Metal Dataset Generation Complete!")
    print("="*60)
    print("\nNormalization parameters for WGAN training:")
    print(f"  σ: [{sigma_min:.2e}, {sigma_max:.2e}] S/m")
    print(f"  μ: [{mu_min:.2f}, {mu_max:.2f}]")
    print(f"\nData characteristics:")
    print(f"  - Reduced variation compared to multi-metal dataset")
    print(f"  - Single profile type (LINEAR) for consistency")
    print(f"  - Narrower parameter ranges")
    print(f"\nReady for training with train_dual_wgan.py and train_improved_wgan.py")


if __name__ == '__main__':
    main()
