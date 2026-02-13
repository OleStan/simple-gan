#!/usr/bin/env python
"""Generate training dataset with opposite sigma/mu profiles for WGAN training.

This generator creates a single class dataset where:
- One profile type is used consistently
- Sigma and mu have opposite (inverse) relationships
- When sigma increases radially, mu decreases (and vice versa)
"""

import numpy as np
import json
from pathlib import Path
from eddy_current_data_generator.core.material_profiles import ProfileType, make_profile
from eddy_current_data_generator.core.discretization import discretize_dual_profiles


def generate_opposite_profiles(r, sigma_min, sigma_max, mu_min, mu_max, profile_type, shape_param):
    """Generate sigma and mu profiles with opposite relationships."""
    sigma_profile = make_profile(r, profile_type, sigma_min, sigma_max, shape_param)
    mu_profile = make_profile(r, profile_type, mu_max, mu_min, shape_param)
    return sigma_profile, mu_profile


def main():
    output_dir = Path('./training_data')
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Generating Training Dataset with Opposite σ/μ Profiles")
    print("="*60)
    
    N = 2000
    K = 50
    r_min = 0.0
    r_max = 1.0
    profile_type = ProfileType.LINEAR
    
    sigma_center = 3.5e7
    sigma_variation = 1.0e7
    sigma_min_range = (sigma_center - sigma_variation, sigma_center - sigma_variation/2)
    sigma_max_range = (sigma_center + sigma_variation/2, sigma_center + sigma_variation)
    
    mu_center = 50.0
    mu_variation = 20.0
    mu_min_range = (mu_center - mu_variation, mu_center - mu_variation/2)
    mu_max_range = (mu_center + mu_variation/2, mu_center + mu_variation)
    
    shape_param_range = (1.0, 2.0)
    
    print(f"\nConfiguration:")
    print(f"  N (samples): {N}")
    print(f"  K (layers): {K}")
    print(f"  Profile Type: {profile_type.value}")
    print(f"  Relationship: OPPOSITE (σ↑ → μ↓)")
    print(f"  σ range: [{sigma_min_range[0]:.2e}, {sigma_max_range[1]:.2e}]")
    print(f"  μ range: {mu_min_range} to {mu_max_range}")
    
    np.random.seed(42)
    
    sigma_min_vals = np.random.uniform(sigma_min_range[0], sigma_min_range[1], N)
    sigma_max_vals = np.random.uniform(sigma_max_range[0], sigma_max_range[1], N)
    mu_min_vals = np.random.uniform(mu_min_range[0], mu_min_range[1], N)
    mu_max_vals = np.random.uniform(mu_max_range[0], mu_max_range[1], N)
    shape_param_vals = np.random.uniform(shape_param_range[0], shape_param_range[1], N)
    
    n_points = 1000
    r = np.linspace(r_min, r_max, n_points)
    
    feature_dim = 2 * K
    X = np.zeros((N, feature_dim))
    
    print("\nGenerating profiles...")
    for i in range(N):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{N}")
        
        sigma_profile, mu_profile = generate_opposite_profiles(
            r,
            sigma_min_vals[i], sigma_max_vals[i],
            mu_min_vals[i], mu_max_vals[i],
            profile_type,
            shape_param_vals[i]
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
    print(f"  μ layers: {mu_layers_all.shape}")
    print(f"    Range: [{mu_layers_all.min():.2f}, {mu_layers_all.max():.2f}]")
    
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
        'profile_type': profile_type.value,
        'relationship': 'opposite',
        'sigma_center': sigma_center,
        'sigma_variation': sigma_variation,
        'mu_center': mu_center,
        'mu_variation': mu_variation,
        'discretization_mode': 'centers',
        'seed': 42
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    normalization_params = {
        'K': K,
        'sigma_min': float(sigma_layers_all.min()),
        'sigma_max': float(sigma_layers_all.max()),
        'mu_min': float(mu_layers_all.min()),
        'mu_max': float(mu_layers_all.max()),
        'N': N
    }
    
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(normalization_params, f, indent=2)
    
    print(f"\n✓ Files saved to {output_dir}/")
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
