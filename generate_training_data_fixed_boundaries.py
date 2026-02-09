#!/usr/bin/env python
"""Generate training dataset with fixed boundary conditions based on Table 3.5.

Fixed boundaries:
- σ₁ = 1.88×10⁷ S/m (first layer, fixed)
- σ₅₁ = 3.766×10⁷ S/m (last layer, fixed)
- μ₁ = 1 (first layer, fixed)
- μ₅₁ varies around 8.8 (last layer, small variation)

Layers 2-50 vary according to profile functions with opposite relationship.
"""

import numpy as np
import json
from pathlib import Path
from eddy_current_data_generator.core.material_profiles import ProfileType, make_profile
from eddy_current_data_generator.core.discretization import discretize_dual_profiles


def generate_fixed_boundary_profiles(r, sigma_1, sigma_51, mu_1, mu_51, profile_type, shape_param):
    """Generate sigma and mu profiles with fixed boundaries and opposite relationship.
    
    When sigma goes from sigma_1 to sigma_51 (increasing),
    mu goes from mu_51 to mu_1 (decreasing) - TRUE OPPOSITE relationship.
    """
    sigma_profile = make_profile(r, profile_type, sigma_1, sigma_51, shape_param)
    mu_profile = make_profile(r, profile_type, mu_51, mu_1, shape_param)
    return sigma_profile, mu_profile


def main():
    output_dir = Path('./training_data')
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Generating Training Dataset with Fixed Boundaries")
    print("Based on Table 3.5 specifications")
    print("="*60)
    
    N = 2000
    K = 51
    r_min = 0.0
    r_max = 1.0
    profile_type = ProfileType.SIGMOID
    
    sigma_1_fixed = 1.88e7
    sigma_51_center = 3.766e7
    sigma_51_variation_percent = 0.075
    sigma_51_variation = sigma_51_center * sigma_51_variation_percent
    sigma_51_range = (sigma_51_center - sigma_51_variation, sigma_51_center + sigma_51_variation)
    
    mu_1_fixed = 1.0
    mu_51_center = 8.8
    mu_51_variation_percent = 0.075
    mu_51_variation = mu_51_center * mu_51_variation_percent
    mu_51_range = (mu_51_center - mu_51_variation, mu_51_center + mu_51_variation)
    
    shape_param_range = (8.0, 15.0)
    
    print(f"\nConfiguration:")
    print(f"  N (samples): {N}")
    print(f"  K (layers): {K}")
    print(f"  Profile Type: {profile_type.value}")
    print(f"  Relationship: OPPOSITE (σ↑ → μ↓)")
    print(f"\nFixed Boundaries (from Table 3.5):")
    print(f"  σ₁ (layer 1):  {sigma_1_fixed:.3e} S/m (FIXED)")
    print(f"  σ₅₁ (layer 51): {sigma_51_center:.3e} S/m ± {sigma_51_variation_percent*100:.1f}% (VARIABLE)")
    print(f"    Range: [{sigma_51_range[0]:.3e}, {sigma_51_range[1]:.3e}]")
    print(f"  μ₁ (layer 1):  {mu_1_fixed:.1f} (FIXED)")
    print(f"  μ₅₁ (layer 51): {mu_51_center:.1f} ± {mu_51_variation_percent*100:.1f}% (VARIABLE)")
    print(f"    Range: [{mu_51_range[0]:.2f}, {mu_51_range[1]:.2f}]")
    print(f"\nShape parameter range: {shape_param_range} (sigmoid steepness)")
    
    np.random.seed(42)
    
    sigma_51_vals = np.random.uniform(sigma_51_range[0], sigma_51_range[1], N)
    mu_51_vals = np.random.uniform(mu_51_range[0], mu_51_range[1], N)
    shape_param_vals = np.random.uniform(shape_param_range[0], shape_param_range[1], N)
    
    n_points = 1000
    r = np.linspace(r_min, r_max, n_points)
    
    feature_dim = 2 * K
    X = np.zeros((N, feature_dim))
    
    print("\nGenerating profiles...")
    for i in range(N):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{N}")
        
        sigma_profile, mu_profile = generate_fixed_boundary_profiles(
            r,
            sigma_1_fixed, sigma_51_vals[i],
            mu_1_fixed, mu_51_vals[i],
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
    print(f"    Range: [{sigma_layers_all.min():.3e}, {sigma_layers_all.max():.3e}] S/m")
    print(f"    σ₁ values: min={sigma_layers_all[:, 0].min():.3e}, max={sigma_layers_all[:, 0].max():.3e}")
    print(f"    σ₅₁ values: min={sigma_layers_all[:, -1].min():.3e}, max={sigma_layers_all[:, -1].max():.3e}")
    print(f"  μ layers: {mu_layers_all.shape}")
    print(f"    Range: [{mu_layers_all.min():.2f}, {mu_layers_all.max():.2f}]")
    print(f"    μ₁ values: min={mu_layers_all[:, 0].min():.2f}, max={mu_layers_all[:, 0].max():.2f}")
    print(f"    μ₅₁ values: min={mu_layers_all[:, -1].min():.2f}, max={mu_layers_all[:, -1].max():.2f}")
    
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
        'fixed_boundaries': {
            'sigma_1': sigma_1_fixed,
            'sigma_51_center': sigma_51_center,
            'sigma_51_variation_percent': sigma_51_variation_percent,
            'mu_1': mu_1_fixed,
            'mu_51_center': mu_51_center,
            'mu_51_variation_percent': mu_51_variation_percent
        },
        'discretization_mode': 'centers',
        'seed': 42,
        'based_on': 'Table 3.5 specifications'
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
    
    print("\nBoundary Verification:")
    print("-" * 60)
    print(f"σ₁ (layer 1):  Expected={sigma_1_fixed:.3e}, Got={sigma_layers_all[:, 0].mean():.3e}")
    print(f"σ₅₁ (layer 51): Expected≈{sigma_51_center:.3e}, Got={sigma_layers_all[:, -1].mean():.3e}")
    print(f"  σ₅₁ range: [{sigma_layers_all[:, -1].min():.3e}, {sigma_layers_all[:, -1].max():.3e}]")
    print(f"μ₁ (layer 1):  Expected={mu_1_fixed:.2f}, Got={mu_layers_all[:, 0].mean():.2f}")
    print(f"μ₅₁ (layer 51): Expected≈{mu_51_center:.2f}, Got={mu_layers_all[:, -1].mean():.2f}")
    print(f"  μ₅₁ range: [{mu_layers_all[:, -1].min():.2f}, {mu_layers_all[:, -1].max():.2f}]")
    
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
