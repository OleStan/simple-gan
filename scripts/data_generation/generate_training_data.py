#!/usr/bin/env python
"""Generate training dataset for WGAN dual-head training."""

import numpy as np
import json
from pathlib import Path

from eddy_current_data_generator import (
    DatasetConfig,
    build_dataset
)


def main():
    output_dir = Path('./training_data')
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Generating Training Dataset for WGAN")
    print("="*60)
    
    config = DatasetConfig(
        N=2000,
        K=50,
        r_min=0.0,
        r_max=1.0,
        sigma_bounds=(1e6, 6e7),
        mu_bounds=(1.0, 100.0),
        include_frequency=False,
        include_resistance=False,
        discretization_mode='centers',
        seed=42
    )
    
    print(f"\nConfiguration:")
    print(f"  N (samples): {config.N}")
    print(f"  K (layers): {config.K}")
    print(f"  σ range: [{config.sigma_bounds[0]:.2e}, {config.sigma_bounds[1]:.2e}] S/m")
    print(f"  μ range: {config.mu_bounds}")
    
    print("\nBuilding dataset...")
    dataset, metadata = build_dataset(config)
    
    print(f"\n✓ Dataset generated:")
    print(f"  Shape: {dataset.shape}")
    print(f"  Features per sample: {dataset.shape[1]}")
    
    K = config.K
    sigma_layers = dataset[:, :K]
    mu_layers = dataset[:, K:2*K]
    
    print(f"\n  σ layers: {sigma_layers.shape}")
    print(f"    Range: [{sigma_layers.min():.2e}, {sigma_layers.max():.2e}]")
    print(f"  μ layers: {mu_layers.shape}")
    print(f"    Range: [{mu_layers.min():.2f}, {mu_layers.max():.2f}]")
    
    print("\nSaving dataset...")
    np.save(output_dir / 'X_raw.npy', dataset)
    np.save(output_dir / 'sigma_layers.npy', sigma_layers)
    np.save(output_dir / 'mu_layers.npy', mu_layers)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    sigma_min = float(sigma_layers.min())
    sigma_max = float(sigma_layers.max())
    mu_min = float(mu_layers.min())
    mu_max = float(mu_layers.max())
    
    normalization_params = {
        'K': K,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'mu_min': mu_min,
        'mu_max': mu_max,
        'N': config.N
    }
    
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(normalization_params, f, indent=2)
    
    print(f"\n✓ Files saved to {output_dir}/:")
    print(f"  - X_raw.npy ({dataset.shape})")
    print(f"  - sigma_layers.npy ({sigma_layers.shape})")
    print(f"  - mu_layers.npy ({mu_layers.shape})")
    print(f"  - metadata.json")
    print(f"  - normalization_params.json")
    
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)
    print("\nNormalization parameters for WGAN training:")
    print(f"  σ: [{sigma_min:.2e}, {sigma_max:.2e}] S/m")
    print(f"  μ: [{mu_min:.2f}, {mu_max:.2f}]")
    print(f"\nReady for training with train_dual_wgan.py")


if __name__ == '__main__':
    main()
