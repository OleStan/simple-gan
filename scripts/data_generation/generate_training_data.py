#!/usr/bin/env python
"""Generate two-class training dataset: sigmoid vs linear sigma profiles."""

import numpy as np
import json
from pathlib import Path

from eddy_current_data_generator.core.material_profiles import (
    ProfileType, generate_dual_profiles,
)
from eddy_current_data_generator.core.discretization import discretize_dual_profiles
from eddy_current_data_generator.core.roberts_sequence import generate_roberts_plan

N_PER_CLASS = 2000
K = 50
R_MIN = 0.0
R_MAX = 1.0
N_POINTS = 1000

SIGMA_BOUNDS = (1e6, 6e7)
MU_BOUNDS = (1.0, 100.0)

CLASS_CONFIGS = [
    {
        'name': 'sigmoid_sigma',
        'sigma_profile_type': ProfileType.SIGMOID,
        'mu_profile_type': ProfileType.LINEAR,
        'sigma_shape_bounds': (5.0, 20.0),
        'mu_shape_bounds': (0.5, 2.0),
    },
    {
        'name': 'linear_sigma',
        'sigma_profile_type': ProfileType.LINEAR,
        'mu_profile_type': ProfileType.LINEAR,
        'sigma_shape_bounds': (0.5, 2.0),
        'mu_shape_bounds': (0.5, 2.0),
    },
]


def _generate_class_samples(cfg: dict, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.linspace(R_MIN, R_MAX, N_POINTS)

    sigma_min_vals = rng.uniform(SIGMA_BOUNDS[0], SIGMA_BOUNDS[1] * 0.3, n)
    sigma_max_vals = rng.uniform(SIGMA_BOUNDS[1] * 0.4, SIGMA_BOUNDS[1], n)
    mu_min_vals = rng.uniform(MU_BOUNDS[0], MU_BOUNDS[1] * 0.2, n)
    mu_max_vals = rng.uniform(MU_BOUNDS[1] * 0.3, MU_BOUNDS[1], n)
    sigma_shapes = rng.uniform(*cfg['sigma_shape_bounds'], n)
    mu_shapes = rng.uniform(*cfg['mu_shape_bounds'], n)

    X = np.zeros((n, 2 * K))
    for i in range(n):
        sigma_profile, mu_profile = generate_dual_profiles(
            r,
            sigma_min_vals[i], sigma_max_vals[i],
            mu_min_vals[i], mu_max_vals[i],
            cfg['sigma_profile_type'], cfg['mu_profile_type'],
            sigma_shapes[i], mu_shapes[i],
        )
        sigma_layers, mu_layers = discretize_dual_profiles(r, sigma_profile, mu_profile, K)
        X[i, :K] = sigma_layers
        X[i, K:] = mu_layers

    return X


def main():
    output_dir = Path('./training_data')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Sigmoid vs Linear Profile Dataset for WGAN")
    print("=" * 60)

    n_classes = len(CLASS_CONFIGS)
    print(f"\n{n_classes} classes:")
    for i, cfg in enumerate(CLASS_CONFIGS):
        print(f"  Class {i} ({cfg['name']}): "
              f"sigma_profile={cfg['sigma_profile_type'].value}")

    all_X = []
    all_labels = []

    for class_idx, cfg in enumerate(CLASS_CONFIGS):
        print(f"\nBuilding Class {class_idx} ({cfg['name']}) — {N_PER_CLASS} samples...")
        X = _generate_class_samples(cfg, N_PER_CLASS, seed=42 + class_idx)
        labels = np.full(N_PER_CLASS, class_idx, dtype=np.int64)
        all_X.append(X)
        all_labels.append(labels)
        print(f"  ✓ Shape: {X.shape}")

    X_raw = np.concatenate(all_X, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X_raw))
    X_raw = X_raw[idx]
    y_labels = y_labels[idx]

    sigma_layers = X_raw[:, :K]
    mu_layers = X_raw[:, K:2 * K]

    sigma_min = float(sigma_layers.min())
    sigma_max = float(sigma_layers.max())
    mu_min = float(mu_layers.min())
    mu_max = float(mu_layers.max())

    print(f"\n✓ Combined dataset:")
    print(f"  Shape:   {X_raw.shape}")
    print(f"  σ range: [{sigma_min:.2e}, {sigma_max:.2e}] S/m")
    print(f"  μ range: [{mu_min:.2f}, {mu_max:.2f}]")
    for i in range(n_classes):
        count = int((y_labels == i).sum())
        print(f"  Class {i} ({CLASS_CONFIGS[i]['name']}): {count} samples")

    print("\nSaving dataset...")
    np.save(output_dir / 'X_raw.npy', X_raw)
    np.save(output_dir / 'y_labels.npy', y_labels)
    np.save(output_dir / 'sigma_layers.npy', sigma_layers)
    np.save(output_dir / 'mu_layers.npy', mu_layers)

    serializable_class_configs = [
        {k: (v.value if isinstance(v, ProfileType) else v) for k, v in cfg.items()}
        for cfg in CLASS_CONFIGS
    ]

    normalization_params = {
        'K': K,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'mu_min': mu_min,
        'mu_max': mu_max,
        'N': len(X_raw),
        'n_classes': n_classes,
        'class_configs': serializable_class_configs,
    }

    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(normalization_params, f, indent=2)

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump({
            'n_classes': n_classes,
            'N_per_class': N_PER_CLASS,
            'N_total': len(X_raw),
            'K': K,
            'class_configs': serializable_class_configs,
        }, f, indent=2)

    print(f"\n✓ Files saved to {output_dir}/:")
    print(f"  - X_raw.npy               {X_raw.shape}")
    print(f"  - y_labels.npy            {y_labels.shape}")
    print(f"  - normalization_params.json  (n_classes={n_classes})")
    print(f"\nClasses: {[c['name'] for c in CLASS_CONFIGS]}")
    print(f"\nReady for conditional WGAN training.")


if __name__ == '__main__':
    main()
