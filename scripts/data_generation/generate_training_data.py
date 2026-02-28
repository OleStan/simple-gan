#!/usr/bin/env python
"""Generate two-class training dataset: sigmoid (both) vs linear (both) profiles."""

import numpy as np
import json
from pathlib import Path

# Constants
N_PER_CLASS = 2000
K = 50
R_MIN = 0.0
R_MAX = 1.0
N_POINTS = 1000

SIGMA_BOUNDS = (1e6, 6e7)
MU_BOUNDS = (1.0, 100.0)

# Common starting value ranges for layer 1
SIGMA1_BASE = 1e7
MU1_BASE = 1.0

def _linear_profile(r, P_min, P_max, a):
    r_max = r[-1]
    return P_min + (P_max - P_min) * (r / r_max) ** a

def _sigmoid_profile(r, P_min, P_max, d, r_0=None):
    r_max = r[-1]
    if r_0 is None:
        r_0 = r_max / 2
    raw = 1 / (1 + np.exp(-d * (r - r_0)))
    v_min = 1 / (1 + np.exp(-d * (0 - r_0)))
    v_max = 1 / (1 + np.exp(-d * (r_max - r_0)))
    norm = (raw - v_min) / (v_max - v_min)
    return P_min + (P_max - P_min) * norm

def main():
    output_dir = Path('./data/training')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Sigmoid (both) vs Linear (both) Dataset")
    print("=" * 60)

    rng = np.random.default_rng(42)
    r = np.linspace(R_MIN, R_MAX, N_POINTS)
    
    # Pre-generate common starting points for Layer 1
    # We want corresponding samples in Class 0 and Class 1 to have same start
    s1_vals = SIGMA1_BASE * rng.uniform(0.8, 1.2, N_PER_CLASS)
    m1_vals = MU1_BASE * rng.uniform(0.9, 1.1, N_PER_CLASS)

    all_X = []
    all_labels = []

    # --- Class 0: Sigmoid (Both) ---
    print(f"Building Class 0 (Sigmoid) — {N_PER_CLASS} samples...")
    X0 = np.zeros((N_PER_CLASS, 2 * K))
    for i in range(N_PER_CLASS):
        s_end = 5e7 * rng.uniform(0.7, 1.3)
        m_end = 50.0 * rng.uniform(0.7, 1.3)
        d_val = rng.uniform(10, 25)
        
        s_prof = _sigmoid_profile(r, s1_vals[i], s_end, d=d_val)
        m_prof = _sigmoid_profile(r, m1_vals[i], m_end, d=d_val)
        
        # Discretize to K layers (simple mean pooling)
        s_layers = np.mean(s_prof.reshape(K, -1), axis=1)
        m_layers = np.mean(m_prof.reshape(K, -1), axis=1)
        X0[i, :K] = s_layers
        X0[i, K:] = m_layers
    
    all_X.append(X0)
    all_labels.append(np.zeros(N_PER_CLASS, dtype=np.int64))

    # --- Class 1: Linear (Both) ---
    print(f"Building Class 1 (Linear) — {N_PER_CLASS} samples...")
    X1 = np.zeros((N_PER_CLASS, 2 * K))
    for i in range(N_PER_CLASS):
        s_end = 3e7 * rng.uniform(0.7, 1.3)
        m_end = 80.0 * rng.uniform(0.7, 1.3)
        a_val = rng.uniform(0.8, 1.5)
        
        s_prof = _linear_profile(r, s1_vals[i], s_end, a=a_val)
        m_prof = _linear_profile(r, m1_vals[i], m_end, a=a_val)
        
        s_layers = np.mean(s_prof.reshape(K, -1), axis=1)
        m_layers = np.mean(m_prof.reshape(K, -1), axis=1)
        X1[i, :K] = s_layers
        X1[i, K:] = m_layers

    all_X.append(X1)
    all_labels.append(np.ones(N_PER_CLASS, dtype=np.int64))

    X_raw = np.concatenate(all_X, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)

    # Shuffle
    shuffle_idx = rng.permutation(len(X_raw))
    X_raw = X_raw[shuffle_idx]
    y_labels = y_labels[shuffle_idx]

    sigma_layers = X_raw[:, :K]
    mu_layers = X_raw[:, K:2 * K]

    sigma_min, sigma_max = float(sigma_layers.min()), float(sigma_layers.max())
    mu_min, mu_max = float(mu_layers.min()), float(mu_layers.max())

    print(f"\n✓ Dataset Complete:")
    print(f"  Shape:   {X_raw.shape}")
    print(f"  σ range: [{sigma_min:.2e}, {sigma_max:.2e}]")
    print(f"  μ range: [{mu_min:.2f}, {mu_max:.2f}]")

    print("Saving dataset...")
    np.save(output_dir / 'X_raw.npy', X_raw)
    np.save(output_dir / 'y_labels.npy', y_labels)
    
    norm_params = {
        'K': K,
        'sigma_min': sigma_min, 'sigma_max': sigma_max,
        'mu_min': mu_min, 'mu_max': mu_max,
        'N': len(X_raw),
        'n_classes': 2,
        'class_names': ['sigmoid', 'linear']
    }
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    print(f"Files saved to {output_dir}/")

if __name__ == '__main__':
    main()
