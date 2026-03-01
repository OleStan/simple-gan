#!/usr/bin/env python
"""Generate four-class training dataset: 2 Sigmoid, 2 Linear with different starts/ends."""

import numpy as np
import json
from pathlib import Path

# Constants
N_PER_CLASS = 2000
K = 50
R_MIN = 0.0
R_MAX = 1.0
N_POINTS = 1000

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
    print("Generating 4-Class Dataset: 2 Sigmoid, 2 Linear")
    print("=" * 60)

    rng = np.random.default_rng(42)
    r = np.linspace(R_MIN, R_MAX, N_POINTS)
    
    configs = [
        {'name': 'sigmoid_low',  'type': 'sigmoid', 's1': 1.0e7, 'm1': 1.0, 's51': 3.0e7, 'm51': 30},
        {'name': 'sigmoid_high', 'type': 'sigmoid', 's1': 2.5e7, 'm1': 5.0, 's51': 5.5e7, 'm51': 80},
        {'name': 'linear_low',   'type': 'linear',  's1': 1.0e7, 'm1': 1.0, 's51': 3.0e7, 'm51': 30},
        {'name': 'linear_high',  'type': 'linear',  's1': 2.5e7, 'm1': 5.0, 's51': 5.5e7, 'm51': 80},
    ]

    all_X = []
    all_labels = []

    for class_idx, cfg in enumerate(configs):
        print(f"Building Class {class_idx} ({cfg['name']}) — {N_PER_CLASS} samples...")
        X = np.zeros((N_PER_CLASS, 2 * K))
        
        for i in range(N_PER_CLASS):
            # 8% dispersion around base values
            disp = lambda val: val * rng.uniform(0.92, 1.08)
            s1 = disp(cfg['s1'])
            m1 = disp(cfg['m1'])
            s51 = disp(cfg['s51'])
            m51 = disp(cfg['m51'])
            
            if cfg['type'] == 'sigmoid':
                d_val = rng.uniform(12, 20)
                s_prof = _sigmoid_profile(r, s1, s51, d=d_val)
                m_prof = _sigmoid_profile(r, m1, m51, d=d_val)
            else:
                a_val = rng.uniform(0.9, 1.1)
                s_prof = _linear_profile(r, s1, s51, a=a_val)
                m_prof = _linear_profile(r, m1, m51, a=a_val)
            
            # Discretize to K layers
            s_layers = np.mean(s_prof.reshape(K, -1), axis=1)
            m_layers = np.mean(m_prof.reshape(K, -1), axis=1)
            X[i, :K] = s_layers
            X[i, K:] = m_layers
            
        all_X.append(X)
        all_labels.append(np.full(N_PER_CLASS, class_idx, dtype=np.int64))

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
    print(f"  Total samples: {len(X_raw)}")
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
        'n_classes': 4,
        'class_names': [c['name'] for c in configs]
    }
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    print(f"Files saved to {output_dir}/")

if __name__ == '__main__':
    main()
