#!/usr/bin/env python
"""Generate two-class training dataset: sigmoid (both) vs linear (both) with matched starts."""

import numpy as np
import json
from pathlib import Path

N_PER_CLASS = 2000
K = 50
R_MIN = 0.0
R_MAX = 1.0
N_POINTS = 1000
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
    rng = np.random.default_rng(42)
    r = np.linspace(R_MIN, R_MAX, N_POINTS)
    s1_vals = SIGMA1_BASE * rng.uniform(0.8, 1.2, N_PER_CLASS)
    m1_vals = MU1_BASE * rng.uniform(0.9, 1.1, N_PER_CLASS)
    all_X, all_labels = [], []

    # Class 0: Sigmoid
    X0 = np.zeros((N_PER_CLASS, 2 * K))
    for i in range(N_PER_CLASS):
        s_prof = _sigmoid_profile(r, s1_vals[i], 5e7 * rng.uniform(0.7, 1.3), d=rng.uniform(10, 25))
        m_prof = _sigmoid_profile(r, m1_vals[i], 50.0 * rng.uniform(0.7, 1.3), d=rng.uniform(10, 25))
        X0[i, :K] = np.mean(s_prof.reshape(K, -1), axis=1)
        X0[i, K:] = np.mean(m_prof.reshape(K, -1), axis=1)
    all_X.append(X0); all_labels.append(np.zeros(N_PER_CLASS, dtype=np.int64))

    # Class 1: Linear
    X1 = np.zeros((N_PER_CLASS, 2 * K))
    for i in range(N_PER_CLASS):
        s_prof = _linear_profile(r, s1_vals[i], 3e7 * rng.uniform(0.7, 1.3), a=rng.uniform(0.8, 1.5))
        m_prof = _linear_profile(r, m1_vals[i], 80.0 * rng.uniform(0.7, 1.3), a=rng.uniform(0.8, 1.5))
        X1[i, :K] = np.mean(s_prof.reshape(K, -1), axis=1)
        X1[i, K:] = np.mean(m_prof.reshape(K, -1), axis=1)
    all_X.append(X1); all_labels.append(np.ones(N_PER_CLASS, dtype=np.int64))

    X_raw = np.concatenate(all_X, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)
    idx = rng.permutation(len(X_raw))
    X_raw, y_labels = X_raw[idx], y_labels[idx]
    
    np.save(output_dir / 'X_raw.npy', X_raw)
    np.save(output_dir / 'y_labels.npy', y_labels)
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump({
            'K': K, 'sigma_min': float(X_raw[:, :K].min()), 'sigma_max': float(X_raw[:, :K].max()),
            'mu_min': float(X_raw[:, K:].min()), 'mu_max': float(X_raw[:, K:].max()),
            'N': len(X_raw), 'n_classes': 2, 'class_names': ['sigmoid', 'linear']
        }, f, indent=2)
    print(f"2-class data generated in {output_dir}")

if __name__ == '__main__':
    main()
