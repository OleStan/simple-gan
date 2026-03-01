#!/usr/bin/env python
"""Generate four-class training dataset: 2 Sigmoid, 2 Linear with dispersion."""

import numpy as np
import json
from pathlib import Path

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
    rng = np.random.default_rng(42)
    r = np.linspace(R_MIN, R_MAX, N_POINTS)
    configs = [
        {'name': 'sigmoid_low',  'type': 'sigmoid', 's1': 1.0e7, 'm1': 1.0, 's51': 3.0e7, 'm51': 30},
        {'name': 'sigmoid_high', 'type': 'sigmoid', 's1': 2.5e7, 'm1': 5.0, 's51': 5.5e7, 'm51': 80},
        {'name': 'linear_low',   'type': 'linear',  's1': 1.0e7, 'm1': 1.0, 's51': 3.0e7, 'm51': 30},
        {'name': 'linear_high',  'type': 'linear',  's1': 2.5e7, 'm1': 5.0, 's51': 5.5e7, 'm51': 80},
    ]
    all_X, all_labels = [], []

    for class_idx, cfg in enumerate(configs):
        X = np.zeros((N_PER_CLASS, 2 * K))
        for i in range(N_PER_CLASS):
            disp = lambda val: val * rng.uniform(0.92, 1.08)
            s1, m1, s51, m51 = disp(cfg['s1']), disp(cfg['m1']), disp(cfg['s51']), disp(cfg['m51'])
            if cfg['type'] == 'sigmoid':
                s_prof = _sigmoid_profile(r, s1, s51, d=rng.uniform(12, 20))
                m_prof = _sigmoid_profile(r, m1, m51, d=rng.uniform(12, 20))
            else:
                s_prof = _linear_profile(r, s1, s51, a=rng.uniform(0.9, 1.1))
                m_prof = _linear_profile(r, m1, m51, a=rng.uniform(0.9, 1.1))
            X[i, :K] = np.mean(s_prof.reshape(K, -1), axis=1)
            X[i, K:] = np.mean(m_prof.reshape(K, -1), axis=1)
        all_X.append(X); all_labels.append(np.full(N_PER_CLASS, class_idx, dtype=np.int64))

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
            'N': len(X_raw), 'n_classes': 4, 'class_names': [c['name'] for c in configs]
        }, f, indent=2)
    print(f"4-class data generated in {output_dir}")

if __name__ == '__main__':
    main()
