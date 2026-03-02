#!/usr/bin/env python
"""
Generate 8-class training dataset: 
Standard (Increasing) and Mirror (Decreasing) versions of 
Sigmoid and Linear profiles with two ranges (Low/High).
"""

import numpy as np
import json
import argparse
from pathlib import Path

N_PER_CLASS = 1500  # Total 12,000 samples
K = 50
R_MIN = 0.0
R_MAX = 1.0
N_POINTS = 1000

# Base Physical Ranges
SIGMA_RANGES = {
    'low': (1e7, 3e7),
    'high': (3e7, 6e7)
}
MU_RANGES = {
    'low': (1.0, 40.0),
    'high': (40.0, 85.0)
}

def _linear_profile(r, P_start, P_end, a):
    r_max = r[-1]
    return P_start + (P_end - P_start) * (r / r_max) ** a

def _sigmoid_profile(r, P_start, P_end, d, r_0=None):
    r_max = r[-1]
    if r_0 is None:
        r_0 = r_max / 2
    raw = 1 / (1 + np.exp(-d * (r - r_0)))
    v_min = 1 / (1 + np.exp(-d * (0 - r_0)))
    v_max = 1 / (1 + np.exp(-d * (r_max - r_0)))
    norm = (raw - v_min) / (v_max - v_min)
    return P_start + (P_end - P_start) * norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./data/training_8class')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    r = np.linspace(R_MIN, R_MAX, N_POINTS)
    
    all_X = []
    all_labels = []
    class_names = []

    # Helper to generate a batch
    def add_batch(name, p_type, s_range, m_range, inverse=False):
        X = np.zeros((N_PER_CLASS, 2 * K))
        for i in range(N_PER_CLASS):
            # Randomize start/end within range
            s_start = rng.uniform(s_range[0] * 0.8, s_range[0] * 1.2)
            s_end = rng.uniform(s_range[1] * 0.8, s_range[1] * 1.2)
            m_start = rng.uniform(m_range[0] * 0.9, m_range[0] * 1.1)
            m_end = rng.uniform(m_range[1] * 0.9, m_range[1] * 1.1)
            
            if inverse:
                s_start, s_end = s_end, s_start
                m_start, m_end = m_end, m_start
            
            if p_type == 'sigmoid':
                d = rng.uniform(10, 25)
                s_prof = _sigmoid_profile(r, s_start, s_end, d)
                m_prof = _sigmoid_profile(r, m_start, m_end, d)
            else:
                a = rng.uniform(0.8, 1.5)
                s_prof = _linear_profile(r, s_start, s_end, a)
                m_prof = _linear_profile(r, m_start, m_end, a)
                
            X[i, :K] = np.mean(s_prof.reshape(K, -1), axis=1)
            X[i, K:] = np.mean(m_prof.reshape(K, -1), axis=1)
            
        all_X.append(X)
        all_labels.append(np.full(N_PER_CLASS, len(class_names), dtype=np.int64))
        class_names.append(name)

    # 1-4: Standard Increasing
    add_batch('sigmoid_low', 'sigmoid', SIGMA_RANGES['low'], MU_RANGES['low'])
    add_batch('sigmoid_high', 'sigmoid', SIGMA_RANGES['high'], MU_RANGES['high'])
    add_batch('linear_low', 'linear', SIGMA_RANGES['low'], MU_RANGES['low'])
    add_batch('linear_high', 'linear', SIGMA_RANGES['high'], MU_RANGES['high'])

    # 5-8: Mirror Decreasing
    add_batch('sigmoid_low_inv', 'sigmoid', SIGMA_RANGES['low'], MU_RANGES['low'], inverse=True)
    add_batch('sigmoid_high_inv', 'sigmoid', SIGMA_RANGES['high'], MU_RANGES['high'], inverse=True)
    add_batch('linear_low_inv', 'linear', SIGMA_RANGES['low'], MU_RANGES['low'], inverse=True)
    add_batch('linear_high_inv', 'linear', SIGMA_RANGES['high'], MU_RANGES['high'], inverse=True)

    X_raw = np.concatenate(all_X, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)
    
    # Shuffle
    idx = rng.permutation(len(X_raw))
    X_raw, y_labels = X_raw[idx], y_labels[idx]
    
    np.save(output_dir / 'X_raw.npy', X_raw)
    np.save(output_dir / 'y_labels.npy', y_labels)
    
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump({
            'K': K,
            'sigma_min': float(X_raw[:, :K].min()),
            'sigma_max': float(X_raw[:, :K].max()),
            'mu_min': float(X_raw[:, K:].min()),
            'mu_max': float(X_raw[:, K:].max()),
            'N': len(X_raw),
            'n_classes': len(class_names),
            'class_names': class_names
        }, f, indent=2)
    
    print(f"8-class 'Mirror' dataset generated in {output_dir}")

if __name__ == '__main__':
    main()
