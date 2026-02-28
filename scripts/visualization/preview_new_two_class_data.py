
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path

class ProfileType(Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"

def _linear_profile(r, P_min, P_max, a):
    r_max = r[-1]
    return P_min + (P_max - P_min) * (r / r_max) ** a

def _sigmoid_profile(r, P_min, P_max, d, r_0=None):
    r_max = r[-1]
    if r_0 is None:
        r_0 = r_max / 2
    # Standard sigmoid
    raw = 1 / (1 + np.exp(-d * (r - r_0)))
    # Rescale to strictly [P_min, P_max]
    v_min = 1 / (1 + np.exp(-d * (0 - r_0)))
    v_max = 1 / (1 + np.exp(-d * (r_max - r_0)))
    norm = (raw - v_min) / (v_max - v_min)
    return P_min + (P_max - P_min) * norm

def main():
    r = np.linspace(0, 1, 100)
    rng = np.random.default_rng(42)
    n_samples = 10
    
    # 5x2 grid of subplots
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    # Common starting value ranges
    sigma1_base = 1e7
    mu1_base = 1.0
    
    for i in range(n_samples):
        ax = axes[i]
        ax_mu = ax.twinx()
        
        # Pick common starting points for this "pair" of samples across classes
        s1 = sigma1_base * rng.uniform(0.8, 1.2)
        m1 = mu1_base * rng.uniform(0.9, 1.1)
        
        # Class 1: Sigmoid
        s51_sig = 5e7 * rng.uniform(0.7, 1.3)
        m51_sig = 50.0 * rng.uniform(0.7, 1.3)
        d_val = rng.uniform(10, 25)
        
        # Class 2: Linear
        s51_lin = 3e7 * rng.uniform(0.7, 1.3)
        m51_lin = 80.0 * rng.uniform(0.7, 1.3)
        a_val = rng.uniform(0.8, 1.5)
        
        # Generate
        # Class 1: BOTH sigma and mu are SIGMOID
        s_sig = _sigmoid_profile(r, s1, s51_sig, d=d_val)
        m_sig = _sigmoid_profile(r, m1, m51_sig, d=d_val)
        
        # Class 2: BOTH sigma and mu are LINEAR
        s_lin = _linear_profile(r, s1, s51_lin, a=a_val)
        m_lin = _linear_profile(r, m1, m51_lin, a=a_val)
        
        # Plot Sigma (left axis)
        ln1 = ax.plot(r, s_sig, 'b-', alpha=0.8, label='Sigmoid Sigma')
        ln2 = ax.plot(r, s_lin, 'b--', alpha=0.6, label='Linear Sigma')
        ax.set_ylabel('Sigma (S/m)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot Mu (right axis)
        ln3 = ax_mu.plot(r, m_sig, 'r-', alpha=0.8, label='Sigmoid Mu')
        ln4 = ax_mu.plot(r, m_lin, 'r--', alpha=0.6, label='Linear Mu')
        ax_mu.set_ylabel('Mu (relative)', color='r')
        ax_mu.tick_params(axis='y', labelcolor='r')
        
        ax.set_title(f'Sample {i+1}: Common Start ({s1:.1e}, {m1:.1f})')
        ax.grid(True, alpha=0.2)
        
        if i == 0:
            lines = ln1 + ln2 + ln3 + ln4
            labs = [l.get_label() for l in lines]
            ax.legend(lines, labs, loc='upper left', fontsize='small')

    plt.suptitle('New Data: Comparison of Sigmoid vs Linear Classes\n(Each plot shows 2 classes starting from same layer 1)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    save_path = Path('new_data_grid_preview.png')
    plt.savefig(save_path, dpi=120)
    print(f"Grid preview saved to {save_path}")

if __name__ == '__main__':
    main()
