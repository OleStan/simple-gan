
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    r = np.linspace(0, 1, 100)
    rng = np.random.default_rng(42)
    n_samples_per_class = 5
    
    # Define 4 classes
    # Group A: Low Start, Med End
    # Group B: High Start, High End
    
    configs = [
        {'name': 'Sigmoid Low', 'type': 'sigmoid', 's1': 1.0e7, 'm1': 1.0, 's51': 3.0e7, 'm51': 30, 'color': 'blue', 'style': '-'},
        {'name': 'Sigmoid High', 'type': 'sigmoid', 's1': 2.5e7, 'm1': 5.0, 's51': 5.5e7, 'm51': 80, 'color': 'cyan', 'style': '-'},
        {'name': 'Linear Low', 'type': 'linear', 's1': 1.0e7, 'm1': 1.0, 's51': 3.0e7, 'm51': 30, 'color': 'red', 'style': '--'},
        {'name': 'Linear High', 'type': 'linear', 's1': 2.5e7, 'm1': 5.0, 's51': 5.5e7, 'm51': 80, 'color': 'orange', 'style': '--'},
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    for cfg in configs:
        for i in range(n_samples_per_class):
            # 5-10% dispersion
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
                
            ax1.plot(r, s_prof, color=cfg['color'], linestyle=cfg['style'], alpha=0.6, 
                    label=cfg['name'] if i == 0 else "")
            ax2.plot(r, m_prof, color=cfg['color'], linestyle=cfg['style'], alpha=0.6, 
                    label=cfg['name'] if i == 0 else "")

    ax1.set_title('Conductivity (sigma) - 4 Classes')
    ax1.set_ylabel('S/m')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize='small')
    
    ax2.set_title('Permeability (mu) - 4 Classes')
    ax2.set_ylabel('Relative Permeability')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize='small')
    
    plt.suptitle('4-Class Dataset Preview: 2 Sigmoid, 2 Linear\nDifferent starting points and end points with 8% dispersion', fontsize=14)
    plt.tight_layout()
    
    save_path = Path('new_4_class_preview.png')
    plt.savefig(save_path, dpi=150)
    print(f"4-class preview saved to {save_path}")

if __name__ == '__main__':
    main()
