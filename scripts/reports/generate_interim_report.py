#!/usr/bin/env python
"""Generate interim report from checkpoint while training continues."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

from wgan_dual_profiles import DualHeadGenerator


def generate_interim_report(model_dir, checkpoint_epoch=100):
    """Generate report from intermediate checkpoint."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print(f"Generating Interim Report (Epoch {checkpoint_epoch})")
    print("="*60)
    
    with open(f'{model_dir}/normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    K = norm_params['K']
    nz = 100
    
    netG = DualHeadGenerator(nz=nz, K=K).to(device)
    checkpoint_path = f'{model_dir}/models/netG_epoch_{checkpoint_epoch}.pth'
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint_epoch}")
    
    n_samples = 100
    with torch.no_grad():
        noise = torch.randn(n_samples, nz, device=device)
        fake_data, fake_sigma, fake_mu = netG(noise)
        fake_data_np = fake_data.cpu().numpy()
    
    sigma_normalized = fake_data_np[:, :K]
    mu_normalized = fake_data_np[:, K:2*K]
    
    sigma_min = norm_params['sigma_min']
    sigma_max = norm_params['sigma_max']
    mu_min = norm_params['mu_min']
    mu_max = norm_params['mu_max']
    
    sigma_gen = (sigma_normalized + 1) / 2 * (sigma_max - sigma_min) + sigma_min
    mu_gen = (mu_normalized + 1) / 2 * (mu_max - mu_min) + mu_min
    
    print(f"\nGenerated {n_samples} samples:")
    print(f"  σ: [{sigma_gen.min():.2e}, {sigma_gen.max():.2e}] S/m")
    print(f"  μ: [{mu_gen.min():.2f}, {mu_gen.max():.2f}]")
    
    report_dir = f'{model_dir}/interim_report_epoch_{checkpoint_epoch}'
    Path(report_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            ax = axes[i, j]
            ax2 = ax.twinx()
            
            ax.plot(sigma_gen[idx], 'b-', linewidth=1.5, alpha=0.8)
            ax2.plot(mu_gen[idx], 'r-', linewidth=1.5, alpha=0.8)
            
            ax.set_ylabel('σ (S/m)', color='b', fontsize=9)
            ax2.set_ylabel('μᵣ', color='r', fontsize=9)
            
            ax.tick_params(axis='y', labelcolor='b', labelsize=8)
            ax2.tick_params(axis='y', labelcolor='r', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
            
            ax.set_xlabel('Layer', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Generated Profiles (Epoch {checkpoint_epoch})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{report_dir}/sample_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Report saved to {report_dir}/")
    
    return sigma_gen, mu_gen, report_dir


if __name__ == '__main__':
    results_dirs = sorted(Path('./results').glob('dual_wgan_*'))
    if not results_dirs:
        print("No training results found")
        sys.exit(1)
    
    model_dir = str(results_dirs[-1])
    print(f"Using: {model_dir}")
    
    sigma_gen, mu_gen, report_dir = generate_interim_report(model_dir, checkpoint_epoch=100)
