
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path

# V3 Architecture imports
from model import ContinuousConditionalGenerator

def generate_v3_report(model_dir):
    model_dir = Path(model_dir)
    output_dir = model_dir / "report_final"
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load config and norm_params
    with open(model_dir / "config.json", 'r') as f:
        config = json.load(f)
    nz = config['nz']
    n_classes = config['n_classes']
    K = config['K']
    
    with open(model_dir.parent.parent.parent / "data/training/normalization_params.json", 'r') as f:
        norm_params = json.load(f)
    
    # Init Model
    netG = ContinuousConditionalGenerator(nz=nz, K=K, n_classes=n_classes).to(device)
    model_path = model_dir / "models" / "netG_final.pt"
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    print(f"✓ Model loaded from {model_path}")
    
    # Generate Samples
    with torch.no_grad():
        noise = torch.randn(100, nz, device=device)
        # Generate some from each class
        labels = torch.randint(0, n_classes, (100,), device=device)
        output, sigma_out, mu_out = netG(noise, labels=labels)
        
        # Denormalize
        s_min, s_max = norm_params['sigma_min'], norm_params['sigma_max']
        m_min, m_max = norm_params['mu_min'], norm_params['mu_max']
        
        sigma_denorm = (sigma_out.cpu().numpy() + 1) / 2 * (s_max - s_min) + s_min
        mu_denorm = (mu_out.cpu().numpy() + 1) / 2 * (m_max - m_min) + m_min

    # Visualizations
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    for i in range(10):
        plt.plot(sigma_denorm[i], alpha=0.5)
    plt.title("Generated Sigma (V3 ISR)")
    plt.xlabel("Layer")
    plt.ylabel("Sigma (S/m)")
    
    plt.subplot(1, 2, 2)
    for i in range(10):
        plt.plot(mu_denorm[i], alpha=0.5)
    plt.title("Generated Mu (V3 ISR)")
    plt.xlabel("Layer")
    plt.ylabel("Relative Mu")
    
    plt.savefig(output_dir / "samples.png")
    plt.close()
    
    print(f"✓ Report saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    args = parser.parse_args()
    generate_v3_report(args.model_dir)
