
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path

def preview_8_class(data_dir):
    data_dir = Path(data_dir)
    X = np.load(data_dir / 'X_raw.npy')
    y = np.load(data_dir / 'y_labels.npy')
    with open(data_dir / 'normalization_params.json', 'r') as f:
        params = json.load(f)
    
    K = params['K']
    class_names = params['class_names']
    n_classes = params['n_classes']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i in range(n_classes):
        ax = axes[i]
        mask = (y == i)
        X_class = X[mask]
        
        # Show first 5 samples
        for j in range(min(5, len(X_class))):
            ax.plot(X_class[j, :K], 'b-', alpha=0.3, label='Sigma' if j==0 else "")
            ax2 = ax.twinx()
            ax2.plot(X_class[j, K:], 'r--', alpha=0.3, label='Mu' if j==0 else "")
            
        ax.set_title(f"Class {i}: {class_names[i]}")
        if i >= 4: ax.set_xlabel("Layer")
        if i % 4 == 0: ax.set_ylabel("Sigma (S/m)")
        
    plt.tight_layout()
    plt.savefig(data_dir / 'preview_8_class.png')
    print(f"✓ Preview saved to {data_dir / 'preview_8_class.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/training_8class')
    args = parser.parse_args()
    preview_8_class(args.data_dir)
