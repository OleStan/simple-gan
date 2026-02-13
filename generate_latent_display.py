#!/usr/bin/env python
"""
Generate and display latent vectors (noise) for GAN models in console.
Shows the structure and values of random latent vectors used for generation.
"""

import torch
import numpy as np

def generate_latent_vectors(nz=100, batch_size=5, device='cpu'):
    """Generate and display latent vectors."""
    print(f"=== GENERATING LATENT VECTORS ===")
    print(f"Latent dimension (nz): {nz}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print()
    
    # Generate random latent vectors
    z = torch.randn(batch_size, nz, device=device)
    
    print(f"Generated tensor shape: {z.shape}")
    print(f"Data type: {z.dtype}")
    print(f"Device: {z.device}")
    print()
    
    # Display statistics
    print("=== LATENT VECTOR STATISTICS ===")
    print(f"Mean: {z.mean().item():.6f}")
    print(f"Std: {z.std().item():.6f}")
    print(f"Min: {z.min().item():.6f}")
    print(f"Max: {z.max().item():.6f}")
    print()
    
    # Display each latent vector
    print("=== INDIVIDUAL LATENT VECTORS ===")
    for i in range(batch_size):
        print(f"\nLatent vector {i+1}/{batch_size}:")
        vector = z[i].cpu().numpy()
        
        # Show first 10 and last 10 values
        print(f"  First 10 values: {vector[:10]}")
        if len(vector) > 20:
            print(f"  ...")
            print(f"  Last 10 values: {vector[-10:]}")
        else:
            print(f"  All values: {vector}")
        
        print(f"  Norm: {np.linalg.norm(vector):.6f}")
        print(f"  Mean: {vector.mean():.6f}")
        print(f"  Std: {vector.std():.6f}")

def compare_different_nz():
    """Compare latent vectors with different dimensions."""
    print("\n" + "="*60)
    print("COMPARING DIFFERENT LATENT DIMENSIONS")
    print("="*60)
    
    nz_values = [10, 50, 100, 200]
    
    for nz in nz_values:
        print(f"\n--- nz = {nz} ---")
        z = torch.randn(1, nz)
        print(f"Shape: {z.shape}")
        print(f"Sample values: {z[0][:5].tolist()}...")
        print(f"Statistics: mean={z.mean().item():.4f}, std={z.std().item():.4f}")

def show_model_latent_usage():
    """Show how latent vectors are used in different models."""
    print("\n" + "="*60)
    print("LATENT USAGE IN DIFFERENT MODELS")
    print("="*60)
    
    models_config = [
        {"name": "DualHeadGenerator", "nz": 100, "K": 50},
        {"name": "ConditionalConv1DGenerator", "nz": 100, "K": 50, "conditional": False},
        {"name": "Conv1DGenerator", "nz": 100, "K": 50},
    ]
    
    for config in models_config:
        print(f"\n--- {config['name']} ---")
        print(f"Latent dim (nz): {config['nz']}")
        print(f"Profile layers (K): {config['K']}")
        
        # Generate latent vector
        z = torch.randn(1, config['nz'])
        print(f"Latent vector shape: {z.shape}")
        print(f"Sample latent values: {z[0][:5].tolist()}...")
        
        # Show expected output dimensions
        if 'conditional' in config and config['conditional']:
            expected_output = (1, 2 * config['K'])  # sigma + mu profiles
        else:
            expected_output = (1, 2 * config['K'])  # sigma + mu profiles
        
        print(f"Expected output shape: {expected_output}")

if __name__ == "__main__":
    print("GAN LATENT VECTOR GENERATOR AND ANALYZER")
    print("=" * 50)
    
    # Generate standard latent vectors
    generate_latent_vectors(nz=100, batch_size=3)
    
    # Compare different dimensions
    compare_different_nz()
    
    # Show model usage
    show_model_latent_usage()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("- Latent vectors are random noise (typically normal distribution)")
    print("- Standard dimension used: nz=100")
    print("- Each vector encodes information for generating σ and μ profiles")
    print("- Different models may use different latent dimensions")
    print("- The generator transforms latent vectors into realistic profiles")
