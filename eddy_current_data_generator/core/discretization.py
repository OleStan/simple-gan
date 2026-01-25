"""Discretization of continuous profiles into piecewise-constant layers."""

import numpy as np


def discretize_profile(r: np.ndarray, 
                       profile: np.ndarray, 
                       K: int,
                       mode: str = 'centers') -> np.ndarray:
    """
    Discretize a continuous profile into K piecewise-constant layers.
    
    Args:
        r: Radial coordinates (N points)
        profile: Parameter values at each r (N points)
        K: Number of discrete layers
        mode: Discretization mode
            - 'centers': Use value at layer center
            - 'average': Use average value over layer
            - 'start': Use value at layer start
            
    Returns:
        Array of K layer values
        
    Example:
        >>> r = np.linspace(0, 1, 1000)
        >>> sigma = 1e6 + (6e7 - 1e6) * r
        >>> layers = discretize_profile(r, sigma, K=50, mode='centers')
        >>> layers.shape
        (50,)
    """
    r_min = r[0]
    r_max = r[-1]
    
    layer_boundaries = np.linspace(r_min, r_max, K + 1)
    
    layer_values = np.zeros(K)
    
    for k in range(K):
        r_start = layer_boundaries[k]
        r_end = layer_boundaries[k + 1]
        
        if mode == 'centers':
            r_center = (r_start + r_end) / 2
            idx = np.argmin(np.abs(r - r_center))
            layer_values[k] = profile[idx]
            
        elif mode == 'average':
            mask = (r >= r_start) & (r <= r_end)
            if np.sum(mask) > 0:
                layer_values[k] = np.mean(profile[mask])
            else:
                idx = np.argmin(np.abs(r - (r_start + r_end) / 2))
                layer_values[k] = profile[idx]
                
        elif mode == 'start':
            idx = np.argmin(np.abs(r - r_start))
            layer_values[k] = profile[idx]
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'centers', 'average', or 'start'")
    
    return layer_values


def discretize_dual_profiles(r: np.ndarray,
                             sigma_profile: np.ndarray,
                             mu_profile: np.ndarray,
                             K: int,
                             mode: str = 'centers') -> tuple:
    """
    Discretize both σ and μ profiles.
    
    Args:
        r: Radial coordinates
        sigma_profile: Continuous σ profile
        mu_profile: Continuous μ profile
        K: Number of layers
        mode: Discretization mode
        
    Returns:
        Tuple of (sigma_layers, mu_layers), each with K values
        
    Example:
        >>> r = np.linspace(0, 1, 1000)
        >>> sigma = 1e6 + (6e7 - 1e6) * r
        >>> mu = 1 + (100 - 1) * r
        >>> sigma_layers, mu_layers = discretize_dual_profiles(r, sigma, mu, K=50)
    """
    sigma_layers = discretize_profile(r, sigma_profile, K, mode)
    mu_layers = discretize_profile(r, mu_profile, K, mode)
    
    return sigma_layers, mu_layers
