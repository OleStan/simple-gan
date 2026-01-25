"""Complete dataset builder integrating all components."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from .roberts_sequence import generate_roberts_plan
from .material_profiles import ProfileType, generate_dual_profiles
from .discretization import discretize_dual_profiles


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    N: int = 1000
    K: int = 50
    r_min: float = 0.0
    r_max: float = 1.0
    sigma_bounds: Tuple[float, float] = (1e6, 6e7)
    mu_bounds: Tuple[float, float] = (1.0, 100.0)
    include_frequency: bool = False
    frequency_bounds: Optional[Tuple[float, float]] = None
    include_resistance: bool = False
    resistance_bounds: Optional[Tuple[float, float]] = None
    discretization_mode: str = 'centers'
    seed: int = 42


def build_dataset(config: DatasetConfig) -> Tuple[np.ndarray, Dict]:
    """
    Build complete dataset for training.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Tuple of (X, metadata) where:
            X: Array of shape (N, feature_dim)
            metadata: Dictionary with dataset information
            
    Example:
        >>> config = DatasetConfig(N=1000, K=50)
        >>> X, metadata = build_dataset(config)
        >>> X.shape
        (1000, 100)  # 50 sigma + 50 mu layers
    """
    d = 6
    if config.include_frequency:
        d += 1
    if config.include_resistance:
        d += 1
    
    bounds = [
        config.sigma_bounds,
        config.sigma_bounds,
        config.mu_bounds,
        config.mu_bounds,
        (0.5, 2.0),
        (0.5, 2.0)
    ]
    
    if config.include_frequency:
        if config.frequency_bounds is None:
            raise ValueError("frequency_bounds required when include_frequency=True")
        bounds.append(config.frequency_bounds)
    
    if config.include_resistance:
        if config.resistance_bounds is None:
            raise ValueError("resistance_bounds required when include_resistance=True")
        bounds.append(config.resistance_bounds)
    
    plan = generate_roberts_plan(config.N, d, bounds=bounds, seed=config.seed)
    
    sigma_min_vals = plan[:, 0]
    sigma_max_vals = plan[:, 1]
    mu_min_vals = plan[:, 2]
    mu_max_vals = plan[:, 3]
    sigma_shape_vals = plan[:, 4]
    mu_shape_vals = plan[:, 5]
    
    profile_types = [ProfileType.LINEAR, ProfileType.EXPONENTIAL, 
                    ProfileType.POWER, ProfileType.SIGMOID]
    
    n_points = 1000
    r = np.linspace(config.r_min, config.r_max, n_points)
    
    feature_dim = 2 * config.K
    if config.include_frequency:
        feature_dim += 1
    if config.include_resistance:
        feature_dim += 1
    
    X = np.zeros((config.N, feature_dim))
    
    shape_params = np.zeros((config.N, 6))
    
    for i in range(config.N):
        sigma_type = profile_types[i % 4]
        mu_type = profile_types[(i + 1) % 4]
        
        sigma_profile, mu_profile = generate_dual_profiles(
            r,
            sigma_min_vals[i], sigma_max_vals[i],
            mu_min_vals[i], mu_max_vals[i],
            sigma_type, mu_type,
            sigma_shape_vals[i], mu_shape_vals[i]
        )
        
        sigma_layers, mu_layers = discretize_dual_profiles(
            r, sigma_profile, mu_profile, config.K, config.discretization_mode
        )
        
        X[i, :config.K] = sigma_layers
        X[i, config.K:2*config.K] = mu_layers
        
        shape_params[i, 0] = sigma_min_vals[i]
        shape_params[i, 1] = sigma_max_vals[i]
        shape_params[i, 2] = mu_min_vals[i]
        shape_params[i, 3] = mu_max_vals[i]
        shape_params[i, 4] = sigma_shape_vals[i]
        shape_params[i, 5] = mu_shape_vals[i]
        
        col_idx = 2 * config.K
        
        if config.include_frequency:
            X[i, col_idx] = plan[i, 6]
            col_idx += 1
        
        if config.include_resistance:
            X[i, col_idx] = plan[i, 6 + (1 if config.include_frequency else 0)]
    
    metadata = {
        'N': config.N,
        'K': config.K,
        'feature_dim': feature_dim,
        'r_min': config.r_min,
        'r_max': config.r_max,
        'sigma_bounds': config.sigma_bounds,
        'mu_bounds': config.mu_bounds,
        'discretization_mode': config.discretization_mode,
        'include_frequency': config.include_frequency,
        'include_resistance': config.include_resistance,
        'seed': config.seed,
        'profile_types_used': [pt.value for pt in profile_types],
        'shape_params': {
            'sigma_min_range': [float(sigma_min_vals.min()), float(sigma_min_vals.max())],
            'sigma_max_range': [float(sigma_max_vals.min()), float(sigma_max_vals.max())],
            'mu_min_range': [float(mu_min_vals.min()), float(mu_min_vals.max())],
            'mu_max_range': [float(mu_max_vals.min()), float(mu_max_vals.max())],
            'sigma_shape_range': [float(sigma_shape_vals.min()), float(sigma_shape_vals.max())],
            'mu_shape_range': [float(mu_shape_vals.min()), float(mu_shape_vals.max())]
        }
    }
    
    return X, metadata
