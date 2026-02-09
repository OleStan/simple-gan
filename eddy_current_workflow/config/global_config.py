"""Global configuration - single source of truth for all pipeline parameters."""

from dataclasses import dataclass, field
from typing import Tuple
import json
from pathlib import Path


@dataclass
class GlobalConfig:
    """
    Global configuration for eddy-current workflow.
    
    This class serves as the single source of truth for all parameters
    used across the pipeline. All components should reference CONFIG
    rather than hardcoding values.
    """
    
    # Discretization parameters
    K: int = 51
    r_min: float = 0.0
    r_max: float = 1.0
    physical_depth: float = 1e-3
    
    # Material property bounds
    sigma_bounds: Tuple[float, float] = (1e6, 6e7)
    mu_bounds: Tuple[float, float] = (1.0, 100.0)
    
    # Normalization range (for GAN training)
    normalize_to_range: Tuple[float, float] = (-1.0, 1.0)
    
    # Probe settings (defaults, Dodd-Deeds coil geometry)
    default_frequency: float = 1e6
    default_inner_radius: float = 4e-3
    default_outer_radius: float = 6e-3
    default_lift_off: float = 0.5e-3
    default_coil_height: float = 2e-3
    default_n_turns: int = 100
    
    # Optimization parameters
    inverse_max_iter: int = 1000
    inverse_n_starts: int = 10
    inverse_tol: float = 1e-6
    inverse_method: str = 'L-BFGS-B'
    
    # Regularization weights
    lambda_smooth: float = 1e-3
    lambda_monotonic: float = 1e-2
    
    # Database settings
    db_path: str = './profile_database.h5'
    db_k_neighbors: int = 5
    
    # Training data paths
    training_data_dir: str = './training_data'
    results_dir: str = './results'
    
    # Random seed for reproducibility
    seed: int = 42
    
    @property
    def layer_thickness(self) -> float:
        """Calculate thickness of each layer in meters."""
        return self.physical_depth / self.K
    
    @property
    def r_array(self):
        """Generate normalized depth array."""
        import numpy as np
        return np.linspace(self.r_min, self.r_max, 1000)
    
    def validate(self):
        """
        Validate configuration consistency.
        
        Raises:
            AssertionError: If configuration is invalid
        """
        assert self.K > 0, "K must be positive"
        assert self.r_max > self.r_min, "r_max must be > r_min"
        assert self.sigma_bounds[1] > self.sigma_bounds[0], "Invalid sigma bounds"
        assert self.mu_bounds[1] > self.mu_bounds[0], "Invalid mu bounds"
        assert self.physical_depth > 0, "Physical depth must be positive"
        assert self.normalize_to_range[1] > self.normalize_to_range[0], "Invalid normalization range"
        assert self.default_frequency > 0, "Frequency must be positive"
        assert self.inverse_max_iter > 0, "Max iterations must be positive"
        assert self.inverse_n_starts > 0, "Number of starts must be positive"
    
    def save(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        config_dict = {
            'K': self.K,
            'r_min': self.r_min,
            'r_max': self.r_max,
            'physical_depth': self.physical_depth,
            'sigma_bounds': list(self.sigma_bounds),
            'mu_bounds': list(self.mu_bounds),
            'normalize_to_range': list(self.normalize_to_range),
            'default_frequency': self.default_frequency,
            'default_inner_radius': self.default_inner_radius,
            'default_outer_radius': self.default_outer_radius,
            'default_lift_off': self.default_lift_off,
            'default_coil_height': self.default_coil_height,
            'default_n_turns': self.default_n_turns,
            'inverse_max_iter': self.inverse_max_iter,
            'inverse_n_starts': self.inverse_n_starts,
            'inverse_tol': self.inverse_tol,
            'inverse_method': self.inverse_method,
            'lambda_smooth': self.lambda_smooth,
            'lambda_monotonic': self.lambda_monotonic,
            'db_path': self.db_path,
            'db_k_neighbors': self.db_k_neighbors,
            'training_data_dir': self.training_data_dir,
            'results_dir': self.results_dir,
            'seed': self.seed
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'GlobalConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            GlobalConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config_dict['sigma_bounds'] = tuple(config_dict['sigma_bounds'])
        config_dict['mu_bounds'] = tuple(config_dict['mu_bounds'])
        config_dict['normalize_to_range'] = tuple(config_dict['normalize_to_range'])
        
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"GlobalConfig(\n"
            f"  Discretization: K={self.K}, depth={self.physical_depth*1e3:.2f}mm\n"
            f"  Sigma bounds: [{self.sigma_bounds[0]:.2e}, {self.sigma_bounds[1]:.2e}] S/m\n"
            f"  Mu bounds: [{self.mu_bounds[0]:.2f}, {self.mu_bounds[1]:.2f}]\n"
            f"  Probe: f={self.default_frequency/1e6:.2f}MHz, r=[{self.default_inner_radius*1e3:.1f},{self.default_outer_radius*1e3:.1f}]mm\n"
            f"  Inverse: method={self.inverse_method}, n_starts={self.inverse_n_starts}\n"
            f")"
        )


CONFIG = GlobalConfig()
CONFIG.validate()
