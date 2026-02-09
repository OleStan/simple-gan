"""Profile normalization and denormalization utilities."""

import numpy as np
from typing import Tuple, Optional
import json
from pathlib import Path


class ProfileNormalizer:
    """
    Handles normalization/denormalization of material profiles.
    
    Normalization is required for GAN training, which expects inputs
    in a specific range (typically [-1, 1]). This class ensures
    lossless round-trip conversion between physical units and
    normalized values.
    
    Example:
        >>> normalizer = ProfileNormalizer(1e6, 6e7, 1.0, 100.0)
        >>> sigma_norm, mu_norm = normalizer.normalize(sigma, mu)
        >>> sigma_recovered, mu_recovered = normalizer.denormalize(sigma_norm, mu_norm)
        >>> assert np.allclose(sigma, sigma_recovered)
    """
    
    def __init__(self, 
                 sigma_min: float, 
                 sigma_max: float,
                 mu_min: float, 
                 mu_max: float,
                 target_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize normalizer with physical bounds.
        
        Args:
            sigma_min: Minimum conductivity in dataset (S/m)
            sigma_max: Maximum conductivity in dataset (S/m)
            mu_min: Minimum permeability in dataset
            mu_max: Maximum permeability in dataset
            target_range: Target range for normalization (default: [-1, 1])
        """
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.target_min = float(target_range[0])
        self.target_max = float(target_range[1])
        
        if self.sigma_max <= self.sigma_min:
            raise ValueError(f"sigma_max ({sigma_max}) must be > sigma_min ({sigma_min})")
        if self.mu_max <= self.mu_min:
            raise ValueError(f"mu_max ({mu_max}) must be > mu_min ({mu_min})")
        if self.target_max <= self.target_min:
            raise ValueError(f"target_max must be > target_min")
    
    def normalize(self, 
                  sigma: np.ndarray, 
                  mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize profiles to target range.
        
        Formula: x_norm = (x - x_min) / (x_max - x_min) * (t_max - t_min) + t_min
        
        Args:
            sigma: Conductivity profile (K,) in S/m
            mu: Permeability profile (K,)
            
        Returns:
            Tuple of (sigma_norm, mu_norm) in target range
        """
        sigma = np.asarray(sigma, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        
        sigma_norm = ((sigma - self.sigma_min) / (self.sigma_max - self.sigma_min) * 
                     (self.target_max - self.target_min) + self.target_min)
        
        mu_norm = ((mu - self.mu_min) / (self.mu_max - self.mu_min) * 
                  (self.target_max - self.target_min) + self.target_min)
        
        return sigma_norm, mu_norm
    
    def denormalize(self,
                    sigma_norm: np.ndarray,
                    mu_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Denormalize profiles from target range to physical units.
        
        Formula: x = (x_norm - t_min) / (t_max - t_min) * (x_max - x_min) + x_min
        
        Args:
            sigma_norm: Normalized conductivity in target range
            mu_norm: Normalized permeability in target range
            
        Returns:
            Tuple of (sigma, mu) in physical units
        """
        sigma_norm = np.asarray(sigma_norm, dtype=np.float64)
        mu_norm = np.asarray(mu_norm, dtype=np.float64)
        
        sigma = ((sigma_norm - self.target_min) / (self.target_max - self.target_min) * 
                (self.sigma_max - self.sigma_min) + self.sigma_min)
        
        mu = ((mu_norm - self.target_min) / (self.target_max - self.target_min) * 
             (self.mu_max - self.mu_min) + self.mu_min)
        
        return sigma, mu
    
    def normalize_sigma(self, sigma: np.ndarray) -> np.ndarray:
        """Normalize only sigma values."""
        sigma = np.asarray(sigma, dtype=np.float64)
        return ((sigma - self.sigma_min) / (self.sigma_max - self.sigma_min) * 
               (self.target_max - self.target_min) + self.target_min)
    
    def normalize_mu(self, mu: np.ndarray) -> np.ndarray:
        """Normalize only mu values."""
        mu = np.asarray(mu, dtype=np.float64)
        return ((mu - self.mu_min) / (self.mu_max - self.mu_min) * 
               (self.target_max - self.target_min) + self.target_min)
    
    def denormalize_sigma(self, sigma_norm: np.ndarray) -> np.ndarray:
        """Denormalize only sigma values."""
        sigma_norm = np.asarray(sigma_norm, dtype=np.float64)
        return ((sigma_norm - self.target_min) / (self.target_max - self.target_min) * 
               (self.sigma_max - self.sigma_min) + self.sigma_min)
    
    def denormalize_mu(self, mu_norm: np.ndarray) -> np.ndarray:
        """Denormalize only mu values."""
        mu_norm = np.asarray(mu_norm, dtype=np.float64)
        return ((mu_norm - self.target_min) / (self.target_max - self.target_min) * 
               (self.mu_max - self.mu_min) + self.mu_min)
    
    def validate_roundtrip(self, 
                          sigma: np.ndarray, 
                          mu: np.ndarray, 
                          tol: float = 1e-8) -> Tuple[bool, dict]:
        """
        Verify normalization is lossless within tolerance.
        
        Args:
            sigma: Original conductivity profile
            mu: Original permeability profile
            tol: Maximum allowed absolute error
            
        Returns:
            Tuple of (is_valid, error_dict) where error_dict contains:
                - sigma_max_error: Maximum absolute error in sigma
                - mu_max_error: Maximum absolute error in mu
                - sigma_rel_error: Relative error in sigma
                - mu_rel_error: Relative error in mu
        """
        sigma_norm, mu_norm = self.normalize(sigma, mu)
        sigma_recovered, mu_recovered = self.denormalize(sigma_norm, mu_norm)
        
        sigma_error = np.abs(sigma - sigma_recovered)
        mu_error = np.abs(mu - mu_recovered)
        
        sigma_max_error = np.max(sigma_error)
        mu_max_error = np.max(mu_error)
        
        sigma_rel_error = np.max(sigma_error / (np.abs(sigma) + 1e-10))
        mu_rel_error = np.max(mu_error / (np.abs(mu) + 1e-10))
        
        is_valid = (sigma_max_error < tol) and (mu_max_error < tol)
        
        error_dict = {
            'sigma_max_error': float(sigma_max_error),
            'mu_max_error': float(mu_max_error),
            'sigma_rel_error': float(sigma_rel_error),
            'mu_rel_error': float(mu_rel_error),
            'is_valid': is_valid,
            'tolerance': tol
        }
        
        return is_valid, error_dict
    
    def save(self, filepath: str):
        """
        Save normalization parameters to JSON.
        
        Args:
            filepath: Path to save parameters
        """
        params = {
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'mu_min': self.mu_min,
            'mu_max': self.mu_max,
            'target_range': [self.target_min, self.target_max]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ProfileNormalizer':
        """
        Load normalization parameters from JSON.
        
        Args:
            filepath: Path to parameters file
            
        Returns:
            ProfileNormalizer instance
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        return cls(
            sigma_min=params['sigma_min'],
            sigma_max=params['sigma_max'],
            mu_min=params['mu_min'],
            mu_max=params['mu_max'],
            target_range=tuple(params['target_range'])
        )
    
    @classmethod
    def from_data(cls, 
                  sigma_data: np.ndarray, 
                  mu_data: np.ndarray,
                  margin: float = 0.05,
                  target_range: Tuple[float, float] = (-1.0, 1.0)) -> 'ProfileNormalizer':
        """
        Create normalizer from actual data with optional margin.
        
        Args:
            sigma_data: Array of sigma values (N, K) or (K,)
            mu_data: Array of mu values (N, K) or (K,)
            margin: Percentage margin to add to bounds (default: 5%)
            target_range: Target normalization range
            
        Returns:
            ProfileNormalizer instance
        """
        sigma_min = np.min(sigma_data)
        sigma_max = np.max(sigma_data)
        mu_min = np.min(mu_data)
        mu_max = np.max(mu_data)
        
        sigma_range = sigma_max - sigma_min
        mu_range = mu_max - mu_min
        
        sigma_min -= margin * sigma_range
        sigma_max += margin * sigma_range
        mu_min -= margin * mu_range
        mu_max += margin * mu_range
        
        mu_min = max(mu_min, 0.1)
        
        return cls(sigma_min, sigma_max, mu_min, mu_max, target_range)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProfileNormalizer(\n"
            f"  sigma: [{self.sigma_min:.2e}, {self.sigma_max:.2e}] S/m\n"
            f"  mu: [{self.mu_min:.2f}, {self.mu_max:.2f}]\n"
            f"  target: [{self.target_min:.2f}, {self.target_max:.2f}]\n"
            f")"
        )
