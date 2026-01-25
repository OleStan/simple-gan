"""Material profile generation functions."""

import numpy as np
from enum import Enum


class ProfileType(Enum):
    """Types of material parameter profiles."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POWER = "power"
    SIGMOID = "sigmoid"


def _linear_profile(r: np.ndarray, P_min: float, P_max: float, a: float) -> np.ndarray:
    """
    Linear profile: P(r) = P_min + (P_max - P_min) * (r/r_max)^a
    
    Args:
        r: Radial coordinates [0, r_max]
        P_min: Minimum parameter value
        P_max: Maximum parameter value
        a: Shape parameter (typically 0.5 to 2.0)
    """
    r_max = r[-1]
    return P_min + (P_max - P_min) * (r / r_max) ** a


def _exponential_profile(r: np.ndarray, P_min: float, P_max: float, b: float) -> np.ndarray:
    """
    Exponential profile: P(r) = P_min * exp(b * r/r_max)
    
    Args:
        r: Radial coordinates [0, r_max]
        P_min: Minimum parameter value
        P_max: Maximum parameter value (used to determine b if not provided)
        b: Shape parameter (typically 0.5 to 3.0)
    """
    r_max = r[-1]
    return P_min * np.exp(b * r / r_max)


def _power_profile(r: np.ndarray, P_min: float, P_max: float, c: float) -> np.ndarray:
    """
    Power profile: P(r) = P_min + (P_max - P_min) * (1 - exp(-c * r/r_max))
    
    Args:
        r: Radial coordinates [0, r_max]
        P_min: Minimum parameter value
        P_max: Maximum parameter value
        c: Shape parameter (typically 1.0 to 5.0)
    """
    r_max = r[-1]
    return P_min + (P_max - P_min) * (1 - np.exp(-c * r / r_max))


def _sigmoid_profile(r: np.ndarray, P_min: float, P_max: float, d: float, r_0: float = None) -> np.ndarray:
    """
    Sigmoid profile: P(r) = P_min + (P_max - P_min) / (1 + exp(-d * (r - r_0)))
    
    Args:
        r: Radial coordinates [0, r_max]
        P_min: Minimum parameter value
        P_max: Maximum parameter value
        d: Steepness parameter (typically 5.0 to 20.0)
        r_0: Inflection point (default: r_max/2)
    """
    r_max = r[-1]
    if r_0 is None:
        r_0 = r_max / 2
    
    return P_min + (P_max - P_min) / (1 + np.exp(-d * (r - r_0)))


def make_profile(r: np.ndarray, 
                profile_type: ProfileType,
                P_min: float,
                P_max: float,
                shape_param: float,
                **kwargs) -> np.ndarray:
    """
    Generate a material parameter profile.
    
    Args:
        r: Radial coordinates array
        profile_type: Type of profile (LINEAR, EXPONENTIAL, POWER, SIGMOID)
        P_min: Minimum parameter value
        P_max: Maximum parameter value
        shape_param: Shape parameter (meaning depends on profile type)
        **kwargs: Additional parameters (e.g., r_0 for sigmoid)
        
    Returns:
        Array of parameter values at each r coordinate
        
    Example:
        >>> r = np.linspace(0, 1, 100)
        >>> sigma = make_profile(r, ProfileType.LINEAR, 1e6, 6e7, 1.5)
    """
    if profile_type == ProfileType.LINEAR:
        return _linear_profile(r, P_min, P_max, shape_param)
    elif profile_type == ProfileType.EXPONENTIAL:
        return _exponential_profile(r, P_min, P_max, shape_param)
    elif profile_type == ProfileType.POWER:
        return _power_profile(r, P_min, P_max, shape_param)
    elif profile_type == ProfileType.SIGMOID:
        r_0 = kwargs.get('r_0', None)
        return _sigmoid_profile(r, P_min, P_max, shape_param, r_0)
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")


def generate_dual_profiles(r: np.ndarray,
                          sigma_min: float,
                          sigma_max: float,
                          mu_min: float,
                          mu_max: float,
                          sigma_type: ProfileType,
                          mu_type: ProfileType,
                          sigma_shape: float,
                          mu_shape: float,
                          **kwargs) -> tuple:
    """
    Generate both σ and μ profiles.
    
    Args:
        r: Radial coordinates
        sigma_min: Minimum electrical conductivity (S/m)
        sigma_max: Maximum electrical conductivity (S/m)
        mu_min: Minimum relative permeability
        mu_max: Maximum relative permeability
        sigma_type: Profile type for σ
        mu_type: Profile type for μ
        sigma_shape: Shape parameter for σ
        mu_shape: Shape parameter for μ
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (sigma_profile, mu_profile)
        
    Example:
        >>> r = np.linspace(0, 1, 100)
        >>> sigma, mu = generate_dual_profiles(
        ...     r, 1e6, 6e7, 1, 100,
        ...     ProfileType.LINEAR, ProfileType.EXPONENTIAL,
        ...     1.5, 2.0
        ... )
    """
    sigma_profile = make_profile(r, sigma_type, sigma_min, sigma_max, sigma_shape, **kwargs)
    mu_profile = make_profile(r, mu_type, mu_min, mu_max, mu_shape, **kwargs)
    
    return sigma_profile, mu_profile
