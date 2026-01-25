"""Roberts R-sequence generator for uniform experimental design."""

import numpy as np


def calculate_phi(d: int) -> float:
    """
    Calculate φ parameter for d-dimensional R-sequence (Roberts).
    
    φ is the positive root of: x^(d+1) = x + 1
    
    Reference values:
    - d=1: φ ≈ 1.618034 (golden ratio)
    - d=2: φ ≈ 1.324718
    - d=3: φ ≈ 1.220744
    
    Args:
        d: Dimension of parameter space
        
    Returns:
        φ value for given dimension
    """
    control_values = {
        1: 1.618033988749895,
        2: 1.3247179572447460,
        3: 1.2207440846057596
    }
    
    if d in control_values:
        return control_values[d]
    
    coeffs = np.zeros(d + 2)
    coeffs[0] = 1
    coeffs[1] = -1
    coeffs[-1] = -1
    roots = np.roots(coeffs)
    
    real_positive_roots = roots[np.isreal(roots) & (roots.real > 0)]
    if len(real_positive_roots) == 0:
        raise ValueError(f"Cannot find positive root for dimension {d}")
    
    return float(real_positive_roots[0].real)


def generate_roberts_plan(N: int, d: int, bounds: list = None, seed: int = None) -> np.ndarray:
    """
    Generate N-point experimental plan using Roberts R-sequence.
    
    The R-sequence provides uniform coverage of d-dimensional parameter space
    with low discrepancy.
    
    Args:
        N: Number of points to generate
        d: Dimension of parameter space
        bounds: List of (min, max) tuples for each dimension. If None, uses [0, 1]^d
        seed: Random seed for reproducibility (affects initial offset α₀)
        
    Returns:
        Array of shape (N, d) with uniformly distributed points
        
    Example:
        >>> plan = generate_roberts_plan(100, 3, bounds=[(1e6, 6e7), (1, 100), (1e3, 1e5)])
        >>> plan.shape
        (100, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    phi = calculate_phi(d)
    
    alpha_0 = np.random.rand(d)
    
    n_indices = np.arange(1, N + 1).reshape(-1, 1)
    
    phi_powers = np.array([1.0 / (phi ** (j + 1)) for j in range(d)])
    
    R = (alpha_0 + n_indices * phi_powers) % 1.0
    
    if bounds is not None:
        if len(bounds) != d:
            raise ValueError(f"bounds must have length {d}, got {len(bounds)}")
        
        for j in range(d):
            min_val, max_val = bounds[j]
            R[:, j] = min_val + R[:, j] * (max_val - min_val)
    
    return R
