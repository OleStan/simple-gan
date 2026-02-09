# Forward-Inverse Eddy-Current Workflow Architecture

## Document Analysis Summary

### 1. EXPERIMENT PLAN LOGIC (From Codebase)

**Roberts R-Sequence Implementation:**
- ✅ **CONFIRMED IN CODEBASE**: `roberts_sequence.py` implements space-filling experimental design
- **Purpose**: Uniform coverage of d-dimensional parameter space with low discrepancy
- **Mathematical Foundation**: φ is the positive root of x^(d+1) = x + 1
  - d=1: φ ≈ 1.618034 (golden ratio)
  - d=2: φ ≈ 1.324718
  - d=3: φ ≈ 1.220744
- **Generation Formula**: R_n = (α₀ + n · φ^(-j)) mod 1.0
- **Parameter Dimensionality**: Currently d=6 (σ_min, σ_max, μ_min, μ_max, σ_shape, μ_shape)
- **Extensible**: Can add frequency, resistance as additional dimensions

**Why Space-Filling Plan is Required:**
- Ensures uniform sampling across entire parameter space
- Avoids clustering and gaps in training data
- Provides better coverage than random sampling
- Critical for LUT (Look-Up Table) based identification

---

### 2. MATERIAL MODEL (From Codebase)

**Continuous Depth Functions:**
- σ(r): Electrical conductivity as function of normalized depth r ∈ [0, r_max]
- μ(r): Relative magnetic permeability as function of normalized depth r

**Discretization Approach:**
- ✅ **CONFIRMED**: Piecewise-constant conditional layers
- **Implementation**: `discretization.py` converts continuous profiles to K discrete layers
- **Typical Configuration**: 
  - K = 50 or 51 layers (current implementation)
  - r_max = 1.0 (normalized depth)
  - Discretization modes: 'centers', 'edges', 'average'

**Physical Interpretation:**
- Each layer represents a thin slice of material with constant σ and μ
- Approximates continuous variation in real materials
- Trade-off: More layers → better approximation, higher computational cost

---

### 3. PROFILE PARAMETERIZATION (From Codebase)

**✅ CONFIRMED: Four Profile Types Implemented**

#### 3.1 Linear Profile
```
P(r) = P_min + (P_max - P_min) * (r/r_max)^a
```
- Parameters: P_min, P_max, a (shape parameter, typically 0.5-2.0)
- Use case: Simple monotonic variation

#### 3.2 Exponential Profile
```
P(r) = P_min * exp(b * r/r_max)
```
- Parameters: P_min, P_max (for scaling), b (shape parameter, typically 0.5-3.0)
- Use case: Rapid initial change, then saturation

#### 3.3 Power Profile
```
P(r) = P_min + (P_max - P_min) * (1 - exp(-c * r/r_max))
```
- Parameters: P_min, P_max, c (shape parameter, typically 1.0-5.0)
- Use case: Gradual approach to maximum

#### 3.4 Sigmoid Profile
```
P(r) = P_min + (P_max - P_min) / (1 + exp(-d * (r - r_0)))
```
- Parameters: P_min, P_max, d (steepness, typically 5.0-20.0), r_0 (inflection point)
- Use case: S-shaped transition, smooth boundaries

**Parameter Meaning:**
- σ₁ ≡ σ_min: Conductivity at surface (r=0)
- σ₂ ≡ σ_max: Conductivity at depth (r=r_max)
- μ₁ ≡ μ_min: Permeability at surface
- μ₂ ≡ μ_max: Permeability at depth
- Shape parameters: Control curvature/steepness of transition

**Current Implementation Bounds:**
- σ: [1×10⁶, 6×10⁷] S/m (configurable)
- μ: [1.0, 100.0] (configurable)
- Shape parameters: [0.5, 2.0] (varies by profile type)

---

### 4. LUT / CANDIDATE GENERATION CONCEPT

**✅ CONFIRMED IN CODEBASE**: `dataset_builder.py` implements dense parameter generation

**Why Dense, Wide Parameter Set:**
- Roberts plan generates N uniformly distributed parameter combinations
- Each combination produces unique (σ, μ) profile pair
- Dense coverage ensures any real material has nearby candidates
- Wide bounds account for real scatter in material properties

**Current Implementation:**
- N = 1000-2000 samples typical
- Covers full parameter space defined by bounds
- Rotating profile types (LINEAR, EXPONENTIAL, POWER, SIGMOID)
- Each sample: unique combination of (σ_min, σ_max, μ_min, μ_max, shapes)

---

## ⚠️ NOT DEFINED IN CURRENT CODEBASE

The following components are **NOT IMPLEMENTED** and must be designed:

### 1. Forward EDC Calculation
- **MISSING**: Actual eddy-current response calculation
- **MISSING**: Physical model relating (σ, μ) → EDC
- **MISSING**: Probe geometry and excitation parameters
- **MISSING**: EDC representation format (complex impedance, amplitude/phase, etc.)

### 2. Inverse Problem Formulation
- **MISSING**: Mismatch functional Z(·)
- **MISSING**: Optimization strategy
- **MISSING**: Constraint handling

### 3. Database/LUT Storage
- **MISSING**: Profile storage format
- **MISSING**: Nearest-neighbor search implementation
- **MISSING**: Distance metric in profile space

---

## PROPOSED ARCHITECTURE

### Package Structure

```
eddy_current_workflow/
├── profiles/
│   ├── __init__.py
│   ├── generation.py          # Profile generation (existing)
│   ├── discretization.py      # Continuous → discrete (existing)
│   ├── normalization.py       # NEW: Normalize/denormalize
│   └── validation.py          # NEW: Physical constraints
│
├── forward/
│   ├── __init__.py
│   ├── edc_solver.py          # NEW: Forward EDC calculation
│   ├── probe_config.py        # NEW: Probe geometry/settings
│   └── response_format.py     # NEW: EDC representation
│
├── models/
│   ├── __init__.py
│   ├── gan_generators.py      # Existing GAN models
│   └── model_utils.py         # Load/save utilities
│
├── inverse/
│   ├── __init__.py
│   ├── objective.py           # NEW: Mismatch functionals
│   ├── optimizers.py          # NEW: Optimization algorithms
│   ├── constraints.py         # NEW: Physical constraints
│   └── recovery.py            # NEW: Main inverse solver
│
├── database/
│   ├── __init__.py
│   ├── storage.py             # NEW: Profile database
│   ├── search.py              # NEW: Nearest-neighbor search
│   └── metrics.py             # NEW: Distance metrics
│
├── pipelines/
│   ├── __init__.py
│   ├── forward_pipeline.py    # NEW: Profile → EDC
│   ├── inverse_pipeline.py    # NEW: EDC → Profile
│   └── validation_pipeline.py # NEW: Round-trip testing
│
└── config/
    ├── __init__.py
    ├── global_config.py       # NEW: Single source of truth
    └── experiment_plans.py    # Roberts plan configurations
```

---

## IMPLEMENTATION SPECIFICATIONS

### 1. FORWARD EDC CALCULATION MODULE

```python
# forward/edc_solver.py

from dataclasses import dataclass
import numpy as np
from typing import Union, Tuple

@dataclass
class ProbeSettings:
    """Eddy-current probe configuration."""
    frequency: float  # Hz
    coil_radius: float  # m
    lift_off: float  # m
    excitation_current: float  # A
    # Add other probe-specific parameters

@dataclass
class EDCResponse:
    """Eddy-current response representation."""
    frequency: float
    impedance_real: float
    impedance_imag: float
    
    @property
    def amplitude(self) -> float:
        return np.sqrt(self.impedance_real**2 + self.impedance_imag**2)
    
    @property
    def phase(self) -> float:
        return np.arctan2(self.impedance_imag, self.impedance_real)
    
    def to_vector(self) -> np.ndarray:
        """Convert to vector for optimization."""
        return np.array([self.impedance_real, self.impedance_imag])


def edc_forward(
    sigma_layers: np.ndarray,  # (K,) S/m
    mu_layers: np.ndarray,     # (K,) relative
    settings: ProbeSettings,
    layer_thickness: float = None  # m, if None: auto-calculate
) -> EDCResponse:
    """
    Calculate eddy-current response from material profile.
    
    Args:
        sigma_layers: Electrical conductivity for each layer
        mu_layers: Relative permeability for each layer
        settings: Probe configuration
        layer_thickness: Thickness of each layer (m)
        
    Returns:
        EDCResponse object with impedance
        
    Notes:
        THIS IS A PLACEHOLDER - MUST BE REPLACED WITH ACTUAL PHYSICS MODEL
        Options:
        1. Analytical solution (if available for layered media)
        2. Finite element solver
        3. Integral equation method
        4. Pre-computed lookup with interpolation
    """
    # PLACEHOLDER IMPLEMENTATION
    # Replace with actual eddy-current physics
    
    K = len(sigma_layers)
    if layer_thickness is None:
        # Assume 1mm total depth, uniform layers
        layer_thickness = 1e-3 / K
    
    # Simplified placeholder: weighted sum (NOT PHYSICAL)
    # Real implementation needs Maxwell equations solution
    weights = np.exp(-np.arange(K) * 0.1)  # Skin depth approximation
    
    sigma_eff = np.sum(sigma_layers * weights) / np.sum(weights)
    mu_eff = np.sum(mu_layers * weights) / np.sum(weights)
    
    # Placeholder impedance calculation
    omega = 2 * np.pi * settings.frequency
    skin_depth = np.sqrt(2 / (omega * sigma_eff * mu_eff * 4e-7 * np.pi))
    
    Z_real = sigma_eff * skin_depth * 1e-6  # Arbitrary scaling
    Z_imag = omega * mu_eff * skin_depth * 1e-6
    
    return EDCResponse(
        frequency=settings.frequency,
        impedance_real=Z_real,
        impedance_imag=Z_imag
    )


# CRITICAL: Replace placeholder with one of:
# - Dodd-Deeds analytical solution for layered media
# - FEM solver (COMSOL, ANSYS, custom)
# - Boundary element method
# - Neural network surrogate (if trained on physics)
```

---

### 2. NORMALIZATION / DENORMALIZATION

```python
# profiles/normalization.py

import numpy as np
from typing import Tuple, Dict
import json

class ProfileNormalizer:
    """Handles normalization/denormalization of profiles for GAN training."""
    
    def __init__(self, 
                 sigma_min: float, sigma_max: float,
                 mu_min: float, mu_max: float):
        """
        Initialize normalizer with physical bounds.
        
        Args:
            sigma_min: Minimum conductivity in dataset (S/m)
            sigma_max: Maximum conductivity in dataset (S/m)
            mu_min: Minimum permeability in dataset
            mu_max: Maximum permeability in dataset
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu_min = mu_min
        self.mu_max = mu_max
    
    def normalize(self, 
                  sigma: np.ndarray, 
                  mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize profiles to [-1, 1] range for GAN training.
        
        Args:
            sigma: Conductivity profile (K,) in S/m
            mu: Permeability profile (K,)
            
        Returns:
            Tuple of (sigma_norm, mu_norm) in [-1, 1]
        """
        sigma_norm = 2 * (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min) - 1
        mu_norm = 2 * (mu - self.mu_min) / (self.mu_max - self.mu_min) - 1
        return sigma_norm, mu_norm
    
    def denormalize(self,
                    sigma_norm: np.ndarray,
                    mu_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Denormalize profiles from [-1, 1] to physical units.
        
        Args:
            sigma_norm: Normalized conductivity in [-1, 1]
            mu_norm: Normalized permeability in [-1, 1]
            
        Returns:
            Tuple of (sigma, mu) in physical units
        """
        sigma = (sigma_norm + 1) / 2 * (self.sigma_max - self.sigma_min) + self.sigma_min
        mu = (mu_norm + 1) / 2 * (self.mu_max - self.mu_min) + self.mu_min
        return sigma, mu
    
    def save(self, filepath: str):
        """Save normalization parameters to JSON."""
        params = {
            'sigma_min': float(self.sigma_min),
            'sigma_max': float(self.sigma_max),
            'mu_min': float(self.mu_min),
            'mu_max': float(self.mu_max)
        }
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ProfileNormalizer':
        """Load normalization parameters from JSON."""
        with open(filepath, 'r') as f:
            params = json.load(f)
        return cls(**params)
    
    def validate_roundtrip(self, sigma: np.ndarray, mu: np.ndarray, 
                          tol: float = 1e-10) -> bool:
        """
        Verify normalization is lossless within tolerance.
        
        Args:
            sigma: Original conductivity profile
            mu: Original permeability profile
            tol: Maximum allowed error
            
        Returns:
            True if round-trip error < tol
        """
        sigma_norm, mu_norm = self.normalize(sigma, mu)
        sigma_recovered, mu_recovered = self.denormalize(sigma_norm, mu_norm)
        
        sigma_error = np.max(np.abs(sigma - sigma_recovered))
        mu_error = np.max(np.abs(mu - mu_recovered))
        
        return sigma_error < tol and mu_error < tol
```

---

### 3. INVERSE PROBLEM SOLVER

```python
# inverse/objective.py

import numpy as np
from typing import Callable, Optional
from ..forward.edc_solver import edc_forward, EDCResponse, ProbeSettings

class EDCMismatchObjective:
    """Objective function for inverse problem: min Z(θ)."""
    
    def __init__(self,
                 edc_measured: EDCResponse,
                 probe_settings: ProbeSettings,
                 K: int,
                 normalizer: Optional['ProfileNormalizer'] = None):
        """
        Initialize mismatch objective.
        
        Args:
            edc_measured: Measured EDC from real sample
            probe_settings: Probe configuration used in measurement
            K: Number of layers in discretization
            normalizer: Optional normalizer for GAN-based optimization
        """
        self.edc_measured = edc_measured
        self.probe_settings = probe_settings
        self.K = K
        self.normalizer = normalizer
        
        # Store measured response as vector
        self.y_measured = edc_measured.to_vector()
    
    def __call__(self, theta: np.ndarray) -> float:
        """
        Evaluate mismatch for given parameters.
        
        Args:
            theta: Parameter vector
                  - Direct: [sigma_1, ..., sigma_K, mu_1, ..., mu_K]
                  - GAN: latent vector z
                  
        Returns:
            Scalar mismatch value (lower is better)
        """
        # Extract sigma and mu from theta
        sigma_layers, mu_layers = self._theta_to_profiles(theta)
        
        # Calculate EDC response
        edc_generated = edc_forward(sigma_layers, mu_layers, self.probe_settings)
        y_generated = edc_generated.to_vector()
        
        # Compute mismatch (L2 norm)
        mismatch = np.linalg.norm(y_generated - self.y_measured)
        
        return mismatch
    
    def _theta_to_profiles(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert parameter vector to sigma/mu profiles."""
        if len(theta) == 2 * self.K:
            # Direct parametrization
            sigma_layers = theta[:self.K]
            mu_layers = theta[self.K:]
        else:
            # Assume GAN latent vector - needs generator
            raise NotImplementedError("GAN-based parametrization requires generator")
        
        return sigma_layers, mu_layers
    
    def with_regularization(self, 
                           lambda_smooth: float = 0.0,
                           lambda_monotonic: float = 0.0) -> Callable:
        """
        Add regularization terms to objective.
        
        Args:
            lambda_smooth: Weight for smoothness penalty
            lambda_monotonic: Weight for monotonicity penalty
            
        Returns:
            Regularized objective function
        """
        def regularized_objective(theta: np.ndarray) -> float:
            # Base mismatch
            mismatch = self(theta)
            
            sigma_layers, mu_layers = self._theta_to_profiles(theta)
            
            # Smoothness penalty (penalize large gradients)
            if lambda_smooth > 0:
                sigma_grad = np.diff(sigma_layers)
                mu_grad = np.diff(mu_layers)
                smoothness = np.sum(sigma_grad**2) + np.sum(mu_grad**2)
                mismatch += lambda_smooth * smoothness
            
            # Monotonicity penalty (if profiles should be monotonic)
            if lambda_monotonic > 0:
                # Penalize non-monotonic behavior
                sigma_violations = np.sum(np.maximum(0, -np.diff(sigma_layers)))
                mu_violations = np.sum(np.maximum(0, np.diff(mu_layers)))  # Opposite for mu
                mismatch += lambda_monotonic * (sigma_violations + mu_violations)
            
            return mismatch
        
        return regularized_objective
```

```python
# inverse/optimizers.py

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Callable, Tuple, Dict, Optional

class InverseSolver:
    """Solve inverse problem: recover (σ, μ) from EDC_measured."""
    
    def __init__(self,
                 objective: Callable,
                 K: int,
                 sigma_bounds: Tuple[float, float],
                 mu_bounds: Tuple[float, float]):
        """
        Initialize inverse solver.
        
        Args:
            objective: Mismatch function to minimize
            K: Number of layers
            sigma_bounds: (min, max) for conductivity
            mu_bounds: (min, max) for permeability
        """
        self.objective = objective
        self.K = K
        self.sigma_bounds = sigma_bounds
        self.mu_bounds = mu_bounds
    
    def solve_multistart(self,
                        n_starts: int = 10,
                        method: str = 'L-BFGS-B',
                        seed: int = 42) -> Dict:
        """
        Solve using multiple random initializations.
        
        Args:
            n_starts: Number of random starts
            method: Scipy optimization method
            seed: Random seed
            
        Returns:
            Dictionary with best solution and statistics
        """
        np.random.seed(seed)
        
        # Bounds for all parameters
        bounds = [(self.sigma_bounds[0], self.sigma_bounds[1])] * self.K + \
                 [(self.mu_bounds[0], self.mu_bounds[1])] * self.K
        
        best_result = None
        best_mismatch = np.inf
        all_results = []
        
        for i in range(n_starts):
            # Random initialization
            theta_0 = np.concatenate([
                np.random.uniform(self.sigma_bounds[0], self.sigma_bounds[1], self.K),
                np.random.uniform(self.mu_bounds[0], self.mu_bounds[1], self.K)
            ])
            
            # Optimize
            result = minimize(
                self.objective,
                theta_0,
                method=method,
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            all_results.append(result)
            
            if result.fun < best_mismatch:
                best_mismatch = result.fun
                best_result = result
        
        # Extract best profiles
        theta_best = best_result.x
        sigma_best = theta_best[:self.K]
        mu_best = theta_best[self.K:]
        
        return {
            'sigma': sigma_best,
            'mu': mu_best,
            'mismatch': best_mismatch,
            'success': best_result.success,
            'n_iterations': best_result.nit,
            'all_results': all_results,
            'convergence_rate': sum(r.success for r in all_results) / n_starts
        }
    
    def solve_global(self, 
                    maxiter: int = 1000,
                    seed: int = 42) -> Dict:
        """
        Solve using global optimization (differential evolution).
        
        Args:
            maxiter: Maximum iterations
            seed: Random seed
            
        Returns:
            Dictionary with solution
        """
        bounds = [(self.sigma_bounds[0], self.sigma_bounds[1])] * self.K + \
                 [(self.mu_bounds[0], self.mu_bounds[1])] * self.K
        
        result = differential_evolution(
            self.objective,
            bounds,
            maxiter=maxiter,
            seed=seed,
            polish=True
        )
        
        theta_best = result.x
        sigma_best = theta_best[:self.K]
        mu_best = theta_best[self.K:]
        
        return {
            'sigma': sigma_best,
            'mu': mu_best,
            'mismatch': result.fun,
            'success': result.success,
            'n_iterations': result.nit
        }
```

---

### 4. DATABASE MATCHING

```python
# database/storage.py

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class ProfileRecord:
    """Single profile record in database."""
    id: int
    sigma: np.ndarray  # (K,)
    mu: np.ndarray     # (K,)
    edc_response: Optional[np.ndarray] = None  # Pre-computed EDC
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'sigma': self.sigma,
            'mu': self.mu,
            'edc_response': self.edc_response,
            'metadata': self.metadata or {}
        }


class ProfileDatabase:
    """Storage and retrieval of material profiles."""
    
    def __init__(self, db_path: str):
        """
        Initialize database.
        
        Args:
            db_path: Path to HDF5 database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def create(self, K: int):
        """Create new database."""
        with h5py.File(self.db_path, 'w') as f:
            f.attrs['K'] = K
            f.attrs['n_records'] = 0
            f.create_group('profiles')
    
    def add_record(self, record: ProfileRecord):
        """Add profile record to database."""
        with h5py.File(self.db_path, 'a') as f:
            grp = f['profiles'].create_group(f'record_{record.id}')
            grp.create_dataset('sigma', data=record.sigma)
            grp.create_dataset('mu', data=record.mu)
            
            if record.edc_response is not None:
                grp.create_dataset('edc_response', data=record.edc_response)
            
            if record.metadata:
                for key, value in record.metadata.items():
                    grp.attrs[key] = value
            
            f.attrs['n_records'] = f.attrs['n_records'] + 1
    
    def add_batch(self, sigma_batch: np.ndarray, mu_batch: np.ndarray,
                  metadata_batch: Optional[List[Dict]] = None):
        """
        Add multiple profiles at once.
        
        Args:
            sigma_batch: (N, K) array of conductivity profiles
            mu_batch: (N, K) array of permeability profiles
            metadata_batch: Optional list of metadata dicts
        """
        N = len(sigma_batch)
        
        with h5py.File(self.db_path, 'a') as f:
            start_id = f.attrs['n_records']
            
            for i in range(N):
                record = ProfileRecord(
                    id=start_id + i,
                    sigma=sigma_batch[i],
                    mu=mu_batch[i],
                    metadata=metadata_batch[i] if metadata_batch else None
                )
                self.add_record(record)
    
    def get_all_profiles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve all profiles from database.
        
        Returns:
            Tuple of (sigma_all, mu_all) arrays of shape (N, K)
        """
        with h5py.File(self.db_path, 'r') as f:
            n_records = f.attrs['n_records']
            K = f.attrs['K']
            
            sigma_all = np.zeros((n_records, K))
            mu_all = np.zeros((n_records, K))
            
            for i in range(n_records):
                grp = f['profiles'][f'record_{i}']
                sigma_all[i] = grp['sigma'][:]
                mu_all[i] = grp['mu'][:]
        
        return sigma_all, mu_all
    
    def get_record(self, record_id: int) -> ProfileRecord:
        """Retrieve specific record by ID."""
        with h5py.File(self.db_path, 'r') as f:
            grp = f['profiles'][f'record_{record_id}']
            
            sigma = grp['sigma'][:]
            mu = grp['mu'][:]
            edc = grp['edc_response'][:] if 'edc_response' in grp else None
            metadata = dict(grp.attrs)
            
            return ProfileRecord(
                id=record_id,
                sigma=sigma,
                mu=mu,
                edc_response=edc,
                metadata=metadata
            )
```

```python
# database/search.py

import numpy as np
from typing import List, Tuple, Callable
from .storage import ProfileDatabase, ProfileRecord

class ProfileSearchEngine:
    """Nearest-neighbor search in profile space."""
    
    def __init__(self, database: ProfileDatabase):
        """
        Initialize search engine.
        
        Args:
            database: Profile database to search
        """
        self.database = database
        self.sigma_all, self.mu_all = database.get_all_profiles()
    
    def search_nearest(self,
                      sigma_query: np.ndarray,
                      mu_query: np.ndarray,
                      k: int = 5,
                      metric: str = 'euclidean',
                      weights: Tuple[float, float] = (1.0, 1.0)) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors to query profile.
        
        Args:
            sigma_query: Query conductivity profile (K,)
            mu_query: Query permeability profile (K,)
            k: Number of neighbors to return
            metric: Distance metric ('euclidean', 'cosine', 'correlation')
            weights: (w_sigma, w_mu) weights for combining distances
            
        Returns:
            List of (record_id, distance) tuples, sorted by distance
        """
        N = len(self.sigma_all)
        distances = np.zeros(N)
        
        for i in range(N):
            if metric == 'euclidean':
                d_sigma = np.linalg.norm(self.sigma_all[i] - sigma_query)
                d_mu = np.linalg.norm(self.mu_all[i] - mu_query)
            elif metric == 'cosine':
                d_sigma = 1 - np.dot(self.sigma_all[i], sigma_query) / \
                          (np.linalg.norm(self.sigma_all[i]) * np.linalg.norm(sigma_query))
                d_mu = 1 - np.dot(self.mu_all[i], mu_query) / \
                       (np.linalg.norm(self.mu_all[i]) * np.linalg.norm(mu_query))
            elif metric == 'correlation':
                d_sigma = 1 - np.corrcoef(self.sigma_all[i], sigma_query)[0, 1]
                d_mu = 1 - np.corrcoef(self.mu_all[i], mu_query)[0, 1]
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Weighted combination
            distances[i] = weights[0] * d_sigma + weights[1] * d_mu
        
        # Find k nearest
        nearest_indices = np.argsort(distances)[:k]
        
        results = [(int(idx), float(distances[idx])) for idx in nearest_indices]
        
        return results
    
    def search_threshold(self,
                        sigma_query: np.ndarray,
                        mu_query: np.ndarray,
                        threshold: float,
                        metric: str = 'euclidean') -> List[Tuple[int, float]]:
        """
        Find all profiles within distance threshold.
        
        Args:
            sigma_query: Query conductivity profile
            mu_query: Query permeability profile
            threshold: Maximum distance
            metric: Distance metric
            
        Returns:
            List of (record_id, distance) tuples within threshold
        """
        all_neighbors = self.search_nearest(
            sigma_query, mu_query, 
            k=len(self.sigma_all), 
            metric=metric
        )
        
        return [(idx, dist) for idx, dist in all_neighbors if dist <= threshold]
```

---

### 5. GLOBAL CONFIGURATION

```python
# config/global_config.py

"""Single source of truth for all pipeline parameters."""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class GlobalConfig:
    """Global configuration for eddy-current workflow."""
    
    # Discretization
    K: int = 51  # Number of layers
    r_min: float = 0.0  # Normalized depth start
    r_max: float = 1.0  # Normalized depth end
    physical_depth: float = 1e-3  # Physical depth in meters (1mm)
    
    # Material bounds
    sigma_bounds: Tuple[float, float] = (1e6, 6e7)  # S/m
    mu_bounds: Tuple[float, float] = (1.0, 100.0)  # Relative
    
    # Normalization (for GAN)
    normalize_to_range: Tuple[float, float] = (-1.0, 1.0)
    
    # Probe settings (default)
    default_frequency: float = 1e6  # 1 MHz
    default_coil_radius: float = 5e-3  # 5mm
    default_lift_off: float = 0.5e-3  # 0.5mm
    
    # Optimization
    inverse_max_iter: int = 1000
    inverse_n_starts: int = 10
    inverse_tol: float = 1e-6
    
    # Database
    db_path: str = './profile_database.h5'
    
    @property
    def layer_thickness(self) -> float:
        """Calculate thickness of each layer."""
        return self.physical_depth / self.K
    
    def validate(self):
        """Validate configuration consistency."""
        assert self.K > 0, "K must be positive"
        assert self.r_max > self.r_min, "r_max must be > r_min"
        assert self.sigma_bounds[1] > self.sigma_bounds[0], "Invalid sigma bounds"
        assert self.mu_bounds[1] > self.mu_bounds[0], "Invalid mu bounds"
        assert self.physical_depth > 0, "Physical depth must be positive"


# Global instance
CONFIG = GlobalConfig()
```

---

### 6. END-TO-END PIPELINES

```python
# pipelines/forward_pipeline.py

"""Forward pipeline: Profile → EDC."""

import numpy as np
from typing import Union, Dict
from ..profiles.generation import make_profile, ProfileType
from ..profiles.discretization import discretize_dual_profiles
from ..forward.edc_solver import edc_forward, ProbeSettings
from ..config.global_config import CONFIG

def profile_to_edc(
    sigma_profile: np.ndarray,
    mu_profile: np.ndarray,
    probe_settings: ProbeSettings,
    already_discretized: bool = False
) -> Dict:
    """
    Complete forward pipeline: continuous/discrete profile → EDC.
    
    Args:
        sigma_profile: Conductivity profile
        mu_profile: Permeability profile
        probe_settings: Probe configuration
        already_discretized: If True, profiles are already K layers
        
    Returns:
        Dictionary with EDC response and metadata
    """
    if not already_discretized:
        # Discretize continuous profiles
        r = np.linspace(CONFIG.r_min, CONFIG.r_max, 1000)
        sigma_layers, mu_layers = discretize_dual_profiles(
            r, sigma_profile, mu_profile, CONFIG.K, 'centers'
        )
    else:
        sigma_layers = sigma_profile
        mu_layers = mu_profile
    
    # Calculate EDC
    edc_response = edc_forward(sigma_layers, mu_layers, probe_settings)
    
    return {
        'edc_response': edc_response,
        'sigma_layers': sigma_layers,
        'mu_layers': mu_layers,
        'K': CONFIG.K,
        'probe_settings': probe_settings
    }
```

```python
# pipelines/inverse_pipeline.py

"""Inverse pipeline: EDC → Profile."""

import numpy as np
from typing import Dict, Optional
from ..forward.edc_solver import EDCResponse, ProbeSettings
from ..inverse.objective import EDCMismatchObjective
from ..inverse.optimizers import InverseSolver
from ..database.search import ProfileSearchEngine
from ..database.storage import ProfileDatabase
from ..config.global_config import CONFIG

def edc_to_profile(
    edc_measured: EDCResponse,
    probe_settings: ProbeSettings,
    method: str = 'multistart',
    use_database: bool = True,
    db_path: Optional[str] = None
) -> Dict:
    """
    Complete inverse pipeline: EDC → recovered (σ, μ) → database match.
    
    Args:
        edc_measured: Measured EDC from real sample
        probe_settings: Probe configuration
        method: 'multistart' or 'global'
        use_database: If True, search database for nearest match
        db_path: Path to profile database
        
    Returns:
        Dictionary with recovered profiles and matches
    """
    # Step 1: Set up inverse problem
    objective = EDCMismatchObjective(
        edc_measured=edc_measured,
        probe_settings=probe_settings,
        K=CONFIG.K
    )
    
    # Add regularization
    regularized_obj = objective.with_regularization(
        lambda_smooth=1e-3,
        lambda_monotonic=1e-2
    )
    
    # Step 2: Solve inverse problem
    solver = InverseSolver(
        objective=regularized_obj,
        K=CONFIG.K,
        sigma_bounds=CONFIG.sigma_bounds,
        mu_bounds=CONFIG.mu_bounds
    )
    
    if method == 'multistart':
        result = solver.solve_multistart(n_starts=CONFIG.inverse_n_starts)
    elif method == 'global':
        result = solver.solve_global(maxiter=CONFIG.inverse_max_iter)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    sigma_recovered = result['sigma']
    mu_recovered = result['mu']
    
    # Step 3: Database search (optional)
    matches = None
    if use_database:
        db_path = db_path or CONFIG.db_path
        database = ProfileDatabase(db_path)
        search_engine = ProfileSearchEngine(database)
        
        matches = search_engine.search_nearest(
            sigma_recovered, mu_recovered,
            k=5,
            metric='euclidean'
        )
    
    return {
        'sigma_recovered': sigma_recovered,
        'mu_recovered': mu_recovered,
        'mismatch': result['mismatch'],
        'success': result['success'],
        'database_matches': matches,
        'optimization_result': result
    }
```

```python
# pipelines/validation_pipeline.py

"""Validation: round-trip testing."""

import numpy as np
from typing import Dict
from .forward_pipeline import profile_to_edc
from .inverse_pipeline import edc_to_profile
from ..forward.edc_solver import ProbeSettings
from ..config.global_config import CONFIG

def validate_roundtrip(
    sigma_true: np.ndarray,
    mu_true: np.ndarray,
    probe_settings: ProbeSettings,
    noise_level: float = 0.0
) -> Dict:
    """
    Validate forward-inverse pipeline with known ground truth.
    
    Args:
        sigma_true: True conductivity profile (K,)
        mu_true: True permeability profile (K,)
        probe_settings: Probe configuration
        noise_level: Relative noise to add to EDC (0.0 = no noise)
        
    Returns:
        Dictionary with validation metrics
    """
    # Forward: true profile → EDC
    forward_result = profile_to_edc(
        sigma_true, mu_true, probe_settings, already_discretized=True
    )
    edc_clean = forward_result['edc_response']
    
    # Add noise
    if noise_level > 0:
        noise_real = np.random.normal(0, noise_level * abs(edc_clean.impedance_real))
        noise_imag = np.random.normal(0, noise_level * abs(edc_clean.impedance_imag))
        
        from ..forward.edc_solver import EDCResponse
        edc_noisy = EDCResponse(
            frequency=edc_clean.frequency,
            impedance_real=edc_clean.impedance_real + noise_real,
            impedance_imag=edc_clean.impedance_imag + noise_imag
        )
    else:
        edc_noisy = edc_clean
    
    # Inverse: EDC → recovered profile
    inverse_result = edc_to_profile(
        edc_noisy, probe_settings, method='multistart', use_database=False
    )
    
    sigma_recovered = inverse_result['sigma_recovered']
    mu_recovered = inverse_result['mu_recovered']
    
    # Compute errors
    sigma_error = np.linalg.norm(sigma_recovered - sigma_true) / np.linalg.norm(sigma_true)
    mu_error = np.linalg.norm(mu_recovered - mu_true) / np.linalg.norm(mu_true)
    
    sigma_max_error = np.max(np.abs(sigma_recovered - sigma_true))
    mu_max_error = np.max(np.abs(mu_recovered - mu_true))
    
    return {
        'sigma_true': sigma_true,
        'mu_true': mu_true,
        'sigma_recovered': sigma_recovered,
        'mu_recovered': mu_recovered,
        'sigma_relative_error': sigma_error,
        'mu_relative_error': mu_error,
        'sigma_max_error': sigma_max_error,
        'mu_max_error': mu_max_error,
        'edc_mismatch': inverse_result['mismatch'],
        'noise_level': noise_level,
        'success': inverse_result['success']
    }
```

---

## CRITICAL ASSUMPTIONS & MISSING COMPONENTS

### ⚠️ MUST BE PROVIDED EXTERNALLY

1. **Forward EDC Solver Physics**
   - Current implementation is a PLACEHOLDER
   - Requires actual electromagnetic solver:
     * Analytical (Dodd-Deeds for layered media)
     * Numerical (FEM, BEM, FDTD)
     * Surrogate model (trained neural network)
   - Must handle:
     * Multi-layer geometry
     * Frequency-dependent skin effect
     * Probe geometry and lift-off
     * Magnetic and electric field coupling

2. **Probe Calibration Data**
   - Actual probe specifications
   - Calibration curves
   - Measurement noise characteristics

3. **Validation Data**
   - Real measurements from known samples
   - Ground truth profiles from destructive testing
   - Benchmark problems

---

## IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1-2)
- [ ] Implement `ProfileNormalizer` class
- [ ] Create `GlobalConfig` with validation
- [ ] Set up package structure
- [ ] Write unit tests for normalization

### Phase 2: Forward Model (Week 3-4)
- [ ] Define `ProbeSettings` and `EDCResponse` classes
- [ ] Implement placeholder `edc_forward` function
- [ ] **CRITICAL**: Replace with actual physics solver
- [ ] Validate against known analytical solutions

### Phase 3: Inverse Solver (Week 5-6)
- [ ] Implement `EDCMismatchObjective`
- [ ] Implement `InverseSolver` with multistart
- [ ] Add regularization options
- [ ] Test on synthetic data

### Phase 4: Database (Week 7-8)
- [ ] Implement `ProfileDatabase` with HDF5
- [ ] Implement `ProfileSearchEngine`
- [ ] Populate database with generated profiles
- [ ] Benchmark search performance

### Phase 5: Integration (Week 9-10)
- [ ] Implement end-to-end pipelines
- [ ] Create validation suite
- [ ] Document all APIs
- [ ] Write user guide

### Phase 6: Validation (Week 11-12)
- [ ] Round-trip testing on synthetic data
- [ ] Noise sensitivity analysis
- [ ] Compare with existing methods
- [ ] Real data validation (if available)

---

## SUCCESS CRITERIA

✅ **Architecture is successful if:**

1. **Modularity**: Each component can be developed/tested independently
2. **Reusability**: Forward solver works with any profile source (generator, GAN, manual)
3. **Extensibility**: Easy to swap optimization algorithms, distance metrics, etc.
4. **Correctness**: Round-trip validation shows < 1% error on noise-free synthetic data
5. **Documentation**: All assumptions explicitly stated, APIs clearly documented
6. **Single Source of Truth**: All parameters defined in `GlobalConfig`

---

## NEXT STEPS

1. **Immediate**: Implement normalization utilities and global config
2. **Critical**: Identify/implement actual EDC forward solver
3. **Validation**: Create synthetic test cases with known solutions
4. **Integration**: Connect with existing GAN generators
5. **Documentation**: Expand this document with physics equations

---

## REFERENCES & RESOURCES

### Existing Codebase
- `eddy_current_data_generator/`: Profile generation and discretization
- `wgan_dual_profiles.py`: Dual-head GAN generator
- `wgan_improved_v2.py`: Improved GAN with physics losses

### Required External Resources
- Eddy-current physics textbook/papers
- Dodd-Deeds analytical solution (if applicable)
- FEM solver documentation (COMSOL, ANSYS, or custom)
- Benchmark datasets for validation

### Future Enhancements
- Physics-informed neural network (PINN) for forward solver
- Bayesian optimization for inverse problem
- Uncertainty quantification
- Multi-frequency inversion
- Real-time processing pipeline

---

**END OF ARCHITECTURE DOCUMENT**
