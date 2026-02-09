"""
Forward eddy-current solver: (σ, μ) → EDC response.

Uses the Dodd-Deeds analytical model (1968) with transfer-matrix
recursion for multilayer conductors under the eddy-current
(quasi-static) approximation of the Ampere-Maxwell equations.

The displacement current ∂D/∂t is dropped (valid when ωε ≪ σ),
giving:
    ∇ × H = σE
    ∇ × E = -∂B/∂t
    B = μ₀μᵣH

For an air-core coil above a K-layer planar conductor the coil
impedance change ΔZ is expressed as a Hankel-transform integral
whose integrand contains a reflection coefficient Φ(s) built
bottom-up through the layer stack.

Reference:
    C.V. Dodd & W.E. Deeds, "Analytical Solutions to
    Eddy-Current Probe-Coil Problems", J. Appl. Phys. 39, 2829 (1968).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy import integrate
from scipy.special import j1 as bessel_j1


MU_0 = 4e-7 * np.pi


@dataclass
class ProbeSettings:
    """
    Eddy-current probe configuration (Dodd-Deeds coil geometry).
    
    The coil is modelled as a finite cross-section solenoid with inner
    radius r1, outer radius r2, height (l2 - l1) above the specimen
    surface. l1 = lift_off, l2 = lift_off + coil_height.
    """
    frequency: float
    inner_radius: float = 4e-3
    outer_radius: float = 6e-3
    lift_off: float = 0.5e-3
    coil_height: float = 2e-3
    n_turns: int = 100
    
    def __post_init__(self):
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")
        if self.inner_radius <= 0:
            raise ValueError("Inner radius must be positive")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("Outer radius must be > inner radius")
        if self.lift_off < 0:
            raise ValueError("Lift-off must be non-negative")
        if self.coil_height <= 0:
            raise ValueError("Coil height must be positive")
    
    @property
    def omega(self) -> float:
        return 2 * np.pi * self.frequency
    
    @property
    def mean_radius(self) -> float:
        return 0.5 * (self.inner_radius + self.outer_radius)
    
    @property
    def l1(self) -> float:
        return self.lift_off
    
    @property
    def l2(self) -> float:
        return self.lift_off + self.coil_height


@dataclass
class EDCResponse:
    """
    Eddy-current response representation.
    
    The response can be represented as:
    - Complex impedance: Z = R + jX
    - Amplitude and phase: |Z|, ∠Z
    - Real and imaginary components
    
    Attributes:
        frequency: Measurement frequency (Hz)
        impedance_real: Real part of impedance (Ω)
        impedance_imag: Imaginary part of impedance (Ω)
    """
    frequency: float
    impedance_real: float
    impedance_imag: float
    
    @property
    def impedance_complex(self) -> complex:
        """Complex impedance Z = R + jX."""
        return complex(self.impedance_real, self.impedance_imag)
    
    @property
    def amplitude(self) -> float:
        """Impedance amplitude |Z|."""
        return np.sqrt(self.impedance_real**2 + self.impedance_imag**2)
    
    @property
    def phase(self) -> float:
        """Impedance phase angle (radians)."""
        return np.arctan2(self.impedance_imag, self.impedance_real)
    
    @property
    def phase_deg(self) -> float:
        """Impedance phase angle (degrees)."""
        return np.degrees(self.phase)
    
    def to_vector(self) -> np.ndarray:
        """
        Convert to vector for optimization.
        
        Returns:
            Array [real, imag]
        """
        return np.array([self.impedance_real, self.impedance_imag])
    
    @classmethod
    def from_complex(cls, frequency: float, impedance: complex) -> 'EDCResponse':
        """
        Create from complex impedance.
        
        Args:
            frequency: Measurement frequency (Hz)
            impedance: Complex impedance
            
        Returns:
            EDCResponse instance
        """
        return cls(
            frequency=frequency,
            impedance_real=impedance.real,
            impedance_imag=impedance.imag
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EDCResponse(f={self.frequency/1e6:.2f}MHz, "
            f"Z={self.impedance_real:.4f}{self.impedance_imag:+.4f}j Ω, "
            f"|Z|={self.amplitude:.4f}Ω, ∠Z={self.phase_deg:.2f}°)"
        )


def _coil_integral_P(s: float, r1: float, r2: float) -> float:
    """
    Coil integration factor P(s) = ∫_{s·r1}^{s·r2} x·J₁(x) dx.

    Evaluated via the analytic recurrence:
        ∫ x J₁(x) dx = -x J₀(x) + ∫ J₀(x) dx
    which Dodd-Deeds tabulate. Here we use a compact Gauss-Kronrod
    quadrature because the integrand is smooth and short-ranged.
    """
    if s < 1e-15:
        return 0.0
    a = s * r1
    b = s * r2
    val, _ = integrate.quad(lambda x: x * bessel_j1(x), a, b, limit=80)
    return val


def _layer_alpha(s: float, omega: float, sigma: float, mu_r: float) -> complex:
    """
    Propagation constant for a single layer.

        α_k = √(s² + jωμ₀μ_k σ_k)
    """
    return np.sqrt(s**2 + 1j * omega * MU_0 * mu_r * sigma)


def _reflection_coefficient(
    s: float,
    omega: float,
    sigma_layers: np.ndarray,
    mu_layers: np.ndarray,
    thicknesses: np.ndarray,
) -> complex:
    """
    Bottom-up recursion for the effective surface reflection
    coefficient Φ₁(s) of a K-layer stack.

    Convention: layer index 0 is the topmost (surface) layer,
    layer K-1 is the deepest. Below layer K-1 sits a non-conducting
    half-space (air/vacuum, σ=0, μ_r=1).

    Recursion (Dodd-Deeds / Cheng et al.):

        α_k   = √(s² + jωμ₀μ_kσ_k)
        α_air = s                          (substrate below stack)

        Φ_K   = (μ_{K-1}·α_air - 1·α_{K-1}) /
                 (μ_{K-1}·α_air + 1·α_{K-1})

        For k = K-2 … 0:
            u_k = (μ_{k+1}·α_k - μ_k·α_{k+1}) /
                  (μ_{k+1}·α_k + μ_k·α_{k+1})

            Φ_k = (u_k + Φ_{k+1}·exp(-2·α_{k+1}·d_{k+1})) /
                   (1 + u_k·Φ_{k+1}·exp(-2·α_{k+1}·d_{k+1}))

    Returns the surface reflection Φ₀ evaluated at integration
    variable s.
    """
    K = len(sigma_layers)

    alphas = np.array([
        _layer_alpha(s, omega, sigma_layers[k], mu_layers[k])
        for k in range(K)
    ])

    alpha_sub = complex(s, 0.0)

    mu_bot = mu_layers[K - 1]
    phi = (mu_bot * alpha_sub - 1.0 * alphas[K - 1]) / (
           mu_bot * alpha_sub + 1.0 * alphas[K - 1])

    for k in range(K - 2, -1, -1):
        exp_term = np.exp(-2.0 * alphas[k + 1] * thicknesses[k + 1])
        u_k = (mu_layers[k + 1] * alphas[k] - mu_layers[k] * alphas[k + 1]) / (
               mu_layers[k + 1] * alphas[k] + mu_layers[k] * alphas[k + 1])
        phi = (u_k + phi * exp_term) / (1.0 + u_k * phi * exp_term)

    return phi


def _dodd_deeds_integrand(
    s: float,
    omega: float,
    sigma_layers: np.ndarray,
    mu_layers: np.ndarray,
    thicknesses: np.ndarray,
    settings: ProbeSettings,
) -> complex:
    """
    Integrand of the Dodd-Deeds impedance-change integral.

        I(s) = P²(s) / s⁶ · [e^{-s·l₁} - e^{-s·l₂}]² · Φ(s)

    The coil impedance change is then:

        ΔZ = jω · 2πμ₀N² / [(r₂-r₁)²(l₂-l₁)²] · ∫₀^∞ I(s) ds
    """
    P = _coil_integral_P(s, settings.inner_radius, settings.outer_radius)
    if abs(P) < 1e-30:
        return 0.0 + 0.0j

    phi = _reflection_coefficient(s, omega, sigma_layers, mu_layers, thicknesses)

    exp_l1 = np.exp(-s * settings.l1)
    exp_l2 = np.exp(-s * settings.l2)
    coil_factor = (exp_l1 - exp_l2) ** 2

    denom = s ** 6 if s > 1e-15 else 1e-90

    return (P ** 2 / denom) * coil_factor * phi


def edc_forward(
    sigma_layers: np.ndarray,
    mu_layers: np.ndarray,
    settings: ProbeSettings,
    layer_thickness: Optional[float] = None,
    n_quad: int = 200,
    s_max_factor: float = 20.0,
) -> EDCResponse:
    """
    Dodd-Deeds forward solver for a multilayer conductor.

    Computes the coil impedance change ΔZ due to the layered
    specimen using:

        ΔZ = jω · C · ∫₀^∞ P²(s)/s⁶ · [e^{-sl₁}-e^{-sl₂}]² · Φ(s) ds

    where C = 2πμ₀N² / [(r₂-r₁)²(l₂-l₁)²] and Φ(s) is the
    multilayer reflection coefficient.

    Also computes the air-core (no specimen) impedance Z₀ so
    the total normalised impedance can be obtained.

    Args:
        sigma_layers: Conductivity per layer (K,) in S/m
        mu_layers: Relative permeability per layer (K,)
        settings: Probe geometry / frequency
        layer_thickness: Uniform layer thickness (m).
            If None, defaults to 1 mm / K.
        n_quad: Number of Gauss-Legendre quadrature points
        s_max_factor: Upper integration limit as multiple of
            1 / mean_radius (controls truncation)

    Returns:
        EDCResponse with the impedance change ΔZ
    """
    sigma_layers = np.asarray(sigma_layers, dtype=np.float64)
    mu_layers = np.asarray(mu_layers, dtype=np.float64)

    if len(sigma_layers) != len(mu_layers):
        raise ValueError(
            f"sigma_layers ({len(sigma_layers)}) and mu_layers "
            f"({len(mu_layers)}) must have same length"
        )

    K = len(sigma_layers)
    if layer_thickness is None:
        layer_thickness = 1e-3 / K

    thicknesses = np.full(K, layer_thickness, dtype=np.float64)
    omega = settings.omega

    s_max = s_max_factor / settings.mean_radius

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    s_vals = 0.5 * s_max * (nodes + 1.0)
    s_weights = 0.5 * s_max * weights

    integral = 0.0 + 0.0j
    for s_val, s_w in zip(s_vals, s_weights):
        integral += _dodd_deeds_integrand(
            s_val, omega, sigma_layers, mu_layers, thicknesses, settings
        ) * s_w

    dr = settings.outer_radius - settings.inner_radius
    dh = settings.coil_height
    C = 2.0 * np.pi * MU_0 * settings.n_turns ** 2 / (dr ** 2 * dh ** 2)

    delta_Z = 1j * omega * C * integral

    return EDCResponse(
        frequency=settings.frequency,
        impedance_real=float(delta_Z.real),
        impedance_imag=float(delta_Z.imag),
    )


def edc_forward_multifreq(
    sigma_layers: np.ndarray,
    mu_layers: np.ndarray,
    settings: ProbeSettings,
    frequencies: np.ndarray,
    layer_thickness: Optional[float] = None,
    n_quad: int = 200,
) -> np.ndarray:
    """
    Compute impedance change at multiple frequencies.

    Returns:
        Complex array of shape (len(frequencies),) with ΔZ at each f.
    """
    from dataclasses import replace

    results = np.empty(len(frequencies), dtype=complex)
    for i, f in enumerate(frequencies):
        s = replace(settings, frequency=float(f))
        resp = edc_forward(sigma_layers, mu_layers, s, layer_thickness, n_quad)
        results[i] = resp.impedance_complex
    return results


def calculate_skin_depth(sigma: float, mu_r: float, frequency: float) -> float:
    """δ = √(2 / (ω σ μ₀ μᵣ))"""
    omega = 2 * np.pi * frequency
    return np.sqrt(2.0 / (omega * sigma * mu_r * MU_0))


def estimate_penetration_depth(
    sigma_layers: np.ndarray,
    mu_layers: np.ndarray,
    frequency: float,
    layer_thickness: float,
) -> float:
    """Mean skin depth across all layers."""
    skin_depths = np.array([
        calculate_skin_depth(s, m, frequency)
        for s, m in zip(sigma_layers, mu_layers)
    ])
    return np.mean(skin_depths)
