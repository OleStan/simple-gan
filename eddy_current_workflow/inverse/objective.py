"""
Mismatch objective functionals for the EDC inverse problem.

The inverse problem seeks material profiles (σ, μ) that reproduce
a measured impedance change ΔZ_meas.  The mismatch functional is:

    J(θ) = ‖ΔZ(θ) - ΔZ_meas‖² + λ_s R_smooth(θ) + λ_m R_mono(θ)

where θ parameterises the layered profile either directly
(θ = [σ₁…σ_K, μ₁…μ_K]) or through a GAN latent vector.
"""

import numpy as np
from typing import Optional, Tuple, Callable

from ..forward.edc_solver import edc_forward, EDCResponse, ProbeSettings


class EDCMismatchObjective:
    """Callable objective: θ → scalar mismatch."""

    def __init__(
        self,
        edc_measured: EDCResponse,
        probe_settings: ProbeSettings,
        K: int,
        layer_thickness: Optional[float] = None,
        n_quad: int = 100,
    ):
        self.edc_measured = edc_measured
        self.probe_settings = probe_settings
        self.K = K
        self.layer_thickness = layer_thickness
        self.n_quad = n_quad
        self._y_meas = edc_measured.to_vector()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_count(self) -> None:
        self._call_count = 0

    def _theta_to_profiles(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(theta) != 2 * self.K:
            raise ValueError(
                f"Expected theta of length 2K={2*self.K}, got {len(theta)}"
            )
        return theta[: self.K], theta[self.K :]

    def __call__(self, theta: np.ndarray) -> float:
        self._call_count += 1
        sigma, mu = self._theta_to_profiles(theta)

        resp = edc_forward(
            sigma, mu, self.probe_settings,
            layer_thickness=self.layer_thickness,
            n_quad=self.n_quad,
        )
        y_pred = resp.to_vector()

        return float(np.sum((y_pred - self._y_meas) ** 2))

    def gradient_fd(
        self, theta: np.ndarray, eps: float = 1e-7
    ) -> np.ndarray:
        """Central finite-difference gradient (for debugging / fallback)."""
        grad = np.zeros_like(theta)
        f0 = self(theta)
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_p[i] += eps
            grad[i] = (self(theta_p) - f0) / eps
        return grad


def smoothness_penalty(
    sigma: np.ndarray, mu: np.ndarray
) -> float:
    """Sum of squared first differences (Tikhonov-style)."""
    d_sigma = np.diff(sigma)
    d_mu = np.diff(mu)
    return float(np.sum(d_sigma ** 2) + np.sum(d_mu ** 2))


def monotonicity_penalty(
    sigma: np.ndarray,
    mu: np.ndarray,
    sigma_increasing: bool = True,
    mu_decreasing: bool = True,
) -> float:
    """
    Soft penalty for non-monotonic profiles.

    Penalises violations of the expected monotonicity direction.
    Default: σ increasing with depth, μ decreasing with depth.
    """
    penalty = 0.0
    d_sigma = np.diff(sigma)
    d_mu = np.diff(mu)

    if sigma_increasing:
        penalty += float(np.sum(np.maximum(0.0, -d_sigma) ** 2))
    else:
        penalty += float(np.sum(np.maximum(0.0, d_sigma) ** 2))

    if mu_decreasing:
        penalty += float(np.sum(np.maximum(0.0, d_mu) ** 2))
    else:
        penalty += float(np.sum(np.maximum(0.0, -d_mu) ** 2))

    return penalty


class RegularisedObjective:
    """Wraps an EDCMismatchObjective with smoothness + monotonicity terms."""

    def __init__(
        self,
        base: EDCMismatchObjective,
        lambda_smooth: float = 0.0,
        lambda_mono: float = 0.0,
        sigma_increasing: bool = True,
        mu_decreasing: bool = True,
    ):
        self.base = base
        self.lambda_smooth = lambda_smooth
        self.lambda_mono = lambda_mono
        self.sigma_increasing = sigma_increasing
        self.mu_decreasing = mu_decreasing

    @property
    def K(self) -> int:
        return self.base.K

    @property
    def call_count(self) -> int:
        return self.base.call_count

    def reset_count(self) -> None:
        self.base.reset_count()

    def __call__(self, theta: np.ndarray) -> float:
        mismatch = self.base(theta)
        sigma, mu = self.base._theta_to_profiles(theta)

        if self.lambda_smooth > 0:
            mismatch += self.lambda_smooth * smoothness_penalty(sigma, mu)

        if self.lambda_mono > 0:
            mismatch += self.lambda_mono * monotonicity_penalty(
                sigma, mu, self.sigma_increasing, self.mu_decreasing
            )

        return mismatch
