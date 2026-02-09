"""
High-level inverse recovery API.

Provides a single entry-point `recover_profiles` that:
1. Builds the mismatch objective from a measured EDC response.
2. Optionally wraps it with regularisation.
3. Dispatches to the chosen optimiser.
4. Returns the recovered (σ, μ) profiles together with diagnostics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from ..forward.edc_solver import EDCResponse, ProbeSettings, edc_forward
from .objective import EDCMismatchObjective, RegularisedObjective
from .optimizers import (
    InverseResult,
    solve_multistart,
    solve_global,
)


@dataclass
class RecoveryConfig:
    """All tunables for a single inverse recovery run."""

    K: int = 51
    sigma_bounds: Tuple[float, float] = (1e6, 6e7)
    mu_bounds: Tuple[float, float] = (1.0, 100.0)
    layer_thickness: Optional[float] = None
    n_quad: int = 100
    method: str = "multistart"
    local_method: str = "L-BFGS-B"
    n_starts: int = 10
    max_iter: int = 1000
    tol: float = 1e-10
    seed: int = 42
    lambda_smooth: float = 0.0
    lambda_mono: float = 0.0
    sigma_increasing: bool = True
    mu_decreasing: bool = True
    verbose: bool = False


def recover_profiles(
    edc_measured: EDCResponse,
    probe_settings: ProbeSettings,
    config: Optional[RecoveryConfig] = None,
) -> InverseResult:
    """
    Recover (σ, μ) layer profiles from a measured impedance change.

    Args:
        edc_measured: Target impedance (from measurement or synthetic).
        probe_settings: Probe geometry / frequency used during measurement.
        config: Recovery hyper-parameters.  Uses defaults when None.

    Returns:
        InverseResult with recovered sigma, mu, mismatch, etc.
    """
    if config is None:
        config = RecoveryConfig()

    base_obj = EDCMismatchObjective(
        edc_measured=edc_measured,
        probe_settings=probe_settings,
        K=config.K,
        layer_thickness=config.layer_thickness,
        n_quad=config.n_quad,
    )

    has_reg = config.lambda_smooth > 0 or config.lambda_mono > 0
    if has_reg:
        objective = RegularisedObjective(
            base=base_obj,
            lambda_smooth=config.lambda_smooth,
            lambda_mono=config.lambda_mono,
            sigma_increasing=config.sigma_increasing,
            mu_decreasing=config.mu_decreasing,
        )
    else:
        objective = base_obj

    if config.method == "multistart":
        result = solve_multistart(
            objective=objective,
            K=config.K,
            sigma_bounds=config.sigma_bounds,
            mu_bounds=config.mu_bounds,
            n_starts=config.n_starts,
            method=config.local_method,
            max_iter=config.max_iter,
            tol=config.tol,
            seed=config.seed,
            verbose=config.verbose,
        )
    elif config.method == "global":
        result = solve_global(
            objective=objective,
            K=config.K,
            sigma_bounds=config.sigma_bounds,
            mu_bounds=config.mu_bounds,
            max_iter=config.max_iter,
            tol=config.tol,
            seed=config.seed,
            verbose=config.verbose,
        )
    else:
        raise ValueError(f"Unknown method: {config.method!r}")

    return result


def round_trip_error(
    sigma_true: np.ndarray,
    mu_true: np.ndarray,
    sigma_recovered: np.ndarray,
    mu_recovered: np.ndarray,
) -> dict:
    """
    Quantify recovery quality with several metrics.

    Returns dict with:
        sigma_rmse, mu_rmse, sigma_max_err, mu_max_err,
        sigma_rel_rmse, mu_rel_rmse
    """
    sigma_diff = sigma_true - sigma_recovered
    mu_diff = mu_true - mu_recovered

    sigma_rmse = float(np.sqrt(np.mean(sigma_diff ** 2)))
    mu_rmse = float(np.sqrt(np.mean(mu_diff ** 2)))

    sigma_range = float(np.ptp(sigma_true)) if np.ptp(sigma_true) > 0 else 1.0
    mu_range = float(np.ptp(mu_true)) if np.ptp(mu_true) > 0 else 1.0

    return {
        "sigma_rmse": sigma_rmse,
        "mu_rmse": mu_rmse,
        "sigma_max_err": float(np.max(np.abs(sigma_diff))),
        "mu_max_err": float(np.max(np.abs(mu_diff))),
        "sigma_rel_rmse": sigma_rmse / sigma_range,
        "mu_rel_rmse": mu_rmse / mu_range,
    }
