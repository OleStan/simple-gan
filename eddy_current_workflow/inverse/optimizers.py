"""
Optimization back-ends for the EDC inverse problem.

Provides multi-start local (L-BFGS-B) and global (differential
evolution) solvers that operate on the mismatch objective.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional
from scipy.optimize import minimize, differential_evolution


@dataclass
class InverseResult:
    """Container for an inverse-problem solution."""

    sigma: np.ndarray
    mu: np.ndarray
    mismatch: float
    success: bool
    n_feval: int
    n_iterations: int
    method: str
    all_mismatches: List[float] = field(default_factory=list)
    convergence_rate: float = 0.0

    def __repr__(self) -> str:
        return (
            f"InverseResult(mismatch={self.mismatch:.6e}, "
            f"success={self.success}, feval={self.n_feval}, "
            f"method={self.method})"
        )


def _build_bounds(
    K: int,
    sigma_bounds: Tuple[float, float],
    mu_bounds: Tuple[float, float],
) -> list:
    return (
        [(sigma_bounds[0], sigma_bounds[1])] * K
        + [(mu_bounds[0], mu_bounds[1])] * K
    )


def _random_theta0(
    K: int,
    sigma_bounds: Tuple[float, float],
    mu_bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    sigma_init = rng.uniform(sigma_bounds[0], sigma_bounds[1], K)
    mu_init = rng.uniform(mu_bounds[0], mu_bounds[1], K)
    return np.concatenate([sigma_init, mu_init])


def solve_multistart(
    objective: Callable[[np.ndarray], float],
    K: int,
    sigma_bounds: Tuple[float, float],
    mu_bounds: Tuple[float, float],
    n_starts: int = 10,
    method: str = "L-BFGS-B",
    max_iter: int = 1000,
    tol: float = 1e-10,
    seed: int = 42,
    verbose: bool = False,
) -> InverseResult:
    """
    Multi-start local optimisation.

    Launches *n_starts* independent L-BFGS-B runs from random initial
    points and keeps the best.
    """
    rng = np.random.default_rng(seed)
    bounds = _build_bounds(K, sigma_bounds, mu_bounds)

    best_result = None
    best_mismatch = np.inf
    all_mismatches: List[float] = []
    total_feval = 0
    n_success = 0

    for i in range(n_starts):
        theta0 = _random_theta0(K, sigma_bounds, mu_bounds, rng)

        res = minimize(
            objective,
            theta0,
            method=method,
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": tol},
        )

        total_feval += res.nfev
        all_mismatches.append(float(res.fun))

        if res.success:
            n_success += 1

        if verbose:
            tag = "✓" if res.success else "✗"
            print(f"  start {i+1}/{n_starts} {tag}  J={res.fun:.6e}  nit={res.nit}")

        if res.fun < best_mismatch:
            best_mismatch = res.fun
            best_result = res

    sigma_best = best_result.x[:K]
    mu_best = best_result.x[K:]

    return InverseResult(
        sigma=sigma_best,
        mu=mu_best,
        mismatch=best_mismatch,
        success=best_result.success,
        n_feval=total_feval,
        n_iterations=best_result.nit,
        method=method,
        all_mismatches=all_mismatches,
        convergence_rate=n_success / n_starts,
    )


def solve_global(
    objective: Callable[[np.ndarray], float],
    K: int,
    sigma_bounds: Tuple[float, float],
    mu_bounds: Tuple[float, float],
    max_iter: int = 500,
    tol: float = 1e-10,
    seed: int = 42,
    popsize: int = 15,
    polish: bool = True,
    verbose: bool = False,
) -> InverseResult:
    """
    Global optimisation via differential evolution.

    Slower but more robust for highly non-convex landscapes.
    """
    bounds = _build_bounds(K, sigma_bounds, mu_bounds)

    res = differential_evolution(
        objective,
        bounds,
        maxiter=max_iter,
        tol=tol,
        seed=seed,
        popsize=popsize,
        polish=polish,
        disp=verbose,
    )

    sigma_best = res.x[:K]
    mu_best = res.x[K:]

    return InverseResult(
        sigma=sigma_best,
        mu=mu_best,
        mismatch=float(res.fun),
        success=res.success,
        n_feval=res.nfev,
        n_iterations=res.nit,
        method="differential_evolution",
    )
