"""
Statistical distribution metrics for GAN quality evaluation.

Implements criteria from GAN_Quality_Guide.md sections 2.1 and 2.2:
- Mean consistency between real and generated distributions
- Variance consistency (mode collapse / noise amplification detection)
- Wasserstein distance (Earth Mover Distance)
- Maximum Mean Discrepancy (MMD) with Gaussian RBF kernel
"""

import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MomentComparisonResult:
    real_mean: np.ndarray
    generated_mean: np.ndarray
    mean_abs_diff: float
    mean_rel_diff: float
    real_variance: np.ndarray
    generated_variance: np.ndarray
    variance_abs_diff: float
    variance_ratio: float
    mode_collapse_detected: bool
    noise_amplification_detected: bool

    @property
    def passed(self) -> bool:
        return not self.mode_collapse_detected and not self.noise_amplification_detected


@dataclass
class DistributionDistanceResult:
    wasserstein_per_dim: np.ndarray
    wasserstein_mean: float
    mmd_score: float
    mmd_sigma_component: float
    mmd_mu_component: float


def compute_moment_comparison(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    K: int,
    variance_ratio_threshold: float = 0.3,
) -> MomentComparisonResult:
    real_sigma = real_data[:, :K]
    real_mu = real_data[:, K:2*K]
    gen_sigma = generated_data[:, :K]
    gen_mu = generated_data[:, K:2*K]

    real_mean = np.concatenate([real_sigma.mean(axis=0), real_mu.mean(axis=0)])
    gen_mean = np.concatenate([gen_sigma.mean(axis=0), gen_mu.mean(axis=0)])

    real_var = np.concatenate([real_sigma.var(axis=0), real_mu.var(axis=0)])
    gen_var = np.concatenate([gen_sigma.var(axis=0), gen_mu.var(axis=0)])

    mean_abs_diff = float(np.mean(np.abs(real_mean - gen_mean)))
    mean_rel_diff = float(np.mean(np.abs(real_mean - gen_mean) / (np.abs(real_mean) + 1e-10)))

    variance_ratio = float(np.mean(gen_var / (real_var + 1e-10)))

    mode_collapse = variance_ratio < (1.0 - variance_ratio_threshold)
    noise_amplification = variance_ratio > (1.0 + variance_ratio_threshold * 2)

    return MomentComparisonResult(
        real_mean=real_mean,
        generated_mean=gen_mean,
        mean_abs_diff=mean_abs_diff,
        mean_rel_diff=mean_rel_diff,
        real_variance=real_var,
        generated_variance=gen_var,
        variance_abs_diff=float(np.mean(np.abs(real_var - gen_var))),
        variance_ratio=variance_ratio,
        mode_collapse_detected=mode_collapse,
        noise_amplification_detected=noise_amplification,
    )


def _gaussian_rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    diff = x - y
    return float(np.exp(-np.dot(diff, diff) / (2 * sigma ** 2)))


def compute_mmd(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    K: int,
    kernel_bandwidth: float = 1.0,
    max_samples: int = 500,
) -> Tuple[float, float, float]:
    n_real = min(len(real_data), max_samples)
    n_gen = min(len(generated_data), max_samples)

    rng = np.random.default_rng(42)
    real_idx = rng.choice(len(real_data), n_real, replace=False) if len(real_data) > n_real else np.arange(n_real)
    gen_idx = rng.choice(len(generated_data), n_gen, replace=False) if len(generated_data) > n_gen else np.arange(n_gen)

    real_sub = real_data[real_idx]
    gen_sub = generated_data[gen_idx]

    def _mmd_component(real_part: np.ndarray, gen_part: np.ndarray) -> float:
        bandwidth = kernel_bandwidth * np.median(
            np.linalg.norm(real_part[:min(100, len(real_part))] - real_part[:min(100, len(real_part))].mean(axis=0), axis=1)
        )
        if bandwidth < 1e-10:
            bandwidth = 1.0

        kxx = 0.0
        for i in range(n_real):
            for j in range(i + 1, n_real):
                kxx += _gaussian_rbf_kernel(real_part[i], real_part[j], bandwidth)
        kxx = 2.0 * kxx / (n_real * (n_real - 1)) if n_real > 1 else 0.0

        kyy = 0.0
        for i in range(n_gen):
            for j in range(i + 1, n_gen):
                kyy += _gaussian_rbf_kernel(gen_part[i], gen_part[j], bandwidth)
        kyy = 2.0 * kyy / (n_gen * (n_gen - 1)) if n_gen > 1 else 0.0

        kxy = 0.0
        for i in range(n_real):
            for j in range(n_gen):
                kxy += _gaussian_rbf_kernel(real_part[i], gen_part[j], bandwidth)
        kxy = kxy / (n_real * n_gen)

        return kxx + kyy - 2.0 * kxy

    mmd_sigma = _mmd_component(real_sub[:, :K], gen_sub[:, :K])
    mmd_mu = _mmd_component(real_sub[:, K:2*K], gen_sub[:, K:2*K])
    mmd_total = mmd_sigma + mmd_mu

    return mmd_total, mmd_sigma, mmd_mu


def compute_wasserstein_per_dimension(
    real_data: np.ndarray,
    generated_data: np.ndarray,
) -> np.ndarray:
    n_dims = real_data.shape[1]
    distances = np.zeros(n_dims)
    for d in range(n_dims):
        distances[d] = scipy_wasserstein(real_data[:, d], generated_data[:, d])
    return distances


def compute_distribution_distances(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    K: int,
    mmd_max_samples: int = 300,
) -> DistributionDistanceResult:
    w_distances = compute_wasserstein_per_dimension(real_data, generated_data)

    mmd_total, mmd_sigma, mmd_mu = compute_mmd(
        real_data, generated_data, K, max_samples=mmd_max_samples
    )

    return DistributionDistanceResult(
        wasserstein_per_dim=w_distances,
        wasserstein_mean=float(np.mean(w_distances)),
        mmd_score=mmd_total,
        mmd_sigma_component=mmd_sigma,
        mmd_mu_component=mmd_mu,
    )
