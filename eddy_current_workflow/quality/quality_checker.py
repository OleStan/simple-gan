"""
GAN Quality Checker - orchestrates all validation criteria.

Runs the five quality validation tests from GAN_Quality_Guide.md:
1. Moment matching (mean & variance consistency)
2. Distribution distances (Wasserstein / MMD)
3. Latent space traversal (smoothness, active dimensions)
4. Physics consistency (forward model, bounds)
5. Noise robustness (Lipschitz stability)
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .metrics import (
    compute_moment_comparison,
    compute_distribution_distances,
    compute_class_separation,
    MomentComparisonResult,
    DistributionDistanceResult,
    ClassSeparationResult,
)
from .latent_analysis import (
    compute_latent_traversal,
    compute_noise_robustness,
    LatentTraversalResult,
    NoiseRobustnessResult,
)
from .physics_consistency import (
    check_physics_consistency,
    PhysicsConsistencyResult,
)
from ..forward.edc_solver import ProbeSettings


@dataclass
class GANQualityReport:
    model_name: str
    n_real_samples: int
    n_generated_samples: int
    K: int
    nz: int
    moment_comparison: MomentComparisonResult
    distribution_distances: DistributionDistanceResult
    latent_traversal: LatentTraversalResult
    noise_robustness: NoiseRobustnessResult
    physics_consistency: PhysicsConsistencyResult
    class_separation: Optional[ClassSeparationResult] = None

    @property
    def overall_passed(self) -> bool:
        class_sep_ok = self.class_separation.passed if self.class_separation is not None else True
        return (
            self.moment_comparison.passed
            and self.latent_traversal.passed
            and self.noise_robustness.passed
            and self.physics_consistency.passed
            and class_sep_ok
        )

    @property
    def summary(self) -> dict:
        criteria: Dict = {
            '1_moment_matching': {
                'passed': self.moment_comparison.passed,
                'mean_rel_diff': self.moment_comparison.mean_rel_diff,
                'variance_ratio': self.moment_comparison.variance_ratio,
                'mode_collapse': self.moment_comparison.mode_collapse_detected,
                'noise_amplification': self.moment_comparison.noise_amplification_detected,
            },
            '2_distribution_distance': {
                'wasserstein_mean': self.distribution_distances.wasserstein_mean,
                'mmd_score': self.distribution_distances.mmd_score,
                'mmd_sigma': self.distribution_distances.mmd_sigma_component,
                'mmd_mu': self.distribution_distances.mmd_mu_component,
            },
            '3_latent_traversal': {
                'passed': self.latent_traversal.passed,
                'active_dimensions': self.latent_traversal.n_active_dimensions,
                'total_tested': self.latent_traversal.n_dimensions_tested,
                'inactive_ratio': self.latent_traversal.inactive_ratio,
                'smoothness_score': self.latent_traversal.mean_smoothness_score,
            },
            '4_physics_consistency': {
                'passed': self.physics_consistency.passed,
                'bounds_sigma_ratio': self.physics_consistency.bounds_result.sigma_in_bounds_ratio,
                'bounds_mu_ratio': self.physics_consistency.bounds_result.mu_in_bounds_ratio,
                'forward_valid': self.physics_consistency.forward_result.n_valid_responses,
                'forward_total': self.physics_consistency.forward_result.n_samples_tested,
            },
            '5_noise_robustness': {
                'passed': self.noise_robustness.passed,
                'mean_lipschitz': self.noise_robustness.mean_lipschitz,
            },
        }
        if self.class_separation is not None:
            criteria['6_class_separation'] = {
                'passed': self.class_separation.passed,
                'n_classes': self.class_separation.n_classes,
                'separation_ratio': self.class_separation.separation_ratio,
                'inter_class_wasserstein_real': self.class_separation.inter_class_wasserstein_real,
                'inter_class_wasserstein_gen': self.class_separation.inter_class_wasserstein_gen,
                'per_class_real': self.class_separation.per_class_real_moments,
                'per_class_gen': self.class_separation.per_class_gen_moments,
            }
        return {
            'model_name': self.model_name,
            'overall_passed': self.overall_passed,
            'criteria': criteria,
        }


class GANQualityChecker:
    def __init__(
        self,
        K: int,
        nz: int,
        sigma_bounds: Tuple[float, float] = (1e6, 6e7),
        mu_bounds: Tuple[float, float] = (1.0, 100.0),
        device: Optional[torch.device] = None,
        n_classes: int = 1,
    ):
        self.K = K
        self.nz = nz
        self.sigma_bounds = sigma_bounds
        self.mu_bounds = mu_bounds
        self.device = device or torch.device('cpu')
        self.n_classes = n_classes

    def run_all_checks(
        self,
        generator: torch.nn.Module,
        real_data: np.ndarray,
        generated_data: np.ndarray,
        model_name: str = "GAN",
        probe_settings: Optional[ProbeSettings] = None,
        real_data_physical: Optional[np.ndarray] = None,
        generated_data_physical: Optional[np.ndarray] = None,
        mmd_max_samples: int = 300,
        n_latent_dims_to_test: int = 20,
        n_forward_samples: int = 20,
        real_labels: Optional[np.ndarray] = None,
        gen_labels: Optional[np.ndarray] = None,
    ) -> GANQualityReport:
        n_criteria = 6 if self.n_classes > 1 else 5
        print(f"[Quality Check] Running all {n_criteria} criteria for '{model_name}'...")

        print("  [1] Moment comparison (mean & variance)...")
        moment_result = compute_moment_comparison(real_data, generated_data, self.K)

        print("  [2] Distribution distances (Wasserstein & MMD)...")
        dist_result = compute_distribution_distances(
            real_data, generated_data, self.K, mmd_max_samples=mmd_max_samples,
        )

        print("  [3] Latent space traversal...")
        traversal_result = compute_latent_traversal(
            generator, self.nz, self.device,
            n_dimensions_to_test=n_latent_dims_to_test,
            labels=gen_labels,
        )

        print("  [4] Noise robustness...")
        robustness_result = compute_noise_robustness(
            generator, self.nz, self.device,
            labels=gen_labels,
        )

        phys_gen = generated_data_physical if generated_data_physical is not None else generated_data
        phys_ref = real_data_physical if real_data_physical is not None else real_data

        print("  [5] Physics consistency...")
        physics_result = check_physics_consistency(
            phys_gen, self.K,
            sigma_bounds=self.sigma_bounds,
            mu_bounds=self.mu_bounds,
            probe_settings=probe_settings,
            reference_data=phys_ref,
            n_forward_samples=n_forward_samples,
        )

        class_sep_result: Optional[ClassSeparationResult] = None
        if self.n_classes > 1 and real_labels is not None and gen_labels is not None:
            print("  [6] Class separation...")
            class_sep_result = compute_class_separation(
                real_data, generated_data, real_labels, gen_labels, self.K,
            )

        report = GANQualityReport(
            model_name=model_name,
            n_real_samples=len(real_data),
            n_generated_samples=len(generated_data),
            K=self.K,
            nz=self.nz,
            moment_comparison=moment_result,
            distribution_distances=dist_result,
            latent_traversal=traversal_result,
            noise_robustness=robustness_result,
            physics_consistency=physics_result,
            class_separation=class_sep_result,
        )

        status = "PASSED" if report.overall_passed else "FAILED"
        print(f"  [Done] Overall: {status}")

        return report

    def generate_samples(
        self,
        generator: torch.nn.Module,
        n_samples: int = 1000,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (samples, labels_used) arrays."""
        generator.eval()
        all_samples = []
        all_labels = []
        batch_size = 64

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                current_batch = min(batch_size, n_samples - start)
                z = torch.randn(current_batch, self.nz, device=self.device)

                if self.n_classes > 1:
                    if labels is not None:
                        batch_labels_np = labels[start:start + current_batch]
                        batch_labels = torch.from_numpy(batch_labels_np).to(self.device)
                    else:
                        batch_labels = torch.arange(current_batch, device=self.device) % self.n_classes
                    output, _, _ = generator(z, labels=batch_labels)
                    all_labels.append(batch_labels.cpu().numpy())
                else:
                    output, _, _ = generator(z)
                    all_labels.append(np.zeros(current_batch, dtype=np.int64))

                all_samples.append(output.cpu().numpy())

        return np.concatenate(all_samples, axis=0), np.concatenate(all_labels, axis=0)
