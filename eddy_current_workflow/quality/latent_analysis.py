"""
Latent space analysis for GAN quality evaluation.

Implements criteria from GAN_Quality_Guide.md sections 3.2 and 5:
- Latent space traversal: smooth interpolation along individual dimensions
- Inactive dimension detection
- Noise robustness: small perturbations should produce small output changes
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TraversalDimensionResult:
    dimension_index: int
    output_change_norm: float
    max_gradient_norm: float
    is_smooth: bool
    is_active: bool


@dataclass
class LatentTraversalResult:
    n_dimensions_tested: int
    n_active_dimensions: int
    n_smooth_dimensions: int
    inactive_ratio: float
    mean_smoothness_score: float
    dimension_results: List[TraversalDimensionResult]

    @property
    def passed(self) -> bool:
        return self.inactive_ratio < 0.5 and self.mean_smoothness_score > 0.5


@dataclass
class NoiseRobustnessResult:
    noise_levels: List[float]
    mean_output_changes: List[float]
    max_output_changes: List[float]
    lipschitz_estimates: List[float]
    mean_lipschitz: float
    is_robust: bool

    @property
    def passed(self) -> bool:
        return self.is_robust


def compute_latent_traversal(
    generator: torch.nn.Module,
    nz: int,
    device: torch.device,
    n_dimensions_to_test: int = 20,
    n_alpha_steps: int = 21,
    alpha_range: float = 3.0,
    activity_threshold: float = 0.01,
    smoothness_threshold: float = 5.0,
) -> LatentTraversalResult:
    generator.eval()
    rng = np.random.default_rng(42)

    dims_to_test = min(n_dimensions_to_test, nz)
    test_dims = rng.choice(nz, dims_to_test, replace=False)

    z0 = torch.randn(1, nz, device=device)
    alphas = np.linspace(-alpha_range, alpha_range, n_alpha_steps)

    dimension_results = []

    with torch.no_grad():
        base_output, _, _ = generator(z0)
        base_output_np = base_output.cpu().numpy().flatten()

        for dim_idx in test_dims:
            outputs = []
            for alpha in alphas:
                z_traversed = z0.clone()
                z_traversed[0, dim_idx] = z0[0, dim_idx] + alpha
                out, _, _ = generator(z_traversed)
                outputs.append(out.cpu().numpy().flatten())

            outputs = np.array(outputs)

            total_change = np.linalg.norm(outputs[-1] - outputs[0])

            gradients = []
            for i in range(1, len(outputs)):
                grad_norm = np.linalg.norm(outputs[i] - outputs[i-1]) / (alphas[i] - alphas[i-1])
                gradients.append(grad_norm)

            gradients = np.array(gradients)
            max_grad = float(np.max(gradients)) if len(gradients) > 0 else 0.0
            mean_grad = float(np.mean(gradients)) if len(gradients) > 0 else 0.0

            is_active = total_change > activity_threshold
            is_smooth = max_grad < smoothness_threshold * mean_grad if mean_grad > 1e-10 else True

            dimension_results.append(TraversalDimensionResult(
                dimension_index=int(dim_idx),
                output_change_norm=float(total_change),
                max_gradient_norm=max_grad,
                is_smooth=is_smooth,
                is_active=is_active,
            ))

    n_active = sum(1 for r in dimension_results if r.is_active)
    n_smooth = sum(1 for r in dimension_results if r.is_smooth and r.is_active)
    inactive_ratio = 1.0 - (n_active / dims_to_test) if dims_to_test > 0 else 1.0

    smoothness_scores = []
    for r in dimension_results:
        if r.is_active:
            smoothness_scores.append(1.0 if r.is_smooth else 0.0)
    mean_smoothness = float(np.mean(smoothness_scores)) if smoothness_scores else 0.0

    return LatentTraversalResult(
        n_dimensions_tested=dims_to_test,
        n_active_dimensions=n_active,
        n_smooth_dimensions=n_smooth,
        inactive_ratio=inactive_ratio,
        mean_smoothness_score=mean_smoothness,
        dimension_results=dimension_results,
    )


def compute_noise_robustness(
    generator: torch.nn.Module,
    nz: int,
    device: torch.device,
    noise_levels: Optional[List[float]] = None,
    n_base_samples: int = 50,
    n_perturbations: int = 10,
    lipschitz_threshold: float = 10.0,
) -> NoiseRobustnessResult:
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

    generator.eval()

    z_base = torch.randn(n_base_samples, nz, device=device)

    with torch.no_grad():
        base_outputs, _, _ = generator(z_base)
        base_outputs_np = base_outputs.cpu().numpy()

    mean_changes = []
    max_changes = []
    lipschitz_estimates = []

    for sigma in noise_levels:
        all_changes = []

        with torch.no_grad():
            for _ in range(n_perturbations):
                epsilon = torch.randn_like(z_base) * sigma
                z_perturbed = z_base + epsilon

                perturbed_outputs, _, _ = generator(z_perturbed)
                perturbed_np = perturbed_outputs.cpu().numpy()

                output_diffs = np.linalg.norm(perturbed_np - base_outputs_np, axis=1)
                input_diffs = np.linalg.norm(epsilon.cpu().numpy(), axis=1)

                all_changes.extend(output_diffs.tolist())

                ratios = output_diffs / (input_diffs + 1e-10)
                lipschitz_estimates.append(float(np.max(ratios)))

        mean_changes.append(float(np.mean(all_changes)))
        max_changes.append(float(np.max(all_changes)))

    mean_lipschitz = float(np.mean(lipschitz_estimates))
    is_robust = mean_lipschitz < lipschitz_threshold

    return NoiseRobustnessResult(
        noise_levels=noise_levels,
        mean_output_changes=mean_changes,
        max_output_changes=max_changes,
        lipschitz_estimates=lipschitz_estimates,
        mean_lipschitz=mean_lipschitz,
        is_robust=is_robust,
    )
