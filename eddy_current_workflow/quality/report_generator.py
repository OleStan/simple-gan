"""
Quality report generator with visualizations and detailed descriptions.

Produces a comprehensive report for each of the 5 GAN quality criteria,
including plots, numerical results, and interpretive descriptions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from .quality_checker import GANQualityReport


_STATUS_PASS = "PASS"
_STATUS_FAIL = "FAIL"
_STATUS_WARN = "WARNING"


class QualityReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)

    def generate_full_report(
        self,
        report: GANQualityReport,
        real_data: Optional[np.ndarray] = None,
        generated_data: Optional[np.ndarray] = None,
    ) -> str:
        print(f"Generating quality report in {self.output_dir}...")

        if real_data is not None and generated_data is not None:
            self._plot_moment_comparison(report, real_data, generated_data)
            self._plot_distribution_distances(report, real_data, generated_data)
            self._plot_sample_comparison(report, real_data, generated_data)

        self._plot_latent_traversal(report)
        self._plot_noise_robustness(report)

        report_text = self._build_text_report(report)
        report_path = self.output_dir / 'quality_report.md'
        report_path.write_text(report_text)

        summary_path = self.output_dir / 'quality_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(report.summary, f, indent=2, default=str)

        print(f"Report saved to {report_path}")
        return str(report_path)

    def _plot_moment_comparison(
        self, report: GANQualityReport,
        real_data: np.ndarray, generated_data: np.ndarray,
    ):
        K = report.K
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{report.model_name} - Moment Comparison', fontsize=14, fontweight='bold')

        x = np.arange(K)

        axes[0, 0].plot(x, report.moment_comparison.real_mean[:K], 'b-', label='Real', linewidth=2)
        axes[0, 0].plot(x, report.moment_comparison.generated_mean[:K], 'r--', label='Generated', linewidth=2)
        axes[0, 0].set_title('Mean σ Profile')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(x, report.moment_comparison.real_mean[K:], 'b-', label='Real', linewidth=2)
        axes[0, 1].plot(x, report.moment_comparison.generated_mean[K:], 'r--', label='Generated', linewidth=2)
        axes[0, 1].set_title('Mean μ Profile')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Mean Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(x, report.moment_comparison.real_variance[:K], 'b-', label='Real', linewidth=2)
        axes[1, 0].plot(x, report.moment_comparison.generated_variance[:K], 'r--', label='Generated', linewidth=2)
        axes[1, 0].set_title('Variance σ Profile')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(x, report.moment_comparison.real_variance[K:], 'b-', label='Real', linewidth=2)
        axes[1, 1].plot(x, report.moment_comparison.generated_variance[K:], 'r--', label='Generated', linewidth=2)
        axes[1, 1].set_title('Variance μ Profile')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'moment_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_distribution_distances(
        self, report: GANQualityReport,
        real_data: np.ndarray, generated_data: np.ndarray,
    ):
        K = report.K
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{report.model_name} - Distribution Distances', fontsize=14, fontweight='bold')

        w_dist = report.distribution_distances.wasserstein_per_dim
        axes[0].bar(range(K), w_dist[:K], alpha=0.7, label='σ dimensions', color='steelblue')
        axes[0].bar(range(K, 2*K), w_dist[K:], alpha=0.7, label='μ dimensions', color='coral')
        axes[0].set_title('Wasserstein Distance per Dimension')
        axes[0].set_xlabel('Dimension')
        axes[0].set_ylabel('Wasserstein Distance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        mmd_vals = [
            report.distribution_distances.mmd_sigma_component,
            report.distribution_distances.mmd_mu_component,
            report.distribution_distances.mmd_score,
        ]
        bars = axes[1].bar(['σ MMD', 'μ MMD', 'Total MMD'], mmd_vals,
                           color=['steelblue', 'coral', 'mediumpurple'], alpha=0.8)
        axes[1].set_title('Maximum Mean Discrepancy')
        axes[1].set_ylabel('MMD Score')
        axes[1].grid(True, alpha=0.3)
        for bar, val in zip(bars, mmd_vals):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        for dim_offset, label, color in [(0, 'σ', 'steelblue'), (K, 'μ', 'coral')]:
            for i in range(min(5, len(real_data))):
                axes[2].plot(real_data[i, dim_offset:dim_offset+K], color=color, alpha=0.2)
            for i in range(min(5, len(generated_data))):
                axes[2].plot(generated_data[i, dim_offset:dim_offset+K], color=color,
                            alpha=0.2, linestyle='--')
        axes[2].plot([], [], 'b-', label='Real')
        axes[2].plot([], [], 'b--', label='Generated')
        axes[2].set_title('Sample Overlay (first 5)')
        axes[2].set_xlabel('Layer Index')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'distribution_distances.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_sample_comparison(
        self, report: GANQualityReport,
        real_data: np.ndarray, generated_data: np.ndarray,
    ):
        K = report.K
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(f'{report.model_name} - Real vs Generated Samples', fontsize=14, fontweight='bold')

        for i in range(4):
            axes[0, i].plot(real_data[i, :K], 'b-', label='σ', linewidth=1.5)
            ax2 = axes[0, i].twinx()
            ax2.plot(real_data[i, K:2*K], 'r-', label='μ', linewidth=1.5)
            axes[0, i].set_title(f'Real #{i+1}')
            axes[0, i].grid(True, alpha=0.3)
            if i == 0:
                axes[0, i].set_ylabel('σ', color='b')
                ax2.set_ylabel('μ', color='r')

        for i in range(4):
            axes[1, i].plot(generated_data[i, :K], 'b-', label='σ', linewidth=1.5)
            ax2 = axes[1, i].twinx()
            ax2.plot(generated_data[i, K:2*K], 'r-', label='μ', linewidth=1.5)
            axes[1, i].set_title(f'Generated #{i+1}')
            axes[1, i].grid(True, alpha=0.3)
            if i == 0:
                axes[1, i].set_ylabel('σ', color='b')
                ax2.set_ylabel('μ', color='r')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'sample_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_latent_traversal(self, report: GANQualityReport):
        traversal = report.latent_traversal
        dim_results = traversal.dimension_results

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'{report.model_name} - Latent Space Analysis', fontsize=14, fontweight='bold')

        changes = [r.output_change_norm for r in dim_results]
        colors = ['green' if r.is_active else 'red' for r in dim_results]
        axes[0].bar(range(len(changes)), changes, color=colors, alpha=0.7)
        axes[0].set_title('Output Change per Latent Dimension')
        axes[0].set_xlabel('Tested Dimension')
        axes[0].set_ylabel('||G(z+αe_i) - G(z-αe_i)||')
        axes[0].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Activity threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        max_grads = [r.max_gradient_norm for r in dim_results if r.is_active]
        if max_grads:
            axes[1].hist(max_grads, bins=min(20, len(max_grads)), color='steelblue', alpha=0.7, edgecolor='black')
        axes[1].set_title('Max Gradient Distribution (Active Dims)')
        axes[1].set_xlabel('Max Gradient Norm')
        axes[1].set_ylabel('Count')
        axes[1].grid(True, alpha=0.3)

        labels = ['Active', 'Inactive', 'Smooth', 'Non-smooth']
        values = [
            traversal.n_active_dimensions,
            traversal.n_dimensions_tested - traversal.n_active_dimensions,
            traversal.n_smooth_dimensions,
            traversal.n_active_dimensions - traversal.n_smooth_dimensions,
        ]
        bar_colors = ['green', 'red', 'steelblue', 'orange']
        axes[2].bar(labels, values, color=bar_colors, alpha=0.7)
        axes[2].set_title('Latent Dimension Summary')
        axes[2].set_ylabel('Count')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'latent_traversal.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_noise_robustness(self, report: GANQualityReport):
        robustness = report.noise_robustness

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{report.model_name} - Noise Robustness', fontsize=14, fontweight='bold')

        axes[0].plot(robustness.noise_levels, robustness.mean_output_changes,
                    'bo-', label='Mean Δ output', linewidth=2, markersize=8)
        axes[0].plot(robustness.noise_levels, robustness.max_output_changes,
                    'r^--', label='Max Δ output', linewidth=2, markersize=8)
        axes[0].set_title('Output Change vs Noise Level')
        axes[0].set_xlabel('Noise σ')
        axes[0].set_ylabel('||G(z+ε) - G(z)||')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(range(len(robustness.lipschitz_estimates)),
                   robustness.lipschitz_estimates, alpha=0.6, color='steelblue')
        axes[1].axhline(y=robustness.mean_lipschitz, color='red', linestyle='--',
                       label=f'Mean = {robustness.mean_lipschitz:.2f}')
        axes[1].axhline(y=10.0, color='orange', linestyle=':', label='Threshold = 10.0')
        axes[1].set_title('Lipschitz Constant Estimates')
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('||ΔG|| / ||Δz||')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'noise_robustness.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _build_text_report(self, report: GANQualityReport) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overall = _STATUS_PASS if report.overall_passed else _STATUS_FAIL

        lines = [
            f"# GAN Quality Validation Report",
            f"",
            f"**Model:** {report.model_name}",
            f"**Date:** {timestamp}",
            f"**Overall Status:** {overall}",
            f"**Real samples:** {report.n_real_samples}",
            f"**Generated samples:** {report.n_generated_samples}",
            f"**K (layers):** {report.K}",
            f"**nz (latent dim):** {report.nz}",
            f"",
            f"---",
            f"",
        ]

        lines.extend(self._section_moments(report))
        lines.extend(self._section_distances(report))
        lines.extend(self._section_traversal(report))
        lines.extend(self._section_physics(report))
        lines.extend(self._section_robustness(report))

        return "\n".join(lines)

    def _section_moments(self, report: GANQualityReport) -> list:
        m = report.moment_comparison
        status = _STATUS_PASS if m.passed else _STATUS_FAIL
        lines = [
            f"## 1. Moment Matching (Mean & Variance Consistency) — {status}",
            f"",
            f"**Description:** Compares first-order (mean) and second-order (variance) statistics",
            f"between real and generated distributions. If E[x] ≈ E[x̂] and Var(x) ≈ Var(x̂),",
            f"the generator captures the central tendency and spread of the data.",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean absolute difference | {m.mean_abs_diff:.6f} |",
            f"| Mean relative difference | {m.mean_rel_diff:.4f} ({m.mean_rel_diff*100:.2f}%) |",
            f"| Variance ratio (gen/real) | {m.variance_ratio:.4f} |",
            f"| Variance absolute difference | {m.variance_abs_diff:.6f} |",
            f"| Mode collapse detected | {'Yes' if m.mode_collapse_detected else 'No'} |",
            f"| Noise amplification detected | {'Yes' if m.noise_amplification_detected else 'No'} |",
            f"",
            f"**Interpretation:**",
            f"- Variance ratio < 0.7 → mode collapse (generator produces too-similar outputs)",
            f"- Variance ratio > 1.6 → noise amplification (generator is unstable)",
            f"- Ideal variance ratio ≈ 1.0",
            f"",
            f"![Moment Comparison](plots/moment_comparison.png)",
            f"",
            f"---",
            f"",
        ]
        return lines

    def _section_distances(self, report: GANQualityReport) -> list:
        d = report.distribution_distances
        lines = [
            f"## 2. Distribution Distances (Wasserstein & MMD)",
            f"",
            f"**Description:** Measures how close the full generated distribution is to the real",
            f"distribution, beyond just first two moments.",
            f"",
            f"### Wasserstein Distance (Earth Mover Distance)",
            f"",
            f"Measures the minimum cost of transforming one distribution into another.",
            f"Lower values indicate closer distributions.",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean Wasserstein distance | {d.wasserstein_mean:.6f} |",
            f"| σ dimensions mean | {np.mean(d.wasserstein_per_dim[:report.K]):.6f} |",
            f"| μ dimensions mean | {np.mean(d.wasserstein_per_dim[report.K:]):.6f} |",
            f"",
            f"### Maximum Mean Discrepancy (MMD)",
            f"",
            f"Kernel-based distance using Gaussian RBF kernel. MMD² = 0 iff distributions are identical.",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total MMD | {d.mmd_score:.6f} |",
            f"| σ component | {d.mmd_sigma_component:.6f} |",
            f"| μ component | {d.mmd_mu_component:.6f} |",
            f"",
            f"![Distribution Distances](plots/distribution_distances.png)",
            f"",
            f"---",
            f"",
        ]
        return lines

    def _section_traversal(self, report: GANQualityReport) -> list:
        t = report.latent_traversal
        status = _STATUS_PASS if t.passed else _STATUS_FAIL
        lines = [
            f"## 3. Latent Space Traversal — {status}",
            f"",
            f"**Description:** Tests smoothness and disentanglement by traversing individual",
            f"latent dimensions: z(α) = z₀ + α·eᵢ. A well-trained GAN should produce smooth",
            f"output changes. Sudden jumps indicate instability; no change indicates inactive dimensions.",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Dimensions tested | {t.n_dimensions_tested} |",
            f"| Active dimensions | {t.n_active_dimensions} |",
            f"| Smooth dimensions | {t.n_smooth_dimensions} |",
            f"| Inactive ratio | {t.inactive_ratio:.2f} ({t.inactive_ratio*100:.1f}%) |",
            f"| Mean smoothness score | {t.mean_smoothness_score:.4f} |",
            f"",
            f"**Interpretation:**",
            f"- Inactive ratio > 50% → too many latent dimensions are unused (consider smaller nz)",
            f"- Low smoothness → generator has discontinuities in latent space",
            f"",
            f"![Latent Traversal](plots/latent_traversal.png)",
            f"",
            f"---",
            f"",
        ]
        return lines

    def _section_physics(self, report: GANQualityReport) -> list:
        p = report.physics_consistency
        b = p.bounds_result
        f = p.forward_result
        status = _STATUS_PASS if p.passed else _STATUS_FAIL
        lines = [
            f"## 4. Physics Consistency — {status}",
            f"",
            f"**Description:** Validates that generated profiles are physically plausible by:",
            f"1. Checking material property bounds (σ > 0, μ ≥ 1)",
            f"2. Running generated profiles through the Dodd-Deeds forward solver F(G(z))",
            f"   and verifying the impedance response is finite and comparable to real data.",
            f"",
            f"### Bounds Check",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Samples checked | {b.n_samples_checked} |",
            f"| σ in bounds ratio | {b.sigma_in_bounds_ratio:.4f} ({b.sigma_in_bounds_ratio*100:.1f}%) |",
            f"| μ in bounds ratio | {b.mu_in_bounds_ratio:.4f} ({b.mu_in_bounds_ratio*100:.1f}%) |",
            f"| σ positive ratio | {b.sigma_positive_ratio:.4f} |",
            f"| μ valid (≥1) ratio | {b.mu_valid_ratio:.4f} |",
            f"",
            f"### Forward Model Consistency",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Samples tested | {f.n_samples_tested} |",
            f"| Valid responses | {f.n_valid_responses} |",
            f"| NaN responses | {f.n_nan_responses} |",
            f"| Inf responses | {f.n_inf_responses} |",
            f"| Mean |Z| | {f.mean_amplitude:.6e} |",
            f"| Std |Z| | {f.std_amplitude:.6e} |",
            f"| Impedance real range | [{f.impedance_real_range[0]:.6e}, {f.impedance_real_range[1]:.6e}] |",
            f"| Impedance imag range | [{f.impedance_imag_range[0]:.6e}, {f.impedance_imag_range[1]:.6e}] |",
        ]

        if f.reference_amplitude is not None:
            lines.append(f"| Reference |Z| (real data) | {f.reference_amplitude:.6e} |")
        if f.amplitude_relative_error is not None:
            lines.append(f"| Amplitude relative error | {f.amplitude_relative_error:.4f} ({f.amplitude_relative_error*100:.2f}%) |")

        lines.extend([
            f"",
            f"**Interpretation:**",
            f"- All forward responses should be finite (no NaN/Inf)",
            f"- Generated impedance amplitude should be in same order of magnitude as real data",
            f"",
            f"---",
            f"",
        ])
        return lines

    def _section_robustness(self, report: GANQualityReport) -> list:
        r = report.noise_robustness
        status = _STATUS_PASS if r.passed else _STATUS_FAIL
        lines = [
            f"## 5. Noise Robustness — {status}",
            f"",
            f"**Description:** Tests generator stability by injecting small perturbations",
            f"z' = z + ε where ε ~ N(0, σ²I). A robust generator satisfies:",
            f"||G(z') - G(z)|| ≤ C·||ε|| (bounded Lipschitz constant).",
            f"",
            f"| Noise Level (σ) | Mean Δ Output | Max Δ Output |",
            f"|-----------------|---------------|--------------|",
        ]

        for sigma, mean_c, max_c in zip(r.noise_levels, r.mean_output_changes, r.max_output_changes):
            lines.append(f"| {sigma:.3f} | {mean_c:.6f} | {max_c:.6f} |")

        lines.extend([
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean Lipschitz estimate | {r.mean_lipschitz:.4f} |",
            f"| Robust (Lipschitz < 10) | {'Yes' if r.is_robust else 'No'} |",
            f"",
            f"**Interpretation:**",
            f"- Lipschitz constant > 10 → small latent noise causes large output distortion",
            f"- Output change should scale approximately linearly with noise level",
            f"",
            f"![Noise Robustness](plots/noise_robustness.png)",
            f"",
        ])
        return lines
