"""
GAN Experiment 03 — Noise robustness.

Tests how well the GAN inverse solver handles noisy impedance measurements.
At each noise level, the solver must recover a profile from a corrupted ΔZ.

Unlike classical experiments, the GAN prior acts as an implicit denoiser:
the latent space forces all solutions to be physically plausible profiles.

Run from project root:
    python experiments/inverse_solver/gan/experiment_03_noise_robustness.py
    python experiments/inverse_solver/gan/experiment_03_noise_robustness.py --sample 5 --n_restarts 8
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

from eddy_current_workflow.forward.edc_solver import edc_forward, EDCResponse
from base import GANInverseExperiment, GANInverseResult, PROBE, LAYER_THICKNESS, RESULTS_DIR

DATA_DIR = Path(__file__).parents[3] / "data" / "training"

NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]


def add_noise(edc: EDCResponse, noise_frac: float, rng: np.random.Generator) -> EDCResponse:
    """Add complex Gaussian noise as a fraction of |ΔZ|."""
    if noise_frac == 0.0:
        return edc
    magnitude = abs(edc.impedance_complex)
    noise_std = noise_frac * magnitude
    noisy_real = edc.impedance_real + rng.normal(0, noise_std)
    noisy_imag = edc.impedance_imag + rng.normal(0, noise_std)

    return replace(edc, impedance_real=noisy_real, impedance_imag=noisy_imag)


class NoiseRobustnessExperiment(GANInverseExperiment):
    """
    Test GAN inverse solver robustness to measurement noise.

    For each noise level σ_noise/|ΔZ|:
      1. Add complex Gaussian noise to the target impedance
      2. Run GAN latent-space optimization on the noisy measurement
      3. Report |ΔZ| error and profile RMSE
    """

    description = "Noise robustness with GAN prior"

    def __init__(self, sample_idx: int = 0, noise_levels: list = None, **kwargs):
        self.sample_idx = sample_idx
        self.noise_levels = noise_levels or NOISE_LEVELS
        super().__init__(**kwargs)
        self.name = f"gan_exp03_noise_sample{sample_idx:04d}"

    def load_target(self) -> tuple:
        sigma_all = np.load(DATA_DIR / "sigma_layers.npy")
        mu_all = np.load(DATA_DIR / "mu_layers.npy")
        sigma_true = sigma_all[self.sample_idx]
        mu_true = mu_all[self.sample_idx]
        edc_clean = edc_forward(sigma_true, mu_true, PROBE, LAYER_THICKNESS, n_quad=self.n_quad_verify)
        return sigma_true, mu_true, edc_clean

    def explain(self) -> None:
        print("\n" + "=" * 60)
        print("EXPLANATION")
        print("=" * 60)
        print("""
What this experiment tests:
  How does GAN-based inverse solver perform as measurement noise increases?

Classical inverse (exp 01-05) vs GAN inverse:
  Classical: explicit smoothness regularization fights noise
  GAN prior: all solutions stay on the GAN manifold (implicit denoiser)

  The GAN prior is a stronger constraint than smoothness regularization
  because it learned the actual distribution of physical profiles.

Expected behavior:
  - At 0% noise: best impedance match, good profile recovery
  - At 1-5% noise: slight degradation, but GAN prior stabilizes solution
  - At >5% noise: error grows but profile remains physically plausible

Key difference from classical:
  The GAN cannot generate unphysical profiles regardless of noise level.
  Classical solvers can drift to unphysical regions under heavy noise.
""")

    def run_noise_sweep(self) -> list:
        print("=" * 60)
        print(f"{self.name} — {self.description}")
        print(f"  Sample: {self.sample_idx}  |  Noise levels: {self.noise_levels}")
        print(f"  Restarts per level: {self.n_restarts}")
        print("=" * 60)

        sigma_true, mu_true, edc_clean = self.load_target()
        noise_rng = np.random.default_rng(99)
        sweep_results = []

        for noise_frac in self.noise_levels:
            edc_noisy = add_noise(edc_clean, noise_frac, noise_rng)
            result = self.optimize(edc_noisy, verbose=False)

            s_rmse = float(np.sqrt(np.mean((sigma_true - result.sigma_rec) ** 2)))
            m_rmse = float(np.sqrt(np.mean((mu_true - result.mu_rec) ** 2)))
            sweep_results.append((noise_frac, result, s_rmse, m_rmse))

            status = "PASS ✓" if result.passed else "FAIL ✗"
            print(f"  noise={noise_frac*100:.1f}%  {status}  "
                  f"|ΔZ|={result.best_mismatch:.3e}  "
                  f"σ_rmse={s_rmse:.3e}  μ_rmse={m_rmse:.3e}")

        self.explain()
        self._visualize_noise_sweep(sigma_true, mu_true, sweep_results)
        self._save_noise_json(sigma_true, mu_true, sweep_results)
        return sweep_results

    def _save_noise_json(self, sigma_true, mu_true, sweep_results) -> None:
        import json
        levels = []
        for noise_frac, result, s_rmse, m_rmse in sweep_results:
            levels.append({
                "noise_frac": noise_frac,
                "noise_pct": round(noise_frac * 100, 2),
                "passed": result.passed,
                "best_mismatch_ohm": float(result.best_mismatch),
                "all_mismatches_ohm": [float(m) for m in result.all_mismatches],
                "sigma_rmse": s_rmse,
                "sigma_rmse_pct": float(s_rmse / float(np.mean(sigma_true)) * 100),
                "mu_rmse": m_rmse,
                "mu_rmse_pct": float(m_rmse / float(np.mean(mu_true)) * 100),
                "elapsed_s": float(result.elapsed),
            })

        mismatches = [r.best_mismatch for _, r, _, _ in sweep_results]
        record = {
            "experiment": self.name,
            "description": self.description,
            "timestamp": self._run_timestamp,
            "config": {
                "sample_idx": self.sample_idx,
                "noise_levels": self.noise_levels,
                "n_restarts": self.n_restarts,
                "fd_epsilon": self.fd_epsilon,
                "n_quad_opt": self.n_quad_opt,
                "n_quad_refine": self.n_quad_refine,
                "n_quad_verify": self.n_quad_verify,
                "impedance_tolerance": self.impedance_tolerance,
            },
            "aggregate": {
                "n_levels": len(sweep_results),
                "n_passed": sum(1 for _, r, _, _ in sweep_results if r.passed),
                "best_mismatch_ohm": float(min(mismatches)),
                "worst_mismatch_ohm": float(max(mismatches)),
                "noise_breakdown_factor": (
                    float(max(mismatches) / min(mismatches)) if min(mismatches) > 0 else None
                ),
            },
            "levels": levels,
        }

        path = self.run_dir / "results.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  Saved: {path}")

    def _visualize_noise_sweep(self, sigma_true, mu_true, sweep_results) -> None:
        import matplotlib.pyplot as plt
        from visualize import plot_sigma_comparison, plot_mu_comparison

        save_dir = self.run_dir

        noise_pcts = [n * 100 for n, _, _, _ in sweep_results]
        mismatches = [r.best_mismatch for _, r, _, _ in sweep_results]
        sigma_rmses = [s for _, _, s, _ in sweep_results]
        mu_rmses = [m for _, _, _, m in sweep_results]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].semilogy(noise_pcts, mismatches, "o-", color="#2E86AB", linewidth=2.5, markersize=9)
        axes[0].axhline(self.impedance_tolerance, color="green", linestyle="--", linewidth=2,
                        label=f"Threshold {self.impedance_tolerance:.0e}")
        axes[0].set_xlabel("Noise level (%)", fontweight="bold")
        axes[0].set_ylabel("|ΔZ error| (Ω)", fontweight="bold")
        axes[0].set_title("|ΔZ| Error vs Noise", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, linestyle=":")

        axes[1].semilogy(noise_pcts, sigma_rmses, "s-", color="#A23B72", linewidth=2.5, markersize=9)
        axes[1].set_xlabel("Noise level (%)", fontweight="bold")
        axes[1].set_ylabel("σ RMSE (S/m)", fontweight="bold")
        axes[1].set_title("Conductivity RMSE vs Noise", fontweight="bold")
        axes[1].grid(True, alpha=0.3, linestyle=":")

        axes[2].semilogy(noise_pcts, mu_rmses, "^-", color="#F18F01", linewidth=2.5, markersize=9)
        axes[2].set_xlabel("Noise level (%)", fontweight="bold")
        axes[2].set_ylabel("μ RMSE", fontweight="bold")
        axes[2].set_title("Permeability RMSE vs Noise", fontweight="bold")
        axes[2].grid(True, alpha=0.3, linestyle=":")

        fig.suptitle(
            f"GAN Inverse — Noise Robustness (sample {self.sample_idx}, {self.n_restarts} restarts)",
            fontsize=13, fontweight="bold",
        )
        fig.text(0.5, 0.01, "Model: improved_wgan_v2 (nz=32) | GAN prior acts as implicit denoiser",
                 ha="center", fontsize=8, style="italic", color="gray")
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])

        save_path = save_dir / f"{self.name}_noise_sweep.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()

        best_result = min(sweep_results, key=lambda x: x[1].best_mismatch)[1]
        plot_sigma_comparison(
            sigma_true, best_result.sigma_rec, LAYER_THICKNESS,
            save_path=save_dir / f"{self.name}_best_sigma.png",
            title=f"{self.name} — Best Conductivity σ (0% noise)",
        )
        plot_mu_comparison(
            mu_true, best_result.mu_rec, LAYER_THICKNESS,
            save_path=save_dir / f"{self.name}_best_mu.png",
            title=f"{self.name} — Best Permeability μ (0% noise)",
        )
        print(f"\nPlots saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Training sample index (0-1999)")
    parser.add_argument("--n_restarts", type=int, default=8, help="Restarts per noise level")
    args = parser.parse_args()

    exp = NoiseRobustnessExperiment(sample_idx=args.sample, n_restarts=args.n_restarts)
    exp.run_noise_sweep()
