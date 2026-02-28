"""
GAN Experiment 02 — Multi-sample sweep.

Runs the GAN inverse solver on N randomly chosen training samples
and reports aggregate statistics: pass rate, median |ΔZ| error,
σ/μ RMSE distribution.

Run from project root:
    python experiments/inverse_solver/gan/experiment_02_multi_sample.py
    python experiments/inverse_solver/gan/experiment_02_multi_sample.py --n_samples 20 --n_restarts 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

from eddy_current_workflow.forward.edc_solver import edc_forward
from base import GANInverseExperiment, GANInverseResult, PROBE, LAYER_THICKNESS, RESULTS_DIR

DATA_DIR = Path(__file__).parents[3] / "data" / "training"


class MultiSampleExperiment(GANInverseExperiment):
    """
    Sweep the GAN inverse solver over multiple training samples.

    Reports aggregate statistics to understand overall solver performance
    across the training data distribution.
    """

    name = "gan_exp02_multi_sample"
    description = "Multi-sample performance sweep"

    def __init__(self, sample_indices: list, **kwargs):
        self.sample_indices = sample_indices
        super().__init__(**kwargs)

    def load_target(self) -> tuple:
        raise NotImplementedError("Use run_sweep() instead of run() for this experiment.")

    def explain(self) -> None:
        print("\n" + "=" * 60)
        print("EXPLANATION")
        print("=" * 60)
        print("""
What this experiment measures:
  Runs the GAN inverse solver on multiple real training samples
  and collects aggregate statistics.

Why aggregate statistics matter:
  A single sample result (Exp 01) may be lucky or unlucky.
  Running on N samples gives a reliable estimate of:
    - Pass rate: fraction of samples where |ΔZ| < 1e-5
    - Median impedance error
    - Profile RMSE distribution

Expected results:
  - Pass rate depends on how well the GAN covers the data manifold
  - If GAN is well-trained: high pass rate (>70%)
  - Samples near GAN mode collapse regions may fail
  - Profile RMSE is expected to be nonzero (ill-posed problem)
""")

    def _load_dataset(self) -> tuple:
        return (
            np.load(DATA_DIR / "sigma_layers.npy"),
            np.load(DATA_DIR / "mu_layers.npy"),
        )

    def _load_sample(self, idx: int, sigma_all: np.ndarray, mu_all: np.ndarray) -> tuple:
        sigma_true = sigma_all[idx]
        mu_true = mu_all[idx]
        edc_target = edc_forward(sigma_true, mu_true, PROBE, LAYER_THICKNESS, n_quad=self.n_quad_verify)
        return sigma_true, mu_true, edc_target

    def run_sweep(self) -> list:
        print("=" * 60)
        print(f"{self.name} — {self.description}")
        print(f"  Samples: {len(self.sample_indices)}")
        print(f"  Restarts per sample: {self.n_restarts}")
        print(f"  Model: improved_wgan_v2 | nz=32 | K=51")
        print("=" * 60)
        print(f"\nDevice: {self.device}")
        print(f"Generator: loaded from {RESULTS_DIR.parent.parent / 'results' / 'improved_wgan_v2_nz32_20260214_140817'}")

        all_results = []
        sigma_rmses = []
        mu_rmses = []

        sigma_all, mu_all = self._load_dataset()

        for i, sample_idx in enumerate(self.sample_indices):
            print(f"\n[{i+1}/{len(self.sample_indices)}] Sample {sample_idx}")
            sigma_true, mu_true, edc_target = self._load_sample(sample_idx, sigma_all, mu_all)

            result = self.optimize(edc_target, verbose=False)
            all_results.append((sample_idx, sigma_true, mu_true, result))

            s_rmse = float(np.sqrt(np.mean((sigma_true - result.sigma_rec) ** 2)))
            m_rmse = float(np.sqrt(np.mean((mu_true - result.mu_rec) ** 2)))
            sigma_rmses.append(s_rmse)
            mu_rmses.append(m_rmse)

            status = "PASS ✓" if result.passed else "FAIL ✗"
            print(f"  {status}  |ΔZ|={result.best_mismatch:.3e}  "
                  f"σ_rmse={s_rmse:.3e}  μ_rmse={m_rmse:.3e}  ({result.elapsed:.1f}s)")

        self._report_aggregate(all_results, sigma_rmses, mu_rmses)
        self.explain()
        self._visualize_sweep(all_results, sigma_rmses, mu_rmses)
        self._save_sweep_json(all_results, sigma_rmses, mu_rmses)
        return all_results

    def _report_aggregate(self, all_results, sigma_rmses, mu_rmses) -> None:
        mismatches = [r.best_mismatch for _, _, _, r in all_results]
        n_pass = sum(1 for _, _, _, r in all_results if r.passed)
        n_total = len(all_results)

        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(f"  Samples tested:    {n_total}")
        print(f"  Pass rate:         {n_pass}/{n_total} ({n_pass/n_total*100:.1f}%)")
        print(f"  Median |ΔZ| error: {float(np.median(mismatches)):.3e} Ω")
        print(f"  Best  |ΔZ| error:  {float(np.min(mismatches)):.3e} Ω")
        print(f"  Worst |ΔZ| error:  {float(np.max(mismatches)):.3e} Ω")
        print(f"  Median σ RMSE:     {float(np.median(sigma_rmses)):.3e} S/m")
        print(f"  Median μ RMSE:     {float(np.median(mu_rmses)):.3e}")

    def _save_sweep_json(self, all_results, sigma_rmses, mu_rmses) -> None:
        import json
        mismatches = [r.best_mismatch for _, _, _, r in all_results]
        n_pass = sum(1 for m in mismatches if m < self.impedance_tolerance)
        n_total = len(all_results)

        per_sample = []
        for (idx, sigma_true, mu_true, result), s_rmse, m_rmse in zip(all_results, sigma_rmses, mu_rmses):
            per_sample.append({
                "sample_idx": idx,
                "passed": result.passed,
                "best_mismatch_ohm": float(result.best_mismatch),
                "all_mismatches_ohm": [float(m) for m in result.all_mismatches],
                "sigma_rmse": s_rmse,
                "sigma_rmse_pct": float(s_rmse / float(np.mean(sigma_true)) * 100),
                "mu_rmse": m_rmse,
                "mu_rmse_pct": float(m_rmse / float(np.mean(mu_true)) * 100),
                "elapsed_s": float(result.elapsed),
            })

        record = {
            "experiment": self.name,
            "description": self.description,
            "timestamp": self._run_timestamp,
            "config": {
                "n_restarts": self.n_restarts,
                "n_iter": self.n_iter,
                "seed": self.seed,
                "fd_epsilon": self.fd_epsilon,
                "n_quad_opt": self.n_quad_opt,
                "n_quad_refine": self.n_quad_refine,
                "n_quad_verify": self.n_quad_verify,
                "impedance_tolerance": self.impedance_tolerance,
                "sample_indices": self.sample_indices,
            },
            "aggregate": {
                "n_samples": n_total,
                "n_passed": n_pass,
                "pass_rate_pct": round(n_pass / n_total * 100, 1),
                "median_mismatch_ohm": float(np.median(mismatches)),
                "best_mismatch_ohm": float(np.min(mismatches)),
                "worst_mismatch_ohm": float(np.max(mismatches)),
                "median_sigma_rmse": float(np.median(sigma_rmses)),
                "median_mu_rmse": float(np.median(mu_rmses)),
            },
            "per_sample": per_sample,
        }

        path = self.run_dir / "results.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  Saved: {path}")

    def _visualize_sweep(self, all_results, sigma_rmses, mu_rmses) -> None:
        import matplotlib.pyplot as plt

        save_dir = self.run_dir

        mismatches = [r.best_mismatch for _, _, _, r in all_results]
        sample_ids = [idx for idx, _, _, _ in all_results]
        n_pass = sum(1 for m in mismatches if m < self.impedance_tolerance)
        n_total = len(mismatches)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        colors = ["#2E86AB" if m < self.impedance_tolerance else "#C73E1D" for m in mismatches]
        axes[0].bar(range(n_total), mismatches, color=colors, alpha=0.75, edgecolor="black", linewidth=0.8)
        axes[0].axhline(self.impedance_tolerance, color="green", linestyle="--", linewidth=2,
                        label=f"Threshold {self.impedance_tolerance:.0e}")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Sample", fontweight="bold")
        axes[0].set_ylabel("|ΔZ error| (Ω)", fontweight="bold")
        axes[0].set_title(f"|ΔZ| Error per Sample\n{n_pass}/{n_total} passed", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, linestyle=":")

        axes[1].hist(np.log10(np.array(sigma_rmses)), bins=10, color="#2E86AB", alpha=0.75, edgecolor="black")
        axes[1].set_xlabel("log₁₀(σ RMSE [S/m])", fontweight="bold")
        axes[1].set_ylabel("Count", fontweight="bold")
        axes[1].set_title("Conductivity RMSE Distribution", fontweight="bold")
        axes[1].grid(True, alpha=0.3, linestyle=":")

        axes[2].hist(np.log10(np.array(mu_rmses)), bins=10, color="#A23B72", alpha=0.75, edgecolor="black")
        axes[2].set_xlabel("log₁₀(μ RMSE)", fontweight="bold")
        axes[2].set_ylabel("Count", fontweight="bold")
        axes[2].set_title("Permeability RMSE Distribution", fontweight="bold")
        axes[2].grid(True, alpha=0.3, linestyle=":")

        fig.suptitle(
            f"GAN Inverse — Multi-Sample Sweep ({n_total} samples, {self.n_restarts} restarts each)",
            fontsize=13, fontweight="bold",
        )
        fig.text(0.5, 0.01, "Model: improved_wgan_v2 (nz=32) | Optimizer: scipy L-BFGS-B on z ∈ ℝ³²",
                 ha="center", fontsize=8, style="italic", color="gray")
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])

        save_path = save_dir / f"{self.name}_sweep.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()

        best_idx = int(np.argmin(mismatches))
        _, sigma_true, mu_true, best_result = all_results[best_idx]
        from visualize import plot_profile_comparison, plot_sigma_comparison, plot_mu_comparison
        plot_profile_comparison(
            sigma_true, mu_true, best_result.sigma_rec, best_result.mu_rec, LAYER_THICKNESS,
            save_path=save_dir / f"{self.name}_best_profiles.png",
            title=f"{self.name} — Best Sample (idx={sample_ids[best_idx]})",
        )
        plot_sigma_comparison(
            sigma_true, best_result.sigma_rec, LAYER_THICKNESS,
            save_path=save_dir / f"{self.name}_best_sigma.png",
            title=f"{self.name} — Best Conductivity σ (sample {sample_ids[best_idx]})",
        )
        plot_mu_comparison(
            mu_true, best_result.mu_rec, LAYER_THICKNESS,
            save_path=save_dir / f"{self.name}_best_mu.png",
            title=f"{self.name} — Best Permeability μ (sample {sample_ids[best_idx]})",
        )
        print(f"\nPlots saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--n_restarts", type=int, default=8, help="Restarts per sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sample selection")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(2000, size=args.n_samples, replace=False).tolist()
    print(f"Testing samples: {indices}")

    exp = MultiSampleExperiment(sample_indices=indices, n_restarts=args.n_restarts)
    exp.run_sweep()
