"""
Dodd-GAN Experiment 01 — Single training-data sample recovery.

Uses dodd_analytical_model (VectorPotentialInsideCoilGreenFunction, ORNL-5220)
as the forward solver and improved_wgan_v2_nz32 as the inverse solver.

Loads one profile from the training dataset and tests how well the GAN
latent-space optimizer can reproduce its voltage measurement when the
forward model is the full Dodd-Deeds analytical model.

Run from experiments/inverse_solver/dodd_gan/:
    python experiment_01_single_sample.py
    python experiment_01_single_sample.py --sample 42
    python experiment_01_single_sample.py --sample 42 --n_restarts 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3]))

from base import GANDoddExperiment, PROBE, LAYER_THICKNESS
from dodd_forward import dodd_forward

DATA_DIR = Path(__file__).parents[3] / "data" / "training"


class SingleSampleExperiment(GANDoddExperiment):
    """
    Recover a single profile from the training dataset.

    Target: one (sigma, mu) pair from data/training/sigma_layers.npy
    Forward: dodd_analytical_model (VectorPotentialInsideCoilGreenFunction)
    Inverse: improved_wgan_v2_nz32, L-BFGS-B on z ∈ ℝ³²
    """

    description = "Single training-data sample recovery (dodd_analytical_model forward)"

    def __init__(self, sample_idx: int = 0, **kwargs):
        self.sample_idx = sample_idx
        super().__init__(**kwargs)
        self.name = f"dodd_gan_exp01_sample{sample_idx:04d}"

    def load_target(self) -> tuple:
        sigma_all = np.load(DATA_DIR / "sigma_layers.npy")
        mu_all = np.load(DATA_DIR / "mu_layers.npy")

        sigma_true = sigma_all[self.sample_idx]
        mu_true = mu_all[self.sample_idx]

        print(
            f"\nLoaded sample {self.sample_idx} from training data "
            f"({sigma_all.shape[0]} total samples)"
        )

        target_response = dodd_forward(
            sigma_true, mu_true, PROBE, LAYER_THICKNESS,
            integ_top_range=self.integ_range_verify,
        )
        return sigma_true, mu_true, target_response

    def explain(self) -> None:
        print("\n" + "=" * 60)
        print("EXPLANATION")
        print("=" * 60)
        print(f"""
Target: real training data sample {self.sample_idx}
  - K={51} layers, layer thickness = {LAYER_THICKNESS*1e6:.2f} µm
  - Profiles from data/training/sigma_layers.npy + mu_layers.npy

Forward solver: dodd_analytical_model (VectorPotentialInsideCoilGreenFunction, ORNL-5220)
  - Full Dodd-Deeds analytical solution, N-layer cylindrical geometry
  - Adaptive Gauss integration (integ_range_opt={self.integ_range_opt})
  - More physically faithful than the lightweight edc_solver

Inverse solver: improved_wgan_v2_nz32
  - Generator G: z ∈ ℝ³² → (σ, μᵣ) ∈ ℝ⁵¹×ℝ⁵¹
  - Optimizer: scipy L-BFGS-B, {self.n_restarts} restarts
  - Objective: J(z) = |ΔZ_pred - ΔZ_target|² / |ΔZ_target|²

Expected results:
  - If |ΔZ error| < 1e-5 Ω: PASS
    The GAN latent space contains a profile matching this impedance.
  - Profile RMSE may be nonzero: the inverse problem is ill-posed.
    Multiple profiles can share the same impedance signature.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dodd-GAN inverse solver: single sample recovery"
    )
    parser.add_argument("--sample", type=int, default=0, help="Training sample index (0-1999)")
    parser.add_argument("--n_restarts", type=int, default=10, help="Number of latent restarts")
    parser.add_argument("--integ_range_opt", type=int, default=20, help="Integration range during optimization")
    parser.add_argument("--integ_range_verify", type=int, default=50, help="Integration range for verification")
    args = parser.parse_args()

    exp = SingleSampleExperiment(
        sample_idx=args.sample,
        n_restarts=args.n_restarts,
        integ_range_opt=args.integ_range_opt,
        integ_range_verify=args.integ_range_verify,
    )
    exp.run()
