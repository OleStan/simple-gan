"""
Dodd-GAN Experiment 02 — Synthetic profile recovery.

Uses dodd_analytical_model (VectorPotentialInsideCoilGreenFunction, ORNL-5220)
as the forward solver and improved_wgan_v2_nz32 as the inverse solver.

Constructs a synthetic graded profile (sigmoid transition in σ and μ)
and tests whether the GAN inverse solver can recover it from the
impedance computed by the full Dodd-Deeds analytical model.

Run from experiments/inverse_solver/dodd_gan/:
    python experiment_02_synthetic_profile.py
    python experiment_02_synthetic_profile.py --n_restarts 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3]))

from base import GANDoddExperiment, PROBE, LAYER_THICKNESS, K
from dodd_forward import dodd_forward

NORM_PARAMS = {
    "sigma_min": 18810564.546982836,
    "sigma_max": 40430204.630051635,
    "mu_min": 1.0046969416375982,
    "mu_max": 9.45226732538497,
}


def make_sigmoid_profile(
    sigma_surface: float,
    sigma_deep: float,
    mu_surface: float,
    mu_deep: float,
    k: int = K,
    steepness: float = 10.0,
) -> tuple:
    """
    Construct a sigmoid-graded (σ, μ) profile over K layers.

    The transition is centred at mid-depth; steepness controls how sharp it is.
    """
    x = np.linspace(-1, 1, k)
    weight = 1.0 / (1.0 + np.exp(-steepness * x))

    sigma = sigma_surface + (sigma_deep - sigma_surface) * weight
    mu = mu_surface + (mu_deep - mu_surface) * weight
    return sigma, mu


class SyntheticProfileExperiment(GANDoddExperiment):
    """
    Recovery of a synthetic graded profile.

    The profile is constructed as a sigmoid transition between
    high-σ/low-μ (surface) and low-σ/high-μ (deep) regions,
    consistent with the training distribution of improved_wgan_v2.

    Forward: dodd_analytical_model (VectorPotentialInsideCoilGreenFunction)
    Inverse: improved_wgan_v2_nz32, L-BFGS-B on z ∈ ℝ³²
    """

    name = "dodd_gan_exp02_synthetic"
    description = "Synthetic sigmoid profile recovery (dodd_analytical_model forward)"

    def __init__(self, steepness: float = 10.0, **kwargs):
        self.steepness = steepness
        super().__init__(**kwargs)

    def load_target(self) -> tuple:
        sigma_surface = NORM_PARAMS["sigma_max"] * 0.92
        sigma_deep = NORM_PARAMS["sigma_min"] * 1.05
        mu_surface = NORM_PARAMS["mu_min"] * 1.01
        mu_deep = NORM_PARAMS["mu_max"] * 0.75

        sigma_true, mu_true = make_sigmoid_profile(
            sigma_surface, sigma_deep,
            mu_surface, mu_deep,
            steepness=self.steepness,
        )

        print(f"\nSynthetic sigmoid profile:")
        print(f"  σ: {sigma_surface:.3e} → {sigma_deep:.3e} S/m")
        print(f"  μ: {mu_surface:.3f} → {mu_deep:.3f}")
        print(f"  Steepness: {self.steepness}")

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
Target: synthetic sigmoid profile (not from training data)
  - Sigmoid transition σ_high→σ_low, μ_low→μ_high
  - Steepness={self.steepness}, K={K} layers
  - Within the training distribution of improved_wgan_v2

Forward solver: dodd_analytical_model (VectorPotentialInsideCoilGreenFunction, ORNL-5220)
  - Full Dodd-Deeds analytical solution, N-layer cylindrical geometry
  - integ_range_opt={self.integ_range_opt} (optimization), {self.integ_range_verify} (verification)

Inverse solver: improved_wgan_v2_nz32
  - Generator G: z ∈ ℝ³² → (σ, μᵣ) ∈ ℝ⁵¹×ℝ⁵¹
  - Optimizer: scipy L-BFGS-B, {self.n_restarts} restarts

This experiment tests whether a physically meaningful profile outside
the training set can still be recovered via the GAN prior, using
the full analytical forward model as the measurement oracle.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dodd-GAN inverse solver: synthetic profile recovery"
    )
    parser.add_argument("--n_restarts", type=int, default=10, help="Number of latent restarts")
    parser.add_argument("--steepness", type=float, default=10.0, help="Sigmoid steepness")
    parser.add_argument("--integ_range_opt", type=int, default=20, help="Integration range during optimization")
    parser.add_argument("--integ_range_verify", type=int, default=50, help="Integration range for verification")
    args = parser.parse_args()

    exp = SyntheticProfileExperiment(
        steepness=args.steepness,
        n_restarts=args.n_restarts,
        integ_range_opt=args.integ_range_opt,
        integ_range_verify=args.integ_range_verify,
    )
    exp.run()
