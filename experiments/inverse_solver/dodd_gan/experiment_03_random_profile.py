"""
Dodd-GAN Experiment 03 — Random profile recovery.

Uses dodd_analytical_model (VectorPotentialInsideCoilGreenFunction) as the
forward solver and improved_wgan_v2_nz32 as the inverse solver.

Constructs a random (σ, μ) profile by sampling uniformly within the
training distribution bounds, with no assumed shape (no sigmoid).
This is the hardest case: the profile may not lie in the GAN manifold.

Run from the project root:
    python experiments/inverse_solver/dodd_gan/experiment_03_random_profile.py
    python experiments/inverse_solver/dodd_gan/experiment_03_random_profile.py --seed 7 --n_restarts 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3]))

from base import GANDoddExperiment, PROBE, LAYER_THICKNESS, K
from dodd_forward import dodd_forward


class RandomProfileExperiment(GANDoddExperiment):
    """
    Recovery of a fully random (σ, μ) profile.

    σ and μ per layer are sampled independently and uniformly within
    the training distribution bounds. No spatial smoothness is enforced,
    so the profile is unlikely to lie on the GAN manifold.

    This tests the worst-case behaviour: how well can the GAN latent-space
    optimizer approximate an arbitrary measurement with the nearest profile
    it can generate?

    Forward: dodd_analytical_model (VectorPotentialInsideCoilGreenFunction)
    Inverse: improved_wgan_v2_nz32, L-BFGS-B on z ∈ ℝ³²
    """

    description = "Random profile recovery (dodd_analytical_model forward)"

    def __init__(self, profile_seed: int = 0, **kwargs):
        self.profile_seed = profile_seed
        super().__init__(**kwargs)
        self.name = f"dodd_gan_exp03_random_seed{profile_seed:04d}"

    def load_target(self) -> tuple:
        n = self.norm
        rng = np.random.default_rng(self.profile_seed)

        sigma_true = rng.uniform(n["sigma_min"], n["sigma_max"], size=K)
        mu_true = rng.uniform(n["mu_min"], n["mu_max"], size=K)

        print(f"\nRandom profile (seed={self.profile_seed}):")
        print(f"  σ ∈ [{sigma_true.min():.3e}, {sigma_true.max():.3e}] S/m")
        print(f"  μ ∈ [{mu_true.min():.3f}, {mu_true.max():.3f}]")

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
Target: fully random (σ, μ) profile, seed={self.profile_seed}
  - Each of K={K} layers sampled independently within training bounds
  - No spatial structure assumed — profile likely off the GAN manifold
  - Hardest test: inverse solver must find the nearest representable profile

Forward solver: dodd_analytical_model (VectorPotentialInsideCoilGreenFunction, ORNL-5220)
  - N-layer cylindrical geometry, adaptive Gauss integration
  - integ_range_opt={self.integ_range_opt} (optimization), {self.integ_range_verify} (verification)

Inverse solver: improved_wgan_v2_nz32
  - Generator G: z ∈ ℝ³² → (σ, μᵣ) ∈ ℝ⁵¹×ℝ⁵¹
  - Optimizer: scipy L-BFGS-B, {self.n_restarts} restarts

Expected behaviour:
  - Voltage mismatch likely larger than for structured profiles
  - Profile RMSE will be significant — the GAN cannot represent arbitrary noise
  - The recovered profile will be the best smooth approximation on the GAN manifold
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dodd-GAN inverse solver: random profile recovery"
    )
    parser.add_argument("--profile_seed", type=int, default=0, help="Seed for random profile generation")
    parser.add_argument("--n_restarts", type=int, default=10, help="Number of latent restarts")
    parser.add_argument("--integ_range_opt", type=int, default=20)
    parser.add_argument("--integ_range_verify", type=int, default=50)
    args = parser.parse_args()

    exp = RandomProfileExperiment(
        profile_seed=args.profile_seed,
        n_restarts=args.n_restarts,
        integ_range_opt=args.integ_range_opt,
        integ_range_verify=args.integ_range_verify,
    )
    exp.run()
