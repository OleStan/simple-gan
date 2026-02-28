"""
GAN Experiment 01 — Single training-data sample recovery.

Loads one profile from the training dataset and tests how well the GAN
latent-space optimizer can reproduce its impedance measurement.

Run from project root:
    python experiments/inverse_solver/gan/experiment_01_single_sample.py
    python experiments/inverse_solver/gan/experiment_01_single_sample.py --sample 42
    python experiments/inverse_solver/gan/experiment_01_single_sample.py --sample 42 --n_restarts 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

from eddy_current_workflow.forward.edc_solver import edc_forward
from base import GANInverseExperiment, PROBE, LAYER_THICKNESS

DATA_DIR = Path(__file__).parents[3] / "data" / "training"


class SingleSampleExperiment(GANInverseExperiment):
    """
    Recover a single profile from the training dataset.

    Target: one (sigma, mu) pair from data/training/sigma_layers.npy
    Question: can the GAN latent space reproduce the measured impedance?
    """

    description = "Single training-data sample recovery"

    def __init__(self, sample_idx: int = 0, **kwargs):
        self.sample_idx = sample_idx
        super().__init__(**kwargs)
        self.name = f"gan_exp01_sample{sample_idx:04d}"

    def load_target(self) -> tuple:
        sigma_all = np.load(DATA_DIR / "sigma_layers.npy")
        mu_all = np.load(DATA_DIR / "mu_layers.npy")

        sigma_true = sigma_all[self.sample_idx]
        mu_true = mu_all[self.sample_idx]

        print(f"\nLoaded sample {self.sample_idx} from training data "
              f"({sigma_all.shape[0]} total samples)")

        edc_target = edc_forward(sigma_true, mu_true, PROBE, LAYER_THICKNESS, n_quad=self.n_quad_verify)
        return sigma_true, mu_true, edc_target

    def explain(self) -> None:
        print("\n" + "=" * 60)
        print("EXPLANATION")
        print("=" * 60)
        print(f"""
Target: real training data sample {self.sample_idx}
  - Sigmoid profile, K=51 layers
  - Opposite σ/μ relationship (σ↑ → μ↓)
  - Fixed boundaries from Table 3.5

What we test:
  Does the GAN latent space contain a profile that produces
  the same impedance as this real training sample?

Why this is the right approach:
  Classical inverse (exp 01-05): 102 unknowns, explicit smoothness
  GAN inverse:                    32 unknowns, implicit GAN prior

  The GAN was trained on exactly this type of profile, so z* should
  exist in ℝ³² such that G(z*) ≈ the true profile.

Expected results:
  - If impedance matched (|ΔZ| < 1e-5): PASS
    The GAN can represent profiles with this impedance signature.
  - Profile RMSE may still be nonzero: the inverse problem is
    ill-posed — multiple profiles share the same ΔZ.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Training sample index (0-1999)")
    parser.add_argument("--n_restarts", type=int, default=10, help="Number of latent restarts")
    args = parser.parse_args()

    exp = SingleSampleExperiment(sample_idx=args.sample, n_restarts=args.n_restarts)
    exp.run()
