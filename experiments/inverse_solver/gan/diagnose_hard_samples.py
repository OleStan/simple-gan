"""
Diagnostic: Why do hard samples fail?

For each failing sample, compute:
  1. Nearest-neighbour distance in GAN latent space
     — encodes the sample via gradient descent (no prior) and checks |ΔZ|
  2. Best impedance achievable by any GAN output (random search, 5000 candidates)
     — if best random |ΔZ| >> 1e-5 → GAN coverage gap
     — if best random |ΔZ| < 1e-5 → optimizer failure (basin problem)

Run from project root:
    python experiments/inverse_solver/gan/diagnose_hard_samples.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

from eddy_current_workflow.forward.edc_solver import edc_forward
from base import GANInverseExperiment, PROBE, LAYER_THICKNESS, NZ

DATA_DIR = Path(__file__).parents[3] / "data" / "training"

HARD_SAMPLES = [1664, 1153, 1881, 111, 449, 1363]
EASY_SAMPLES = [1245, 1789, 1548, 600]
N_RANDOM = 5000
N_QUAD_CHECK = 50


class DiagnosticExperiment(GANInverseExperiment):
    """Minimal subclass just to load the GAN generator."""

    description = "Diagnostic"

    def load_target(self):
        return None, None, None

    def explain(self):
        pass


def load_sample(idx, sigma_all, mu_all, n_quad=N_QUAD_CHECK):
    sigma = sigma_all[idx]
    mu = mu_all[idx]
    edc = edc_forward(sigma, mu, PROBE, LAYER_THICKNESS, n_quad=n_quad)
    return sigma, mu, edc


def random_search_best(exp, edc_target, n_candidates=N_RANDOM, seed=0):
    """Best |ΔZ| achieved by random z sampling from N(0,1)."""
    rng = np.random.default_rng(seed)
    Z_candidates = rng.standard_normal((n_candidates, NZ)).astype(np.float32)

    best = np.inf
    for i in range(n_candidates):
        sigma, mu = exp._decode_z(Z_candidates[i])
        try:
            pred = edc_forward(sigma, mu, PROBE, LAYER_THICKNESS, n_quad=N_QUAD_CHECK)
            mm = abs(pred.impedance_complex - edc_target.impedance_complex)
            if mm < best:
                best = mm
        except Exception:
            pass
    return best


def main():
    print("Loading GAN experiment base ...")
    exp = DiagnosticExperiment(n_restarts=1)

    print("Loading training data ...")
    sigma_all = np.load(DATA_DIR / "sigma_layers.npy")
    mu_all = np.load(DATA_DIR / "mu_layers.npy")

    all_indices = HARD_SAMPLES + EASY_SAMPLES
    labels = ["HARD"] * len(HARD_SAMPLES) + ["EASY"] * len(EASY_SAMPLES)

    print(f"\nRandom search diagnostic ({N_RANDOM} candidates, n_quad={N_QUAD_CHECK})\n")
    print(f"{'Label':<6} {'Idx':>5}  {'Best random |ΔZ|':>18}  {'Diagnosis'}")
    print("-" * 62)

    results = []
    for idx, label in zip(all_indices, labels):
        sigma, mu, edc_target = load_sample(idx, sigma_all, mu_all)
        t0 = time.perf_counter()
        best_mm = random_search_best(exp, edc_target, n_candidates=N_RANDOM)
        elapsed = time.perf_counter() - t0

        if best_mm < 1e-5:
            diagnosis = "optimizer failure (GAN can represent it)"
        elif best_mm < 1e-4:
            diagnosis = "partial coverage (GAN barely reaches it)"
        else:
            diagnosis = "GAN coverage gap"

        print(f"{label:<6} {idx:>5}  {best_mm:>18.3e}  {diagnosis}  ({elapsed:.1f}s)")
        results.append({"idx": idx, "label": label, "best_random_mm": float(best_mm), "diagnosis": diagnosis})

    print("\nSummary:")
    hard_results = [r for r in results if r["label"] == "HARD"]
    n_coverage_gap = sum(1 for r in hard_results if "coverage gap" in r["diagnosis"])
    n_opt_failure = sum(1 for r in hard_results if "optimizer failure" in r["diagnosis"])
    n_partial = sum(1 for r in hard_results if "partial" in r["diagnosis"])
    print(f"  Hard samples ({len(HARD_SAMPLES)} total):")
    print(f"    GAN coverage gap:    {n_coverage_gap}")
    print(f"    Partial coverage:    {n_partial}")
    print(f"    Optimizer failures:  {n_opt_failure}")


if __name__ == "__main__":
    main()
