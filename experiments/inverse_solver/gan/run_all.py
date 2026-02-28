"""
Run all GAN inverse solver experiments.

Run from project root:
    python experiments/inverse_solver/gan/run_all.py
    python experiments/inverse_solver/gan/run_all.py --sample 5 --n_restarts 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_01_single_sample import SingleSampleExperiment
from experiment_02_multi_sample import MultiSampleExperiment
from experiment_03_noise_robustness import NoiseRobustnessExperiment


def main():
    parser = argparse.ArgumentParser(description="Run all GAN inverse solver experiments")
    parser.add_argument("--sample", type=int, default=0, help="Primary sample index (0-1999)")
    parser.add_argument("--n_restarts", type=int, default=8, help="Restarts per experiment")
    parser.add_argument("--n_sweep_samples", type=int, default=5, help="Samples for multi-sample sweep")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GAN INVERSE SOLVER EXPERIMENTS")
    print("=" * 60 + "\n")

    passed = []
    failed = []

    experiments = [
        ("GAN Exp 01 — Single sample recovery", lambda: SingleSampleExperiment(
            sample_idx=args.sample, n_restarts=args.n_restarts).run()),
        ("GAN Exp 02 — Multi-sample sweep", lambda: MultiSampleExperiment(
            sample_indices=np.random.default_rng(0).choice(2000, args.n_sweep_samples, replace=False).tolist(),
            n_restarts=args.n_restarts).run_sweep()),
        ("GAN Exp 03 — Noise robustness", lambda: NoiseRobustnessExperiment(
            sample_idx=args.sample, n_restarts=args.n_restarts).run_noise_sweep()),
    ]

    for name, fn in experiments:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print("=" * 60)
        t0 = time.perf_counter()
        try:
            fn()
            elapsed = time.perf_counter() - t0
            passed.append((name, elapsed))
            print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            failed.append((name, str(exc)))
            print(f"\n[FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, elapsed in passed:
        print(f"  PASS  {name}  ({elapsed:.1f}s)")
    for name, err in failed:
        print(f"  FAIL  {name}  — {err}")
    print(f"\n{len(passed)}/{len(experiments)} experiments completed")


if __name__ == "__main__":
    main()
