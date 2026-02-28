"""
Run all dodd_gan experiments sequentially.

Run from the project root:
    python experiments/inverse_solver/dodd_gan/run_all.py
    python experiments/inverse_solver/dodd_gan/run_all.py --n_restarts 5 --sample 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3]))

from experiment_01_single_sample import SingleSampleExperiment
from experiment_02_synthetic_profile import SyntheticProfileExperiment
from experiment_03_random_profile import RandomProfileExperiment


def main():
    parser = argparse.ArgumentParser(description="Run all dodd_gan experiments")
    parser.add_argument("--sample", type=int, default=0, help="Training sample index for exp01")
    parser.add_argument("--profile_seed", type=int, default=0, help="Random profile seed for exp03")
    parser.add_argument("--n_restarts", type=int, default=10, help="Number of latent restarts")
    parser.add_argument("--integ_range_opt", type=int, default=20)
    parser.add_argument("--integ_range_verify", type=int, default=50)
    args = parser.parse_args()

    shared = dict(
        n_restarts=args.n_restarts,
        integ_range_opt=args.integ_range_opt,
        integ_range_verify=args.integ_range_verify,
    )

    experiments = [
        SingleSampleExperiment(sample_idx=args.sample, **shared),
        SyntheticProfileExperiment(**shared),
        RandomProfileExperiment(profile_seed=args.profile_seed, **shared),
    ]

    results = []
    for exp in experiments:
        print(f"\n{'#' * 60}")
        print(f"# {exp.name}")
        print(f"{'#' * 60}")
        result = exp.run()
        results.append((exp.name, result))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, res in results:
        status = "PASS ✓" if res.passed else "FAIL ✗"
        print(f"  {name:50s} {status}  |ΔZ|={res.best_mismatch:.3e} Ω  ({res.elapsed:.1f}s)")


if __name__ == "__main__":
    main()
