"""
Run all inverse solver experiments with visualizations.

This script runs all three experiments sequentially and generates
comprehensive plots for each one, saved to the results/ folder.

Run from project root:
    python experiments/inverse_solver/run_with_visualizations.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import experiment_01_homogeneous as exp01
import experiment_02_multifreq_graded as exp02
import experiment_03_noise_robustness as exp03
import experiment_04_sigmoid_realistic as exp04
import experiment_05_real_training_data as exp05


def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("INVERSE SOLVER EXPERIMENTS WITH VISUALIZATIONS")
    print("=" * 70)
    print(f"\nResults will be saved to: {results_dir}")
    print("=" * 70 + "\n")

    experiments = [
        ("Experiment 01 — Homogeneous Recovery", exp01.run),
        ("Experiment 02 — Multi-Frequency Graded Recovery", exp02.run),
        ("Experiment 03 — Noise Robustness", exp03.run),
        ("Experiment 04 — Sigmoid Profile Recovery (GAN-realistic)", exp04.run),
        ("Experiment 05 — Real Training Data", exp05.run),
    ]

    results = []
    timings = []

    for name, fn in experiments:
        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print("=" * 70)
        
        t0 = time.perf_counter()
        try:
            result = fn()
            elapsed = time.perf_counter() - t0
            results.append((name, result, None))
            timings.append((name, elapsed))
            print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results.append((name, None, str(exc)))
            timings.append((name, elapsed))
            print(f"\n[FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r, e in results if e is None)
    total = len(results)
    
    print(f"\nExperiments: {passed}/{total} passed\n")
    
    for name, elapsed in timings:
        status = "✓" if any(n == name and e is None for n, _, e in results) else "✗"
        print(f"  {status} {name:<50} {elapsed:>6.1f}s")
    
    print(f"\nAll visualizations saved to: {results_dir}/")
    print("\nGenerated plots:")
    plot_files = sorted(results_dir.glob("*.png"))
    for pf in plot_files:
        print(f"  - {pf.name}")
    
    if passed == total:
        print("\n" + "=" * 70)
        print("ALL EXPERIMENTS PASSED ✓")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"SOME EXPERIMENTS FAILED ({total - passed}/{total})")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
