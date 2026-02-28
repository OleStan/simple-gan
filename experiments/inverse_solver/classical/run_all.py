"""
Run all inverse solver experiments in sequence.

Run from project root:
    python experiments/inverse_solver/classical/run_all.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))
sys.path.insert(0, str(Path(__file__).parent))

import experiment_01_homogeneous as exp01
import experiment_02_multifreq_graded as exp02
import experiment_03_noise_robustness as exp03
import experiment_04_sigmoid_realistic as exp04
import experiment_05_real_training_data as exp05

EXPERIMENTS = [
    ("01 — Homogeneous recovery", exp01.run),
    ("02 — Multi-frequency graded recovery", exp02.run),
    ("03 — Noise robustness", exp03.run),
    ("04 — Sigmoid profile recovery (GAN-realistic)", exp04.run),
    ("05 — Real training data", exp05.run),
]


RESULTS_DIR = Path(__file__).parent / "results"


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{timestamp}_classical_run_all"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("CLASSICAL INVERSE SOLVER EXPERIMENTS")
    print(f"Run directory: {run_dir}")
    print("=" * 60 + "\n")

    passed = []
    failed = []
    experiment_records = []

    for name, fn in EXPERIMENTS:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print("=" * 60)
        t0 = time.perf_counter()
        try:
            fn()
            elapsed = time.perf_counter() - t0
            passed.append((name, elapsed))
            experiment_records.append({"name": name, "status": "PASS", "elapsed_s": elapsed})
            print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            failed.append((name, str(exc)))
            experiment_records.append({"name": name, "status": "FAIL", "elapsed_s": elapsed, "error": str(exc)})
            print(f"\n[FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()

    total = len(EXPERIMENTS)
    summary = {
        "timestamp": timestamp,
        "approach": "classical",
        "total": total,
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate_pct": round(len(passed) / total * 100, 1),
        "total_elapsed_s": sum(r["elapsed_s"] for r in experiment_records),
        "experiments": experiment_records,
    }

    json_path = run_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, elapsed in passed:
        print(f"  PASS  {name}  ({elapsed:.1f}s)")
    for name, err in failed:
        print(f"  FAIL  {name}  — {err}")
    print(f"\n{len(passed)}/{total} experiments passed")
    print(f"Results saved to: {json_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
