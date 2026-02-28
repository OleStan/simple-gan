"""
Experiment 01 — Homogeneous profile recovery (single frequency).

Smoke test: can the inverse solver recover a uniform (σ, μ) profile
from a single-frequency impedance measurement?

Expected outcome:
  - Mismatch J < 1e-8 (impedance reproduced)
  - Note: profile non-uniqueness is normal — the EDC inverse problem
    is ill-posed; many (σ, μ) can produce the same ΔZ at one frequency.

Run from project root:
    python experiments/inverse_solver/experiment_01_homogeneous.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

from eddy_current_workflow.forward.edc_solver import edc_forward, ProbeSettings
from eddy_current_workflow.inverse.recovery import RecoveryConfig, recover_profiles, round_trip_error
from visualize import create_experiment_summary, plot_decision_criteria

PROBE = ProbeSettings(
    frequency=1e6,
    inner_radius=4e-3,
    outer_radius=6e-3,
    lift_off=0.5e-3,
    coil_height=2e-3,
    n_turns=100,
)

K = 5
SIGMA_TRUE = np.full(K, 1e7)
MU_TRUE = np.full(K, 1.0)


def run():
    print("=" * 60)
    print("Experiment 01 — Homogeneous profile recovery")
    print(f"  K={K} layers | f={PROBE.frequency/1e6:.1f} MHz")
    print(f"  σ_true = {SIGMA_TRUE[0]:.2e} S/m (uniform)")
    print(f"  μ_true = {MU_TRUE[0]:.2f} (uniform)")
    print("=" * 60)

    edc_target = edc_forward(SIGMA_TRUE, MU_TRUE, PROBE, n_quad=100)
    print(f"\nTarget ΔZ = {edc_target.impedance_real:.6e} {edc_target.impedance_imag:+.6e}j Ω")

    cfg = RecoveryConfig(
        K=K,
        sigma_bounds=(1e6, 6e7),
        mu_bounds=(1.0, 50.0),
        n_quad=100,
        method="multistart",
        n_starts=8,
        max_iter=500,
        tol=1e-12,
        seed=42,
        verbose=True,
    )

    t0 = time.perf_counter()
    result = recover_profiles(edc_target, PROBE, cfg)
    elapsed = time.perf_counter() - t0

    print(f"\nResult: {result}")
    print(f"Time:   {elapsed:.2f}s")
    print(f"\nRecovered profiles:")
    print(f"  σ_rec = {np.array2string(result.sigma, precision=3, separator=', ')}")
    print(f"  μ_rec = {np.array2string(result.mu, precision=3, separator=', ')}")

    edc_check = edc_forward(result.sigma, result.mu, PROBE, n_quad=100)
    z_err = abs(edc_check.impedance_complex - edc_target.impedance_complex)
    print(f"\n|ΔZ_rec - ΔZ_meas| = {z_err:.6e} Ω")

    errors = round_trip_error(SIGMA_TRUE, MU_TRUE, result.sigma, result.mu)
    print(f"\nProfile errors (vs true — ill-posed, may differ):")
    print(f"  σ RMSE = {errors['sigma_rmse']:.4e} S/m  ({errors['sigma_rel_rmse']:.2%} relative)")
    print(f"  μ RMSE = {errors['mu_rmse']:.4e}       ({errors['mu_rel_rmse']:.2%} relative)")

    status = "PASS" if z_err < 1e-6 else "FAIL"
    print(f"\nStatus: {status} (impedance tolerance 1e-6 Ω)")
    
    print("\n" + "=" * 60)
    print("EXPLANATION: Why PASS/FAIL?")
    print("=" * 60)
    print("""This experiment tests the MOST ILL-POSED case: single-frequency
recovery of a homogeneous profile.

Pass criterion: |ΔZ_rec - ΔZ_meas| < 1e-6 Ω
  → Can the solver reproduce the measured impedance?

Why profile RMSE is huge (expected):
  - Single frequency → 2 constraints (Re(ΔZ), Im(ΔZ))
  - K=5 layers → 10 unknowns (5 σ + 5 μ)
  - Severely underdetermined: many (σ, μ) produce same ΔZ
  - The solver finds ONE valid solution, not THE true solution

Key insight:
  The EDC inverse problem is fundamentally ill-posed at one frequency.
  The solver is working correctly — it minimizes impedance mismatch.
  To recover the actual profile, you need:
    • More frequencies (Experiment 02)
    • Physical priors (regularization, GAN latent space)
    • Additional measurement modalities
""")
    
    if status == "PASS":
        print("\n✓ PASS: Impedance reproduced within tolerance.")
        print("  The solver correctly minimizes the mismatch objective.")
    else:
        print("\n✗ FAIL: Impedance error exceeds tolerance.")
        print("  Check: optimization settings, quadrature points, bounds.")
    
    print("\nGenerating visualizations...")
    save_dir = Path(__file__).parent / "results"
    
    create_experiment_summary(
        exp_name="exp01_homogeneous",
        sigma_true=SIGMA_TRUE,
        mu_true=MU_TRUE,
        result=result,
        probe=PROBE,
        layer_thickness=1e-3 / K,
        save_dir=save_dir,
    )
    
    plot_decision_criteria(
        edc_target=edc_target,
        edc_recovered=edc_check,
        tolerance=1e-6,
        criterion_type="absolute",
        save_path=save_dir / "exp01_homogeneous_decision.png",
    )
    
    return result


if __name__ == "__main__":
    run()
