#!/usr/bin/env python
"""
Phase 3 validation: Inverse EDC solver round-trip tests.

Strategy:
1. Define a known (σ, μ) profile.
2. Run the forward solver to get ΔZ.
3. Feed ΔZ into the inverse solver.
4. Compare recovered profiles to the originals.
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eddy_current_workflow.forward import ProbeSettings, EDCResponse, edc_forward
from eddy_current_workflow.inverse import (
    EDCMismatchObjective,
    RegularisedObjective,
    smoothness_penalty,
    monotonicity_penalty,
    InverseResult,
    RecoveryConfig,
    recover_profiles,
    round_trip_error,
)


PROBE = ProbeSettings(frequency=1e6)


def test_objective_evaluates():
    print("=" * 60)
    print("Test 1: Objective function evaluates correctly")
    print("=" * 60)

    K = 5
    sigma_true = np.full(K, 1e7)
    mu_true = np.full(K, 1.0)

    edc_target = edc_forward(sigma_true, mu_true, PROBE, n_quad=100)
    obj = EDCMismatchObjective(edc_target, PROBE, K, n_quad=100)

    theta_true = np.concatenate([sigma_true, mu_true])
    mismatch_at_truth = obj(theta_true)
    print(f"  J(θ_true) = {mismatch_at_truth:.6e}")
    assert mismatch_at_truth < 1e-20, f"Mismatch at true params should be ~0, got {mismatch_at_truth}"
    print("✓ Objective is ~0 at true parameters")

    theta_wrong = np.concatenate([np.full(K, 5e7), np.full(K, 50.0)])
    mismatch_wrong = obj(theta_wrong)
    print(f"  J(θ_wrong) = {mismatch_wrong:.6e}")
    assert mismatch_wrong > mismatch_at_truth
    print("✓ Objective increases for wrong parameters")
    print()


def test_regularisation_penalties():
    print("=" * 60)
    print("Test 2: Regularisation penalty functions")
    print("=" * 60)

    K = 10
    sigma_smooth = np.linspace(1e7, 2e7, K)
    mu_smooth = np.linspace(5.0, 1.0, K)
    sigma_rough = np.array([1e7, 3e7, 1e7, 3e7, 1e7, 3e7, 1e7, 3e7, 1e7, 3e7])
    mu_rough = np.array([1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0])

    s_smooth = smoothness_penalty(sigma_smooth, mu_smooth)
    s_rough = smoothness_penalty(sigma_rough, mu_rough)
    print(f"  Smoothness (smooth profile): {s_smooth:.6e}")
    print(f"  Smoothness (rough profile):  {s_rough:.6e}")
    assert s_rough > s_smooth
    print("✓ Smoothness penalty correctly distinguishes smooth vs rough")

    m_mono = monotonicity_penalty(sigma_smooth, mu_smooth, sigma_increasing=True, mu_decreasing=True)
    sigma_nonmono = np.array([1e7, 2e7, 1.5e7, 3e7, 2e7, 4e7, 3e7, 5e7, 4e7, 6e7])
    m_nonmono = monotonicity_penalty(sigma_nonmono, mu_smooth, sigma_increasing=True, mu_decreasing=True)
    print(f"  Monotonicity (monotonic):     {m_mono:.6e}")
    print(f"  Monotonicity (non-monotonic): {m_nonmono:.6e}")
    assert m_mono < 1e-20, "Monotonic profile should have zero penalty"
    assert m_nonmono > 0
    print("✓ Monotonicity penalty works correctly")
    print()


def test_regularised_objective():
    print("=" * 60)
    print("Test 3: Regularised objective wrapper")
    print("=" * 60)

    K = 5
    sigma_true = np.full(K, 1e7)
    mu_true = np.full(K, 1.0)
    edc_target = edc_forward(sigma_true, mu_true, PROBE, n_quad=100)

    base_obj = EDCMismatchObjective(edc_target, PROBE, K, n_quad=100)
    reg_obj = RegularisedObjective(base_obj, lambda_smooth=1e-3, lambda_mono=1e-2)

    theta = np.concatenate([sigma_true, mu_true])
    j_base = base_obj(theta)
    j_reg = reg_obj(theta)
    print(f"  J_base(θ_true) = {j_base:.6e}")
    print(f"  J_reg(θ_true)  = {j_reg:.6e}")

    assert j_reg >= j_base
    print("✓ Regularised objective >= base objective")
    print(f"  Call count: {reg_obj.call_count}")
    print()


def test_roundtrip_homogeneous():
    """Recover a homogeneous profile — simplest possible case."""
    print("=" * 60)
    print("Test 4: Round-trip recovery (homogeneous, K=5)")
    print("=" * 60)

    K = 5
    sigma_true = np.full(K, 1e7)
    mu_true = np.full(K, 1.0)

    edc_target = edc_forward(sigma_true, mu_true, PROBE, n_quad=100)
    print(f"  Target ΔZ = {edc_target.impedance_real:.6e} {edc_target.impedance_imag:+.6e}j")

    cfg = RecoveryConfig(
        K=K,
        sigma_bounds=(5e6, 5e7),
        mu_bounds=(1.0, 10.0),
        n_quad=100,
        method="multistart",
        n_starts=5,
        max_iter=500,
        seed=42,
        verbose=True,
    )

    t0 = time.perf_counter()
    result = recover_profiles(edc_target, PROBE, cfg)
    elapsed = time.perf_counter() - t0

    print(f"\n  {result}")
    print(f"  Time: {elapsed:.2f}s")

    errors = round_trip_error(sigma_true, mu_true, result.sigma, result.mu)
    print(f"  σ RMSE: {errors['sigma_rmse']:.4e} S/m")
    print(f"  μ RMSE: {errors['mu_rmse']:.4e}")
    print(f"  σ max err: {errors['sigma_max_err']:.4e}")
    print(f"  μ max err: {errors['mu_max_err']:.4e}")

    edc_check = edc_forward(result.sigma, result.mu, PROBE, n_quad=100)
    z_err = abs(edc_check.impedance_complex - edc_target.impedance_complex)
    print(f"  |ΔZ_rec - ΔZ_meas| = {z_err:.6e}")

    assert result.mismatch < 1e-6, f"Mismatch too large: {result.mismatch}"
    print("✓ Homogeneous round-trip: impedance reproduced")
    print("  (Note: profile non-uniqueness is expected — EDC inverse problem is ill-posed)")
    print()


def test_roundtrip_graded():
    """Recover a linearly graded profile."""
    print("=" * 60)
    print("Test 5: Round-trip recovery (graded, K=5)")
    print("=" * 60)

    K = 5
    sigma_true = np.linspace(1e7, 3e7, K)
    mu_true = np.linspace(5.0, 1.0, K)

    edc_target = edc_forward(sigma_true, mu_true, PROBE, n_quad=100)
    print(f"  Target ΔZ = {edc_target.impedance_real:.6e} {edc_target.impedance_imag:+.6e}j")
    print(f"  σ_true = {np.array2string(sigma_true, precision=2, separator=', ')}")
    print(f"  μ_true = {np.array2string(mu_true, precision=2, separator=', ')}")

    cfg = RecoveryConfig(
        K=K,
        sigma_bounds=(5e6, 5e7),
        mu_bounds=(0.5, 10.0),
        n_quad=100,
        method="multistart",
        n_starts=10,
        max_iter=1000,
        seed=42,
        lambda_smooth=1e-6,
        verbose=True,
    )

    t0 = time.perf_counter()
    result = recover_profiles(edc_target, PROBE, cfg)
    elapsed = time.perf_counter() - t0

    print(f"\n  {result}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  σ_rec = {np.array2string(result.sigma, precision=2, separator=', ')}")
    print(f"  μ_rec = {np.array2string(result.mu, precision=2, separator=', ')}")

    edc_check = edc_forward(result.sigma, result.mu, PROBE, n_quad=100)
    z_err = abs(edc_check.impedance_complex - edc_target.impedance_complex)
    print(f"  |ΔZ_rec - ΔZ_meas| = {z_err:.6e}")

    errors = round_trip_error(sigma_true, mu_true, result.sigma, result.mu)
    print(f"  σ relative RMSE: {errors['sigma_rel_rmse']:.4%}")
    print(f"  μ relative RMSE: {errors['mu_rel_rmse']:.4%}")

    assert result.mismatch < 1e-4, f"Mismatch too large: {result.mismatch}"
    print("✓ Graded round-trip: impedance reproduced within tolerance")
    print()


def test_inverse_result_api():
    print("=" * 60)
    print("Test 6: InverseResult API")
    print("=" * 60)

    r = InverseResult(
        sigma=np.ones(5),
        mu=np.ones(5),
        mismatch=1e-8,
        success=True,
        n_feval=500,
        n_iterations=42,
        method="L-BFGS-B",
        all_mismatches=[1e-3, 1e-5, 1e-8],
        convergence_rate=0.67,
    )

    print(f"  {r}")
    assert r.success is True
    assert r.n_feval == 500
    assert r.convergence_rate == 0.67
    print("✓ InverseResult dataclass works")
    print()


def main():
    print("\n" + "=" * 60)
    print("PHASE 3: INVERSE EDC SOLVER — VALIDATION SUITE")
    print("=" * 60 + "\n")

    try:
        test_objective_evaluates()
        test_regularisation_penalties()
        test_regularised_objective()
        test_roundtrip_homogeneous()
        test_roundtrip_graded()
        test_inverse_result_api()

        print("=" * 60)
        print("ALL 6 INVERSE SOLVER TESTS PASSED ✓")
        print("=" * 60 + "\n")
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
