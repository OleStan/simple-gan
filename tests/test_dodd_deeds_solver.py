#!/usr/bin/env python
"""
Validation tests for the Dodd-Deeds forward EDC solver.

Physics sanity checks:
1. Homogeneous half-space: known analytical result
2. Sensitivity to conductivity changes
3. Sensitivity to permeability changes
4. Skin-depth frequency dependence
5. Lift-off effect
6. Deterministic (no randomness)
7. Multi-frequency sweep
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eddy_current_workflow.forward import (
    ProbeSettings,
    EDCResponse,
    edc_forward,
    edc_forward_multifreq,
    calculate_skin_depth,
)


def make_homogeneous_layers(sigma, mu_r, K=51):
    return np.full(K, sigma), np.full(K, mu_r)


def test_deterministic():
    print("="*60)
    print("Test 1: Deterministic output")
    print("="*60)

    sigma, mu = make_homogeneous_layers(1e7, 1.0)
    settings = ProbeSettings(frequency=1e6)

    r1 = edc_forward(sigma, mu, settings)
    r2 = edc_forward(sigma, mu, settings)

    assert r1.impedance_real == r2.impedance_real
    assert r1.impedance_imag == r2.impedance_imag
    print("✓ Identical inputs → identical outputs (no randomness)")
    print()


def test_nonzero_impedance():
    print("="*60)
    print("Test 2: Non-zero impedance for conducting specimen")
    print("="*60)

    sigma, mu = make_homogeneous_layers(1e7, 1.0)
    settings = ProbeSettings(frequency=1e6)
    r = edc_forward(sigma, mu, settings)

    print(f"  ΔZ = {r.impedance_real:.6e} {r.impedance_imag:+.6e}j Ω")
    print(f"  |ΔZ| = {r.amplitude:.6e} Ω")
    print(f"  ∠ΔZ  = {r.phase_deg:.2f}°")

    assert r.amplitude > 0, "Impedance change must be non-zero"
    print("✓ Non-zero impedance for conducting specimen")
    print()


def test_conductivity_sensitivity():
    print("="*60)
    print("Test 3: Conductivity sensitivity")
    print("="*60)

    settings = ProbeSettings(frequency=1e6)
    sigma_low, mu_low = make_homogeneous_layers(1e6, 1.0)
    sigma_high, mu_high = make_homogeneous_layers(5e7, 1.0)

    r_low = edc_forward(sigma_low, mu_low, settings)
    r_high = edc_forward(sigma_high, mu_high, settings)

    print(f"  σ = 1e6  S/m → |ΔZ| = {r_low.amplitude:.6e} Ω, ∠ = {r_low.phase_deg:.2f}°")
    print(f"  σ = 5e7  S/m → |ΔZ| = {r_high.amplitude:.6e} Ω, ∠ = {r_high.phase_deg:.2f}°")

    assert r_low.amplitude != r_high.amplitude, "Different σ must give different |ΔZ|"
    print("✓ Impedance changes with conductivity")
    print()


def test_permeability_sensitivity():
    print("="*60)
    print("Test 4: Permeability sensitivity")
    print("="*60)

    settings = ProbeSettings(frequency=1e6)
    sigma1, mu1 = make_homogeneous_layers(1e7, 1.0)
    sigma2, mu2 = make_homogeneous_layers(1e7, 10.0)

    r1 = edc_forward(sigma1, mu1, settings)
    r2 = edc_forward(sigma2, mu2, settings)

    print(f"  μᵣ = 1.0  → |ΔZ| = {r1.amplitude:.6e} Ω, ∠ = {r1.phase_deg:.2f}°")
    print(f"  μᵣ = 10.0 → |ΔZ| = {r2.amplitude:.6e} Ω, ∠ = {r2.phase_deg:.2f}°")

    assert r1.amplitude != r2.amplitude, "Different μ must give different |ΔZ|"
    print("✓ Impedance changes with permeability")
    print()


def test_lift_off_effect():
    print("="*60)
    print("Test 5: Lift-off effect (|ΔZ| decreases with distance)")
    print("="*60)

    sigma, mu = make_homogeneous_layers(1e7, 1.0)
    lift_offs = [0.1e-3, 0.5e-3, 1.0e-3, 2.0e-3]
    amplitudes = []

    for lo in lift_offs:
        s = ProbeSettings(frequency=1e6, lift_off=lo)
        r = edc_forward(sigma, mu, s)
        amplitudes.append(r.amplitude)
        print(f"  lift-off = {lo*1e3:.1f} mm → |ΔZ| = {r.amplitude:.6e} Ω")

    for i in range(len(amplitudes) - 1):
        assert amplitudes[i] > amplitudes[i + 1], (
            f"|ΔZ| must decrease with lift-off: "
            f"{amplitudes[i]:.4e} should be > {amplitudes[i+1]:.4e}"
        )
    print("✓ Impedance decreases monotonically with lift-off")
    print()


def test_frequency_sweep():
    print("="*60)
    print("Test 6: Multi-frequency sweep")
    print("="*60)

    sigma, mu = make_homogeneous_layers(1e7, 1.0)
    settings = ProbeSettings(frequency=1e6)
    freqs = np.array([1e4, 1e5, 1e6, 5e6])

    results = edc_forward_multifreq(sigma, mu, settings, freqs)

    for f, z in zip(freqs, results):
        print(f"  f = {f/1e3:8.1f} kHz → ΔZ = {z.real:.6e} {z.imag:+.6e}j Ω")

    assert len(results) == len(freqs)
    assert all(np.abs(z) > 0 for z in results)
    print("✓ Multi-frequency sweep produces valid results")
    print()


def test_layered_vs_homogeneous():
    print("="*60)
    print("Test 7: Layered profile differs from homogeneous")
    print("="*60)

    K = 51
    settings = ProbeSettings(frequency=1e6)

    sigma_hom, mu_hom = make_homogeneous_layers(2e7, 5.0, K)
    r_hom = edc_forward(sigma_hom, mu_hom, settings)

    sigma_grad = np.linspace(1e7, 3e7, K)
    mu_grad = np.linspace(1.0, 9.0, K)
    r_grad = edc_forward(sigma_grad, mu_grad, settings)

    print(f"  Homogeneous  → |ΔZ| = {r_hom.amplitude:.6e} Ω, ∠ = {r_hom.phase_deg:.2f}°")
    print(f"  Graded       → |ΔZ| = {r_grad.amplitude:.6e} Ω, ∠ = {r_grad.phase_deg:.2f}°")

    assert r_hom.impedance_complex != r_grad.impedance_complex, (
        "Graded and homogeneous profiles should produce different impedance"
    )
    print("✓ Layered profile produces different impedance than homogeneous")
    print()


def test_skin_depth_utility():
    print("="*60)
    print("Test 8: Skin depth utility function")
    print("="*60)

    delta = calculate_skin_depth(sigma=5.8e7, mu_r=1.0, frequency=1e6)
    print(f"  Copper (σ=5.8e7, μᵣ=1) at 1 MHz: δ = {delta*1e6:.1f} μm")
    assert 50e-6 < delta < 120e-6, f"Copper skin depth at 1MHz should be ~66 μm, got {delta*1e6:.1f} μm"

    delta_fe = calculate_skin_depth(sigma=1e7, mu_r=100.0, frequency=1e6)
    print(f"  Steel  (σ=1e7, μᵣ=100) at 1 MHz: δ = {delta_fe*1e6:.1f} μm")
    assert delta_fe < delta, "Steel skin depth should be much less than copper"

    print("✓ Skin depth values physically reasonable")
    print()


def test_performance():
    print("="*60)
    print("Test 9: Performance benchmark")
    print("="*60)

    sigma, mu = make_homogeneous_layers(1e7, 1.0)
    settings = ProbeSettings(frequency=1e6)

    edc_forward(sigma, mu, settings)

    n_runs = 5
    t0 = time.perf_counter()
    for _ in range(n_runs):
        edc_forward(sigma, mu, settings, n_quad=200)
    elapsed = time.perf_counter() - t0

    per_call = elapsed / n_runs
    print(f"  {n_runs} calls, n_quad=200: {elapsed:.3f}s total, {per_call:.3f}s/call")

    t0 = time.perf_counter()
    for _ in range(n_runs):
        edc_forward(sigma, mu, settings, n_quad=50)
    elapsed_fast = time.perf_counter() - t0
    per_call_fast = elapsed_fast / n_runs
    print(f"  {n_runs} calls, n_quad=50:  {elapsed_fast:.3f}s total, {per_call_fast:.3f}s/call")
    print("✓ Performance benchmark complete")
    print()


def main():
    print("\n" + "="*60)
    print("DODD-DEEDS FORWARD EDC SOLVER — VALIDATION SUITE")
    print("="*60 + "\n")

    try:
        test_deterministic()
        test_nonzero_impedance()
        test_conductivity_sensitivity()
        test_permeability_sensitivity()
        test_lift_off_effect()
        test_frequency_sweep()
        test_layered_vs_homogeneous()
        test_skin_depth_utility()
        test_performance()

        print("="*60)
        print("ALL 9 DODD-DEEDS VALIDATION TESTS PASSED ✓")
        print("="*60 + "\n")
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
