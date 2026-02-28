"""
Experiment 04 — Sigmoid profile recovery (GAN-realistic).

This experiment uses the ACTUAL profile type the GAN was trained on:
  - Sigmoid profiles (not linear/homogeneous)
  - K=51 layers (not 5)
  - Opposite relationship: σ ↑ → μ ↓
  - Fixed boundaries matching Table 3.5

This is the most realistic test for the inverse solver in the context
of the GAN pipeline.

Run from project root:
    python experiments/inverse_solver/experiment_04_sigmoid_realistic.py
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

K = 51

# Fixed boundaries from Table 3.5 (training data spec)
SIGMA_1_FIXED = 1.88e7
SIGMA_51_CENTER = 3.766e7
MU_1_FIXED = 1.0
MU_51_CENTER = 8.8


def sigmoid_profile(n_layers: int, min_val: float, max_val: float, steepness: float = 10.0) -> np.ndarray:
    """
    Generate sigmoid profile matching GAN training data.
    
    Args:
        n_layers: Number of layers (51 for GAN)
        min_val: Starting value (layer 1)
        max_val: Ending value (layer 51)
        steepness: Sigmoid steepness parameter (8-15 in training)
    
    Returns:
        Array of shape (n_layers,) with sigmoid profile
    """
    r = np.linspace(0, 1, n_layers)
    r_0 = 0.5
    sigmoid = 1 / (1 + np.exp(-steepness * (r - r_0)))
    return min_val + (max_val - min_val) * sigmoid


def run():
    print("=" * 60)
    print("Experiment 04 — Sigmoid Profile Recovery (GAN-realistic)")
    print(f"  K={K} layers | f={PROBE.frequency/1e6:.1f} MHz")
    print(f"  Profile: SIGMOID (matches GAN training)")
    print(f"  Relationship: OPPOSITE (σ ↑ → μ ↓)")
    print("=" * 60)
    
    # Generate sigmoid profiles with fixed boundaries
    steepness = 12.0
    SIGMA_TRUE = sigmoid_profile(K, SIGMA_1_FIXED, SIGMA_51_CENTER, steepness)
    MU_TRUE = sigmoid_profile(K, MU_51_CENTER, MU_1_FIXED, steepness)  # Opposite!
    
    print(f"\nTrue profile (sigmoid, steepness={steepness}):")
    print(f"  σ₁ = {SIGMA_TRUE[0]:.3e} S/m (layer 1)")
    print(f"  σ₅₁ = {SIGMA_TRUE[-1]:.3e} S/m (layer 51)")
    print(f"  μ₁ = {MU_TRUE[0]:.2f} (layer 1)")
    print(f"  μ₅₁ = {MU_TRUE[-1]:.2f} (layer 51)")
    print(f"  Relationship: σ increases → μ decreases ✓")
    
    edc_target = edc_forward(SIGMA_TRUE, MU_TRUE, PROBE, layer_thickness=1e-3/K, n_quad=150)
    print(f"\nTarget ΔZ = {edc_target.impedance_real:.6e} {edc_target.impedance_imag:+.6e}j Ω")
    
    cfg = RecoveryConfig(
        K=K,
        sigma_bounds=(1e7, 5e7),
        mu_bounds=(1.0, 15.0),
        layer_thickness=1e-3 / K,
        n_quad=150,
        method="multistart",
        n_starts=12,
        max_iter=800,
        tol=1e-12,
        seed=42,
        lambda_smooth=1e-5,
        verbose=True,
    )
    
    t0 = time.perf_counter()
    result = recover_profiles(edc_target, PROBE, cfg)
    elapsed = time.perf_counter() - t0
    
    print(f"\nResult: {result}")
    print(f"Time:   {elapsed:.2f}s")
    
    print(f"\nRecovered profile boundaries:")
    print(f"  σ₁_rec = {result.sigma[0]:.3e} S/m (true: {SIGMA_TRUE[0]:.3e})")
    print(f"  σ₅₁_rec = {result.sigma[-1]:.3e} S/m (true: {SIGMA_TRUE[-1]:.3e})")
    print(f"  μ₁_rec = {result.mu[0]:.2f} (true: {MU_TRUE[0]:.2f})")
    print(f"  μ₅₁_rec = {result.mu[-1]:.2f} (true: {MU_TRUE[-1]:.2f})")
    
    edc_check = edc_forward(result.sigma, result.mu, PROBE, layer_thickness=1e-3/K, n_quad=150)
    z_err = abs(edc_check.impedance_complex - edc_target.impedance_complex)
    print(f"\n|ΔZ_rec - ΔZ_meas| = {z_err:.6e} Ω")
    
    errors = round_trip_error(SIGMA_TRUE, MU_TRUE, result.sigma, result.mu)
    print(f"\nProfile errors:")
    print(f"  σ RMSE = {errors['sigma_rmse']:.4e} S/m  ({errors['sigma_rel_rmse']:.2%} relative)")
    print(f"  μ RMSE = {errors['mu_rmse']:.4e}       ({errors['mu_rel_rmse']:.2%} relative)")
    
    status = "PASS" if z_err < 1e-5 else "FAIL"
    print(f"\nStatus: {status} (impedance tolerance 1e-5 Ω)")
    
    print("\n" + "=" * 60)
    print("EXPLANATION: Why This Experiment Matters")
    print("=" * 60)
    print("""This experiment uses the ACTUAL profile type the GAN was trained on.

GAN Training Data (from metadata.json):
  - Profile type: SIGMOID
  - K = 51 layers
  - Opposite relationship: σ ↑ → μ ↓
  - Fixed boundaries from Table 3.5

Previous experiments (01-03):
  - Used K=5 layers (10× fewer!)
  - Linear/homogeneous profiles (not sigmoid)
  - No fixed boundaries
  - NOT representative of GAN training distribution

Why sigmoid profiles?
  - Smooth transitions (physically realistic)
  - Controllable steepness (shape parameter 8-15)
  - Matches real material degradation patterns
  - GAN learned to generate these specifically

Inverse solver performance on sigmoid:
  - More challenging than linear (non-trivial curvature)
  - K=51 → 102 unknowns from 2 measurements (51× underdetermined!)
  - Smoothness regularization is CRITICAL here
  - Profile RMSE will be large (expected for single-freq)

Key insight:
  If you're using the inverse solver with GAN-generated profiles,
  THIS is the experiment that matters. Experiments 01-03 were
  pedagogical examples, not production-realistic tests.
""")
    
    if status == "PASS":
        print("\n✓ PASS: Impedance reproduced for sigmoid profile.")
        print("  Solver handles GAN-realistic profiles correctly.")
    else:
        print("\n✗ FAIL: Impedance error too large.")
        print("  Try: more starts, higher λ_smooth, or multi-frequency.")
    
    print("\nGenerating visualizations...")
    save_dir = Path(__file__).parent / "results"
    
    create_experiment_summary(
        exp_name="exp04_sigmoid_realistic",
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
        tolerance=1e-5,
        criterion_type="absolute",
        save_path=save_dir / "exp04_sigmoid_realistic_decision.png",
    )
    
    return result


if __name__ == "__main__":
    run()
