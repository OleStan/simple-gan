"""
Experiment 05 — Recovery from ACTUAL training data signal.

This experiment uses a real profile from the training dataset (not synthetic).
Tests the inverse solver on the exact type of data the GAN was trained on.

Run from project root:
    python experiments/inverse_solver/experiment_05_real_training_data.py
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


def load_training_sample(sample_idx: int = 0):
    """
    Load a specific sample from the training dataset.
    
    Args:
        sample_idx: Index of sample to load (0-1999)
    
    Returns:
        sigma_true: Conductivity profile (51,)
        mu_true: Permeability profile (51,)
    """
    data_dir = Path(__file__).parents[3] / "data" / "training"
    
    sigma_all = np.load(data_dir / "sigma_layers.npy")
    mu_all = np.load(data_dir / "mu_layers.npy")
    
    print(f"\nLoaded training data:")
    print(f"  Total samples: {sigma_all.shape[0]}")
    print(f"  Layers per sample: {sigma_all.shape[1]}")
    print(f"  Using sample index: {sample_idx}")
    
    return sigma_all[sample_idx], mu_all[sample_idx]


def run(sample_idx: int = 0):
    print("=" * 60)
    print("Experiment 05 — Real Training Data Recovery")
    print(f"  K={K} layers | f={PROBE.frequency/1e6:.1f} MHz")
    print(f"  Profile: ACTUAL TRAINING DATA (sample {sample_idx})")
    print("=" * 60)
    
    SIGMA_TRUE, MU_TRUE = load_training_sample(sample_idx)
    
    print(f"\nTrue profile (from training data):")
    print(f"  σ₁ = {SIGMA_TRUE[0]:.3e} S/m (layer 1)")
    print(f"  σ₅₁ = {SIGMA_TRUE[-1]:.3e} S/m (layer 51)")
    print(f"  μ₁ = {MU_TRUE[0]:.2f} (layer 1)")
    print(f"  μ₅₁ = {MU_TRUE[-1]:.2f} (layer 51)")
    
    sigma_increases = SIGMA_TRUE[-1] > SIGMA_TRUE[0]
    mu_increases = MU_TRUE[-1] > MU_TRUE[0]
    opposite = sigma_increases != mu_increases
    print(f"  Relationship: {'OPPOSITE ✓' if opposite else 'SAME ✗'} (σ {'↑' if sigma_increases else '↓'} → μ {'↑' if mu_increases else '↓'})")
    
    edc_target = edc_forward(SIGMA_TRUE, MU_TRUE, PROBE, layer_thickness=1e-3/K, n_quad=150)
    print(f"\nTarget ΔZ = {edc_target.impedance_real:.6e} {edc_target.impedance_imag:+.6e}j Ω")
    
    cfg = RecoveryConfig(
        K=K,
        sigma_bounds=(1e7, 5e7),
        mu_bounds=(1.0, 15.0),
        layer_thickness=1e-3 / K,
        n_quad=150,
        method="multistart",
        n_starts=15,
        max_iter=1000,
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
    print("""
This experiment uses ACTUAL training data, not synthetic profiles.

Key differences from Experiments 01-04:
  - Experiments 01-03: Synthetic simple profiles (pedagogical)
  - Experiment 04: Synthetic sigmoid profile (GAN-like)
  - Experiment 05: REAL training data (production test)

Why use real training data?
  - Tests solver on exact distribution GAN learned from
  - Includes natural variations in shape parameters
  - May have subtle features not in synthetic profiles
  - Most realistic performance evaluation

Training data characteristics:
  - Generated with sigmoid profiles
  - K=51 layers
  - Opposite σ/μ relationship
  - Fixed boundaries with ±7.5% variation
  - Shape parameter (steepness) varies 8-15

Expected behavior:
  - Impedance match: Should achieve |ΔZ_error| < 1e-5 Ω
  - Profile RMSE: Will be large (single-freq, 51× underdetermined)
  - Solver finds A valid profile, not THE true profile

This is the ultimate test: if the inverse solver works on real
training data, it will work in the GAN pipeline.
""")
    
    if status == "PASS":
        print("\n✓ PASS: Impedance reproduced for real training data.")
        print("  Solver handles production data correctly.")
    else:
        print("\n✗ FAIL: Impedance error too large.")
        print("  Try: more starts, higher λ_smooth, or multi-frequency.")
    
    print("\nGenerating visualizations...")
    save_dir = Path(__file__).parent / "results"
    
    create_experiment_summary(
        exp_name=f"exp05_real_data_sample{sample_idx:04d}",
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
        save_path=save_dir / f"exp05_real_data_sample{sample_idx:04d}_decision.png",
    )
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inverse solver on real training data")
    parser.add_argument("--sample", type=int, default=0, help="Training sample index (0-1999)")
    args = parser.parse_args()
    
    run(sample_idx=args.sample)
