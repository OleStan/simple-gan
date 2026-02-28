# GAN Inverse Solver — dodd_analytical_model Forward

This experiment set combines:

- **Forward solver**: `dodd_analytical_model` (`VectorPotentialInsideCoilGreenFunction`, ORNL-5220)
- **Inverse solver**: `improved_wgan_v2_nz32` generator + L-BFGS-B in latent space

## Structure

```text
dodd_gan/
├── dodd_forward.py                   # Adapter: (σ,μ) → complex voltage V
├── base.py                           # GANDoddExperiment base class
├── experiment_01_single_sample.py    # Real training data target
├── experiment_02_synthetic_profile.py # Sigmoid profile target
├── experiment_03_random_profile.py   # Random (σ,μ) target
├── run_all.py                        # Run all experiments
└── results/                          # Auto-created on first run
```

## How it works

```text
z ∈ ℝ³²  →  G(z)  →  (σ, μᵣ) ∈ ℝ⁵¹×ℝ⁵¹
                              │
                              ▼
     VectorPotentialInsideCoilGreenFunction  →  V ∈ ℂ
     VoltageFromVectorPotential
                              │
                              ▼
              J(z) = |V_pred - V_target|² / |V_target|²
                              │
                   L-BFGS-B minimises J
```

The GAN generator acts as a learned prior over physically plausible (σ, μᵣ) profiles.
The 32-dimensional latent vector `z` is optimised to match the target voltage computed
by the full Dodd-Deeds analytical model.

## Experiments

| Exp | File | Target | Purpose |
|---|---|---|---|
| 01 | `experiment_01_single_sample.py` | Real training sample | In-distribution recovery |
| 02 | `experiment_02_synthetic_profile.py` | Synthetic sigmoid | Out-of-training, smooth |
| 03 | `experiment_03_random_profile.py` | Random per-layer σ/μ | Worst-case, off-manifold |

## Difference from `experiments/inverse_solver/gan/`

| | `gan/` | `dodd_gan/` |
|---|---|---|
| Forward solver | `edc_solver.py` (Gauss-Legendre, planar) | `dodd_analytical_model` (adaptive Gauss, cylindrical) |
| Output quantity | ΔZ impedance | V voltage |
| Speed | Fast | Slower (adaptive integration) |
| Fidelity | Lightweight reimplementation | Original ORNL-5220 analytical model |
| Inverse | `improved_wgan_v2_nz32` | `improved_wgan_v2_nz32` |

## Running

```bash
# Exp 01 — real training data sample
python experiments/inverse_solver/dodd_gan/experiment_01_single_sample.py
python experiments/inverse_solver/dodd_gan/experiment_01_single_sample.py --sample 42 --n_restarts 20

# Exp 02 — synthetic sigmoid profile
python experiments/inverse_solver/dodd_gan/experiment_02_synthetic_profile.py --steepness 15

# Exp 03 — random profile
python experiments/inverse_solver/dodd_gan/experiment_03_random_profile.py --profile_seed 7

# All experiments
python experiments/inverse_solver/dodd_gan/run_all.py --n_restarts 10 --sample 0 --profile_seed 0
```

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `n_restarts` | 10 | L-BFGS-B multi-restart count |
| `n_iter` | 500 | Max L-BFGS-B iterations per restart |
| `fd_epsilon` | 1e-3 | Finite-difference step for gradient |
| `integ_range_opt` | 20 | Adaptive integration upper limit during optimisation |
| `integ_range_verify` | 50 | Adaptive integration upper limit for final verification |

`integ_range_opt` controls speed vs accuracy during optimisation.
The final mismatch is always re-evaluated at `integ_range_verify`.

## Output

Each run creates `results/<timestamp>_<name>/`:

- `results.json` — config, target (σ, μ, V), recovered (σ, μ, V), all metrics
- `<name>_results.png` — 4-panel: σ profile, μ profile, voltage phasor, convergence

### JSON structure

```json
{
  "config": { "forward_solver": "...", "probe": {...}, ... },
  "target":  { "sigma_true": [...], "mu_true": [...], "voltage_real": ..., "voltage_imag": ... },
  "result":  { "sigma_rec": [...], "mu_rec": [...], "voltage_rec_real": ..., "best_mismatch_v": ... },
  "metrics": { "sigma_rmse": ..., "mu_rmse": ..., "voltage_error_v": ..., "voltage_error_pct": ... }
}
```

## Model

Generator: `results/improved_wgan_v2_nz32_20260214_140817/models/netG_final.pt`

Training distribution:

- σ ∈ [18.8, 40.4] MS/m
- μᵣ ∈ [1.00, 9.45]
- K = 51 layers, layer thickness = 1mm/51 ≈ 19.6 µm
