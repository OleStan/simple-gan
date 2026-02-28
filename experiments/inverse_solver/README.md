# Inverse Solver Experiments

Experiments for the Dodd-Deeds eddy-current inverse problem: recovering material profiles (σ, μ) from impedance measurements ΔZ.

## Structure

```
experiments/inverse_solver/
├── classical/    — Physics-based optimization (no GAN)
└── gan/          — GAN latent-space optimization (uses trained Generator)
```

## Background

```
Forward:  (σ, μ) + probe  →  ΔZ   (unique, deterministic)
Inverse:  ΔZ + probe       →  (σ, μ)  (ill-posed: many solutions)
```

### Two Approaches

**Classical** (`classical/`): Optimize (σ, μ) directly in ℝ¹⁰² with smoothness regularization.

**GAN-based** (`gan/`): Optimize latent vector z ∈ ℝ³² through the frozen Generator.
All outputs are on the GAN's learned manifold — physically plausible by construction.

| | Classical | GAN-based |
|---|---|---|
| Search space | ℝ¹⁰² (102-dim) | ℝ³² (32-dim) |
| Physical constraint | Smoothness penalty | GAN distribution |
| Requires trained model | No | Yes (`improved_wgan_v2`) |
| Best for | Understanding ill-posedness | Production with known distribution |

---

## Classical Experiments (`classical/`)

| File | What it tests | Profiles |
|---|---|---|
| `experiment_01_homogeneous.py` | Single-freq recovery of uniform (σ, μ) | K=5, homogeneous |
| `experiment_02_multifreq_graded.py` | Multi-freq graded profile recovery | K=5, linear |
| `experiment_03_noise_robustness.py` | Solver degradation under noise | K=5, linear |
| `experiment_04_sigmoid_realistic.py` | Synthetic sigmoid matching GAN spec | K=51, sigmoid |
| `experiment_05_real_training_data.py` | Actual training data sample | K=51, real |

```bash
python experiments/inverse_solver/classical/experiment_01_homogeneous.py
python experiments/inverse_solver/classical/experiment_05_real_training_data.py --sample 42
python experiments/inverse_solver/classical/run_all.py
```

---

## GAN Experiments (`gan/`)

| File | What it tests |
|---|---|
| `experiment_01_single_sample.py` | One training sample — basic recovery test |
| `experiment_02_multi_sample.py` | N samples — aggregate pass rate and RMSE distribution |
| `experiment_03_noise_robustness.py` | Noise robustness with GAN prior as implicit denoiser |

```bash
python experiments/inverse_solver/gan/experiment_01_single_sample.py --sample 0
python experiments/inverse_solver/gan/experiment_02_multi_sample.py --n_samples 20
python experiments/inverse_solver/gan/experiment_03_noise_robustness.py --sample 5
python experiments/inverse_solver/gan/run_all.py
```

Full documentation: `gan/README.md`

---

## Results

Plots are saved to:

- `experiments/inverse_solver/classical/results/`
- `experiments/inverse_solver/gan/results/`

Each experiment generates 5 plots: combined profiles, σ only, μ only, decision criteria, convergence.

## Visualizations

Each experiment generates plots showing:

**Experiment 01 (Homogeneous)**:

- Profile comparison: true vs recovered σ and μ
- Convergence diagnostics: multi-start performance

**Experiment 02 (Multi-frequency)**:

- Profile comparison across depth
- Impedance comparison: Re(ΔZ), Im(ΔZ), |ΔZ|, ∠ΔZ at all frequencies
- Per-frequency reconstruction errors
- Convergence diagnostics

**Experiment 03 (Noise robustness)**:

- Mismatch vs noise level
- Profile RMSE vs noise level
- Impedance error vs noise level
- Linearity check (noise sensitivity)

All plots use publication-quality styling with clear legends and labels.

## Pass/Fail Criteria Explained

### Experiment 01: `|ΔZ_rec - ΔZ_meas| < 1e-6 Ω`

**Why this criterion?**

- Tests if solver can minimize the impedance mismatch objective
- Single frequency → severely ill-posed (2 constraints, 10 unknowns)
- Profile RMSE will be huge — **this is expected and correct**
- Many (σ, μ) combinations produce the same ΔZ

**PASS means**: Solver works correctly, finds a valid solution

**FAIL means**: Optimization issue (check bounds, n_starts, max_iter)

### Experiment 02: `max relative |ΔZ| error < 5%`

**Why this criterion?**

- Multi-frequency reduces ill-posedness (8 constraints, 10 unknowns)
- 5% tolerance accounts for remaining underdetermination
- Different frequencies probe different depths (skin effect)

**PASS means**: Solver handles multi-frequency data, impedance reproduced across spectrum

**FAIL means**: Try more starts, tighter bounds, higher λ_smooth, or more frequencies

### Experiment 03: Convergence at all noise levels

**Why this criterion?**

- Real measurements have 0.1–1% noise
- Solver must handle realistic noise without diverging
- Impedance error should scale ~linearly with noise

**PASS means**: Solver is robust, suitable for real measurements

**FAIL means**: Instability — check regularization, bounds, or solver settings

## Key Observations

### Ill-posedness

A **single frequency** measurement produces one complex number (2 real
constraints). Recovering K-layer profiles means fitting 2K unknowns.
For K=5 that is 10 unknowns from 2 measurements — severely underdetermined.
The solver will find a minimum-mismatch solution but it won't match the
true profile exactly. **This is correct behaviour**, not a bug.

### Multi-frequency improves recovery

Using F frequencies gives 2F constraints. With F=4 and K=5 we have 8
constraints vs 10 unknowns — still underdetermined, but much better
conditioned. Profile RMSE drops significantly (Experiment 02).

### Regularisation

Smoothness (`lambda_smooth`) and monotonicity (`lambda_mono`) penalties
bias the solution towards physically realistic profiles. Increasing these
reduces profile RMSE at the cost of slightly higher impedance mismatch.

### Noise tolerance

The solver handles 0–2% measurement noise gracefully (Experiment 03).
Profile errors scale roughly linearly with noise level, as expected for
a well-regularised inverse solver.

## Tuning Guidance

| Parameter | Effect | Start value |
|---|---|---|
| `n_starts` | More starts → less chance of bad local minimum | 8–20 |
| `lambda_smooth` | Higher → smoother recovered profiles | 1e-7 – 1e-5 |
| `lambda_mono` | Higher → more monotonic profiles | 0 or 1e-6 |
| `n_quad` | Higher → more accurate forward solve | 100–200 |
| `method="global"` | Differential evolution — slower but more thorough | for hard cases |

## Understanding the Results

### Why is profile RMSE so high?

The EDC inverse problem is **fundamentally ill-posed**:

```text
Single frequency:  2 measurements → 10 unknowns  (5× underdetermined)
Four frequencies:  8 measurements → 10 unknowns  (1.25× underdetermined)
```

Many different (σ, μ) profiles produce nearly identical impedance. The solver finds **a** solution, not **the** solution.

### When does the solver "work"?

The solver works correctly if:

1. ✓ Impedance is reproduced within tolerance
2. ✓ Optimization converges (result.success = True)
3. ✓ Recovered profiles are physically plausible (σ > 0, μ ≥ 1)

The solver does NOT guarantee:

- ✗ Exact recovery of the true profile (impossible without more data)
- ✗ Low profile RMSE (ill-posedness dominates)

### How to improve profile recovery?

1. **More frequencies** — each frequency adds 2 constraints
2. **Stronger regularization** — λ_smooth, λ_mono bias towards realistic profiles
3. **Physical priors** — GAN latent space constrains to learned manifold
4. **Additional modalities** — combine with other NDT measurements

### Practical use case

In the GAN pipeline:

- Forward solver validates generated profiles (physics-informed loss)
- Inverse solver is NOT used for direct profile recovery
- Instead: GAN learns low-dimensional latent → profile mapping
- Latent space acts as strong prior, making inverse problem tractable

### GAN Training Data Specifications

From `training_data/metadata.json`:

```json
{
  "profile_type": "sigmoid",
  "K": 51,
  "relationship": "opposite",
  "fixed_boundaries": {
    "sigma_1": 1.88e7,
    "sigma_51_center": 3.766e7,
    "mu_1": 1.0,
    "mu_51_center": 8.8
  }
}
```

**Experiment 04 uses these exact specifications** to test the inverse solver on GAN-realistic profiles.
