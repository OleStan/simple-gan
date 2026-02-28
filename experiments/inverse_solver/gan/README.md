x# GAN-Based Inverse Solver Experiments

## How It Works

### The Classical Problem

The eddy-current inverse problem recovers material profiles (σ, μ) from impedance measurements ΔZ.

At a single frequency, measuring ΔZ gives **2 constraints** (Re, Im), but recovering K=51 layers
requires finding **102 unknowns** (51 σ values + 51 μ values). This is 51× underdetermined.

Classical solvers (see `../classical/`) handle this with explicit smoothness regularization,
but still search a 102-dimensional space with no guarantee of physical plausibility.

### The GAN Prior

The trained Generator `G: ℝ³² → ℝ¹⁰²` maps a 32-dimensional latent vector `z` to a
physically plausible (σ, μ) profile.

**GAN-based inverse**: instead of optimizing 102 unknowns, optimize **z ∈ ℝ³²** so that
`G(z)` produces a profile whose ΔZ matches the measurement.

```
Measurement ΔZ_meas
        ↓
Minimize J(z) = |ΔZ(G(z)) - ΔZ_meas|²
        ↓
z* = argmin J(z)   [32 unknowns, scipy L-BFGS-B]
        ↓
G(z*) → (σ_rec, μ_rec)   [always physically plausible]
```

### Why This Is Better

| | Classical | GAN-based |
|---|---|---|
| **Search space** | ℝ¹⁰² (102-dim) | ℝ³² (32-dim) |
| **Physical plausibility** | Enforced by smoothness penalty | Enforced by GAN distribution |
| **Profile type** | Any smooth profile | Only profiles like training data |
| **Gradient source** | Finite-difference on (σ, μ) | Finite-difference on z |
| **Implicit prior** | None | GAN learned distribution |

### Architecture

```
experiments/inverse_solver/gan/
├── base.py                         # GANInverseExperiment base class
│   ├── GANInverseExperiment        # Abstract base: load_target() + explain()
│   ├── GANInverseResult            # Dataclass: sigma_rec, mu_rec, best_z, mismatches
│   ├── optimize()                  # Shared: multi-restart L-BFGS-B over z
│   ├── _report()                   # Shared: print results
│   └── _visualize()                # Shared: 5 plots per experiment
├── experiment_01_single_sample.py  # SingleSampleExperiment
├── experiment_02_multi_sample.py   # MultiSampleExperiment
├── experiment_03_noise_robustness.py # NoiseRobustnessExperiment
├── run_all.py                      # Run all experiments
├── visualize.py                    # Shared visualization utilities
└── README.md                       # This file
```

### Gradient Flow

`edc_forward()` is numpy-based and not differentiable through PyTorch.
The optimizer uses scipy L-BFGS-B with **finite-difference gradients**:

```
z (numpy, 32-dim)
  → torch.tensor(z)
  → netG(z) [PyTorch, no_grad]   # Generator forward pass
  → denormalize() [numpy]         # tanh [-1,1] → physical units
  → edc_forward() [numpy]         # Dodd-Deeds analytical model
  → J = |ΔZ_pred - ΔZ_meas|²    # scalar mismatch
  → scipy estimates ∂J/∂z via finite differences
  → L-BFGS-B updates z
```

A fully differentiable forward model would allow true autograd through the full pipeline.

---

## Experiments

### Experiment 01 — Single Sample Recovery

**File**: `experiment_01_single_sample.py`

Tests one real training data sample. Choose any sample index 0–1999.

**Question**: Can the GAN latent space reproduce the impedance of a specific training sample?

**Outputs**:
- `gan_exp01_sample{N}_profiles.png` — combined σ and μ
- `gan_exp01_sample{N}_sigma.png` — conductivity only
- `gan_exp01_sample{N}_mu.png` — permeability only
- `gan_exp01_sample{N}_decision.png` — pass/fail criteria
- `gan_exp01_sample{N}_convergence.png` — all restarts

**Run**:

```bash
python experiments/inverse_solver/gan/experiment_01_single_sample.py
python experiments/inverse_solver/gan/experiment_01_single_sample.py --sample 42
python experiments/inverse_solver/gan/experiment_01_single_sample.py --sample 42 --n_restarts 20
```

---

### Experiment 02 — Multi-Sample Sweep

**File**: `experiment_02_multi_sample.py`

Runs GAN inverse on N random training samples and reports aggregate statistics.

**Question**: What is the overall pass rate and error distribution across the training distribution?

**Outputs**:
- `gan_exp02_multi_sample_sweep.png` — per-sample errors + RMSE histograms
- `gan_exp02_multi_sample_best_profiles.png` — best result combined plot
- `gan_exp02_multi_sample_best_sigma.png` — best result conductivity
- `gan_exp02_multi_sample_best_mu.png` — best result permeability

**Run**:

```bash
python experiments/inverse_solver/gan/experiment_02_multi_sample.py
python experiments/inverse_solver/gan/experiment_02_multi_sample.py --n_samples 20 --n_restarts 8
```

---

### Experiment 03 — Noise Robustness

**File**: `experiment_03_noise_robustness.py`

Tests GAN inverse solver at increasing noise levels (0% to 5% of |ΔZ|).

**Question**: How does the GAN prior help maintain physical plausibility under noisy measurements?

**Outputs**:
- `gan_exp03_noise_sample{N}_noise_sweep.png` — error vs noise level (3-panel)
- `gan_exp03_noise_sample{N}_best_sigma.png` — conductivity at best noise level
- `gan_exp03_noise_sample{N}_best_mu.png` — permeability at best noise level

**Run**:

```bash
python experiments/inverse_solver/gan/experiment_03_noise_robustness.py
python experiments/inverse_solver/gan/experiment_03_noise_robustness.py --sample 5 --n_restarts 8
```

---

## Run All

```bash
python experiments/inverse_solver/gan/run_all.py
python experiments/inverse_solver/gan/run_all.py --sample 5 --n_restarts 10 --n_sweep_samples 10
```

All plots are saved to `experiments/inverse_solver/gan/results/`.

---

## Model Details

**Generator**: `ConditionalConv1DGenerator` from `models/improved_wgan_v2/model.py`

```
Input:  z ∈ ℝ³²           (latent vector)
Output: (σ, μ) ∈ ℝ¹⁰²    (51 + 51 layers, tanh-normalized)
```

**Architecture**:
```
z (32)
  → Linear(32 → 1024) → reshape (256, 4)
  → ConvTranspose1d ×4   (256 → 256 → 128 → 64 → 64, upsample ×16)
  → sigma_head: Conv1d ×3 → Tanh   → AdaptiveAvgPool1d(51)
  → mu_head:    Conv1d ×3 → Tanh   → AdaptiveAvgPool1d(51)
```

**Checkpoint**: `results/improved_wgan_v2_nz32_20260214_140817/models/netG_final.pt`

**Training data**: 2000 samples, K=51 layers, sigmoid profiles, opposite σ/μ relationship

**Normalization** (from `normalization_params.json`):
- σ: min-max to [-1, 1], range [1.881×10⁷, 4.043×10⁷] S/m
- μ: min-max to [-1, 1], range [1.005, 9.452]

---

## Extending

To add a new GAN experiment, subclass `GANInverseExperiment` from `base.py`:

```python
from base import GANInverseExperiment, PROBE, LAYER_THICKNESS

class MyExperiment(GANInverseExperiment):
    name = "gan_my_experiment"
    description = "My custom GAN inverse experiment"

    def load_target(self) -> tuple:
        # Return (sigma_true, mu_true, edc_target)
        ...

    def explain(self) -> None:
        print("What this experiment tests...")

# Run it
exp = MyExperiment(n_restarts=10)
result = exp.run()
# result.sigma_rec, result.mu_rec, result.best_mismatch, result.passed
```

The base class handles: Generator loading, optimization loop, result reporting,
and all 5 visualization plots automatically.

---

## Comparison: Classical vs GAN

| Aspect | Classical (`../classical/`) | GAN (`./`) |
|--------|----------------------------|-----------|
| Search space | 102-dim (σ, μ directly) | 32-dim (latent z) |
| Profile constraint | Smoothness penalty λ | GAN manifold |
| Profile type | Any smooth profile | Training-data distribution |
| Requires training data | No | Yes (trained GAN) |
| Interpretability | High (direct parameters) | Medium (via latent space) |
| Physical plausibility | Not guaranteed | Guaranteed by GAN |
| Best for | Understanding ill-posedness | Production use with known distribution |
