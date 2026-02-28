# Visualization Guide

## Quick Start

```bash
# Run all experiments with visualizations (recommended)
python experiments/inverse_solver/run_with_visualizations.py

# Or run individual experiments
python experiments/inverse_solver/experiment_01_homogeneous.py
python experiments/inverse_solver/experiment_02_multifreq_graded.py
python experiments/inverse_solver/experiment_03_noise_robustness.py
```

All plots are saved to `experiments/inverse_solver/results/`

## Generated Plots

### Experiment 01: Homogeneous Recovery

**`exp01_homogeneous_profiles.png`**
- Left panel: Conductivity σ (true vs recovered)
- Right panel: Permeability μ (true vs recovered)
- Shows depth profile from 0 to 0.2 mm (K=5 layers)

**What to look for:**
- Recovered profile will NOT match true profile (expected — ill-posed)
- Check if recovered values are physically plausible (σ > 0, μ ≥ 1)

**`exp01_homogeneous_convergence.png`**
- Left panel: Bar chart of final mismatch for each start
- Right panel: Sorted mismatches (best to worst)
- Shows multi-start optimization performance

**What to look for:**
- Best mismatch should be < 1e-6
- Multiple starts should converge to similar values
- Red bars indicate poor local minima (>10× worse than best)

---

### Experiment 02: Multi-Frequency Graded Recovery

**`exp02_multifreq_graded_profiles.png`**
- Same layout as Exp 01
- Graded profile: σ increases, μ decreases with depth

**What to look for:**
- Recovered profile should show similar trend (increasing/decreasing)
- Multi-freq reduces ambiguity → better profile recovery than Exp 01

**`exp02_multifreq_graded_impedance.png`**
- 6 panels showing impedance comparison across 4 frequencies:
  - Re(ΔZ), Im(ΔZ), |ΔZ|, ∠ΔZ
  - Absolute and relative errors
- Target (blue circles) vs Recovered (purple squares)

**What to look for:**
- All 4 frequencies should be reproduced within ~5% error
- Error should be roughly uniform across frequencies
- Bottom panel: relative error < 5% at all frequencies

**`exp02_multifreq_graded_convergence.png`**
- Same as Exp 01 convergence diagnostics
- More starts (10) → better exploration of parameter space

---

### Experiment 03: Noise Robustness

**`exp03_noise_robustness.png`**
- 4 panels analyzing noise sensitivity:

**Top-left: Mismatch vs Noise**
- Log-scale plot of objective function J(θ)
- Should grow monotonically with noise

**Top-right: Profile RMSE vs Noise**
- σ RMSE (purple, left axis) and μ RMSE (orange, right axis)
- Shows how profile errors degrade with noise

**Bottom-left: Impedance Error vs Noise**
- |ΔZ_rec - ΔZ_clean| as function of noise level
- Should scale roughly linearly

**Bottom-right: Linearity Check**
- Linear fit to impedance error vs noise
- Validates that solver is stable (no super-linear degradation)

**What to look for:**
- All noise levels should converge (no divergence)
- Impedance error should scale ~linearly with noise
- Profile RMSE dominated by ill-posedness, not noise

---

## Interpreting the Plots

### Profile Comparison Plots

```text
✓ Good signs:
  - Recovered σ > 0, μ ≥ 1 (physically plausible)
  - Smooth profiles (no wild oscillations)
  - Similar trend to true profile (increasing/decreasing)

✗ Bad signs:
  - Negative σ or μ < 1 (unphysical)
  - Extreme oscillations (numerical instability)
  - All layers identical when true profile is graded
```

### Impedance Comparison Plots

```text
✓ Good signs:
  - Target and recovered curves overlap
  - Relative error < 5% across all frequencies
  - Error bars small compared to signal magnitude

✗ Bad signs:
  - Large systematic offset (bias)
  - Error grows with frequency (quadrature issue)
  - Phase error > 10° (sign of poor convergence)
```

### Convergence Diagnostics

```text
✓ Good signs:
  - Multiple starts converge to same mismatch
  - Best mismatch meets tolerance (< 1e-6 or < 1e-4)
  - Most starts succeed (convergence rate > 70%)

✗ Bad signs:
  - Wide spread in final mismatches (many local minima)
  - Best mismatch still large (optimization failed)
  - Low convergence rate (< 50% success)
```

### Noise Robustness Plots

```text
✓ Good signs:
  - Solver converges at all noise levels
  - Impedance error scales linearly with noise
  - No catastrophic degradation at 1-2% noise

✗ Bad signs:
  - Divergence at low noise levels (< 1%)
  - Super-linear error growth (instability)
  - Profile becomes unphysical at moderate noise
```

---

## Common Issues and Fixes

### Issue: Profile looks nothing like true profile

**Diagnosis**: This is EXPECTED for single-frequency (Exp 01)

**Explanation**: The inverse problem is severely ill-posed. Many different profiles produce the same impedance. The solver finds **a** valid solution, not **the** true solution.

**Fix**: Not a bug. Use multi-frequency (Exp 02) or stronger regularization.

---

### Issue: Impedance error exceeds tolerance

**Diagnosis**: Optimization failed to converge

**Possible causes**:
1. Bounds too tight → solution outside search space
2. Too few starts → stuck in local minimum
3. max_iter too low → stopped before convergence

**Fix**:
```python
cfg = RecoveryConfig(
    sigma_bounds=(1e6, 1e8),  # Widen bounds
    n_starts=20,              # More starts
    max_iter=2000,            # More iterations
)
```

---

### Issue: Convergence diagnostics show wide spread

**Diagnosis**: Many local minima in objective landscape

**Fix**: Use global optimizer or more starts
```python
cfg = RecoveryConfig(
    method="global",  # Differential evolution
    # OR
    n_starts=50,      # Brute-force with many starts
)
```

---

### Issue: Noise robustness shows divergence

**Diagnosis**: Solver is fragile to measurement noise

**Fix**: Add regularization
```python
cfg = RecoveryConfig(
    lambda_smooth=1e-5,  # Smoothness penalty
    lambda_mono=1e-6,    # Monotonicity penalty
)
```

---

## Customizing Visualizations

The `visualize.py` module provides reusable plotting functions:

```python
from visualize import (
    plot_profile_comparison,
    plot_impedance_comparison,
    plot_noise_robustness,
    plot_convergence_diagnostics,
    create_experiment_summary,  # All-in-one
)

# Example: custom profile plot
plot_profile_comparison(
    sigma_true, mu_true,
    sigma_rec, mu_rec,
    layer_thickness=1e-4,
    save_path=Path("my_custom_plot.png"),
    title="My Custom Experiment"
)
```

All functions accept `save_path=None` to display interactively instead of saving.

---

## Publication-Quality Exports

Plots are saved at 150 DPI with tight bounding boxes, suitable for:
- Technical reports
- Research papers
- Presentations

To increase resolution for publication:

```python
# In visualize.py, change:
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 300 DPI
```

---

## Next Steps

1. **Run all experiments**: `python experiments/inverse_solver/run_with_visualizations.py`
2. **Review plots** in `results/` folder
3. **Read explanations** printed to console
4. **Tune parameters** if needed (see README.md tuning guide)
5. **Integrate learnings** into main GAN pipeline
