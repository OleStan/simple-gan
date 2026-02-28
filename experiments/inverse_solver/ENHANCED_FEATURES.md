# Enhanced Visualization Features

## Summary of Enhancements

The visualization system now includes:

1. **Model identification** on all plots
2. **Tolerance bands** showing acceptable deviation zones
3. **Decision criteria visualization** showing exactly what we consider a "match"
4. **All multi-start attempts** displayed in convergence diagnostics
5. **Enhanced profile plots** with ±20% tolerance zones

---

## 1. Model Used: `edc_forward()` — Dodd-Deeds (1968)

**Location**: `eddy_current_workflow/forward/edc_solver.py`

All plots now display at the bottom:
```
Model: edc_forward() — Dodd-Deeds (1968) lightweight solver
```

This is the **lightweight forward solver** specifically designed for the inverse problem and GAN pipeline. It differs from the original model in `dodd_analytical_model/`:

| Feature | Original Model | Lightweight `edc_forward()` |
|---------|----------------|----------------------------|
| **Purpose** | Research, validation, LUT generation | Inverse solver, GAN integration |
| **Capabilities** | Vector potential, voltage, arbitrary points | Impedance only |
| **Layers** | Inner + outer (U/V matrices) | Single stack (bottom-up recursion) |
| **Speed** | Slower (full transfer matrices) | Fast (optimized for batched evaluation) |
| **Input format** | Nested dictionaries | Simple arrays |
| **Integration** | Standalone scripts | Direct Python API |

**Why `edc_forward()` for inverse?**
- Fast enough for optimization loops (100-200 evaluations/second)
- Simple API: `edc_forward(sigma, mu, probe_settings)`
- Numerically stable with Gauss-Legendre quadrature
- Batched evaluation support for GAN training

---

## 2. Enhanced Profile Plots

### New Features

**Tolerance bands** (shaded regions):
- Default: ±20% around true profile
- Shows acceptable deviation zone
- Helps visualize if recovered profile is "close enough"

**Model annotation**:
- Bottom of plot shows which solver was used
- Confirms we're using `edc_forward()` for inverse

**Improved styling**:
- Larger markers (size 8 for true, 7 for recovered)
- Bold axis labels
- Dotted grid lines
- μ=1 reference line (non-magnetic threshold)

### Example Interpretation

```
σ profile with ±20% tolerance:
  - Recovered inside band → physically plausible
  - Recovered outside band → check optimization settings
  - Note: Band is NOT a pass/fail criterion (ill-posedness!)
```

---

## 3. Decision Criteria Visualization

**New plot**: `exp0X_decision.png` (4-panel layout)

Shows **exactly what we consider a match** between target and recovered impedance.

### Panel 1: Complex Impedance Plane

- Target (blue circle) and Recovered (purple square) in Re-Im plane
- **Tolerance zone** (green if PASS, red if FAIL)
- Error vector connecting the two points
- Visual check: Is recovered point inside tolerance circle?

### Panel 2: Real vs Imaginary Components

- Side-by-side bar chart
- Direct comparison of Re(ΔZ) and Im(ΔZ)
- Shows which component has larger error

### Panel 3: Magnitude and Phase

- |ΔZ| and ∠ΔZ comparison
- Phase error often more sensitive than magnitude

### Panel 4: Decision Summary (Text Box)

Complete metrics printout:
```
DECISION CRITERIA
==================================================

Target Impedance:
  ΔZ = 1.2779e-02 +2.0135e-02j Ω
  |ΔZ| = 2.3857e-02 Ω
  ∠ΔZ = 57.61°

Recovered Impedance:
  ΔZ = 1.2779e-02 +2.0135e-02j Ω
  |ΔZ| = 2.3857e-02 Ω
  ∠ΔZ = 57.61°

Error Metrics:
  Absolute: 8.5714e-07 Ω
  Relative: 0.00%
  Re error: 5.4210e-20 Ω
  Im error: 8.5714e-07 Ω

Absolute error: 8.57e-07 Ω (tolerance: 1.00e-06 Ω)

Result: ✓ PASS

Model: edc_forward() — Dodd-Deeds (1968)
```

**Color coding**:
- Green background → PASS
- Red background → FAIL

---

## 4. All Multi-Start Attempts Visualization

**Enhanced convergence diagnostics** now show **3 panels** instead of 2:

### Panel 1: Multi-Start Final Mismatches (Bar Chart)

- Each bar = one optimization start
- **Color coding**:
  - Blue: converged well (within 10× of best)
  - Red: poor local minimum (>10× worse than best)
- **Best solution** highlighted with thick orange border
- **Threshold lines**:
  - Orange dashed: best mismatch
  - Red dotted: 10× threshold
- **Text annotation**: "X/N converged well"

### Panel 2: Sorted Results (Line Plot)

- Same as before, but now with threshold line
- Shows distribution of final mismatches
- Helps identify if most starts converge to similar values

### Panel 3: Distribution Histogram (NEW!)

- Histogram of log₁₀(mismatch) values
- Shows clustering of solutions
- **Vertical lines**:
  - Orange dashed: best solution
  - Orange dotted: median solution
- Helps diagnose multi-modality (multiple local minima)

**Title includes**:
- Number of starts
- Total function evaluations
- **Success rate** (% of starts that converged well)

**Bottom annotation**:
```
Optimizer: L-BFGS-B | Model: edc_forward() inverse solver
```

---

## 5. How to Interpret the New Plots

### Profile Plots with Tolerance Bands

```text
✓ Good:
  - Recovered profile inside tolerance band
  - Smooth, no wild oscillations
  - Physically plausible (σ > 0, μ ≥ 1)

⚠ Acceptable (ill-posed case):
  - Recovered profile outside band but impedance matches
  - This is NORMAL for single-frequency inverse problem

✗ Bad:
  - Unphysical values (σ < 0, μ < 1)
  - Extreme oscillations
  - Impedance error also large
```

### Decision Criteria Plot

```text
✓ PASS:
  - Recovered point inside tolerance circle
  - Green background on summary panel
  - All error metrics below threshold

✗ FAIL:
  - Recovered point outside tolerance circle
  - Red background on summary panel
  - Check optimization settings
```

### Convergence Diagnostics (3-panel)

```text
✓ Good convergence:
  - Most bars blue (>70% within 10× threshold)
  - Histogram shows tight clustering
  - Median close to best

⚠ Moderate:
  - 50-70% blue bars
  - Some spread in histogram
  - May need more starts

✗ Poor:
  - <50% blue bars
  - Wide histogram spread (multi-modal)
  - Try global optimizer or more starts
```

---

## Generated Files Per Experiment

### Experiment 01 (Homogeneous)
- `exp01_homogeneous_profiles.png` — σ and μ with tolerance bands
- `exp01_homogeneous_convergence.png` — 3-panel convergence (8 starts)
- `exp01_homogeneous_decision.png` — 4-panel decision criteria

### Experiment 02 (Multi-frequency)
- `exp02_multifreq_graded_profiles.png` — graded profiles with bands
- `exp02_multifreq_graded_impedance.png` — 6-panel impedance across 4 frequencies
- `exp02_multifreq_graded_convergence.png` — 3-panel convergence (10 starts)

### Experiment 03 (Noise robustness)
- `exp03_noise_robustness.png` — 4-panel noise sensitivity analysis

---

## Quick Reference: What Each Plot Answers

| Plot | Question Answered |
|------|-------------------|
| **Profiles** | Are recovered σ and μ physically plausible? |
| **Decision** | Did we meet the pass/fail criterion? Why? |
| **Convergence** | Did multi-start optimization work well? |
| **Impedance** | How well does recovered impedance match target? |
| **Noise** | Is the solver robust to measurement noise? |

---

## Model Information Summary

**Forward solver**: `edc_forward()` in `eddy_current_workflow/forward/edc_solver.py`

**Key characteristics**:
- Dodd-Deeds (1968) analytical model
- Transfer-matrix recursion for multilayer conductors
- Gauss-Legendre quadrature (default: 200 points)
- Bottom-up reflection coefficient Φ(s)
- Optimized for inverse problem and GAN integration

**Not used for inverse**:
- Original model in `dodd_analytical_model/` (too slow for optimization)
- COMSOL FEM solver (not available in inverse loop)

**Why this matters**:
- Plots now clearly show which model produced the results
- Confirms consistency across all experiments
- Helps debug if results look unexpected
