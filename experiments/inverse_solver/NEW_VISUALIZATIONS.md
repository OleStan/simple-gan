# New Visualization Features

## Summary of Recent Additions

Two major enhancements have been added to the inverse solver experiments:

1. **Separate σ and μ comparison plots** — Individual images for conductivity and permeability
2. **Experiment 05** — Recovery from actual training data samples

---

## 1. Separate σ and μ Comparison Plots

### What Changed

Previously, `create_experiment_summary()` generated only a combined profile plot showing both σ and μ side-by-side.

Now, **three profile plots** are generated for each experiment:

| Plot File | Content |
|-----------|---------|
| `{exp_name}_profiles.png` | Combined σ and μ (2-panel, side-by-side) |
| `{exp_name}_sigma.png` | **NEW**: Conductivity only (standalone) |
| `{exp_name}_mu.png` | **NEW**: Permeability only (standalone) |

### Features of Standalone Plots

Each standalone plot includes:

- **True vs Recovered profiles** with clear markers
- **Tolerance bands** (±20% shaded zones)
- **RMSE metrics** in text box:
  - Absolute RMSE
  - Relative RMSE (%)
- **Model identification** footer
- **Publication-quality styling** (10×6 inches, 150 DPI)

### Example Output

For `experiment_01_homogeneous.py`:

```
results/
├── exp01_homogeneous_profiles.png    # Combined (existing)
├── exp01_homogeneous_sigma.png       # NEW: Conductivity only
├── exp01_homogeneous_mu.png          # NEW: Permeability only
├── exp01_homogeneous_convergence.png
└── exp01_homogeneous_decision.png
```

### Why This Matters

**Separate plots are better for:**
- Presentations (focus on one parameter at a time)
- Publications (easier to arrange in papers)
- Detailed analysis (larger, clearer view of each parameter)
- Comparing multiple experiments (stack σ plots together, μ plots together)

**Combined plot is still useful for:**
- Quick overview
- Seeing σ/μ relationship at a glance
- Compact summaries

---

## 2. Experiment 05 — Real Training Data

### What It Does

`experiment_05_real_training_data.py` loads an **actual profile from the training dataset** and tests the inverse solver on it.

### Key Features

**Command-line interface:**
```bash
# Test on sample 0 (default)
python experiments/inverse_solver/experiment_05_real_training_data.py

# Test on specific sample
python experiments/inverse_solver/experiment_05_real_training_data.py --sample 42

# Test on random samples
python experiments/inverse_solver/experiment_05_real_training_data.py --sample 123
python experiments/inverse_solver/experiment_05_real_training_data.py --sample 456
```

**Data source:**
- Loads from `data/training/sigma_layers.npy` and `data/training/mu_layers.npy`
- 2000 samples available (indices 0-1999)
- Each sample: K=51 layers, sigmoid profiles, opposite σ/μ relationship

**Output files:**
```
results/
├── exp05_real_data_sample0000_profiles.png
├── exp05_real_data_sample0000_sigma.png       # NEW
├── exp05_real_data_sample0000_mu.png          # NEW
├── exp05_real_data_sample0000_convergence.png
└── exp05_real_data_sample0000_decision.png
```

### Why This Matters

**Experiment hierarchy:**

1. **Experiments 01-03** (Pedagogical)
   - Simple profiles (K=5, linear/homogeneous)
   - Easy to understand
   - NOT representative of GAN training

2. **Experiment 04** (Production — Synthetic)
   - Sigmoid profiles (K=51)
   - Matches GAN training specs
   - Controlled test case

3. **Experiment 05** (Production — Real Data) ⭐
   - ACTUAL training data
   - Natural variations in shape parameters
   - Ultimate realism test
   - **This is what the GAN actually learned from**

**Use cases:**
- Validate solver on production data
- Test robustness across different samples
- Compare performance on easy vs hard samples
- Verify GAN pipeline integration

---

## Implementation Details

### New Functions in `visualize.py`

**`plot_sigma_comparison()`**
```python
def plot_sigma_comparison(
    sigma_true: np.ndarray,
    sigma_rec: np.ndarray,
    layer_thickness: float,
    save_path: Optional[Path] = None,
    title: str = "Conductivity Comparison",
    show_tolerance: bool = True,
    tolerance_pct: float = 20.0,
):
    """Plot true vs recovered conductivity profile (standalone)."""
```

**`plot_mu_comparison()`**
```python
def plot_mu_comparison(
    mu_true: np.ndarray,
    mu_rec: np.ndarray,
    layer_thickness: float,
    save_path: Optional[Path] = None,
    title: str = "Permeability Comparison",
    show_tolerance: bool = True,
    tolerance_pct: float = 20.0,
):
    """Plot true vs recovered permeability profile (standalone)."""
```

### Updated Function

**`create_experiment_summary()`** now calls:
1. `plot_profile_comparison()` — Combined plot (existing)
2. `plot_sigma_comparison()` — NEW: Conductivity only
3. `plot_mu_comparison()` — NEW: Permeability only
4. `plot_impedance_comparison()` — If multi-frequency
5. `plot_convergence_diagnostics()` — If multi-start

---

## Usage Examples

### Generate All Plots for Experiment 01

```python
from visualize import create_experiment_summary

create_experiment_summary(
    exp_name="exp01_homogeneous",
    sigma_true=SIGMA_TRUE,
    mu_true=MU_TRUE,
    result=result,
    probe=PROBE,
    layer_thickness=1e-3 / K,
    save_dir=Path("results"),
)
```

**Output:**
- `exp01_homogeneous_profiles.png` (combined)
- `exp01_homogeneous_sigma.png` (NEW)
- `exp01_homogeneous_mu.png` (NEW)
- `exp01_homogeneous_convergence.png`

### Test Multiple Training Samples

```bash
# Test 5 random samples
for i in 0 10 50 100 500; do
    python experiments/inverse_solver/experiment_05_real_training_data.py --sample $i
done
```

**Output:**
```
results/
├── exp05_real_data_sample0000_*.png
├── exp05_real_data_sample0010_*.png
├── exp05_real_data_sample0050_*.png
├── exp05_real_data_sample0100_*.png
└── exp05_real_data_sample0500_*.png
```

### Compare σ Across Experiments

Now you can easily compare conductivity recovery across different experiments:

```bash
# View all sigma plots side-by-side
open results/exp01_homogeneous_sigma.png
open results/exp04_sigmoid_realistic_sigma.png
open results/exp05_real_data_sample0000_sigma.png
```

---

## Backward Compatibility

**All existing experiments still work** — no breaking changes.

The new plots are **additions**, not replacements:
- Combined profile plot still generated
- All existing plot types unchanged
- New plots are opt-in (automatically included in `create_experiment_summary()`)

---

## Next Steps

### For Users

1. **Run experiments** to generate new plots:
   ```bash
   python experiments/inverse_solver/experiment_01_homogeneous.py
   python experiments/inverse_solver/experiment_05_real_training_data.py
   ```

2. **Review results** in `results/` folder

3. **Test multiple samples** with Experiment 05:
   ```bash
   for i in {0..10}; do
       python experiments/inverse_solver/experiment_05_real_training_data.py --sample $i
   done
   ```

### For Developers

**To add separate plots to custom experiments:**

```python
from visualize import plot_sigma_comparison, plot_mu_comparison

# After recovery
plot_sigma_comparison(
    sigma_true, result.sigma, layer_thickness,
    save_path=save_dir / "my_experiment_sigma.png"
)

plot_mu_comparison(
    mu_true, result.mu, layer_thickness,
    save_path=save_dir / "my_experiment_mu.png"
)
```

**To customize tolerance bands:**

```python
plot_sigma_comparison(
    sigma_true, result.sigma, layer_thickness,
    show_tolerance=True,
    tolerance_pct=10.0,  # ±10% instead of ±20%
    save_path=save_dir / "sigma_tight_tolerance.png"
)
```

---

## Files Modified

### New Files
- `experiment_05_real_training_data.py` — Real training data experiment
- `NEW_VISUALIZATIONS.md` — This document

### Modified Files
- `visualize.py`:
  - Added `plot_sigma_comparison()` function
  - Added `plot_mu_comparison()` function
  - Updated `create_experiment_summary()` to call new functions
- `README.md`:
  - Added Experiment 05 to production experiments
  - Updated run commands

### Generated Plots (Examples)

All experiments now generate **5 plots** instead of 3:

1. `{exp_name}_profiles.png` — Combined σ and μ
2. `{exp_name}_sigma.png` — **NEW**: Conductivity only
3. `{exp_name}_mu.png` — **NEW**: Permeability only
4. `{exp_name}_convergence.png` — Multi-start diagnostics
5. `{exp_name}_decision.png` — Pass/fail criteria

---

## Summary

**What you asked for:**
1. ✅ Separate σ and μ images for experiments
2. ✅ Experiment using actual training data signal

**What you got:**
1. ✅ Two new plotting functions (`plot_sigma_comparison`, `plot_mu_comparison`)
2. ✅ Automatic generation of separate plots in all experiments
3. ✅ New Experiment 05 with command-line sample selection
4. ✅ All 2000 training samples accessible for testing
5. ✅ RMSE metrics displayed on standalone plots
6. ✅ Publication-quality styling maintained
7. ✅ Backward compatibility preserved

**Impact:**
- **Better presentations** — Use focused plots for talks
- **Better publications** — Easier to arrange figures
- **Better analysis** — Larger, clearer views
- **Better testing** — Validate on real production data
- **Better confidence** — Know the solver works on actual GAN training distribution
