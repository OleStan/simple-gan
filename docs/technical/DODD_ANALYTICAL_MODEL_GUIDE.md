# Dodd-Deeds Analytical Model — Forward (Direct) Problem Guide

## 1. What Is the Direct Problem?

In eddy-current non-destructive testing (NDT), the **direct (forward) problem** is:

> **Given**: probe geometry (coil radii, height, turns), excitation frequency, and material properties (conductivity σ, relative permeability μ_r, layer thicknesses) of a specimen  
> **Compute**: the electromagnetic response — coil impedance Z, vector potential A, or induced voltage V

This is the foundation of the entire project pipeline. The GAN generates material property profiles, and the forward solver validates them by computing what a real eddy-current probe would measure.

---

## 2. Physical Model

### 2.1 Geometry

The model describes an **axisymmetric coil above a multilayer cylindrical conductor**:

```text
         Coil (r1, r2, l1, l2, N turns)
         ┌─────────────┐
         │  ◯ ◯ ◯ ◯ ◯  │   ← current-carrying windings
         └─────────────┘
    ─────────────────────────── surface
    ░░░░░ Layer 1: σ₁, μ₁, d₁ ░░░░░
    ▓▓▓▓▓ Layer 2: σ₂, μ₂, d₂ ▓▓▓▓▓
    ░░░░░ Layer K: σ_K, μ_K, d_K ░░░░░
    ─────────────────────────── substrate (air)
```

- **r1, r2** — inner and outer radii of the coil
- **l1, l2** — bottom and top of the coil (lift-off encoded here)
- **N** — number of turns
- **σ_k** — electrical conductivity of layer k (S/m)
- **μ_k** — relative magnetic permeability of layer k
- **d_k** — thickness of layer k (m)

### 2.2 Governing Equations

The analytical solution follows **Dodd & Deeds (1968)** and its extensions in ORNL technical reports:

- **ORNL-4384** (Dodd, Deeds, Luquire, Spoeri, 1969) — impedance formulas for coils near layered conductors
- **ORNL-5220** (Nestor, Dodd, Deeds, 1979) — vector potential via Green's function approach

### 2.3 Key Formulas

#### Impedance Change (Dodd69 — ORNL-4384 eq. 3.155)

```text
ΔZ = jω · C · ∫₀^∞ [P²(s)/s⁶] · [e^{-s·l₁} - e^{-s·l₂}]² · Φ(s) ds
```

where:

- `C = 2πμ₀N² / [(r₂-r₁)²(l₂-l₁)²]` — geometric constant
- `P(s) = ∫_{s·r₁}^{s·r₂} x·J₁(x) dx` — coil integration factor (Bessel integral)
- `Φ(s)` — multilayer reflection coefficient (encodes all material info)
- `ω = 2πf` — angular frequency

#### Vector Potential (ORNL-5220 eq. 49)

```text
A(r,z) = (μ₀·N_c·I / π) · ∫₀^∞ F(α, r, z, material) dα
```

where `F` involves Bessel functions I₁, K₁ and U/V transfer matrices that propagate through each layer.

#### Multilayer Reflection Coefficient

Bottom-up recursion through K layers:

```text
α_k = √(s² + jωμ₀μ_kσ_k)       — propagation constant per layer
Φ_K = (μ_{K-1}·s - α_{K-1}) / (μ_{K-1}·s + α_{K-1})    — deepest interface

For k = K-2 … 0:
  u_k = (μ_{k+1}·α_k - μ_k·α_{k+1}) / (μ_{k+1}·α_k + μ_k·α_{k+1})
  Φ_k = (u_k + Φ_{k+1}·e^{-2α_{k+1}d_{k+1}}) / (1 + u_k·Φ_{k+1}·e^{-2α_{k+1}d_{k+1}})
```

#### Voltage from Vector Potential

```text
V = -jω · A · 2πr
```

Converted to polar form (amplitude, phase) for comparison with experimental data.

---

## 3. Code Architecture

### 3.1 Class Hierarchy (Original Model)

The original implementation in `dodd_analytical_model/` uses **multiple inheritance** to compose the solver from specialized mathematical components:

```text
VectorPotentialInsideCoilGreenFunction (main solver)
├── GaussIntegration                    — N-point Gauss quadrature (N=2,4,6,8,10)
├── AdaptiveIntegrationFunctions        — adaptive integration with convergence test
├── CyclesAdaptiveIntegration           — cyclic integration over sub-ranges
├── NormalizationFunctions              — coil dimension normalization
├── IntegralBesselFunctions             — ∫xJ₁dx, ∫xK₁dx specialized integrals
└── MatricesCalculation                 — U/V transfer matrices for multilayer
    └── BesselFunctions                 — I₀, I₁, K₀, K₁ via scipy.special
```

`ImpendanceDodd69` follows a similar pattern but computes impedance directly rather than vector potential at a point.

### 3.2 Data Flow

```text
Raw physical parameters (SI units)
       │
       ▼
┌──────────────────────────┐
│ NormalizationFunctions   │  Normalize coil dimensions by mean radius
│ normalization_1_coil_    │  Compute m = 2π·μ₀·f·μ_r·σ·r̄²
│ _sigma_mu_r()            │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ calculate()              │  Outer integration loop
│  → constant_expression() │  Computes front constant (μ₀·N_c·I / π)
│  → adapt2_integ_cycle()  │  Adaptive Gauss quadrature from 0 to ∞
│    → integrand()         │  Per-α evaluation
│      → frequency_depend_ │
│        part1()           │  sin/cos terms for coil position
│      → frequency_depend_ │
│        part2()           │  Bessel functions + U/V matrix chain
└──────────────────────────┘
       │
       ▼
Complex vector potential A(r,z) or impedance Z
       │
       ▼
┌──────────────────────────┐
│ VoltageFromVectorPotent. │  V = -jωA·2πr
│ Convertations            │  Complex → (amplitude, phase)
└──────────────────────────┘
```

### 3.3 Normalization

The model normalizes all dimensions by the **mean coil radius** `r̄ = (r₁+r₂)/2`:

- All lengths multiplied by `rrb = 2/(r₁+r₂)`
- Material parameter `m` encodes frequency, conductivity, and permeability:
  - SI version: `m = 2π·μ₀·f·μ_r·σ·r̄²`
  - Legacy (inches/kHz/μΩ·cm): `m = μ_r·509.39792/ρ · r̄²`

This normalization makes the integration variable dimensionless and improves numerical stability.

---

## 4. Two Implementations in the Project

### 4.1 Original Model (`dodd_analytical_model/`)

**Purpose**: Full-featured reference implementation for research, validation, and dataset generation.

**Capabilities**:

- Vector potential at arbitrary (r, z) points
- Impedance calculation (Dodd69)
- Multilayer conductors (inner + outer)
- U/V transfer matrix approach for arbitrary layer count
- LUT pipeline for bulk generation (10,000+ samples with multiprocessing)
- Voltage conversion with polar form output
- COMSOL validation scripts

**Input format**: Nested dictionary with `inner_parameters`, `outer_parameters`, `coils_parameters`, `frequency`, `calc_point`

**Key entry points**:

- `VectorPotentialInsideCoilGreenFunction.calculate()` — vector potential
- `ImpendanceDodd69.calculate()` — impedance
- `method_lut/4_calc_analytic_result.py` — bulk dataset generation

### 4.2 Lightweight Solver (`eddy_current_workflow/forward/edc_solver.py`)

**Purpose**: Streamlined solver optimized for the GAN training/validation pipeline.

**Capabilities**:

- Impedance change ΔZ for multilayer specimen
- Gauss-Legendre quadrature (fixed-point, no adaptive)
- Multi-frequency sweep
- Skin depth estimation
- Dataclass-based API (`ProbeSettings`, `EDCResponse`)

**Input format**: NumPy arrays for σ, μ per layer + `ProbeSettings` dataclass

**Key entry points**:

- `edc_forward(sigma_layers, mu_layers, settings)` — single frequency
- `edc_forward_multifreq(...)` — frequency sweep

### 4.3 When to Use Which

| Scenario | Use |
|---|---|
| Generate training datasets (thousands of samples) | Original model + LUT pipeline |
| Physics loss in GAN training loop | `edc_forward()` from `eddy_current_workflow` |
| Validate GAN output against reference | Original model |
| Research / parameter studies / publications | Original model |
| Quick impedance estimate | `edc_forward()` |

---

## 5. The LUT Pipeline (Lookup Table Method)

The LUT method in `dodd_analytical_model/method_lut/` generates large pre-computed datasets:

### Pipeline Steps

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `1_create_r_sequences.py` | Quantity, variance | Low-discrepancy R-sequences |
| 2 | `2_create_scaled_r.py` | R-sequences | Scaled to physical parameter ranges |
| 3 | `3_create_model_input_data.py` | Scaled sequences | JSON input data for solver |
| 4 | `4_calc_analytic_result.py` | JSON input | JSON with computed A, V, Z (multiprocessing) |
| 5 | `_0_moded_input_data_to_csv.py` | JSON results | CSV for neural network training |

### Output Structure

```text
method_lut/datasets_output/
├── model_input_data/                         calculated/
│   input_data_q10000_var0.3_0.3.json  →    full_data_q10000_var0.3_0.3.json
```

Each sample in the output JSON contains:

```json
{
  "inner_parameters": [...],
  "coils_parameters": [...],
  "frequency": [...],
  "calc_point": {...},
  "result_vector_potential": {"real": ..., "imag": ...},
  "result_voltage": {"real": ..., "imag": ...},
  "result_voltage_polar": {"amplitude": ..., "phase": ...}
}
```

---

## 6. How the Forward Solver Fits Into the GAN Project

```text
┌─────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                        │
│                                                         │
│  Real Data ──→ Discriminator ←── Generator(z)           │
│                     ↑                  │                 │
│                     │            (σ, μ) profiles         │
│                     │                  │                 │
│                     │                  ▼                 │
│                     │         Forward Solver (Dodd-Deeds)│
│                     │                  │                 │
│                     │            Impedance Z             │
│                     │                  │                 │
│                     └── Physics Loss ──┘                 │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                   VALIDATION PHASE                       │
│                                                         │
│  Generated profiles ──→ Forward Solver ──→ Z_generated  │
│  Real profiles ────────→ Forward Solver ──→ Z_real      │
│                                                         │
│  Quality = ||Z_generated - Z_real|| + physics_metrics   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                   INVERSE PROBLEM                        │
│                                                         │
│  Measured Z ──→ Optimizer ──→ (σ, μ) profiles           │
│                    ↑                                     │
│                    │ Forward Solver evaluates objective   │
│                    └─────────────────────────────────────│
└─────────────────────────────────────────────────────────┘
```

The forward solver is the **core physics engine** that connects:

1. **GAN training** — physics-informed loss penalizes non-physical profiles
2. **Quality validation** — checks that generated profiles produce realistic impedance
3. **Inverse problem** — optimization loop calls forward solver repeatedly to match measured data

---

## 7. Running the Forward Solver

### 7.1 Single Sample (Vector Potential)

```python
import sys
sys.path.insert(0, 'dodd_analytical_model')

from vector_potential_calculation.vector_potential_inside_coil_green_function import VectorPotentialInsideCoilGreenFunction
from input_data_generators.get_parameters_okita import get_input_parameters_dodd

sigma = 1e5       # conductivity S/m
freq = 1.25       # frequency kHz

input_data = get_input_parameters_dodd(sigma, freq)
calc = VectorPotentialInsideCoilGreenFunction()
normalized = calc.normalization_1_coil__sigma_mu_r(input_data)
result = calc.calculate(normalized, integration_range=50)

print(f"Vector potential: real={result.real}, imag={result.imag}")
```

### 7.2 Single Sample (Impedance)

```python
import sys
sys.path.insert(0, 'dodd_analytical_model')

from impendance_calculation.impendance_dodd_69 import ImpendanceDodd69
from input_data_generators.get_parameters_okita import get_input_parameters_dodd

sigma = 1e5
freq = 1.25

input_data = get_input_parameters_dodd(sigma, freq)
calc = ImpendanceDodd69()
normalized = calc.normalization(input_data)
result = calc.calculate(normalized, integration_range=200)

print(f"Impedance: real={result.real}, imag={result.imag}")
```

### 7.3 GAN Pipeline (Lightweight)

```python
from eddy_current_workflow.forward.edc_solver import edc_forward, ProbeSettings
import numpy as np

settings = ProbeSettings(frequency=1e6)
sigma_layers = np.array([1e7, 2e7, 1.5e7])
mu_layers = np.array([1.0, 1.0, 1.0])

response = edc_forward(sigma_layers, mu_layers, settings)
print(response)
# EDCResponse(f=1.00MHz, Z=...+...j Ω, |Z|=...Ω, ∠Z=...°)
```

### 7.4 Bulk Dataset (LUT Pipeline)

```bash
cd dodd_analytical_model
python method_lut/1_create_r_sequences.py
python method_lut/2_create_scaled_r.py
python method_lut/3_create_model_input_data.py
python method_lut/4_calc_analytic_result.py
```

---

## 8. Validation and References

### Reference Values

| Test Case | Expected | Source |
|---|---|---|
| Dodd69, σ=1e5, f=1.25kHz | Z ≈ 0.103 + 0.672j (normalized) | 2-coil reference model |
| COMSOL validation | See `csv_compare_comsol_python.py` | FEM cross-check |

### Test Commands

```bash
# Project-level tests
cd tests && python test_dodd_deeds_solver.py

# Model-level tests
cd dodd_analytical_model && python impendance_dodd69_test.py
```

### Scientific References

1. **Dodd, C.V. & Deeds, W.E.** (1968). "Analytical Solutions to Eddy-Current Probe-Coil Problems." *J. Appl. Phys.*, 39(6), 2829–2838.
2. **Dodd, C.V., Deeds, W.E., Luquire, J.W., Spoeri, W.G.** (1969). "Some Eddy-Current Problems and Their Integral Solutions." *ORNL-4384*.
3. **Nestor, C.W., Dodd, C.V., Deeds, W.E.** (1979). "Analysis and Computer Programs for Eddy-Current Coils Concentric with Multiple Cylindrical Conductors." *ORNL-5220*.
