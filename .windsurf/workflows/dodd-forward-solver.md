---
description: How to use the Dodd-Deeds analytical model to solve the direct (forward) eddy-current problem — computing impedance/voltage from known material properties and probe geometry
---

# Dodd-Deeds Forward Solver Workflow

## Context

The **direct (forward) problem** in eddy-current NDT means:

> Given material properties (σ, μ_r) and probe geometry → compute the electromagnetic response (impedance, vector potential, voltage).

Our project has **two implementations** of this forward solver:

| Implementation | Location | Use Case |
|---|---|---|
| **Original analytical model** | `dodd_analytical_model/` | Full-featured reference solver with LUT pipeline, multiprocessing, JSON I/O, voltage conversion |
| **Lightweight forward solver** | `eddy_current_workflow/forward/edc_solver.py` | Streamlined solver for GAN training/validation loop |

Both implement the same physics (Dodd-Deeds), but the original model is the authoritative source with richer capabilities.

---

## Step 1: Understand the Physics Pipeline

The forward solver chain is:

```
Input Parameters (σ, μ_r, geometry, frequency)
  → Normalization (coil dimensions normalized by mean radius)
  → Numerical Integration (adaptive Gauss quadrature over Bessel integrals)
  → Vector Potential A(r,z) [complex]
  → Voltage V = -jωA·2πr [complex → polar: amplitude + phase]
```

Key equations from ORNL-5220 / ORNL-4384:
- **Vector potential**: eq. 49 (ORNL-5220) — Green's function approach for coil above multilayer conductor
- **Impedance**: eq. 3.155 (ORNL-4384) — Dodd69 impedance formula
- **Reflection coefficient**: bottom-up recursion through K material layers

---

## Step 2: Prepare Input Data

Input data is a dictionary with this structure:

```python
input_data = {
    "inner_parameters": [
        {"r": <radius_m>, "mu_r": <relative_permeability>, "sigma": <conductivity_S_m>, "rho": <resistivity>},
        ...  # one entry per cylindrical layer (inside coil)
    ],
    "outer_parameters": [],  # layers outside the coil (if any)
    "coils_parameters": [{
        "r1": <inner_radius_m>,
        "r2": <outer_radius_m>,
        "l1": <bottom_height_m>,
        "l2": <top_height_m>,
        "n": <number_of_turns>
    }],
    "frequency": [<freq_kHz>],
    "electrical_values": {"current": 1},
    "calc_point": {"r": <radial_position>, "z": <axial_position>}
}
```

For bulk dataset generation, use the input data generators:
```
dodd_analytical_model/input_data_generators/get_parameters_okita.py
dodd_analytical_model/input_data_generators/test_data_generator.py
```

---

## Step 3: Run the Forward Solver (Single Calculation)

### Option A — Original model (vector potential + voltage)

```bash
cd dodd_analytical_model
python vector_potential_main_test.py
```

Or programmatically:

```python
from vector_potential_calculation.vector_potential_inside_coil_green_function import VectorPotentialInsideCoilGreenFunction
from calculation_helpers.normalization_functions import NormalizationFunctions

calc = VectorPotentialInsideCoilGreenFunction()
normalized = calc.normalization_1_coil__sigma_mu_r(input_data)
vector_potential = calc.calculate(normalized, integration_range=50)
```

### Option B — Impedance calculation (Dodd69)

```python
from impendance_calculation.impendance_dodd_69 import ImpendanceDodd69

calc = ImpendanceDodd69()
normalized = calc.normalization(input_data)
impedance = calc.calculate(normalized, integration_range=200)
```

### Option C — Lightweight solver (for GAN pipeline)

```python
from eddy_current_workflow.forward.edc_solver import edc_forward, ProbeSettings
import numpy as np

settings = ProbeSettings(frequency=1e6)
sigma_layers = np.array([1e7, 2e7, 1e7])
mu_layers = np.array([1.0, 1.0, 1.0])

response = edc_forward(sigma_layers, mu_layers, settings)
print(response)  # EDCResponse with real/imag impedance
```

---

## Step 4: Run the LUT Pipeline (Bulk Dataset Generation)

The LUT method generates large datasets (up to 10,000 samples) for training:

```
1. Generate low-discrepancy sequences:
   method_lut/1_create_r_sequences.py

2. Scale sequences to physical parameter ranges:
   method_lut/2_create_scaled_r.py

3. Create model input data (JSON):
   method_lut/3_create_model_input_data.py

4. Compute analytical results (multiprocessing):
   method_lut/4_calc_analytic_result.py

5. Convert to CSV for model training:
   method_lut/_0_moded_input_data_to_csv.py
```

// turbo
Run the full pipeline:
```bash
cd dodd_analytical_model
python method_lut/1_create_r_sequences.py
python method_lut/2_create_scaled_r.py
python method_lut/3_create_model_input_data.py
python method_lut/4_calc_analytic_result.py
```

Output: `method_lut/datasets_output/calculated/full_data_q{N}_var{V}_{V}.json`

---

## Step 5: Validate Results

Run the existing tests:

```bash
cd tests
python test_dodd_deeds_solver.py
```

Or from the analytical model directory:

```bash
cd dodd_analytical_model
python impendance_dodd69_test.py
```

Reference values for validation:
- Dodd69 impedance: `real ≈ 0.102937, imaginary ≈ 0.671650` (2-coil reference)
- Compare against COMSOL results in `dodd_analytical_model/csv_compare_comsol_python.py`

---

## Step 6: Connect to the GAN Training Loop

The forward solver feeds into the GAN pipeline:

```
GAN generates (σ, μ) profiles
  → Forward solver computes impedance Z
  → Physics loss = ||Z_generated - Z_expected||
  → Backpropagate to improve generator
```

Use `eddy_current_workflow/forward/edc_solver.py` for this integration — it is designed for batched evaluation within the training loop.

---

## Key Classes Reference

| Class | File | Purpose |
|---|---|---|
| `VectorPotentialInsideCoilGreenFunction` | `vector_potential_calculation/vector_potential_inside_coil_green_function.py` | Main vector potential calculator (ORNL-5220 eq.49) |
| `ImpendanceDodd69` | `impendance_calculation/impendance_dodd_69.py` | Impedance calculator (ORNL-4384 eq.3.155) |
| `VectorPotentialParts` | `vector_potential_calculation/vector_potential_parts.py` | D(α) and S(α) functions for 2-layer model |
| `NormalizationFunctions` | `calculation_helpers/normalization_functions.py` | Parameter normalization by mean coil radius |
| `MatricesCalculation` | `special_functions/matrices_calculation.py` | U/V transfer matrices for multilayer conductors |
| `IntegralBesselFunctions` | `special_functions/integral_bessel_functions.py` | Bessel function integrals ∫xJ₁(x)dx, ∫xK₁(x)dx |
| `GaussIntegration` | `calculation_helpers/gauss_integration.py` | N-point Gauss quadrature |
| `AdaptiveIntegrationFunctions` | `calculation_helpers/adaptive_integration_functions.py` | Adaptive integration with convergence testing |
| `edc_forward` | `eddy_current_workflow/forward/edc_solver.py` | Lightweight forward solver for GAN pipeline |
