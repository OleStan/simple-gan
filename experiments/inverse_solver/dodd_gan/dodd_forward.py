"""
Forward solver adapter for dodd_analytical_model.

Wraps VectorPotentialInsideCoilGreenFunction so it can be called from
outside the dodd_analytical_model directory, accepting (sigma_layers,
mu_layers) arrays used by the rest of the inverse-solver pipeline.

Pipeline (matches the LUT dataset generation in method_lut/):
    1. Build raw param dict with K inner cylindrical layers
    2. normalization_1_coil__sigma_mu_r() → normalised params
    3. VectorPotentialInsideCoilGreenFunction.calculate() → complex A
    4. VoltageFromVectorPotential.from_delta_func() → complex voltage V

The dodd_analytical_model uses cylindrical geometry: K radial shells
inside the coil. The K layers from the GAN (σ, μ profiles) are mapped
to K cylindrical shells starting from the coil inner radius.

Requires:
  - dodd_analytical_model/ on sys.path (relative imports throughout)
  - logs/ directory inside dodd_analytical_model/ (logging setup)
"""

import io
import sys
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parents[3]
DODD_DIR = ROOT / "dodd_analytical_model"


def _ensure_logs_dir() -> None:
    (DODD_DIR / "logs").mkdir(exist_ok=True)


@contextlib.contextmanager
def _dodd_syspath():
    """
    Temporarily prepend dodd_analytical_model/ to sys.path AND chdir into it.

    The chdir is required because gauss_integration.py opens
    'logs/gauss_integration.log' relative to the current working directory
    at module-import time.  Without it the import fails with FileNotFoundError.
    """
    import os
    dodd_str = str(DODD_DIR)
    prev_cwd = os.getcwd()
    inserted = dodd_str not in sys.path
    if inserted:
        sys.path.insert(0, dodd_str)
    try:
        os.chdir(dodd_str)
        yield
    finally:
        os.chdir(prev_cwd)
        if inserted and dodd_str in sys.path:
            sys.path.remove(dodd_str)


@dataclass
class DoddProbeSettings:
    """
    Probe and conductor geometry, all in SI units (metres).

    Fields
    ------
    frequency_hz   : excitation frequency in Hz
    coil_r1        : coil inner radius (m)
    coil_r2        : coil outer radius (m)
    coil_l1        : coil lower axial position (m)  — 0 = surface of conductor
    coil_l2        : coil upper axial position (m)
    conductor_r1   : inner radius of the conductor shell stack (m)
    conductor_r2   : outer radius of the conductor shell stack (m)
    calc_r         : radial position of the calc point (m)
    calc_z         : axial position of the calc point (m)
    n_turns        : number of coil turns
    """
    frequency_hz: float
    coil_r1: float
    coil_r2: float
    coil_l1: float
    coil_l2: float
    conductor_r1: float
    conductor_r2: float
    calc_r: float
    calc_z: float
    n_turns: int

    @property
    def coil_height(self) -> float:
        return self.coil_l2 - self.coil_l1

    @property
    def coil_mean_radius(self) -> float:
        return 0.5 * (self.coil_r1 + self.coil_r2)

    # --- legacy aliases kept for base.py display prints ---
    @property
    def inner_radius(self) -> float:
        return self.coil_r1

    @property
    def outer_radius(self) -> float:
        return self.coil_r2

    @property
    def lift_off(self) -> float:
        return self.coil_l1

    @property
    def l1(self) -> float:
        return self.coil_l1

    @property
    def l2(self) -> float:
        return self.coil_l2

    @property
    def mean_radius(self) -> float:
        return self.coil_mean_radius


@dataclass
class DoddResponse:
    """Complex voltage response from the dodd_analytical_model forward solver."""
    frequency: float
    voltage_real: float
    voltage_imag: float

    @property
    def voltage_complex(self) -> complex:
        return complex(self.voltage_real, self.voltage_imag)

    @property
    def amplitude(self) -> float:
        return abs(self.voltage_complex)

    @property
    def impedance_complex(self) -> complex:
        """Alias so the base class optimizer can use .impedance_complex."""
        return self.voltage_complex


# Probe geometry matching the LUT dataset generation pipeline
# (dodd_analytical_model/method_lut/3_create_model_input_data.py)
# All values in metres (SI).
PROBE_DEFAULT = DoddProbeSettings(
    frequency_hz=2500.0,
    coil_r1=0.016,
    coil_r2=0.021,
    coil_l1=0.0475,
    coil_l2=0.0525,
    conductor_r1=0.009,
    conductor_r2=0.010,
    calc_r=0.0135,
    calc_z=0.05,
    n_turns=100,
)


def _build_raw_params(
    sigma_layers: np.ndarray,
    mu_layers: np.ndarray,
    layer_thickness: float,
    probe: DoddProbeSettings,
) -> dict:
    """
    Build the raw parameter dict expected by normalization_1_coil__sigma_mu_r.

    All geometry values passed in metres (SI), matching the convention used by
    get_raw_input_parameters() in the LUT pipeline.  The normalizer
    (normalization_1_coil__sigma_mu_r) works with raw SI values — it does NOT
    expect inches.  The legacy `normalization()` method expected inches, but
    normalization_1_coil__sigma_mu_r uses σ directly in the m-parameter formula:
        m = 2π·μ₀·f·μᵣ·σ·rb²
    which is unit-consistent in SI.

    inner_parameters: K cylindrical shells spanning conductor_r1 → conductor_r2.
    calc_point: evaluation point for the vector potential (metres).
    """
    K = len(sigma_layers)
    r_start = probe.conductor_r1

    inner_parameters = [
        {
            "r": r_start + k * layer_thickness,
            "mu_r": float(mu_layers[k]),
            "sigma": float(sigma_layers[k]),
        }
        for k in range(K)
    ]

    return {
        "inner_parameters": inner_parameters,
        "outer_parameters": [],
        "coils_parameters": [
            {
                "r1": probe.coil_r1,
                "r2": probe.coil_r2,
                "l1": probe.coil_l1,
                "l2": probe.coil_l2,
                "n": probe.n_turns,
            }
        ],
        "frequency": [probe.frequency_hz],
        "electrical_values": {"current": 1},
        "calc_point": {"r": probe.calc_r, "z": probe.calc_z},
    }


def dodd_forward(
    sigma_layers: np.ndarray,
    mu_layers: np.ndarray,
    probe: DoddProbeSettings,
    layer_thickness: Optional[float] = None,
    integ_top_range: int = 50,
) -> DoddResponse:
    """
    Compute coil voltage response using the original dodd_analytical_model.

    Uses VectorPotentialInsideCoilGreenFunction (N-layer cylindrical path,
    ORNL-5220) followed by VoltageFromVectorPotential conversion.

    This is the same pipeline as the LUT dataset generator
    (dodd_analytical_model/method_lut/4_calc_analytic_result.py).

    Args:
        sigma_layers: Conductivity per layer (K,) in S/m
        mu_layers: Relative permeability per layer (K,)
        probe: Probe geometry and frequency
        layer_thickness: Uniform layer thickness (m). Defaults to 1mm/K.
        integ_top_range: Upper limit of adaptive integration (dimensionless).
            Typical range: 20 (fast, optimisation) to 50 (accurate, verification).

    Returns:
        DoddResponse with complex voltage V = jω·2π·r·A
    """
    sigma_layers = np.asarray(sigma_layers, dtype=np.float64)
    mu_layers = np.asarray(mu_layers, dtype=np.float64)

    K = len(sigma_layers)
    if layer_thickness is None:
        layer_thickness = 1e-3 / K

    _ensure_logs_dir()

    with _dodd_syspath():
        from vector_potential_calculation.vector_potential_inside_coil_green_function import (
            VectorPotentialInsideCoilGreenFunction,
        )
        from calculation_helpers.voltage_from_vector_potential import VoltageFromVectorPotential

        raw_params = _build_raw_params(sigma_layers, mu_layers, layer_thickness, probe)

        calc_obj = VectorPotentialInsideCoilGreenFunction()
        norm_params = calc_obj.normalization_1_coil__sigma_mu_r(raw_params)

        # Suppress all stdout and logging from dodd_analytical_model internals.
        # cycles_adaptive_integration.py contains unconditional print() calls
        # on every integration step that would flood the console.
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                vec_pot = calc_obj.calculate(norm_params, integ_top_range)
        finally:
            logging.disable(logging.NOTSET)

        # VoltageFromVectorPotential uses the normalised calc_r (post rrb scaling)
        # to compute V = jω·2π·r·A.  The LUT pipeline reads calc_point['r'] from
        # the packed_data dict after normalization_1_coil__sigma_mu_r has mutated it
        # (multiplied by rrb).  We replicate that here.
        frequency = raw_params["frequency"][0]
        calc_r_norm = norm_params["calc_point"]["r"]
        voltage = VoltageFromVectorPotential.from_delta_func(vec_pot, frequency, calc_r_norm)

    return DoddResponse(
        frequency=probe.frequency_hz,
        voltage_real=float(voltage.real),
        voltage_imag=float(voltage.imag),
    )
