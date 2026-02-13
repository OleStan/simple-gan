"""
Physics consistency validation for GAN-generated profiles.

Implements criterion from GAN_Quality_Guide.md section 4.2:
- F(G(z)) ≈ y_sim: generated profiles should produce physically plausible
  eddy-current responses when passed through the forward Dodd-Deeds solver.

Also validates basic physical constraints:
- Conductivity must be positive
- Permeability must be >= 1.0
- Profiles should be within expected physical bounds
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from ..forward.edc_solver import edc_forward, ProbeSettings, EDCResponse


@dataclass
class PhysicsBoundsResult:
    sigma_in_bounds_ratio: float
    mu_in_bounds_ratio: float
    sigma_positive_ratio: float
    mu_valid_ratio: float
    n_samples_checked: int

    @property
    def passed(self) -> bool:
        return (
            self.sigma_in_bounds_ratio > 0.9
            and self.mu_in_bounds_ratio > 0.9
            and self.sigma_positive_ratio > 0.99
            and self.mu_valid_ratio > 0.99
        )


@dataclass
class ForwardConsistencyResult:
    n_samples_tested: int
    impedance_real_range: Tuple[float, float]
    impedance_imag_range: Tuple[float, float]
    n_valid_responses: int
    n_nan_responses: int
    n_inf_responses: int
    mean_amplitude: float
    std_amplitude: float
    reference_amplitude: Optional[float]
    amplitude_relative_error: Optional[float]

    @property
    def passed(self) -> bool:
        return (
            self.n_nan_responses == 0
            and self.n_inf_responses == 0
            and self.n_valid_responses == self.n_samples_tested
        )


@dataclass
class PhysicsConsistencyResult:
    bounds_result: PhysicsBoundsResult
    forward_result: ForwardConsistencyResult

    @property
    def passed(self) -> bool:
        return self.bounds_result.passed and self.forward_result.passed


def check_physics_bounds(
    generated_data: np.ndarray,
    K: int,
    sigma_bounds: Tuple[float, float] = (1e6, 6e7),
    mu_bounds: Tuple[float, float] = (1.0, 100.0),
) -> PhysicsBoundsResult:
    gen_sigma = generated_data[:, :K]
    gen_mu = generated_data[:, K:2*K]
    n_samples = len(generated_data)

    sigma_in_bounds = np.all(
        (gen_sigma >= sigma_bounds[0]) & (gen_sigma <= sigma_bounds[1]),
        axis=1,
    )
    mu_in_bounds = np.all(
        (gen_mu >= mu_bounds[0]) & (gen_mu <= mu_bounds[1]),
        axis=1,
    )
    sigma_positive = np.all(gen_sigma > 0, axis=1)
    mu_valid = np.all(gen_mu >= 1.0, axis=1)

    return PhysicsBoundsResult(
        sigma_in_bounds_ratio=float(sigma_in_bounds.sum() / n_samples),
        mu_in_bounds_ratio=float(mu_in_bounds.sum() / n_samples),
        sigma_positive_ratio=float(sigma_positive.sum() / n_samples),
        mu_valid_ratio=float(mu_valid.sum() / n_samples),
        n_samples_checked=n_samples,
    )


def check_forward_consistency(
    generated_data: np.ndarray,
    K: int,
    probe_settings: Optional[ProbeSettings] = None,
    layer_thickness: Optional[float] = None,
    reference_data: Optional[np.ndarray] = None,
    n_samples: int = 20,
) -> ForwardConsistencyResult:
    if probe_settings is None:
        probe_settings = ProbeSettings(frequency=1e6)

    if layer_thickness is None:
        layer_thickness = 1e-3 / K

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(generated_data), min(n_samples, len(generated_data)), replace=False)

    impedances_real = []
    impedances_imag = []
    amplitudes = []
    n_nan = 0
    n_inf = 0
    n_valid = 0

    for idx in sample_idx:
        sigma_layers = generated_data[idx, :K]
        mu_layers = generated_data[idx, K:2*K]

        sigma_layers = np.clip(sigma_layers, 1e4, 1e9)
        mu_layers = np.clip(mu_layers, 1.0, 1000.0)

        try:
            response = edc_forward(
                sigma_layers, mu_layers, probe_settings,
                layer_thickness=layer_thickness, n_quad=100,
            )

            if np.isnan(response.impedance_real) or np.isnan(response.impedance_imag):
                n_nan += 1
                continue
            if np.isinf(response.impedance_real) or np.isinf(response.impedance_imag):
                n_inf += 1
                continue

            impedances_real.append(response.impedance_real)
            impedances_imag.append(response.impedance_imag)
            amplitudes.append(response.amplitude)
            n_valid += 1
        except Exception:
            n_nan += 1

    reference_amplitude = None
    amplitude_rel_error = None

    if reference_data is not None and n_valid > 0:
        ref_amplitudes = []
        ref_idx = rng.choice(len(reference_data), min(n_samples, len(reference_data)), replace=False)
        for idx in ref_idx:
            sigma_layers = reference_data[idx, :K]
            mu_layers = reference_data[idx, K:2*K]
            try:
                response = edc_forward(
                    sigma_layers, mu_layers, probe_settings,
                    layer_thickness=layer_thickness, n_quad=100,
                )
                if not (np.isnan(response.amplitude) or np.isinf(response.amplitude)):
                    ref_amplitudes.append(response.amplitude)
            except Exception:
                pass

        if ref_amplitudes:
            reference_amplitude = float(np.mean(ref_amplitudes))
            gen_mean_amp = float(np.mean(amplitudes))
            amplitude_rel_error = abs(gen_mean_amp - reference_amplitude) / (reference_amplitude + 1e-10)

    real_range = (float(min(impedances_real)), float(max(impedances_real))) if impedances_real else (0.0, 0.0)
    imag_range = (float(min(impedances_imag)), float(max(impedances_imag))) if impedances_imag else (0.0, 0.0)

    return ForwardConsistencyResult(
        n_samples_tested=len(sample_idx),
        impedance_real_range=real_range,
        impedance_imag_range=imag_range,
        n_valid_responses=n_valid,
        n_nan_responses=n_nan,
        n_inf_responses=n_inf,
        mean_amplitude=float(np.mean(amplitudes)) if amplitudes else 0.0,
        std_amplitude=float(np.std(amplitudes)) if amplitudes else 0.0,
        reference_amplitude=reference_amplitude,
        amplitude_relative_error=amplitude_rel_error,
    )


def check_physics_consistency(
    generated_data: np.ndarray,
    K: int,
    sigma_bounds: Tuple[float, float] = (1e6, 6e7),
    mu_bounds: Tuple[float, float] = (1.0, 100.0),
    probe_settings: Optional[ProbeSettings] = None,
    reference_data: Optional[np.ndarray] = None,
    n_forward_samples: int = 20,
) -> PhysicsConsistencyResult:
    bounds_result = check_physics_bounds(generated_data, K, sigma_bounds, mu_bounds)

    forward_result = check_forward_consistency(
        generated_data, K,
        probe_settings=probe_settings,
        reference_data=reference_data,
        n_samples=n_forward_samples,
    )

    return PhysicsConsistencyResult(
        bounds_result=bounds_result,
        forward_result=forward_result,
    )
