"""Forward eddy-current response calculation (Dodd-Deeds model)."""

from .edc_solver import (
    ProbeSettings,
    EDCResponse,
    edc_forward,
    edc_forward_multifreq,
    calculate_skin_depth,
    estimate_penetration_depth,
)

__all__ = [
    'ProbeSettings',
    'EDCResponse',
    'edc_forward',
    'edc_forward_multifreq',
    'calculate_skin_depth',
    'estimate_penetration_depth',
]
