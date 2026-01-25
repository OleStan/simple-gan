"""Eddy Current Data Generator Package.

A Python package for generating synthetic datasets of electrical conductivity (σ)
and magnetic permeability (μ) profiles for eddy current NDT applications.
"""

from .core import (
    calculate_phi,
    generate_roberts_plan,
    ProfileType,
    make_profile,
    generate_dual_profiles,
    discretize_profile,
    discretize_dual_profiles,
    DatasetConfig,
    build_dataset
)

__version__ = '0.1.0'

__all__ = [
    'calculate_phi',
    'generate_roberts_plan',
    'ProfileType',
    'make_profile',
    'generate_dual_profiles',
    'discretize_profile',
    'discretize_dual_profiles',
    'DatasetConfig',
    'build_dataset'
]
