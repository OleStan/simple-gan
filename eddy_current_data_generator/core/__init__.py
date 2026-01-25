"""Core module for eddy current data generation."""

from .roberts_sequence import calculate_phi, generate_roberts_plan
from .material_profiles import (
    ProfileType, 
    make_profile, 
    generate_dual_profiles
)
from .discretization import discretize_profile, discretize_dual_profiles
from .dataset_builder import DatasetConfig, build_dataset

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
