"""Visualization module for eddy current data generation."""

from .profile_visualizer import (
    plot_continuous_profile,
    plot_discretized_profile,
    plot_dual_profiles,
    plot_parameter_space_coverage,
    plot_dataset_statistics,
    plot_multiple_profiles_comparison
)

__all__ = [
    'plot_continuous_profile',
    'plot_discretized_profile',
    'plot_dual_profiles',
    'plot_parameter_space_coverage',
    'plot_dataset_statistics',
    'plot_multiple_profiles_comparison'
]
