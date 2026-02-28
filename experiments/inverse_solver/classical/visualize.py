"""
Visualization utilities for inverse solver experiments.

Provides plotting functions for:
  - Profile comparison (true vs recovered)
  - Impedance comparison (target vs recovered)
  - Convergence diagnostics
  - Multi-frequency analysis
  - Noise robustness trends
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, List, Tuple

from eddy_current_workflow.forward.edc_solver import edc_forward, ProbeSettings, EDCResponse
from eddy_current_workflow.inverse.optimizers import InverseResult


def setup_plot_style():
    """Configure matplotlib for clean, publication-ready plots."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'lines.linewidth': 2,
        'lines.markersize': 6,
    })


def plot_profile_comparison(
    sigma_true: np.ndarray,
    mu_true: np.ndarray,
    sigma_rec: np.ndarray,
    mu_rec: np.ndarray,
    layer_thickness: float,
    save_path: Optional[Path] = None,
    title: str = "Profile Recovery",
    show_tolerance: bool = True,
    tolerance_pct: float = 20.0,
):
    """
    Plot true vs recovered conductivity and permeability profiles.
    
    Args:
        sigma_true: True conductivity profile (K,)
        mu_true: True permeability profile (K,)
        sigma_rec: Recovered conductivity profile (K,)
        mu_rec: Recovered permeability profile (K,)
        layer_thickness: Thickness per layer (m)
        save_path: Optional path to save figure
        title: Figure title
    """
    setup_plot_style()
    
    K = len(sigma_true)
    depths = np.arange(K) * layer_thickness * 1e3
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(depths, sigma_true / 1e6, 'o-', label='True', color='#2E86AB', linewidth=2.5, markersize=8, zorder=3)
    ax1.plot(depths, sigma_rec / 1e6, 's--', label='Recovered', color='#A23B72', linewidth=2, markersize=7, zorder=3)
    
    if show_tolerance:
        sigma_upper = sigma_true * (1 + tolerance_pct / 100) / 1e6
        sigma_lower = sigma_true * (1 - tolerance_pct / 100) / 1e6
        ax1.fill_between(depths, sigma_lower, sigma_upper, alpha=0.15, color='#2E86AB', 
                         label=f'±{tolerance_pct:.0f}% tolerance', zorder=1)
    
    ax1.set_xlabel('Depth (mm)', fontweight='bold')
    ax1.set_ylabel('Conductivity σ (MS/m)', fontweight='bold')
    ax1.set_title('Conductivity Profile', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(depths[0] - 0.01, depths[-1] + 0.01)
    
    ax2.plot(depths, mu_true, 'o-', label='True', color='#2E86AB', linewidth=2.5, markersize=8, zorder=3)
    ax2.plot(depths, mu_rec, 's--', label='Recovered', color='#A23B72', linewidth=2, markersize=7, zorder=3)
    
    if show_tolerance:
        mu_upper = mu_true * (1 + tolerance_pct / 100)
        mu_lower = mu_true * (1 - tolerance_pct / 100)
        ax2.fill_between(depths, mu_lower, mu_upper, alpha=0.15, color='#2E86AB',
                         label=f'±{tolerance_pct:.0f}% tolerance', zorder=1)
    
    ax2.set_xlabel('Depth (mm)', fontweight='bold')
    ax2.set_ylabel('Relative Permeability μᵣ', fontweight='bold')
    ax2.set_title('Permeability Profile', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim(depths[0] - 0.01, depths[-1] + 0.01)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='μ=1 (non-magnetic)')
    
    model_info = "Model: edc_forward() — Dodd-Deeds (1968) lightweight solver"
    fig.text(0.5, 0.02, model_info, ha='center', fontsize=9, style='italic', color='gray')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sigma_comparison(
    sigma_true: np.ndarray,
    sigma_rec: np.ndarray,
    layer_thickness: float,
    save_path: Optional[Path] = None,
    title: str = "Conductivity Comparison",
    show_tolerance: bool = True,
    tolerance_pct: float = 20.0,
):
    """
    Plot true vs recovered conductivity profile (standalone).
    
    Args:
        sigma_true: True conductivity profile (K,)
        sigma_rec: Recovered conductivity profile (K,)
        layer_thickness: Thickness per layer (m)
        save_path: Optional path to save figure
        title: Figure title
        show_tolerance: Show tolerance bands
        tolerance_pct: Tolerance percentage for bands
    """
    setup_plot_style()
    
    K = len(sigma_true)
    depths = np.arange(K) * layer_thickness * 1e3
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(depths, sigma_true / 1e6, 'o-', label='True σ', color='#2E86AB', linewidth=2.5, markersize=8, zorder=3)
    ax.plot(depths, sigma_rec / 1e6, 's--', label='Recovered σ', color='#A23B72', linewidth=2, markersize=7, zorder=3)
    
    if show_tolerance:
        sigma_upper = sigma_true * (1 + tolerance_pct / 100) / 1e6
        sigma_lower = sigma_true * (1 - tolerance_pct / 100) / 1e6
        ax.fill_between(depths, sigma_lower, sigma_upper, alpha=0.15, color='#2E86AB', 
                        label=f'±{tolerance_pct:.0f}% tolerance', zorder=1)
    
    ax.set_xlabel('Depth (mm)', fontweight='bold')
    ax.set_ylabel('Conductivity σ (MS/m)', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(depths[0] - 0.01, depths[-1] + 0.01)
    
    rmse = np.sqrt(np.mean((sigma_true - sigma_rec)**2))
    rel_rmse = rmse / np.mean(sigma_true) * 100
    
    textstr = f'RMSE: {rmse:.3e} S/m\nRelative: {rel_rmse:.2f}%'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    model_info = "Model: edc_forward() — Dodd-Deeds (1968) lightweight solver"
    fig.text(0.5, 0.01, model_info, ha='center', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_mu_comparison(
    mu_true: np.ndarray,
    mu_rec: np.ndarray,
    layer_thickness: float,
    save_path: Optional[Path] = None,
    title: str = "Permeability Comparison",
    show_tolerance: bool = True,
    tolerance_pct: float = 20.0,
):
    """
    Plot true vs recovered permeability profile (standalone).
    
    Args:
        mu_true: True permeability profile (K,)
        mu_rec: Recovered permeability profile (K,)
        layer_thickness: Thickness per layer (m)
        save_path: Optional path to save figure
        title: Figure title
        show_tolerance: Show tolerance bands
        tolerance_pct: Tolerance percentage for bands
    """
    setup_plot_style()
    
    K = len(mu_true)
    depths = np.arange(K) * layer_thickness * 1e3
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(depths, mu_true, 'o-', label='True μ', color='#2E86AB', linewidth=2.5, markersize=8, zorder=3)
    ax.plot(depths, mu_rec, 's--', label='Recovered μ', color='#A23B72', linewidth=2, markersize=7, zorder=3)
    
    if show_tolerance:
        mu_upper = mu_true * (1 + tolerance_pct / 100)
        mu_lower = mu_true * (1 - tolerance_pct / 100)
        ax.fill_between(depths, mu_lower, mu_upper, alpha=0.15, color='#2E86AB',
                        label=f'±{tolerance_pct:.0f}% tolerance', zorder=1)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='μ=1 (non-magnetic)')
    
    ax.set_xlabel('Depth (mm)', fontweight='bold')
    ax.set_ylabel('Relative Permeability μᵣ', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(depths[0] - 0.01, depths[-1] + 0.01)
    
    rmse = np.sqrt(np.mean((mu_true - mu_rec)**2))
    rel_rmse = rmse / np.mean(mu_true) * 100
    
    textstr = f'RMSE: {rmse:.3e}\nRelative: {rel_rmse:.2f}%'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    model_info = "Model: edc_forward() — Dodd-Deeds (1968) lightweight solver"
    fig.text(0.5, 0.01, model_info, ha='center', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_impedance_comparison(
    frequencies: np.ndarray,
    targets: List[EDCResponse],
    recovered: List[EDCResponse],
    save_path: Optional[Path] = None,
    title: str = "Impedance Comparison",
):
    """
    Plot target vs recovered impedance (real, imag, magnitude, phase).
    
    Args:
        frequencies: Measurement frequencies (Hz)
        targets: Target EDC responses
        recovered: Recovered EDC responses
        save_path: Optional path to save figure
        title: Figure title
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    z_target = np.array([t.impedance_complex for t in targets])
    z_rec = np.array([r.impedance_complex for r in recovered])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(frequencies / 1e3, z_target.real, 'o-', label='Target', color='#2E86AB', linewidth=2.5)
    ax1.plot(frequencies / 1e3, z_rec.real, 's--', label='Recovered', color='#A23B72', linewidth=2)
    ax1.set_xlabel('Frequency (kHz)')
    ax1.set_ylabel('Re(ΔZ) (Ω)')
    ax1.set_title('Real Part')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(frequencies / 1e3, z_target.imag, 'o-', label='Target', color='#2E86AB', linewidth=2.5)
    ax2.plot(frequencies / 1e3, z_rec.imag, 's--', label='Recovered', color='#A23B72', linewidth=2)
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Im(ΔZ) (Ω)')
    ax2.set_title('Imaginary Part')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(frequencies / 1e3, np.abs(z_target), 'o-', label='Target', color='#2E86AB', linewidth=2.5)
    ax3.plot(frequencies / 1e3, np.abs(z_rec), 's--', label='Recovered', color='#A23B72', linewidth=2)
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('|ΔZ| (Ω)')
    ax3.set_title('Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(frequencies / 1e3, np.angle(z_target, deg=True), 'o-', label='Target', color='#2E86AB', linewidth=2.5)
    ax4.plot(frequencies / 1e3, np.angle(z_rec, deg=True), 's--', label='Recovered', color='#A23B72', linewidth=2)
    ax4.set_xlabel('Frequency (kHz)')
    ax4.set_ylabel('∠ΔZ (degrees)')
    ax4.set_title('Phase')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, :])
    errors = np.abs(z_rec - z_target)
    rel_errors = errors / np.abs(z_target) * 100
    ax5.semilogy(frequencies / 1e3, errors, 'o-', label='Absolute error', color='#F18F01', linewidth=2)
    ax5_twin = ax5.twinx()
    ax5_twin.plot(frequencies / 1e3, rel_errors, 's--', label='Relative error (%)', color='#C73E1D', linewidth=2)
    ax5.set_xlabel('Frequency (kHz)')
    ax5.set_ylabel('|ΔZ_rec - ΔZ_target| (Ω)', color='#F18F01')
    ax5_twin.set_ylabel('Relative error (%)', color='#C73E1D')
    ax5.set_title('Impedance Reconstruction Error')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_noise_robustness(
    noise_levels: np.ndarray,
    mismatches: np.ndarray,
    sigma_rmse: np.ndarray,
    mu_rmse: np.ndarray,
    z_errors: np.ndarray,
    save_path: Optional[Path] = None,
):
    """
    Plot how solver performance degrades with measurement noise.
    
    Args:
        noise_levels: Noise fractions (0.0 = 0%, 0.01 = 1%)
        mismatches: Objective function values
        sigma_rmse: Conductivity RMSE values
        mu_rmse: Permeability RMSE values
        z_errors: Impedance reconstruction errors
        save_path: Optional path to save figure
    """
    setup_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    noise_pct = noise_levels * 100
    
    ax1.semilogy(noise_pct, mismatches, 'o-', color='#2E86AB', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Noise Level (%)')
    ax1.set_ylabel('Mismatch J(θ)')
    ax1.set_title('Objective Function vs Noise')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(noise_pct, sigma_rmse / 1e6, 'o-', label='σ RMSE', color='#A23B72', linewidth=2.5, markersize=8)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(noise_pct, mu_rmse, 's--', label='μ RMSE', color='#F18F01', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Noise Level (%)')
    ax2.set_ylabel('σ RMSE (MS/m)', color='#A23B72')
    ax2_twin.set_ylabel('μ RMSE', color='#F18F01')
    ax2.set_title('Profile RMSE vs Noise')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    ax3.plot(noise_pct, z_errors, 'o-', color='#C73E1D', linewidth=2.5, markersize=8)
    ax3.set_xlabel('Noise Level (%)')
    ax3.set_ylabel('|ΔZ_rec - ΔZ_clean| (Ω)')
    ax3.set_title('Impedance Error vs Noise')
    ax3.grid(True, alpha=0.3)
    
    if len(noise_levels) > 1:
        fit = np.polyfit(noise_levels[1:], z_errors[1:], 1)
        ax4.plot(noise_pct, z_errors, 'o', label='Measured', color='#C73E1D', markersize=8)
        ax4.plot(noise_pct, np.polyval(fit, noise_levels), '--', label=f'Linear fit (slope={fit[0]:.2f})', 
                 color='#2E86AB', linewidth=2)
        ax4.set_xlabel('Noise Level (%)')
        ax4.set_ylabel('|ΔZ_rec - ΔZ_clean| (Ω)')
        ax4.set_title('Noise Sensitivity (linearity check)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    fig.suptitle('Noise Robustness Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_convergence_diagnostics(
    result: InverseResult,
    save_path: Optional[Path] = None,
    show_all_attempts: bool = True,
):
    """
    Plot convergence diagnostics from multi-start optimization.
    
    Args:
        result: InverseResult with all_mismatches populated
        save_path: Optional path to save figure
    """
    setup_plot_style()
    
    if not result.all_mismatches:
        print("  No convergence data available (all_mismatches empty)")
        return
    
    n_plots = 3 if show_all_attempts else 2
    fig = plt.figure(figsize=(16, 5) if show_all_attempts else (12, 5))
    
    if show_all_attempts:
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
    else:
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
    
    n_starts = len(result.all_mismatches)
    starts = np.arange(1, n_starts + 1)
    
    threshold = result.mismatch * 10
    colors = ['#C73E1D' if m > threshold else '#2E86AB' for m in result.all_mismatches]
    bars = ax1.bar(starts, result.all_mismatches, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    best_idx = np.argmin(result.all_mismatches)
    bars[best_idx].set_edgecolor('#F18F01')
    bars[best_idx].set_linewidth(3)
    
    ax1.axhline(result.mismatch, color='#F18F01', linestyle='--', linewidth=2.5, 
                label=f'Best: {result.mismatch:.2e}', zorder=10)
    ax1.axhline(threshold, color='#C73E1D', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'10× threshold: {threshold:.2e}')
    
    ax1.set_xlabel('Start Number', fontweight='bold')
    ax1.set_ylabel('Final Mismatch J(θ)', fontweight='bold')
    ax1.set_title('Multi-Start Final Mismatches', fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    n_good = sum(1 for m in result.all_mismatches if m <= threshold)
    ax1.text(0.02, 0.98, f'{n_good}/{n_starts} converged well', 
             transform=ax1.transAxes, va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    sorted_mismatches = sorted(result.all_mismatches)
    ax2.semilogy(starts, sorted_mismatches, 'o-', color='#2E86AB', linewidth=2.5, markersize=8)
    ax2.axhline(result.mismatch, color='#F18F01', linestyle='--', linewidth=2.5, label='Best solution')
    ax2.axhline(threshold, color='#C73E1D', linestyle=':', linewidth=1.5, alpha=0.7, label='10× threshold')
    ax2.set_xlabel('Rank', fontweight='bold')
    ax2.set_ylabel('Mismatch J(θ)', fontweight='bold')
    ax2.set_title('Sorted Multi-Start Results', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    if show_all_attempts and len(result.all_mismatches) > 1:
        ax3.hist(np.log10(result.all_mismatches), bins=min(15, n_starts), 
                 color='#2E86AB', alpha=0.7, edgecolor='black')
        ax3.axvline(np.log10(result.mismatch), color='#F18F01', linestyle='--', 
                    linewidth=2.5, label=f'Best: 10^{np.log10(result.mismatch):.1f}')
        ax3.set_xlabel('log₁₀(Mismatch)', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Distribution of Final Mismatches', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y', linestyle=':')
        
        median_mismatch = np.median(result.all_mismatches)
        ax3.axvline(np.log10(median_mismatch), color='orange', linestyle=':', 
                    linewidth=2, label=f'Median: 10^{np.log10(median_mismatch):.1f}')
        ax3.legend()
    
    success_rate = result.convergence_rate * 100 if result.convergence_rate > 0 else (n_good / n_starts * 100)
    fig.suptitle(f'Convergence Diagnostics — {n_starts} starts, {result.n_feval} evaluations, {success_rate:.0f}% success rate', 
                 fontsize=13, fontweight='bold')
    
    model_info = f"Optimizer: {result.method} | Model: edc_forward() inverse solver"
    fig.text(0.5, 0.02, model_info, ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_decision_criteria(
    edc_target: 'EDCResponse',
    edc_recovered: 'EDCResponse',
    tolerance: float,
    criterion_type: str = "absolute",
    save_path: Optional[Path] = None,
):
    """
    Visualize the decision criteria: what we consider a "match".
    
    Args:
        edc_target: Target impedance measurement
        edc_recovered: Recovered impedance from inverse solver
        tolerance: Pass/fail tolerance threshold
        criterion_type: "absolute" (Ω) or "relative" (%)
        save_path: Optional path to save figure
    """
    setup_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    z_target = edc_target.impedance_complex
    z_rec = edc_recovered.impedance_complex
    error = abs(z_rec - z_target)
    
    if criterion_type == "relative":
        error_pct = (error / abs(z_target)) * 100
        passed = error_pct < tolerance
        status_text = f"Relative error: {error_pct:.2f}% (tolerance: {tolerance:.1f}%)"
    else:
        passed = error < tolerance
        status_text = f"Absolute error: {error:.2e} Ω (tolerance: {tolerance:.2e} Ω)"
    
    # Panel 1: Complex plane
    ax1.plot(z_target.real, z_target.imag, 'o', markersize=15, color='#2E86AB', 
             label='Target', zorder=5)
    ax1.plot(z_rec.real, z_rec.imag, 's', markersize=12, color='#A23B72', 
             label='Recovered', zorder=5)
    
    circle = plt.Circle((z_target.real, z_target.imag), tolerance if criterion_type == "absolute" else abs(z_target) * tolerance / 100,
                        color='green' if passed else 'red', alpha=0.2, 
                        label=f'Tolerance zone ({"PASS" if passed else "FAIL"})')
    ax1.add_patch(circle)
    
    ax1.plot([z_target.real, z_rec.real], [z_target.imag, z_rec.imag], 
             'k--', linewidth=1.5, alpha=0.5, label=f'Error: {error:.2e} Ω')
    
    ax1.set_xlabel('Re(ΔZ) (Ω)', fontweight='bold')
    ax1.set_ylabel('Im(ΔZ) (Ω)', fontweight='bold')
    ax1.set_title('Complex Impedance Plane', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.axis('equal')
    
    # Panel 2: Real and Imaginary parts
    parts = ['Real', 'Imag']
    target_vals = [z_target.real, z_target.imag]
    rec_vals = [z_rec.real, z_rec.imag]
    x_pos = np.arange(len(parts))
    
    width = 0.35
    bars1 = ax2.bar(x_pos - width/2, target_vals, width, label='Target', color='#2E86AB', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, rec_vals, width, label='Recovered', color='#A23B72', alpha=0.8)
    
    ax2.set_xlabel('Component', fontweight='bold')
    ax2.set_ylabel('Impedance (Ω)', fontweight='bold')
    ax2.set_title('Real vs Imaginary Components', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(parts)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # Panel 3: Magnitude and Phase
    mag_target = abs(z_target)
    mag_rec = abs(z_rec)
    phase_target = np.angle(z_target, deg=True)
    phase_rec = np.angle(z_rec, deg=True)
    
    metrics = ['|ΔZ| (Ω)', '∠ΔZ (°)']
    target_metrics = [mag_target, phase_target]
    rec_metrics = [mag_rec, phase_rec]
    x_pos = np.arange(len(metrics))
    
    bars1 = ax3.bar(x_pos - width/2, target_metrics, width, label='Target', color='#2E86AB', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, rec_metrics, width, label='Recovered', color='#A23B72', alpha=0.8)
    
    ax3.set_xlabel('Metric', fontweight='bold')
    ax3.set_ylabel('Value', fontweight='bold')
    ax3.set_title('Magnitude and Phase', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # Panel 4: Decision summary
    ax4.axis('off')
    
    summary_text = f"""
    DECISION CRITERIA
    {'='*50}
    
    Target Impedance:
      ΔZ = {z_target.real:.4e} {z_target.imag:+.4e}j Ω
      |ΔZ| = {mag_target:.4e} Ω
      ∠ΔZ = {phase_target:.2f}°
    
    Recovered Impedance:
      ΔZ = {z_rec.real:.4e} {z_rec.imag:+.4e}j Ω
      |ΔZ| = {mag_rec:.4e} Ω
      ∠ΔZ = {phase_rec:.2f}°
    
    Error Metrics:
      Absolute: {error:.4e} Ω
      Relative: {(error/abs(z_target)*100):.2f}%
      Re error: {abs(z_rec.real - z_target.real):.4e} Ω
      Im error: {abs(z_rec.imag - z_target.imag):.4e} Ω
    
    {status_text}
    
    Result: {'✓ PASS' if passed else '✗ FAIL'}
    
    Model: edc_forward() — Dodd-Deeds (1968)
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if passed else 'lightcoral', alpha=0.3))
    
    fig.suptitle('Decision Criteria — What We Consider a Match', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_experiment_summary(
    exp_name: str,
    sigma_true: np.ndarray,
    mu_true: np.ndarray,
    result: InverseResult,
    probe: ProbeSettings,
    layer_thickness: float,
    frequencies: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
):
    """
    Generate all relevant plots for an experiment.
    
    Args:
        exp_name: Experiment name (used in titles and filenames)
        sigma_true: True conductivity profile
        mu_true: True permeability profile
        result: InverseResult from solver
        probe: Probe settings
        layer_thickness: Layer thickness (m)
        frequencies: Optional array of frequencies for multi-freq experiments
        save_dir: Directory to save plots (creates if doesn't exist)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations for {exp_name}...")
    
    plot_profile_comparison(
        sigma_true, mu_true, result.sigma, result.mu, layer_thickness,
        save_path=save_dir / f"{exp_name}_profiles.png" if save_dir else None,
        title=f"{exp_name} — Profile Recovery"
    )
    
    plot_sigma_comparison(
        sigma_true, result.sigma, layer_thickness,
        save_path=save_dir / f"{exp_name}_sigma.png" if save_dir else None,
        title=f"{exp_name} — Conductivity σ"
    )
    
    plot_mu_comparison(
        mu_true, result.mu, layer_thickness,
        save_path=save_dir / f"{exp_name}_mu.png" if save_dir else None,
        title=f"{exp_name} — Permeability μ"
    )
    
    if frequencies is not None and len(frequencies) > 1:
        from dataclasses import replace
        targets = [
            edc_forward(sigma_true, mu_true, replace(probe, frequency=float(f)), layer_thickness, n_quad=100)
            for f in frequencies
        ]
        recovered = [
            edc_forward(result.sigma, result.mu, replace(probe, frequency=float(f)), layer_thickness, n_quad=100)
            for f in frequencies
        ]
        plot_impedance_comparison(
            frequencies, targets, recovered,
            save_path=save_dir / f"{exp_name}_impedance.png" if save_dir else None,
            title=f"{exp_name} — Impedance Comparison"
        )
    
    if result.all_mismatches:
        plot_convergence_diagnostics(
            result,
            save_path=save_dir / f"{exp_name}_convergence.png" if save_dir else None,
        )
    
    print(f"✓ Visualization complete for {exp_name}")
