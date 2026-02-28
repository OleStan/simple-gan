"""
Base class for GAN-based inverse solver experiments.

All GAN experiments share:
  - Loading the trained Generator
  - Normalization / denormalization
  - The core latent-space optimization loop (scipy L-BFGS-B)
  - Result reporting and visualization
"""

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "improved_wgan_v2"))
sys.path.insert(0, str(ROOT / "experiments" / "inverse_solver"))

from eddy_current_workflow.forward.edc_solver import edc_forward, ProbeSettings, EDCResponse
from visualize import plot_profile_comparison, plot_sigma_comparison, plot_mu_comparison, plot_decision_criteria
from optimization_utils import run_single_restart_gan


MODEL_DIR = ROOT / "results" / "improved_wgan_v2_nz32_20260214_140817"
RESULTS_DIR = Path(__file__).parent / "results"

K = 51
NZ = 32
LAYER_THICKNESS = 1e-3 / K

FREQUENCIES_MULTI = np.array([1e5, 3e5, 7e5, 1e6])

PROBE_BASE = ProbeSettings(
    frequency=1e6,
    inner_radius=4e-3,
    outer_radius=6e-3,
    lift_off=0.5e-3,
    coil_height=2e-3,
    n_turns=100,
)

PROBE = PROBE_BASE  # alias kept for backward compat


@dataclass
class GANInverseResult:
    sigma_rec: np.ndarray
    mu_rec: np.ndarray
    best_z: np.ndarray
    best_mismatch: float
    all_mismatches: list
    elapsed: float
    n_restarts: int

    @property
    def passed(self) -> bool:
        return self.best_mismatch < 1e-5

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"GANInverseResult({status} | "
            f"|ΔZ|={self.best_mismatch:.3e} Ω | "
            f"{self.n_restarts} restarts | "
            f"{self.elapsed:.1f}s)"
        )


class GANInverseExperiment(ABC):
    """
    Base class for GAN-based inverse solver experiments.

    Subclasses must implement:
      - name: str — experiment identifier
      - description: str — one-line description
      - load_target() -> (sigma_true, mu_true, edc_target)
      - explain() — print explanation of what the experiment tests

    The optimization loop, visualization, and reporting are shared.
    """

    name: str = "gan_base"
    description: str = "GAN inverse experiment"
    impedance_tolerance: float = 1e-5

    def __init__(self, n_restarts: int = 10, n_iter: int = 500, seed: int = 42,
                 frequencies: Optional[np.ndarray] = None, fd_epsilon: float = 1e-3,
                 n_quad_opt: int = 20, n_quad_refine: int = 50, n_quad_verify: int = 150):
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.seed = seed
        self.frequencies = frequencies  # None → single-freq (backward compat)
        self.fd_epsilon = fd_epsilon    # finite-difference step for L-BFGS-B
        self.n_quad_opt = n_quad_opt        # quadrature points during multi-restart search (fast)
        self.n_quad_refine = n_quad_refine  # quadrature points for final refinement pass (medium)
        self.n_quad_verify = n_quad_verify  # quadrature points for final check (accurate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._netG: Optional[torch.nn.Module] = None
        self._norm: Optional[dict] = None
        self._run_timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def run_dir(self) -> Path:
        d = RESULTS_DIR / f"{self._run_timestamp}_{self.name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def netG(self) -> torch.nn.Module:
        if self._netG is None:
            self._netG = self._load_generator()
        return self._netG

    @property
    def norm(self) -> dict:
        if self._norm is None:
            with open(MODEL_DIR / "normalization_params.json") as f:
                self._norm = json.load(f)
        return self._norm

    def _load_generator(self) -> torch.nn.Module:
        from model import ConditionalConv1DGenerator

        netG = ConditionalConv1DGenerator(nz=NZ, K=K, conditional=False)
        state = torch.load(MODEL_DIR / "models" / "netG_final.pt", map_location=self.device)
        netG.load_state_dict(state)
        netG.to(self.device)
        netG.eval()
        for p in netG.parameters():
            p.requires_grad_(False)
        return netG

    def denormalize(self, sigma_norm: np.ndarray, mu_norm: np.ndarray) -> tuple:
        n = self.norm
        sigma = (sigma_norm + 1) / 2 * (n["sigma_max"] - n["sigma_min"]) + n["sigma_min"]
        mu = (mu_norm + 1) / 2 * (n["mu_max"] - n["mu_min"]) + n["mu_min"]
        return sigma, mu

    def _decode_z(self, z_np: np.ndarray) -> tuple:
        """Run z through Generator → physical (sigma, mu)."""
        z_t = torch.tensor(z_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, sigma_norm, mu_norm = self.netG(z_t)
        return self.denormalize(
            sigma_norm.squeeze().cpu().numpy(),
            mu_norm.squeeze().cpu().numpy(),
        )

    def _forward_multi(self, sigma: np.ndarray, mu: np.ndarray, n_quad: int = None) -> np.ndarray:
        """Run forward solver at all configured frequencies. Returns [Re0,Im0, Re1,Im1, ...]."""
        if n_quad is None:
            n_quad = self.n_quad_opt
        freqs = self.frequencies if self.frequencies is not None else np.array([PROBE_BASE.frequency])
        out = []
        for f in freqs:
            probe = dataclass_replace(PROBE_BASE, frequency=float(f))
            resp = edc_forward(sigma, mu, probe, LAYER_THICKNESS, n_quad=n_quad)
            out.extend([resp.impedance_real, resp.impedance_imag])
        return np.array(out, dtype=np.float64)

    def _objective(self, z_np: np.ndarray, target_vec: np.ndarray, scale: float) -> float:
        """
        Normalised multi-frequency mismatch:
          J(z) = ‖pred_vec - target_vec‖² / scale
        where scale = ‖target_vec‖² keeps J dimensionless.
        """
        sigma, mu = self._decode_z(z_np)
        try:
            pred_vec = self._forward_multi(sigma, mu)
            return float(np.sum((pred_vec - target_vec) ** 2) / scale)
        except Exception:
            return 1e10

    def _build_target_vec(self, edc_target: EDCResponse) -> tuple:
        """
        Build (target_vec, scale) for the normalised objective.

        target_vec: [Re(ΔZ@f0), Im(ΔZ@f0), Re(ΔZ@f1), Im(ΔZ@f1), ...]
        scale:      ‖target_vec‖² (so J is dimensionless)

        For multi-freq experiments subclasses should override `load_target()`
        to return an edc_target whose multi-freq measurements are accessible.
        The single-freq fallback uses edc_target.impedance_{real,imag} directly.
        """
        if self.frequencies is None:
            vec = np.array([edc_target.impedance_real, edc_target.impedance_imag])
        else:
            vec = self._get_multifreq_target(edc_target)
        scale = float(np.sum(vec ** 2))
        if scale == 0:
            scale = 1.0
        return vec, scale

    def _get_multifreq_target(self, edc_target: EDCResponse) -> np.ndarray:
        """
        Subclasses running multi-frequency experiments should override this
        to return the full measured vector. Default: uses single-freq value
        repeated (useful for testing but not physically meaningful).
        """
        return np.array([edc_target.impedance_real, edc_target.impedance_imag])

    def _warm_start_candidates(self, target_vec: np.ndarray, rng: np.random.Generator,
                               n_candidates: int = 50) -> list:
        """
        Draw n_candidates random z vectors, evaluate J(z) cheaply,
        return the n_restarts best as warm-start z₀ values.

        This replaces pure random initialization with informed initialization:
        filter out z₀ values that are clearly in flat/bad regions before
        spending L-BFGS-B iterations on them.
        """
        scale = float(np.sum(target_vec ** 2)) or 1.0
        candidates = []
        for _ in range(n_candidates):
            z = rng.standard_normal(NZ).astype(np.float32)
            j = self._objective(z, target_vec, scale)
            candidates.append((j, z))
        candidates.sort(key=lambda x: x[0])
        return [z for _, z in candidates[: self.n_restarts]]

    def optimize(self, edc_target: EDCResponse, verbose: bool = True) -> GANInverseResult:
        """
        Multi-restart L-BFGS-B in latent space z ∈ ℝ³².
        Parallelized across restarts using ProcessPoolExecutor.
        """
        rng = np.random.default_rng(self.seed)
        target_vec, scale = self._build_target_vec(edc_target)

        best_mismatch = np.inf
        best_sigma = best_mu = best_z = None
        all_mismatches = []

        t0 = time.perf_counter()

        if verbose:
            mode = f"{len(self.frequencies)} freqs" if self.frequencies is not None else "1 freq"
            print(f"  Warm-starting {self.n_restarts} restarts from {self.n_restarts * 10} candidates "
                  f"| fd_eps={self.fd_epsilon:.0e} | {mode}")

        z0_list = self._warm_start_candidates(target_vec, rng, n_candidates=self.n_restarts * 10)

        # Build worker configuration
        config = {
            'root_dir': str(ROOT),
            'model_path': str(MODEL_DIR / "models" / "netG_final.pt"),
            'norm': self.norm,
            'nz': NZ,
            'K': K,
            'use_cuda': False, # Avoid CUDA issues in subprocesses
            'frequencies': self.frequencies if self.frequencies is not None else [PROBE_BASE.frequency],
            'probe_base': PROBE_BASE,
            'layer_thickness': LAYER_THICKNESS,
            'n_quad_opt': self.n_quad_opt,
            'n_iter': self.n_iter,
            'fd_epsilon': self.fd_epsilon,
        }

        if verbose:
            print(f"  Running {self.n_restarts} restarts in parallel...")

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_single_restart_gan, i, z0, target_vec, scale, config)
                for i, z0 in enumerate(z0_list)
            ]
            
            results = [f.result() for f in futures]

        for res in results:
            z_opt = res['z_opt']
            sigma, mu = res['sigma'], res['mu']

            try:
                check = edc_forward(sigma, mu, PROBE, LAYER_THICKNESS, n_quad=self.n_quad_refine)
                mismatch = abs(check.impedance_complex - edc_target.impedance_complex)
            except Exception:
                mismatch = np.inf

            all_mismatches.append(float(mismatch))
            if verbose:
                print(f"  Restart {res['restart_idx']+1:2d}/{self.n_restarts}: |ΔZ| = {mismatch:.3e} Ω  "
                      f"(iters={res['nit']}, J={res['fun']:.3e}, time={res['elapsed']:.1f}s)")

            if mismatch < best_mismatch:
                best_mismatch = mismatch
                best_sigma = sigma.copy()
                best_mu = mu.copy()
                best_z = z_opt.copy()

        if verbose:
            print(f"  Refinement pass from best z* (n_quad {self.n_quad_opt}→{self.n_quad_refine}) ...")

        def _objective_refine(z_np):
            sigma, mu = self._decode_z(z_np.astype(np.float32))
            try:
                pred_vec = self._forward_multi(sigma, mu, n_quad=self.n_quad_refine)
                return float(np.sum((pred_vec - target_vec) ** 2) / scale)
            except Exception:
                return 1e10

        opt_r = minimize(
            _objective_refine,
            best_z,
            method="L-BFGS-B",
            options={"maxiter": self.n_iter, "ftol": 1e-15, "gtol": 1e-10, "eps": self.fd_epsilon},
        )
        z_refined = opt_r.x.astype(np.float32)
        sigma_r, mu_r = self._decode_z(z_refined)
        try:
            check_r = edc_forward(sigma_r, mu_r, PROBE, LAYER_THICKNESS, n_quad=self.n_quad_verify)
            mismatch_r = abs(check_r.impedance_complex - edc_target.impedance_complex)
        except Exception:
            mismatch_r = np.inf

        if verbose:
            print(f"  Refined: |ΔZ| = {mismatch_r:.3e} Ω  (iters={opt_r.nit}, J={opt_r.fun:.3e})")

        if mismatch_r < best_mismatch:
            best_mismatch = mismatch_r
            best_sigma = sigma_r.copy()
            best_mu = mu_r.copy()
            best_z = z_refined.copy()
        else:
            try:
                check_v = edc_forward(best_sigma, best_mu, PROBE, LAYER_THICKNESS, n_quad=self.n_quad_verify)
                best_mismatch = abs(check_v.impedance_complex - edc_target.impedance_complex)
            except Exception:
                pass

        elapsed = time.perf_counter() - t0
        return GANInverseResult(
            sigma_rec=best_sigma,
            mu_rec=best_mu,
            best_z=best_z,
            best_mismatch=best_mismatch,
            all_mismatches=all_mismatches,
            elapsed=elapsed,
            n_restarts=self.n_restarts,
        )

    @abstractmethod
    def load_target(self) -> tuple:
        """Return (sigma_true, mu_true, edc_target)."""

    @abstractmethod
    def explain(self) -> None:
        """Print experiment-specific explanation."""

    def run(self) -> GANInverseResult:
        print("=" * 60)
        print(f"{self.name} — {self.description}")
        print(f"  Model: improved_wgan_v2 | nz={NZ} | K={K}")
        print(f"  Restarts: {self.n_restarts}")
        print("=" * 60)
        print(f"\nDevice: {self.device}")
        print(f"Generator: {MODEL_DIR.name}")
        print(f"  σ ∈ [{self.norm['sigma_min']:.3e}, {self.norm['sigma_max']:.3e}] S/m")
        print(f"  μ ∈ [{self.norm['mu_min']:.3f}, {self.norm['mu_max']:.3f}]")

        sigma_true, mu_true, edc_target = self.load_target()

        print(f"\nTarget profile:")
        print(f"  σ₁={sigma_true[0]:.3e}  σ₅₁={sigma_true[-1]:.3e} S/m")
        print(f"  μ₁={mu_true[0]:.3f}   μ₅₁={mu_true[-1]:.3f}")
        print(f"\nTarget ΔZ = {edc_target.impedance_real:.6e} {edc_target.impedance_imag:+.6e}j Ω")
        print(f"\nOptimizing z ∈ ℝ{NZ} ...")

        result = self.optimize(edc_target, verbose=True)

        sigma_rmse, mu_rmse = self._compute_rmse(sigma_true, mu_true, result)
        self._report(sigma_true, mu_true, edc_target, result)
        self.explain()
        self._visualize(sigma_true, mu_true, edc_target, result)
        self._save_json(sigma_true, mu_true, edc_target, result, sigma_rmse, mu_rmse)

        return result

    def _compute_rmse(self, sigma_true, mu_true, result: GANInverseResult) -> tuple:
        sigma_rmse = float(np.sqrt(np.mean((sigma_true - result.sigma_rec) ** 2)))
        mu_rmse = float(np.sqrt(np.mean((mu_true - result.mu_rec) ** 2)))
        return sigma_rmse, mu_rmse

    def _report(self, sigma_true, mu_true, edc_target, result: GANInverseResult) -> None:
        sigma_rmse, mu_rmse = self._compute_rmse(sigma_true, mu_true, result)

        print(f"\nBest |ΔZ error| = {result.best_mismatch:.6e} Ω  ({result.elapsed:.1f}s)")
        print(f"Recovered boundaries:")
        print(f"  σ₁={result.sigma_rec[0]:.3e}  σ₅₁={result.sigma_rec[-1]:.3e} S/m"
              f"  (true: {sigma_true[0]:.3e} / {sigma_true[-1]:.3e})")
        print(f"  μ₁={result.mu_rec[0]:.3f}   μ₅₁={result.mu_rec[-1]:.3f}"
              f"         (true: {mu_true[0]:.3f} / {mu_true[-1]:.3f})")
        print(f"Profile RMSE:  σ={sigma_rmse:.3e} S/m ({sigma_rmse/np.mean(sigma_true)*100:.2f}%)  "
              f"μ={mu_rmse:.3e} ({mu_rmse/np.mean(mu_true)*100:.2f}%)")
        print(f"\nStatus: {'PASS ✓' if result.passed else 'FAIL ✗'}  "
              f"(tolerance {self.impedance_tolerance:.0e} Ω)")
        print(f"Results directory: {self.run_dir}")

    def _save_json(self, sigma_true, mu_true, edc_target, result: GANInverseResult,
                   sigma_rmse: float, mu_rmse: float) -> None:
        edc_rec = edc_forward(result.sigma_rec, result.mu_rec, PROBE, LAYER_THICKNESS, n_quad=self.n_quad_verify)

        record = {
            "experiment": self.name,
            "description": self.description,
            "timestamp": self._run_timestamp,
            "config": {
                "model": MODEL_DIR.name,
                "nz": NZ,
                "K": K,
                "n_restarts": self.n_restarts,
                "n_iter": self.n_iter,
                "seed": self.seed,
                "fd_epsilon": self.fd_epsilon,
                "n_quad_opt": self.n_quad_opt,
                "n_quad_refine": self.n_quad_refine,
                "n_quad_verify": self.n_quad_verify,
                "frequencies_hz": self.frequencies.tolist() if self.frequencies is not None else [PROBE_BASE.frequency],
                "impedance_tolerance": self.impedance_tolerance,
                "probe": {
                    "frequency_hz": PROBE_BASE.frequency,
                    "inner_radius_m": PROBE_BASE.inner_radius,
                    "outer_radius_m": PROBE_BASE.outer_radius,
                    "lift_off_m": PROBE_BASE.lift_off,
                },
            },
            "target": {
                "sigma_true": sigma_true.tolist(),
                "mu_true": mu_true.tolist(),
                "impedance_real": float(edc_target.impedance_real),
                "impedance_imag": float(edc_target.impedance_imag),
                "impedance_magnitude": float(abs(edc_target.impedance_complex)),
            },
            "result": {
                "sigma_rec": result.sigma_rec.tolist(),
                "mu_rec": result.mu_rec.tolist(),
                "best_z": result.best_z.tolist(),
                "best_mismatch_ohm": float(result.best_mismatch),
                "all_mismatches_ohm": [float(m) for m in result.all_mismatches],
                "elapsed_s": float(result.elapsed),
                "passed": bool(result.passed),
            },
            "metrics": {
                "sigma_rmse": sigma_rmse,
                "sigma_rmse_pct": float(sigma_rmse / np.mean(sigma_true) * 100),
                "mu_rmse": mu_rmse,
                "mu_rmse_pct": float(mu_rmse / np.mean(mu_true) * 100),
                "impedance_error_ohm": float(abs(edc_rec.impedance_complex - edc_target.impedance_complex)),
                "best_restart_idx": int(np.argmin(result.all_mismatches)),
                "n_restarts_passed": sum(1 for m in result.all_mismatches if m < self.impedance_tolerance),
            },
        }

        path = self.run_dir / "results.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  Saved: {path}")

    def _visualize(self, sigma_true, mu_true, edc_target, result: GANInverseResult) -> None:
        save_dir = self.run_dir
        name = self.name

        plot_profile_comparison(
            sigma_true, mu_true, result.sigma_rec, result.mu_rec, LAYER_THICKNESS,
            save_path=save_dir / f"{name}_profiles.png",
            title=f"{name} — Profile Recovery",
        )
        plot_sigma_comparison(
            sigma_true, result.sigma_rec, LAYER_THICKNESS,
            save_path=save_dir / f"{name}_sigma.png",
            title=f"{name} — Conductivity σ",
        )
        plot_mu_comparison(
            mu_true, result.mu_rec, LAYER_THICKNESS,
            save_path=save_dir / f"{name}_mu.png",
            title=f"{name} — Permeability μ",
        )

        edc_rec = edc_forward(result.sigma_rec, result.mu_rec, PROBE, LAYER_THICKNESS, n_quad=self.n_quad_verify)
        plot_decision_criteria(
            edc_target=edc_target,
            edc_recovered=edc_rec,
            tolerance=self.impedance_tolerance,
            criterion_type="absolute",
            save_path=save_dir / f"{name}_decision.png",
        )

        self._plot_convergence(result, save_dir / f"{name}_convergence.png")
        print(f"Plots saved to: {save_dir}")

    def _plot_convergence(self, result: GANInverseResult, save_path: Path) -> None:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        n = result.n_restarts
        ms = result.all_mismatches

        threshold = min(ms) * 10
        colors = ["#C73E1D" if m > threshold else "#2E86AB" for m in ms]
        bars = ax1.bar(range(1, n + 1), ms, color=colors, alpha=0.7, edgecolor="black", linewidth=1.2)
        best_idx = int(np.argmin(ms))
        bars[best_idx].set_edgecolor("#F18F01")
        bars[best_idx].set_linewidth(3)
        ax1.axhline(min(ms), color="#F18F01", linestyle="--", linewidth=2, label=f"Best: {min(ms):.2e} Ω")
        ax1.axhline(self.impedance_tolerance, color="green", linestyle=":", linewidth=1.5,
                    label=f"Threshold: {self.impedance_tolerance:.0e} Ω")
        ax1.set_xlabel("Restart", fontweight="bold")
        ax1.set_ylabel("|ΔZ error| (Ω)", fontweight="bold")
        ax1.set_title("All Restarts", fontweight="bold")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle=":")

        sorted_ms = sorted(ms)
        ax2.semilogy(range(1, n + 1), sorted_ms, "o-", color="#2E86AB", linewidth=2.5, markersize=8)
        ax2.axhline(min(ms), color="#F18F01", linestyle="--", linewidth=2, label="Best solution")
        ax2.axhline(self.impedance_tolerance, color="green", linestyle=":", linewidth=1.5,
                    label=f"Threshold: {self.impedance_tolerance:.0e} Ω")
        ax2.set_xlabel("Rank", fontweight="bold")
        ax2.set_ylabel("|ΔZ error| (Ω)", fontweight="bold")
        ax2.set_title("Sorted Results", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle=":")

        n_pass = sum(1 for m in ms if m < self.impedance_tolerance)
        fig.suptitle(
            f"{self.name} | {n} restarts | {n_pass}/{n} passed",
            fontsize=13, fontweight="bold",
        )
        fig.text(0.5, 0.01, "Model: improved_wgan_v2 (nz=32) | Optimizer: scipy L-BFGS-B on z ∈ ℝ³²",
                 ha="center", fontsize=8, style="italic", color="gray")
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()
