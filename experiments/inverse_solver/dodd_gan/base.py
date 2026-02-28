"""
Base class for GAN inverse solver experiments using the original
dodd_analytical_model as the forward solver.

Architecture:
  - Forward: dodd_analytical_model VectorPotentialInsideCoilGreenFunction (ORNL-5220)
  - Inverse: improved_wgan_v2_nz32 generator, latent-space optimization
  - Optimizer: scipy L-BFGS-B on z ∈ ℝ³²
"""

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "improved_wgan_v2"))

from dodd_forward import DoddProbeSettings, DoddResponse, dodd_forward, PROBE_DEFAULT

MODEL_DIR = ROOT / "results" / "improved_wgan_v2_nz32_20260214_140817"
RESULTS_DIR = Path(__file__).parent / "results"

K = 51
NZ = 32
LAYER_THICKNESS = 1e-3 / K

PROBE = PROBE_DEFAULT


@dataclass
class GANDoddResult:
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
            f"GANDoddResult({status} | "
            f"|V|={self.best_mismatch:.3e} V | "
            f"{self.n_restarts} restarts | "
            f"{self.elapsed:.1f}s)"
        )


class GANDoddExperiment(ABC):
    """
    Base class for GAN inverse solver experiments backed by dodd_analytical_model.

    Forward solver: VectorPotentialInsideCoilGreenFunction from dodd_analytical_model (ORNL-5220).
    Inverse solver: improved_wgan_v2_nz32 generator with L-BFGS-B in latent space.

    Subclasses must implement:
      - name: str
      - description: str
      - load_target() -> (sigma_true, mu_true, DoddResponse)
      - explain() -> None
    """

    name: str = "gan_dodd_base"
    description: str = "GAN inverse experiment (dodd_analytical_model forward)"
    impedance_tolerance: float = 1e-5

    def __init__(
        self,
        n_restarts: int = 10,
        n_iter: int = 500,
        seed: int = 42,
        fd_epsilon: float = 1e-3,
        integ_range_opt: int = 20,
        integ_range_verify: int = 50,
        probe: Optional[DoddProbeSettings] = None,
    ):
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.seed = seed
        self.fd_epsilon = fd_epsilon
        self.integ_range_opt = integ_range_opt
        self.integ_range_verify = integ_range_verify
        self.probe = probe if probe is not None else PROBE
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
        state = torch.load(
            MODEL_DIR / "models" / "netG_final.pt",
            map_location=self.device,
        )
        netG.load_state_dict(state)
        netG.to(self.device)
        netG.eval()
        for p in netG.parameters():
            p.requires_grad_(False)
        return netG

    def _denormalize(self, sigma_norm: np.ndarray, mu_norm: np.ndarray) -> tuple:
        n = self.norm
        sigma = (sigma_norm + 1) / 2 * (n["sigma_max"] - n["sigma_min"]) + n["sigma_min"]
        mu = (mu_norm + 1) / 2 * (n["mu_max"] - n["mu_min"]) + n["mu_min"]
        return sigma, mu

    def _decode_z(self, z_np: np.ndarray) -> tuple:
        """Run z through Generator → physical (sigma, mu)."""
        z_t = torch.tensor(z_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, sigma_norm, mu_norm = self.netG(z_t)
        return self._denormalize(
            sigma_norm.squeeze().cpu().numpy(),
            mu_norm.squeeze().cpu().numpy(),
        )

    def _forward(self, sigma: np.ndarray, mu: np.ndarray, integ_range: int) -> complex:
        """Run dodd_analytical_model forward solver. Returns complex ΔZ."""
        resp = dodd_forward(sigma, mu, self.probe, LAYER_THICKNESS, integ_top_range=integ_range)
        return resp.impedance_complex

    def _objective(self, z_np: np.ndarray, target: complex, scale: float) -> float:
        """
        Normalised mismatch:
          J(z) = |ΔZ_pred - ΔZ_target|² / scale
        """
        sigma, mu = self._decode_z(z_np)
        try:
            pred = self._forward(sigma, mu, self.integ_range_opt)
            diff = pred - target
            return float((diff.real ** 2 + diff.imag ** 2) / scale)
        except Exception:
            return 1e10

    def _warm_start_candidates(self, rng: np.random.Generator, n_screen: int) -> list:
        """
        Cheap GAN-space warm-start: decode n_screen random z vectors through the
        generator (free NN forward pass), rank by the L2 norm of the generated
        (sigma_norm, mu_norm) output as a diversity proxy, and return the top
        n_restarts most spread-out candidates.

        This avoids running n_screen expensive forward solver calls just for
        screening — crucial when each dodd_analytical_model call takes ~10s.
        """
        z_batch = rng.standard_normal((n_screen, NZ)).astype(np.float32)
        z_tensor = torch.tensor(z_batch, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, sigma_norm, mu_norm = self.netG(z_tensor)

        sigma_np = sigma_norm.cpu().numpy()
        mu_np = mu_norm.cpu().numpy()

        # Score: spread across σ and μ range (prefer profiles near distribution edges)
        sigma_range = sigma_np.max(axis=1) - sigma_np.min(axis=1)
        mu_range = mu_np.max(axis=1) - mu_np.min(axis=1)
        scores = sigma_range + mu_range

        top_idx = np.argsort(scores)[::-1][: self.n_restarts]
        return [z_batch[i] for i in top_idx]

    def optimize(self, target_response: DoddResponse, verbose: bool = True) -> GANDoddResult:
        """
        Multi-restart L-BFGS-B in latent space z ∈ ℝ³².

        Warm-start: cheap GAN-space screening (no forward calls).
        Optimization: L-BFGS-B with finite-difference gradients at integ_range_opt.
        Verification: final mismatch evaluated at integ_range_verify.
        """
        rng = np.random.default_rng(self.seed)
        target = target_response.impedance_complex
        scale = abs(target) ** 2 or 1.0

        n_screen = max(self.n_restarts * 10, 200)

        best_mismatch = np.inf
        best_sigma = best_mu = best_z = None
        all_mismatches = []

        t0 = time.perf_counter()

        if verbose:
            print(
                f"  Warm-starting {self.n_restarts} restarts from "
                f"{n_screen} GAN samples (no forward calls) | "
                f"fd_eps={self.fd_epsilon:.0e} | integ_range_opt={self.integ_range_opt}"
            )

        z0_list = self._warm_start_candidates(rng, n_screen)

        for restart, z0 in enumerate(z0_list):
            opt = minimize(
                self._objective,
                z0,
                args=(target, scale),
                method="L-BFGS-B",
                options={
                    "maxiter": self.n_iter,
                    "ftol": 1e-15,
                    "gtol": 1e-10,
                    "eps": self.fd_epsilon,
                },
            )

            z_opt = opt.x.astype(np.float32)
            sigma, mu = self._decode_z(z_opt)

            try:
                pred = self._forward(sigma, mu, self.integ_range_verify)
                mismatch = abs(pred - target)
            except Exception:
                mismatch = np.inf

            all_mismatches.append(float(mismatch))

            if verbose:
                print(
                    f"  Restart {restart + 1:2d}/{self.n_restarts}: "
                    f"|V| = {mismatch:.3e} V  "
                    f"(iters={opt.nit}, J={opt.fun:.3e})"
                )

            if mismatch < best_mismatch:
                best_mismatch = mismatch
                best_sigma = sigma.copy()
                best_mu = mu.copy()
                best_z = z_opt.copy()

        elapsed = time.perf_counter() - t0
        return GANDoddResult(
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
        """Return (sigma_true, mu_true, DoddResponse)."""

    @abstractmethod
    def explain(self) -> None:
        """Print experiment-specific explanation."""

    def run(self) -> GANDoddResult:
        print("=" * 60)
        print(f"{self.name} — {self.description}")
        print(f"  Forward: dodd_analytical_model (VectorPotentialInsideCoilGreenFunction)")
        print(f"  Inverse: improved_wgan_v2_nz32 | nz={NZ} | K={K}")
        print(f"  Restarts: {self.n_restarts}")
        print("=" * 60)
        print(f"\nDevice: {self.device}")
        print(f"Generator: {MODEL_DIR.name}")
        print(f"  σ ∈ [{self.norm['sigma_min']:.3e}, {self.norm['sigma_max']:.3e}] S/m")
        print(f"  μ ∈ [{self.norm['mu_min']:.3f}, {self.norm['mu_max']:.3f}]")
        print(f"\nProbe: f={self.probe.frequency_hz:.0f} Hz | "
              f"coil r=[{self.probe.coil_r1*1e3:.1f}, {self.probe.coil_r2*1e3:.1f}] mm | "
              f"coil z=[{self.probe.coil_l1*1e3:.1f}, {self.probe.coil_l2*1e3:.1f}] mm | "
              f"conductor r=[{self.probe.conductor_r1*1e3:.1f}, {self.probe.conductor_r2*1e3:.1f}] mm | "
              f"calc_r={self.probe.calc_r*1e3:.1f} mm")

        sigma_true, mu_true, target_response = self.load_target()

        print(f"\nTarget profile:")
        print(f"  σ₁={sigma_true[0]:.3e}  σ₅₁={sigma_true[-1]:.3e} S/m")
        print(f"  μ₁={mu_true[0]:.3f}   μ₅₁={mu_true[-1]:.3f}")
        print(f"\nTarget V = {target_response.voltage_real:.6e} "
              f"{target_response.voltage_imag:+.6e}j V  "
              f"|V|={target_response.amplitude:.6e} V")
        print(f"\nOptimizing z ∈ ℝ{NZ} ...")

        result = self.optimize(target_response, verbose=True)

        sigma_rmse, mu_rmse = self._compute_rmse(sigma_true, mu_true, result)
        self._report(sigma_true, mu_true, target_response, result)
        self.explain()
        self._visualize(sigma_true, mu_true, target_response, result)
        self._save_json(sigma_true, mu_true, target_response, result, sigma_rmse, mu_rmse)

        return result

    def _compute_rmse(self, sigma_true, mu_true, result: GANDoddResult) -> tuple:
        sigma_rmse = float(np.sqrt(np.mean((sigma_true - result.sigma_rec) ** 2)))
        mu_rmse = float(np.sqrt(np.mean((mu_true - result.mu_rec) ** 2)))
        return sigma_rmse, mu_rmse

    def _report(self, sigma_true, mu_true, target_response: DoddResponse, result: GANDoddResult) -> None:
        sigma_rmse, mu_rmse = self._compute_rmse(sigma_true, mu_true, result)

        print(f"\nBest |V error| = {result.best_mismatch:.6e} V  ({result.elapsed:.1f}s)")
        print(f"Recovered boundaries:")
        print(
            f"  σ₁={result.sigma_rec[0]:.3e}  σ₅₁={result.sigma_rec[-1]:.3e} S/m"
            f"  (true: {sigma_true[0]:.3e} / {sigma_true[-1]:.3e})"
        )
        print(
            f"  μ₁={result.mu_rec[0]:.3f}   μ₅₁={result.mu_rec[-1]:.3f}"
            f"         (true: {mu_true[0]:.3f} / {mu_true[-1]:.3f})"
        )
        print(
            f"Profile RMSE:  σ={sigma_rmse:.3e} S/m ({sigma_rmse / np.mean(sigma_true) * 100:.2f}%)  "
            f"μ={mu_rmse:.3e} ({mu_rmse / np.mean(mu_true) * 100:.2f}%)"
        )
        print(f"\nStatus: {'PASS ✓' if result.passed else 'FAIL ✗'}  "
              f"(tolerance {self.impedance_tolerance:.0e} V)")
        print(f"Results directory: {self.run_dir}")

    def _save_json(
        self,
        sigma_true: np.ndarray,
        mu_true: np.ndarray,
        target_response: DoddResponse,
        result: GANDoddResult,
        sigma_rmse: float,
        mu_rmse: float,
    ) -> None:
        pred_complex = self._forward(result.sigma_rec, result.mu_rec, self.integ_range_verify)
        target_complex = target_response.impedance_complex

        record = {
            "experiment": self.name,
            "description": self.description,
            "timestamp": self._run_timestamp,
            "config": {
                "model": MODEL_DIR.name,
                "forward_solver": "dodd_analytical_model/VectorPotentialInsideCoilGreenFunction",
                "nz": NZ,
                "K": K,
                "layer_thickness_m": LAYER_THICKNESS,
                "n_restarts": self.n_restarts,
                "n_iter": self.n_iter,
                "seed": self.seed,
                "fd_epsilon": self.fd_epsilon,
                "integ_range_opt": self.integ_range_opt,
                "integ_range_verify": self.integ_range_verify,
                "voltage_tolerance": self.impedance_tolerance,
                "probe": {
                    "frequency_hz": self.probe.frequency_hz,
                    "coil_r1_m": self.probe.coil_r1,
                    "coil_r2_m": self.probe.coil_r2,
                    "coil_l1_m": self.probe.coil_l1,
                    "coil_l2_m": self.probe.coil_l2,
                    "conductor_r1_m": self.probe.conductor_r1,
                    "conductor_r2_m": self.probe.conductor_r2,
                    "calc_r_m": self.probe.calc_r,
                    "calc_z_m": self.probe.calc_z,
                    "n_turns": self.probe.n_turns,
                },
            },
            "target": {
                "sigma_true": sigma_true.tolist(),
                "mu_true": mu_true.tolist(),
                "voltage_real": float(target_response.voltage_real),
                "voltage_imag": float(target_response.voltage_imag),
                "voltage_magnitude": float(target_response.amplitude),
                "voltage_phase_deg": float(
                    np.degrees(np.angle(target_response.voltage_complex))
                ),
            },
            "result": {
                "sigma_rec": result.sigma_rec.tolist(),
                "mu_rec": result.mu_rec.tolist(),
                "best_z": result.best_z.tolist(),
                "voltage_rec_real": float(pred_complex.real),
                "voltage_rec_imag": float(pred_complex.imag),
                "voltage_rec_magnitude": float(abs(pred_complex)),
                "voltage_rec_phase_deg": float(np.degrees(np.angle(pred_complex))),
                "best_mismatch_v": float(result.best_mismatch),
                "all_mismatches_v": [float(m) for m in result.all_mismatches],
                "elapsed_s": float(result.elapsed),
                "passed": bool(result.passed),
            },
            "metrics": {
                "sigma_rmse": float(sigma_rmse),
                "sigma_rmse_pct": float(sigma_rmse / np.mean(sigma_true) * 100),
                "mu_rmse": float(mu_rmse),
                "mu_rmse_pct": float(mu_rmse / np.mean(mu_true) * 100),
                "voltage_error_v": float(abs(pred_complex - target_complex)),
                "voltage_error_pct": float(
                    abs(pred_complex - target_complex) / abs(target_complex) * 100
                    if abs(target_complex) > 0 else 0.0
                ),
                "best_restart_idx": int(np.argmin(result.all_mismatches)),
                "n_restarts_passed": int(
                    sum(1 for m in result.all_mismatches if m < self.impedance_tolerance)
                ),
            },
        }

        path = self.run_dir / "results.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  Saved: {path}")

    def _visualize(
        self,
        sigma_true: np.ndarray,
        mu_true: np.ndarray,
        target_response: DoddResponse,
        result: GANDoddResult,
    ) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        save_dir = self.run_dir
        name = self.name

        pred_complex = self._forward(result.sigma_rec, result.mu_rec, self.integ_range_verify)
        target_complex = target_response.impedance_complex
        mismatch = abs(pred_complex - target_complex)
        status = "PASS ✓" if result.passed else "FAIL ✗"

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

        ax_sigma = fig.add_subplot(gs[0, 0])
        ax_mu    = fig.add_subplot(gs[0, 1])
        ax_phasor = fig.add_subplot(gs[1, 0])
        ax_conv  = fig.add_subplot(gs[1, 1])

        depths = np.arange(K) * LAYER_THICKNESS * 1e3

        # --- σ profile ---
        ax_sigma.plot(depths, sigma_true / 1e6, "b-", linewidth=2.0, label="Target σ")
        ax_sigma.plot(depths, result.sigma_rec / 1e6, "r--", linewidth=2.0, label="Recovered σ")
        sigma_rmse = float(np.sqrt(np.mean((sigma_true - result.sigma_rec) ** 2)))
        ax_sigma.fill_between(
            depths,
            sigma_true / 1e6,
            result.sigma_rec / 1e6,
            alpha=0.12,
            color="red",
            label=f"RMSE={sigma_rmse/1e6:.3f} MS/m",
        )
        ax_sigma.set_xlabel("Layer index (radial)")
        ax_sigma.set_ylabel("σ (MS/m)")
        ax_sigma.set_title("Conductivity Profile", fontweight="bold")
        ax_sigma.legend(fontsize=8)
        ax_sigma.grid(True, alpha=0.3, linestyle=":")

        # --- μ profile ---
        ax_mu.plot(depths, mu_true, "b-", linewidth=2.0, label="Target μᵣ")
        ax_mu.plot(depths, result.mu_rec, "r--", linewidth=2.0, label="Recovered μᵣ")
        mu_rmse = float(np.sqrt(np.mean((mu_true - result.mu_rec) ** 2)))
        ax_mu.fill_between(
            depths,
            mu_true,
            result.mu_rec,
            alpha=0.12,
            color="red",
            label=f"RMSE={mu_rmse:.4f}",
        )
        ax_mu.set_xlabel("Layer index (radial)")
        ax_mu.set_ylabel("μᵣ")
        ax_mu.set_title("Permeability Profile", fontweight="bold")
        ax_mu.legend(fontsize=8)
        ax_mu.grid(True, alpha=0.3, linestyle=":")

        # --- voltage phasor diagram ---
        ax_phasor.axhline(0, color="gray", linewidth=0.8)
        ax_phasor.axvline(0, color="gray", linewidth=0.8)
        scale = max(abs(target_complex), abs(pred_complex), 1e-30)
        t_re = target_complex.real / scale
        t_im = target_complex.imag / scale
        p_re = pred_complex.real / scale
        p_im = pred_complex.imag / scale
        ax_phasor.annotate(
            "", xy=(t_re, t_im), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#2E86AB", lw=2.5),
        )
        ax_phasor.annotate(
            "", xy=(p_re, p_im), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#C73E1D", lw=2.5, linestyle="dashed"),
        )
        ax_phasor.plot([], [], color="#2E86AB", lw=2, label=f"Target  |V|={abs(target_complex):.3e} V")
        ax_phasor.plot([], [], color="#C73E1D", lw=2, linestyle="--",
                       label=f"Recovered |V|={abs(pred_complex):.3e} V")
        err_pct = abs(pred_complex - target_complex) / abs(target_complex) * 100 if abs(target_complex) > 0 else 0.0
        ax_phasor.set_title(
            f"Voltage Phasor  |err|={mismatch:.2e} V  ({err_pct:.1f}%)",
            fontweight="bold",
        )
        ax_phasor.set_xlabel("Re(V) / |V|_target")
        ax_phasor.set_ylabel("Im(V) / |V|_target")
        ax_phasor.legend(fontsize=8)
        ax_phasor.set_aspect("equal", adjustable="datalim")
        ax_phasor.grid(True, alpha=0.3, linestyle=":")

        # --- convergence bar chart ---
        ms = result.all_mismatches
        n = result.n_restarts
        threshold = min(ms) * 10
        colors = ["#C73E1D" if m > threshold else "#2E86AB" for m in ms]
        bars = ax_conv.bar(range(1, n + 1), ms, color=colors, alpha=0.75,
                           edgecolor="black", linewidth=0.8)
        best_idx = int(np.argmin(ms))
        bars[best_idx].set_edgecolor("#F18F01")
        bars[best_idx].set_linewidth(3)
        ax_conv.axhline(min(ms), color="#F18F01", linestyle="--", linewidth=1.8,
                        label=f"Best: {min(ms):.2e} V")
        ax_conv.axhline(self.impedance_tolerance, color="green", linestyle=":",
                        linewidth=1.5, label=f"Threshold: {self.impedance_tolerance:.0e} V")
        n_pass = sum(1 for m in ms if m < self.impedance_tolerance)
        ax_conv.set_xlabel("Restart")
        ax_conv.set_ylabel("|V error| (V)")
        ax_conv.set_title(f"Convergence  ({n_pass}/{n} passed)", fontweight="bold")
        ax_conv.set_yscale("log")
        ax_conv.legend(fontsize=8)
        ax_conv.grid(True, alpha=0.3, linestyle=":")

        fig.suptitle(
            f"{name}  |  {status}  |  |V error| = {mismatch:.3e} V",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.5, 0.005,
            "Forward: dodd_analytical_model (VecPotInsideCoilGF, ORNL-5220)  |  "
            "Inverse: improved_wgan_v2_nz32 (nz=32)  |  Optimizer: L-BFGS-B on z ∈ ℝ³²",
            ha="center", fontsize=7, style="italic", color="gray",
        )

        fig.savefig(save_dir / f"{name}_results.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_dir / f'{name}_results.png'}")
