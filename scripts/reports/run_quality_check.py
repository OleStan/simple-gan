#!/usr/bin/env python
"""
Run GAN quality validation for trained models.

Usage:
    python run_quality_check.py --model dual_wgan --model_dir ./results/dual_wgan_20260131_213107
    python run_quality_check.py --model improved_wgan_v2 --model_dir ./results/improved_wgan_v2_20260126_223129
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path

from eddy_current_workflow.quality import GANQualityChecker, QualityReportGenerator
from eddy_current_workflow.forward.edc_solver import ProbeSettings


def load_dual_wgan_generator(model_dir: str, nz: int, K: int, device: torch.device):
    from models.dual_wgan.model import DualHeadGenerator
    netG = DualHeadGenerator(nz=nz, K=K).to(device)

    model_path = Path(model_dir) / 'models' / 'netG_final.pth'
    if not model_path.exists():
        model_path = Path(model_dir) / 'models' / 'netG_final.pt'

    netG.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    netG.eval()
    return netG


def load_improved_wgan_v2_generator(model_dir: str, nz: int, K: int, device: torch.device):
    from models.improved_wgan_v2.model import ConditionalConv1DGenerator
    netG = ConditionalConv1DGenerator(nz=nz, K=K, conditional=False).to(device)

    model_path = Path(model_dir) / 'models' / 'netG_final.pt'
    if not model_path.exists():
        model_path = Path(model_dir) / 'models' / 'netG_final.pth'

    netG.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    netG.eval()
    return netG


def load_real_data(training_data_dir: str, norm_params_path: str, K: int) -> np.ndarray:
    raw_data = np.load(Path(training_data_dir) / 'X_raw.npy')

    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)

    sigma_data = raw_data[:, :K]
    mu_data = raw_data[:, K:2*K]

    sigma_min = norm_params['sigma_min']
    sigma_max = norm_params['sigma_max']
    mu_min = norm_params['mu_min']
    mu_max = norm_params['mu_max']

    sigma_norm = 2 * (sigma_data - sigma_min) / (sigma_max - sigma_min) - 1
    mu_norm = 2 * (mu_data - mu_min) / (mu_max - mu_min) - 1

    return np.concatenate([sigma_norm, mu_norm], axis=1)


def denormalize_data(data: np.ndarray, norm_params: dict, K: int) -> np.ndarray:
    sigma_norm = data[:, :K]
    mu_norm = data[:, K:2*K]

    sigma = (sigma_norm + 1) / 2 * (norm_params['sigma_max'] - norm_params['sigma_min']) + norm_params['sigma_min']
    mu = (mu_norm + 1) / 2 * (norm_params['mu_max'] - norm_params['mu_min']) + norm_params['mu_min']

    return np.concatenate([sigma, mu], axis=1)


def main():
    parser = argparse.ArgumentParser(description='Run GAN Quality Validation')
    parser.add_argument('--model', type=str, required=True,
                        choices=['dual_wgan', 'improved_wgan_v2'],
                        help='Model type to evaluate')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--training_data', type=str, default='../../data/training',
                        help='Path to training data directory')
    parser.add_argument('--n_generated', type=int, default=1000,
                        help='Number of samples to generate for evaluation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for report (default: model_dir/quality_report)')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    norm_params_path = Path(args.model_dir) / 'normalization_params.json'
    if not norm_params_path.exists():
        norm_params_path = Path(args.training_data) / 'normalization_params.json'

    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)

    K = norm_params['K']

    config_path = Path(args.model_dir) / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        nz = model_config.get('nz', 100)
    else:
        nz = 100

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  K: {K}")
    print(f"  nz: {nz}")
    print(f"  Model dir: {args.model_dir}")

    print("\nLoading generator...")
    if args.model == 'dual_wgan':
        generator = load_dual_wgan_generator(args.model_dir, nz, K, device)
        model_name = "Dual-Head WGAN"
    else:
        generator = load_improved_wgan_v2_generator(args.model_dir, nz, K, device)
        model_name = "Improved WGAN v2"

    print("Loading real data...")
    real_data_normalized = load_real_data(args.training_data, str(norm_params_path), K)
    print(f"  Real data shape: {real_data_normalized.shape}")

    checker = GANQualityChecker(
        K=K, nz=nz,
        sigma_bounds=(norm_params['sigma_min'], norm_params['sigma_max']),
        mu_bounds=(norm_params['mu_min'], norm_params['mu_max']),
        device=device,
    )

    print(f"\nGenerating {args.n_generated} samples...")
    generated_normalized = checker.generate_samples(generator, n_samples=args.n_generated)
    print(f"  Generated data shape: {generated_normalized.shape}")

    real_physical = denormalize_data(real_data_normalized, norm_params, K)
    generated_physical = denormalize_data(generated_normalized, norm_params, K)

    probe = ProbeSettings(frequency=1e6)

    print("\nRunning quality checks...")
    report = checker.run_all_checks(
        generator=generator,
        real_data=real_data_normalized,
        generated_data=generated_normalized,
        real_data_physical=real_physical,
        generated_data_physical=generated_physical,
        model_name=model_name,
        probe_settings=probe,
        mmd_max_samples=300,
        n_forward_samples=20,
    )

    output_dir = args.output_dir or str(Path(args.model_dir) / 'quality_report')
    report_gen = QualityReportGenerator(output_dir)

    report_path = report_gen.generate_full_report(
        report=report,
        real_data=real_data_normalized,
        generated_data=generated_normalized,
    )

    print("\n" + "=" * 60)
    print("QUALITY VALIDATION SUMMARY")
    print("=" * 60)
    summary = report.summary
    for criterion, details in summary['criteria'].items():
        passed = details.get('passed', 'N/A')
        status_str = "PASS" if passed is True else ("FAIL" if passed is False else "INFO")
        print(f"  {criterion}: {status_str}")
        for key, val in details.items():
            if key != 'passed':
                print(f"    {key}: {val}")

    overall = "PASSED" if summary['overall_passed'] else "FAILED"
    print(f"\n  Overall: {overall}")
    print(f"\nFull report: {report_path}")
    print(f"Plots: {output_dir}/plots/")


if __name__ == '__main__':
    main()
