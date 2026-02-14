#!/usr/bin/env python
"""
Analyze all dual WGAN latent space experiments and generate quality reports.

Usage:
    python run_latent_space_analysis.py --results_dir ./results
    python run_latent_space_analysis.py --results_dir ./results --nz_values 6,8,10,12,16,32,48
"""

import argparse
import json
import sys
from pathlib import Path
import subprocess
import numpy as np
import torch
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def find_latent_experiments(results_dir: Path) -> List[Path]:
    """Find all dual_wgan experiment directories."""
    experiments = []
    for exp_dir in results_dir.glob('dual_wgan_nz*'):
        if exp_dir.is_dir() and (exp_dir / 'models' / 'netG_final.pth').exists():
            experiments.append(exp_dir)
    return sorted(experiments)


def extract_nz_from_path(exp_path: Path) -> int:
    """Extract latent dimension from experiment path."""
    import re
    match = re.search(r'nz(\d+)', exp_path.name)
    if match:
        return int(match.group(1))
    return None


def run_quality_check(exp_dir: Path, training_data: str) -> bool:
    """Run quality check on a single experiment."""
    print(f"\n{'='*70}")
    print(f"Running quality check: {exp_dir.name}")
    print(f"{'='*70}")

    cmd = [
        'python', 'scripts/reports/run_quality_check.py',
        '--model', 'dual_wgan',
        '--model_dir', str(exp_dir),
        '--training_data', training_data,
        '--n_generated', '1000'
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Quality check failed for {exp_dir.name}")
        print(f"Error: {e.stderr}")
        return False


def load_quality_summary(exp_dir: Path) -> Dict:
    """Load quality summary from experiment."""
    summary_path = exp_dir / 'quality_report' / 'quality_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def load_training_history(exp_dir: Path) -> Dict:
    """Load training history from experiment."""
    history_path = exp_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return None


def load_config(exp_dir: Path) -> Dict:
    """Load config from experiment."""
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description='Analyze Latent Space Experiments')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing experiment results')
    parser.add_argument('--training_data', type=str, default='./data/training',
                        help='Path to training data directory')
    parser.add_argument('--nz_values', type=str, default=None,
                        help='Comma-separated list of nz values to analyze (default: all found)')
    parser.add_argument('--skip_quality_check', action='store_true',
                        help='Skip quality check (use existing reports)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Find all experiments
    experiments = find_latent_experiments(results_dir)
    if not experiments:
        print(f"No dual WGAN experiments found in {results_dir}")
        sys.exit(1)

    # Filter by nz values if specified
    if args.nz_values:
        nz_filter = [int(x.strip()) for x in args.nz_values.split(',')]
        experiments = [exp for exp in experiments if extract_nz_from_path(exp) in nz_filter]

    print(f"\n{'='*70}")
    print(f"Found {len(experiments)} dual WGAN experiments")
    print(f"{'='*70}")
    for exp in experiments:
        nz = extract_nz_from_path(exp)
        print(f"  • {exp.name} (nz={nz})")
    print()

    # Run quality checks
    if not args.skip_quality_check:
        print("\n" + "="*70)
        print("STEP 1: Running Quality Checks")
        print("="*70)

        for exp in experiments:
            success = run_quality_check(exp, args.training_data)
            if not success:
                print(f"⚠️  Warning: Quality check failed for {exp.name}, continuing...")

    # Collect and analyze results
    print("\n" + "="*70)
    print("STEP 2: Collecting Results")
    print("="*70)

    results = []
    for exp in experiments:
        nz = extract_nz_from_path(exp)
        quality = load_quality_summary(exp)
        history = load_training_history(exp)
        config = load_config(exp)

        if quality and history:
            result = {
                'experiment': exp.name,
                'nz': nz,
                'config': config,
                'quality': quality,
                'training': {
                    'final_loss_G': history['loss_G'][-1] if history['loss_G'] else None,
                    'final_loss_C': history['loss_C'][-1] if history['loss_C'] else None,
                    'final_wasserstein': history['wasserstein_distance'][-1] if history['wasserstein_distance'] else None,
                    'avg_epoch_time': np.mean(history['epoch_time']) if 'epoch_time' in history else None,
                    'total_time': np.sum(history['epoch_time']) if 'epoch_time' in history else None,
                }
            }
            results.append(result)
            print(f"✓ Loaded: {exp.name} (nz={nz})")
        else:
            print(f"⚠️  Missing data for: {exp.name}")

    if not results:
        print("\n❌ No valid results collected")
        sys.exit(1)

    # Save collected results
    output_file = results_dir / 'latent_space_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Generate summary table
    print("\n" + "="*70)
    print("STEP 3: Summary")
    print("="*70)

    results_sorted = sorted(results, key=lambda x: x['nz'])

    print(f"\n{'nz':<6} {'FID↓':<10} {'MMD↓':<10} {'KS↓':<10} {'Corr':<10} {'Epoch(s)':<10}")
    print("-" * 60)

    for r in results_sorted:
        q = r['quality']
        t = r['training']

        fid = q.get('fid_score', {}).get('mean', float('nan'))
        mmd = q.get('mmd_score', {}).get('mean', float('nan'))
        ks = q.get('ks_statistic', {}).get('mean', float('nan'))
        corr = q.get('correlation_quality', {}).get('avg_correlation_match', float('nan'))
        epoch_time = t.get('avg_epoch_time', float('nan'))

        print(f"{r['nz']:<6} {fid:<10.4f} {mmd:<10.4f} {ks:<10.4f} {corr:<10.2f} {epoch_time:<10.2f}")

    # Find best performers
    print("\n" + "="*70)
    print("Best Performers")
    print("="*70)

    best_fid = min(results_sorted, key=lambda x: x['quality'].get('fid_score', {}).get('mean', float('inf')))
    best_mmd = min(results_sorted, key=lambda x: x['quality'].get('mmd_score', {}).get('mean', float('inf')))
    best_corr = max(results_sorted, key=lambda x: x['quality'].get('correlation_quality', {}).get('avg_correlation_match', float('-inf')))

    print(f"Best FID: nz={best_fid['nz']} (FID={best_fid['quality']['fid_score']['mean']:.4f})")
    print(f"Best MMD: nz={best_mmd['nz']} (MMD={best_mmd['quality']['mmd_score']['mean']:.4f})")
    print(f"Best Correlation: nz={best_corr['nz']} (Corr={best_corr['quality']['correlation_quality']['avg_correlation_match']:.2f})")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Review detailed reports in: {results_dir}/<experiment>/quality_report/")
    print(f"  2. Generate comparison report:")
    print(f"     python scripts/reports/generate_latent_comparison_report.py")


if __name__ == '__main__':
    main()
