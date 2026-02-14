#!/usr/bin/env python
"""
Generate comprehensive comparison report for latent space experiments.

Usage:
    python generate_latent_comparison_report.py --analysis_file ./results/latent_space_analysis.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def generate_markdown_report(results: list, output_path: Path):
    """Generate Markdown report comparing all latent space experiments."""

    results_sorted = sorted(results, key=lambda x: x['nz'])

    report = []
    report.append("# Dual WGAN Latent Space Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Executive Summary\n")
    report.append(f"Analyzed {len(results)} dual WGAN models trained with different latent space dimensions.\n")

    latent_sizes = [r['nz'] for r in results_sorted]
    report.append(f"**Latent dimensions tested:** {', '.join(map(str, latent_sizes))}\n")

    # Best performers
    best_fid = min(results_sorted, key=lambda x: x['quality'].get('fid_score', {}).get('mean', float('inf')))
    best_mmd = min(results_sorted, key=lambda x: x['quality'].get('mmd_score', {}).get('mean', float('inf')))
    best_corr = max(results_sorted, key=lambda x: x['quality'].get('correlation_quality', {}).get('avg_correlation_match', float('-inf')))

    report.append("### Best Models\n")
    report.append(f"- **Best FID Score:** nz={best_fid['nz']} (FID={best_fid['quality']['fid_score']['mean']:.4f})")
    report.append(f"- **Best MMD Score:** nz={best_mmd['nz']} (MMD={best_mmd['quality']['mmd_score']['mean']:.4f})")
    report.append(f"- **Best Correlation:** nz={best_corr['nz']} (Corr={best_corr['quality']['correlation_quality']['avg_correlation_match']:.2f})\n")

    # Detailed comparison table
    report.append("## Detailed Comparison\n")
    report.append("| nz | nz/(2K) | FID ↓ | MMD ↓ | KS ↓ | Corr. | W-dist | Epoch(s) | Total(min) |")
    report.append("|---:|--------:|------:|------:|-----:|------:|-------:|---------:|-----------:|")

    for r in results_sorted:
        q = r['quality']
        t = r['training']
        c = r['config']

        nz = r['nz']
        ratio = c.get('nz_to_output_ratio', nz / 102.0)
        fid = q.get('fid_score', {}).get('mean', float('nan'))
        mmd = q.get('mmd_score', {}).get('mean', float('nan'))
        ks = q.get('ks_statistic', {}).get('mean', float('nan'))
        corr = q.get('correlation_quality', {}).get('avg_correlation_match', float('nan'))
        wdist = t.get('final_wasserstein', float('nan'))
        epoch_time = t.get('avg_epoch_time', float('nan'))
        total_time = t.get('total_time', float('nan')) / 60

        report.append(f"| {nz} | {ratio:.3f} | {fid:.4f} | {mmd:.4f} | {ks:.4f} | {corr:.2f} | {wdist:.2f} | {epoch_time:.2f} | {total_time:.1f} |")

    report.append("\n### Metrics Explained\n")
    report.append("- **nz**: Latent space dimension")
    report.append("- **nz/(2K)**: Latent dimension relative to output dimension (102)")
    report.append("- **FID ↓**: Fréchet Inception Distance (lower is better)")
    report.append("- **MMD ↓**: Maximum Mean Discrepancy (lower is better)")
    report.append("- **KS ↓**: Kolmogorov-Smirnov statistic (lower is better)")
    report.append("- **Corr.**: Correlation match percentage (higher is better)")
    report.append("- **W-dist**: Final Wasserstein distance")
    report.append("- **Epoch(s)**: Average epoch training time")
    report.append("- **Total(min)**: Total training time in minutes\n")

    # Analysis by latent size
    report.append("## Analysis by Latent Dimension\n")

    for r in results_sorted:
        q = r['quality']
        nz = r['nz']

        report.append(f"### nz = {nz}\n")

        # Quality summary
        report.append("**Quality Metrics:**\n")
        report.append(f"- FID: {q.get('fid_score', {}).get('mean', 'N/A'):.4f}")
        report.append(f"- MMD: {q.get('mmd_score', {}).get('mean', 'N/A'):.4f}")
        report.append(f"- KS: {q.get('ks_statistic', {}).get('mean', 'N/A'):.4f}")

        corr_qual = q.get('correlation_quality', {})
        report.append(f"- Correlation match: {corr_qual.get('avg_correlation_match', 'N/A'):.2f}%")
        report.append(f"- Physical plausibility: {corr_qual.get('physical_plausibility', 'N/A'):.2f}%\n")

        # Distribution quality
        dist_qual = q.get('distribution_quality', {})
        if dist_qual:
            report.append("**Distribution Quality:**\n")
            report.append(f"- Mean error: {dist_qual.get('mean_error', 'N/A'):.4f}")
            report.append(f"- Std error: {dist_qual.get('std_error', 'N/A'):.4f}")
            report.append(f"- Coverage: {dist_qual.get('coverage', 'N/A'):.2f}%\n")

    # Recommendations
    report.append("## Recommendations\n")

    # Find sweet spot
    # Typically want low FID + high correlation + reasonable training time
    for r in results_sorted:
        r['composite_score'] = (
            1 / (r['quality'].get('fid_score', {}).get('mean', float('inf')) + 0.01) +
            r['quality'].get('correlation_quality', {}).get('avg_correlation_match', 0) / 100
        )

    best_overall = max(results_sorted, key=lambda x: x['composite_score'])

    report.append(f"**Recommended latent dimension: nz={best_overall['nz']}**\n")
    report.append("This configuration provides the best balance between:")
    report.append("- Generation quality (FID)")
    report.append("- Physical correctness (correlation)")
    report.append("- Training efficiency\n")

    # Observations
    report.append("## Key Observations\n")

    # Check if quality improves with size
    fid_scores = [(r['nz'], r['quality'].get('fid_score', {}).get('mean', float('inf'))) for r in results_sorted]
    corr_scores = [(r['nz'], r['quality'].get('correlation_quality', {}).get('avg_correlation_match', 0)) for r in results_sorted]

    if len(fid_scores) >= 3:
        # Simple trend analysis
        fid_improving = fid_scores[0][1] > fid_scores[-1][1]
        corr_improving = corr_scores[0][1] < corr_scores[-1][1]

        if fid_improving and corr_improving:
            report.append("- ✅ Both FID and correlation improve with larger latent dimensions")
        elif fid_improving:
            report.append("- ⚠️  FID improves but correlation may degrade with larger dimensions")
        else:
            report.append("- ⚠️  Smaller latent dimensions may be sufficient for this task")

    # Check for diminishing returns
    if len(fid_scores) >= 4:
        mid_point = len(fid_scores) // 2
        early_improvement = abs(fid_scores[0][1] - fid_scores[mid_point][1])
        late_improvement = abs(fid_scores[mid_point][1] - fid_scores[-1][1])

        if late_improvement < early_improvement * 0.3:
            report.append("- 📊 Diminishing returns observed beyond nz=" + str(fid_scores[mid_point][0]))

    report.append("\n---\n")
    report.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Markdown report saved to: {output_path}")


def generate_comparison_plots(results: list, output_dir: Path):
    """Generate comparison plots."""

    results_sorted = sorted(results, key=lambda x: x['nz'])
    nz_values = [r['nz'] for r in results_sorted]

    # Extract metrics
    fid_scores = [r['quality'].get('fid_score', {}).get('mean', float('nan')) for r in results_sorted]
    mmd_scores = [r['quality'].get('mmd_score', {}).get('mean', float('nan')) for r in results_sorted]
    corr_scores = [r['quality'].get('correlation_quality', {}).get('avg_correlation_match', float('nan')) for r in results_sorted]
    ks_scores = [r['quality'].get('ks_statistic', {}).get('mean', float('nan')) for r in results_sorted]

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # FID scores
    axes[0, 0].plot(nz_values, fid_scores, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Latent Dimension (nz)', fontsize=11)
    axes[0, 0].set_ylabel('FID Score', fontsize=11)
    axes[0, 0].set_title('FID vs Latent Dimension (lower is better)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log', base=2)

    # MMD scores
    axes[0, 1].plot(nz_values, mmd_scores, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Latent Dimension (nz)', fontsize=11)
    axes[0, 1].set_ylabel('MMD Score', fontsize=11)
    axes[0, 1].set_title('MMD vs Latent Dimension (lower is better)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log', base=2)

    # Correlation scores
    axes[1, 0].plot(nz_values, corr_scores, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Latent Dimension (nz)', fontsize=11)
    axes[1, 0].set_ylabel('Correlation Match (%)', fontsize=11)
    axes[1, 0].set_title('Correlation vs Latent Dimension (higher is better)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log', base=2)

    # KS statistic
    axes[1, 1].plot(nz_values, ks_scores, 'o-', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Latent Dimension (nz)', fontsize=11)
    axes[1, 1].set_ylabel('KS Statistic', fontsize=11)
    axes[1, 1].set_title('KS Statistic vs Latent Dimension (lower is better)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log', base=2)

    plt.suptitle('Dual WGAN: Quality Metrics vs Latent Dimension', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    plot_path = output_dir / 'latent_comparison_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison plots saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Latent Space Comparison Report')
    parser.add_argument('--analysis_file', type=str, default='./results/latent_space_analysis.json',
                        help='Path to analysis results JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as analysis file)')
    args = parser.parse_args()

    analysis_file = Path(args.analysis_file)
    if not analysis_file.exists():
        print(f"Error: Analysis file not found: {analysis_file}")
        print("Run run_latent_space_analysis.py first")
        sys.exit(1)

    # Load analysis results
    with open(analysis_file, 'r') as f:
        results = json.load(f)

    if not results:
        print("No results found in analysis file")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = analysis_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Generating Latent Space Comparison Report")
    print(f"{'='*70}")
    print(f"Experiments: {len(results)}")
    print(f"Output: {output_dir}")
    print()

    # Generate markdown report
    report_path = output_dir / 'latent_space_comparison_report.md'
    generate_markdown_report(results, report_path)

    # Generate comparison plots
    generate_comparison_plots(results, output_dir)

    print(f"\n{'='*70}")
    print("Report Generation Complete!")
    print(f"{'='*70}")
    print(f"\nView the report:")
    print(f"  {report_path}")
    print(f"\nView the plots:")
    print(f"  {output_dir / 'latent_comparison_metrics.png'}")


if __name__ == '__main__':
    main()
