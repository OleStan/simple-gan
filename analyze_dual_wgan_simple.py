#!/usr/bin/env python
"""Simple analysis of dual WGAN latent space experiments."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

# Find all dual WGAN experiment directories
results_dir = Path('./results')
experiments = sorted([d for d in results_dir.glob('dual_wgan_nz*')
                     if d.is_dir() and (d / 'models' / 'netG_final.pth').exists()])

print(f"Found {len(experiments)} dual WGAN experiments")
print("="*70)

# Collect results
results = []

for exp_dir in experiments:
    # Extract nz value
    import re
    match = re.search(r'nz(\d+)', exp_dir.name)
    if not match:
        continue
    nz = int(match.group(1))

    # Load config
    config_path = exp_dir / 'config.json'
    if not config_path.exists():
        print(f"⚠️  No config.json for {exp_dir.name}")
        continue

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load training history
    history_path = exp_dir / 'training_history.json'
    if not history_path.exists():
        print(f"⚠️  No training_history.json for {exp_dir.name}")
        continue

    with open(history_path, 'r') as f:
        history = json.load(f)

    # Calculate final metrics (average of last 50 epochs)
    final_loss_C = np.mean(history['loss_C'][-50:]) if history['loss_C'] else None
    final_loss_G = np.mean(history['loss_G'][-50:]) if history['loss_G'] else None
    final_w_dist = np.mean(history['wasserstein_distance'][-50:]) if history['wasserstein_distance'] else None

    # Calculate stability (std of last 100 epochs)
    stability_C = np.std(history['loss_C'][-100:]) if len(history['loss_C']) >= 100 else None
    stability_G = np.std(history['loss_G'][-100:]) if len(history['loss_G']) >= 100 else None

    result = {
        'nz': nz,
        'epochs': len(history['loss_C']),
        'final_loss_C': final_loss_C,
        'final_loss_G': final_loss_G,
        'final_w_dist': final_w_dist,
        'stability_C': stability_C,
        'stability_G': stability_G,
        'batch_size': config.get('batch_size', 'N/A'),
        'lr': config.get('lr', 'N/A')
    }

    results.append(result)
    print(f"✓ nz={nz:3d}: Loss_C={final_loss_C:7.4f}, Loss_G={final_loss_G:7.4f}, W_dist={final_w_dist:7.4f}")

print("="*70)
print(f"\nTotal analyzed: {len(results)} experiments")

# Sort by nz
results = sorted(results, key=lambda x: x['nz'])

# Save results
output_file = results_dir / 'dual_wgan_simple_analysis.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to: {output_file}")

# Create comparison plots
if len(results) > 0:
    nz_values = [r['nz'] for r in results]
    loss_C_values = [r['final_loss_C'] for r in results]
    loss_G_values = [r['final_loss_G'] for r in results]
    w_dist_values = [r['final_w_dist'] for r in results]
    stability_C_values = [r['stability_C'] for r in results]
    stability_G_values = [r['stability_G'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dual WGAN: Latent Space Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Final Losses
    ax = axes[0, 0]
    ax.plot(nz_values, loss_C_values, 'o-', label='Critic Loss', linewidth=2, markersize=8)
    ax.plot(nz_values, loss_G_values, 's-', label='Generator Loss', linewidth=2, markersize=8)
    ax.set_xlabel('Latent Dimension (nz)', fontsize=12)
    ax.set_ylabel('Final Loss (avg last 50 epochs)', fontsize=12)
    ax.set_title('Training Losses vs Latent Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Plot 2: Wasserstein Distance
    ax = axes[0, 1]
    ax.plot(nz_values, w_dist_values, 'o-', color='green', linewidth=2, markersize=8)
    ax.set_xlabel('Latent Dimension (nz)', fontsize=12)
    ax.set_ylabel('Wasserstein Distance', fontsize=12)
    ax.set_title('Wasserstein Distance vs Latent Dimension', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Plot 3: Training Stability
    ax = axes[1, 0]
    ax.plot(nz_values, stability_C_values, 'o-', label='Critic Stability', linewidth=2, markersize=8)
    ax.plot(nz_values, stability_G_values, 's-', label='Generator Stability', linewidth=2, markersize=8)
    ax.set_xlabel('Latent Dimension (nz)', fontsize=12)
    ax.set_ylabel('Std Dev (last 100 epochs)', fontsize=12)
    ax.set_title('Training Stability vs Latent Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Plot 4: Summary Table
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary table
    table_data = []
    for r in results:
        table_data.append([
            f"{r['nz']}",
            f"{r['final_loss_C']:.4f}",
            f"{r['final_loss_G']:.4f}",
            f"{r['final_w_dist']:.4f}"
        ])

    table = ax.table(cellText=table_data,
                    colLabels=['nz', 'Loss C', 'Loss G', 'W Dist'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.set_title('Final Metrics Summary', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    plot_file = results_dir / 'dual_wgan_latent_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_file}")

    plt.close()

# Generate markdown report
report_lines = [
    "# Dual WGAN Latent Space Analysis",
    "",
    f"**Total Experiments:** {len(results)}",
    "",
    "## Summary",
    "",
    "| nz | Epochs | Final Loss C | Final Loss G | W Distance | Stability C | Stability G |",
    "|---:|-------:|-------------:|-------------:|-----------:|------------:|------------:|"
]

for r in results:
    report_lines.append(
        f"| {r['nz']} | {r['epochs']} | {r['final_loss_C']:.4f} | "
        f"{r['final_loss_G']:.4f} | {r['final_w_dist']:.4f} | "
        f"{r['stability_C']:.4f} | {r['stability_G']:.4f} |"
    )

report_lines.extend([
    "",
    "## Key Findings",
    "",
    "### Best Configuration (by Wasserstein Distance)",
    ""
])

# Find best by W distance (closest to 0)
best_w = min(results, key=lambda x: abs(x['final_w_dist']))
report_lines.append(f"- **nz={best_w['nz']}**: W_dist={best_w['final_w_dist']:.4f}")

report_lines.extend([
    "",
    "### Most Stable Training (by Generator Loss Stability)",
    ""
])

# Find most stable
best_stable = min(results, key=lambda x: x['stability_G'])
report_lines.append(f"- **nz={best_stable['nz']}**: Stability={best_stable['stability_G']:.4f}")

report_lines.extend([
    "",
    "## Plots",
    "",
    "![Latent Space Comparison](dual_wgan_latent_comparison.png)",
    ""
])

report_file = results_dir / 'dual_wgan_analysis_report.md'
with open(report_file, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Report saved to: {report_file}")
print("\n" + "="*70)
print("Analysis complete!")
