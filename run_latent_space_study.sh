#!/bin/bash
# Complete workflow for running latent space study on RunPod
# Usage: ./run_latent_space_study.sh

set -euo pipefail

echo "============================================"
echo "  Dual WGAN Latent Space Study"
echo "============================================"
echo ""
echo "This script will:"
echo "  1. Run training experiments on RunPod with different latent sizes"
echo "  2. Download all results when training completes"
echo "  3. Run quality metrics on each model"
echo "  4. Generate comprehensive comparison report"
echo ""

# Latent sizes to test
LATENT_SIZES=(6 8 10 12 16 24 32 48 64 96 128)
echo "Latent dimensions to test: ${LATENT_SIZES[@]}"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Launch experiments on RunPod
echo ""
echo "============================================"
echo "STEP 1: Launching Experiments on RunPod"
echo "============================================"
./models/runpod/run_latent_experiments.sh

echo ""
echo "Experiments launched on RunPod!"
echo ""
echo "Training will take approximately:"
echo "  - 6 minutes per experiment"
echo "  - Total: ~$(( 6 * ${#LATENT_SIZES[@]} )) minutes for all experiments"
echo ""
echo "Monitor progress with:"
echo "  ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 'tmux list-sessions'"
echo ""

read -p "Wait for training to complete, then press Enter to download results..."

# Step 2: Download all results
echo ""
echo "============================================"
echo "STEP 2: Downloading Results from RunPod"
echo "============================================"

echo "Downloading all dual_wgan results..."
scp -P 13572 -i ~/.ssh/id_ed25519 -r \
    root@213.173.108.11:/workspace/GANs-for-1D-Signal/results/dual_wgan_nz* \
    ./results/ || echo "Some downloads may have failed (might be expected)"

echo ""
echo "✓ Results downloaded to ./results/"

# Step 3: Run quality checks
echo ""
echo "============================================"
echo "STEP 3: Running Quality Metrics"
echo "============================================"

python scripts/reports/run_latent_space_analysis.py \
    --results_dir ./results \
    --training_data ./data/training

echo ""
echo "✓ Quality analysis complete"

# Step 4: Generate comparison report
echo ""
echo "============================================"
echo "STEP 4: Generating Comparison Report"
echo "============================================"

python scripts/reports/generate_latent_comparison_report.py \
    --analysis_file ./results/latent_space_analysis.json \
    --output_dir ./results

echo ""
echo "============================================"
echo "  Study Complete!"
echo "============================================"
echo ""
echo "Results:"
echo "  • Summary: ./results/latent_space_analysis.json"
echo "  • Report: ./results/latent_space_comparison_report.md"
echo "  • Plots: ./results/latent_comparison_metrics.png"
echo ""
echo "Individual experiment results:"
ls -d results/dual_wgan_nz* 2>/dev/null | while read dir; do
    echo "  • $dir"
done
echo ""
echo "Next steps:"
echo "  1. Review the comparison report: ./results/latent_space_comparison_report.md"
echo "  2. Examine quality plots: ./results/latent_comparison_metrics.png"
echo "  3. Select the best nz value for your application"
echo ""
