#!/bin/bash
# Generate reports for all dual WGAN experiments

echo "============================================"
echo "  Generating Dual WGAN Reports"
echo "============================================"
echo ""

COUNT=0
for MODEL_DIR in results/dual_wgan_nz*/; do
    if [ -f "$MODEL_DIR/models/netG_final.pth" ]; then
        echo "Generating report for: $(basename $MODEL_DIR)"
        PYTHONPATH=. python scripts/reports/generate_dual_wgan_report.py "$MODEL_DIR" 2>&1 | tail -10
        ((COUNT++))
        echo ""
    fi
done

echo "============================================"
echo "✓ Generated $COUNT dual WGAN reports"
echo "============================================"
