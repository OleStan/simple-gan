#!/bin/bash
# Generate reports for all improved WGAN v2 experiments

echo "============================================"
echo "  Generating Improved WGAN v2 Reports"
echo "============================================"
echo ""

COUNT=0
for MODEL_DIR in results/improved_wgan_v2_nz*/; do
    if [ -f "$MODEL_DIR/models/netG_final.pt" ]; then
        echo "Generating report for: $(basename $MODEL_DIR)"
        PYTHONPATH=. python scripts/reports/generate_improved_wgan_v2_report.py "$MODEL_DIR" 2>&1 | tail -10
        ((COUNT++))
        echo ""
    fi
done

echo "============================================"
echo "✓ Generated $COUNT improved WGAN v2 reports"
echo "============================================"
