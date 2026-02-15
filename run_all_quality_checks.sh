#!/bin/bash
# Run quality checks on all experiments

echo "============================================"
echo "  Running Quality Checks"
echo "============================================"
echo ""

DUAL_COUNT=0
IMPROVED_COUNT=0
FAILED=0

echo "=== Dual WGAN Experiments ==="
echo ""

for MODEL_DIR in results/dual_wgan_nz*/; do
    if [ -f "$MODEL_DIR/models/netG_final.pth" ]; then
        echo "Quality check: $(basename $MODEL_DIR)"
        PYTHONPATH=. python scripts/reports/run_quality_check.py \
            --model dual_wgan \
            --model_dir "$MODEL_DIR" \
            --training_data ./data/training \
            --n_generated 1000 2>&1 | tail -5

        if [ $? -eq 0 ]; then
            ((DUAL_COUNT++))
        else
            ((FAILED++))
        fi
        echo ""
    fi
done

echo "=== Improved WGAN v2 Experiments ==="
echo ""

for MODEL_DIR in results/improved_wgan_v2_nz*/; do
    if [ -f "$MODEL_DIR/models/netG_final.pt" ]; then
        echo "Quality check: $(basename $MODEL_DIR)"
        PYTHONPATH=. python scripts/reports/run_quality_check.py \
            --model improved_wgan_v2 \
            --model_dir "$MODEL_DIR" \
            --training_data ./data/training \
            --n_generated 1000 2>&1 | tail -5

        if [ $? -eq 0 ]; then
            ((IMPROVED_COUNT++))
        else
            ((FAILED++))
        fi
        echo ""
    fi
done

echo "============================================"
echo "✓ Dual WGAN quality checks: $DUAL_COUNT"
echo "✓ Improved WGAN v2 quality checks: $IMPROVED_COUNT"
echo "✗ Failed: $FAILED"
echo "============================================"
