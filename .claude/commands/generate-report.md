# Generate Report Skill

Create comprehensive analysis reports for trained GAN models.

## Report Types

### 1. Training Report

Documents training process and final model performance.

```bash
cd scripts/reports
python generate_dual_wgan_report.py --model_dir ../../models/dual_wgan/results/dual_wgan_TIMESTAMP
```

**Contents**:
- Training configuration
- Loss curves and convergence analysis
- Generated sample visualization
- Distribution comparison
- Model architecture summary

### 2. Quality Report

Detailed quality metrics and physics validation.

```bash
cd scripts/reports
python run_quality_check.py --model improved_wgan_v2 --model_dir ../../models/improved_wgan_v2/results/TIMESTAMP
```

**Contents**:
- Statistical metrics (MSE, KL divergence)
- Physics consistency checks
- EDC response analysis
- Latent space visualization
- Quality score summary

### 3. Comparison Report

Compare multiple models or training runs.

```bash
cd scripts/comparison
python compare_wgan_approaches.py \
    --model1 ../../models/dual_wgan/results/dual_wgan_TIMESTAMP1 \
    --model2 ../../models/improved_wgan_v2/results/improved_wgan_v2_TIMESTAMP2
```

**Contents**:
- Side-by-side metrics
- Training efficiency comparison
- Quality score comparison
- Recommendation summary

## Report Outputs

All reports are saved to:
- PDF format: For presentation and documentation
- JSON format: For programmatic analysis
- PNG figures: For embedding in papers/presentations

### Directory Structure

```
models/{model_type}/results/{timestamp}/
├── training_report.md
├── quality_report/
│   ├── quality_summary.md
│   ├── quality_metrics.json
│   └── figures/
├── comparison_report.md
└── visualizations/
```

## Key Metrics to Monitor

### Training Metrics
- **Generator Loss**: Should decrease and stabilize
- **Critic Loss**: Should oscillate around stable value
- **Wasserstein Distance**: Indicates distribution matching
- **Gradient Penalty**: Should remain stable

### Quality Metrics
- **Distribution Overlap**: Real vs Generated
- **Physics Violation Rate**: < 5% target
- **EDC Response Accuracy**: < 10% deviation
- **Sample Diversity**: High entropy

### Performance Metrics
- **Training Time**: Time per epoch
- **Memory Usage**: Peak GPU memory
- **Model Size**: Parameters and storage
- **Inference Speed**: Samples per second

## Automated Reporting

Set up automated quality checks after training:

```bash
# Add to training script
python train.py && python run_quality_check.py --model_dir ./results/latest
```

## Report Customization

Customize reports by editing:
- `eddy_current_workflow/quality/report_generator.py`
- `eddy_current_workflow/quality/metrics.py`

Add custom metrics:
```python
from eddy_current_workflow.quality import QualityReportGenerator

generator = QualityReportGenerator()
generator.add_custom_metric("my_metric", my_metric_function)
generator.generate_report(model_dir)
```
