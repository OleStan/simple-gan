# Dual WGAN Latent Space Study

Comprehensive workflow for testing different latent space dimensions and finding the optimal configuration.

## Quick Start

### Option 1: Automated Full Study (Recommended)

Run the complete workflow:
```bash
./run_latent_space_study.sh
```

This will:
1. Launch training experiments on RunPod for latent dimensions: 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128
2. Download all results when complete
3. Run quality metrics on each model
4. Generate comprehensive comparison report

**Total time:** ~66 minutes (6 min per experiment × 11 experiments)

### Option 2: Manual Step-by-Step

If you want more control, run each step manually:

#### 1. Launch Training on RunPod
```bash
./models/runpod/run_latent_experiments.sh
```

Monitor progress:
```bash
# List all running sessions
ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 "tmux list-sessions"

# Attach to specific experiment
ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 "tmux attach -t dual_wgan_nz32"
```

#### 2. Download Results
```bash
# Download all results
scp -P 13572 -i ~/.ssh/id_ed25519 -r \
    root@213.173.108.11:/workspace/GANs-for-1D-Signal/results/dual_wgan_nz* \
    ./results/

# Or use the sync script
./models/runpod/sync_from_pod.sh all
```

#### 3. Run Quality Analysis
```bash
python scripts/reports/run_latent_space_analysis.py \
    --results_dir ./results \
    --training_data ./data/training
```

Options:
- `--nz_values 6,8,10,12,16,32`: Analyze only specific sizes
- `--skip_quality_check`: Use existing quality reports

#### 4. Generate Comparison Report
```bash
python scripts/reports/generate_latent_comparison_report.py \
    --analysis_file ./results/latent_space_analysis.json \
    --output_dir ./results
```

## Testing Individual Latent Sizes

### On RunPod
```bash
# Upload script (if not done already)
scp -P 13572 -i ~/.ssh/id_ed25519 \
    models/dual_wgan/train_latent_experiment.py \
    root@213.173.108.11:/workspace/GANs-for-1D-Signal/models/dual_wgan/

# Run single experiment
ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 \
    "cd /workspace/GANs-for-1D-Signal/models/dual_wgan && \
     tmux new-session -d -s dual_wgan_nz16 \
     'python train_latent_experiment.py --nz 16 --epochs 500'"

# Monitor
ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 \
    "tmux attach -t dual_wgan_nz16"
```

### Locally (CPU/GPU)
```bash
cd models/dual_wgan
python train_latent_experiment.py --nz 32 --epochs 500
```

Options:
```bash
--nz 32              # Latent space dimension
--epochs 500         # Number of epochs
--batch_size 256     # Batch size
--lr 1e-4            # Learning rate
--num_workers 4      # Data loading workers
--save_interval 50   # Save every N epochs
--image_interval 10  # Generate images every N epochs
```

## Output Structure

After running experiments, you'll have:

```
results/
├── dual_wgan_nz6_<timestamp>/
│   ├── models/
│   │   ├── netG_final.pth
│   │   ├── netC_final.pth
│   │   └── ...
│   ├── quality_report/
│   │   ├── quality_report.md
│   │   ├── quality_summary.json
│   │   └── *.png
│   ├── training_images/
│   ├── training_history.json
│   ├── training_curves.png
│   └── config.json
├── dual_wgan_nz8_<timestamp>/
│   └── ...
├── dual_wgan_nz10_<timestamp>/
│   └── ...
...
├── latent_space_analysis.json           # All results combined
├── latent_space_comparison_report.md    # Full comparison report
└── latent_comparison_metrics.png        # Comparison plots
```

## Understanding Results

### Key Metrics

1. **FID (Fréchet Inception Distance)** ↓
   - Measures how close generated distribution is to real data
   - Lower is better
   - Good: < 50, Excellent: < 20

2. **MMD (Maximum Mean Discrepancy)** ↓
   - Statistical distance between distributions
   - Lower is better
   - Complements FID

3. **Correlation Match** ↑
   - How well σ-μ correlations are preserved
   - Higher is better (0-100%)
   - Important for physical plausibility

4. **KS Statistic** ↓
   - Distribution similarity test
   - Lower is better
   - Good: < 0.1

### Interpretation

The comparison report (`latent_space_comparison_report.md`) will show:

- **Performance vs Latent Size**: How quality metrics change with nz
- **Sweet Spot**: Optimal nz that balances quality and efficiency
- **Diminishing Returns**: Where increasing nz stops helping
- **Recommendations**: Best configuration for your use case

### Example Findings

Typical patterns:
- **Too small (nz < 10)**: Poor quality, can't capture complexity
- **Optimal (nz ≈ 16-48)**: Best quality/efficiency balance
- **Too large (nz > 96)**: Minimal improvement, longer training

The optimal nz depends on:
- Output dimensionality (2K = 102)
- Data complexity
- Training stability
- Computational budget

## Troubleshooting

### Training Failed
```bash
# Check session
ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 "tmux list-sessions"

# View logs
ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 \
    "tmux capture-pane -t dual_wgan_nz32 -p"

# Restart failed experiment
ssh -p 13572 -i ~/.ssh/id_ed25519 root@213.173.108.11 \
    "tmux kill-session -t dual_wgan_nz32"
# Then relaunch with run_latent_experiments.sh
```

### Quality Check Failed
```bash
# Run quality check manually for one experiment
python scripts/reports/run_quality_check.py \
    --model dual_wgan \
    --model_dir ./results/dual_wgan_nz32_<timestamp> \
    --training_data ./data/training
```

### Missing Data
```bash
# Check what's downloaded
ls -lh results/dual_wgan_nz*/

# Re-download specific result
scp -P 13572 -i ~/.ssh/id_ed25519 -r \
    root@213.173.108.11:/workspace/GANs-for-1D-Signal/results/dual_wgan_nz32_* \
    ./results/
```

## Cost Optimization

Running on RunPod:
- **Single experiment**: ~6 minutes on RTX 4000 Ada
- **Full study (11 experiments)**: ~66 minutes = **~$0.50-1.00**

To minimize costs:
1. Use the automated script (no idle time)
2. Stop the pod immediately after downloading results
3. Test fewer latent sizes initially (e.g., 6, 12, 24, 48)

## Next Steps

After finding the optimal latent dimension:

1. **Use it in production**:
   ```python
   from models.dual_wgan.model import DualHeadGenerator
   netG = DualHeadGenerator(nz=32, K=51)  # Use your optimal nz
   netG.load_state_dict(torch.load('path/to/best_model.pth'))
   ```

2. **Fine-tune hyperparameters** (learning rate, batch size, etc.)

3. **Run longer training** (e.g., 1000 epochs) with optimal nz

4. **Generate production dataset** for downstream tasks

---

**Ready to start?** Run:
```bash
./run_latent_space_study.sh
```
