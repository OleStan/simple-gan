# Train Model Skill

Train a GAN model for generating eddy-current material profiles.

## Usage

This skill helps you train GAN models with proper configuration and monitoring.

## Available Models

1. **Dual WGAN**: Basic dual-head WGAN for correlated sigma and mu profiles
2. **Improved WGAN V2**: Enhanced stability with spectral normalization and physics-informed loss

## Training Process

### 1. Prepare Training Data

Ensure training data exists at `data/training/`:
- `X_raw.npy`: Raw training profiles (N x 2K)
- `normalization_params.json`: Normalization parameters

### 2. Train Dual WGAN

```bash
cd models/dual_wgan
python train.py
```

Default configuration:
- Latent dimension (nz): 100
- Epochs: 500
- Batch size: 32
- Learning rate: 1e-4
- n_critic: 5
- Gradient penalty (λ_gp): 10

### 3. Train Improved WGAN V2

```bash
cd models/improved_wgan_v2
python train.py --epochs 500 --nz 100
```

Additional features:
- Spectral normalization
- Physics-informed loss with warmup
- Separate learning rates for Generator and Critic
- Checkpointing and resumable training

### 4. Monitor Training

Results are saved to:
- `models/{model_type}/results/{timestamp}/`
  - `models/`: Trained model checkpoints
  - `training_images/`: Generated samples during training
  - `training_history.json`: Loss curves and metrics
  - `training_curves.png`: Visualization of training progress

### 5. Validate Quality

```bash
cd scripts/reports
python run_quality_check.py --model dual_wgan --model_dir ../../models/dual_wgan/results/dual_wgan_TIMESTAMP
```

## Quality Metrics

The training should achieve:
- **Wasserstein Distance**: Converging to stable value
- **Generator Loss**: Decreasing trend
- **Critic Loss**: Stable oscillation
- **Physics Consistency**:
  - Profiles within physical bounds
  - Smooth gradients
  - Realistic EDC response

## Troubleshooting

- **Mode collapse**: Reduce learning rate, increase n_critic
- **Training instability**: Enable spectral normalization, adjust gradient penalty
- **Poor quality**: Increase physics loss weight, extend training epochs
