# GANs for Eddy-Current NDT with Physics-Informed Deep Learning

A comprehensive framework for generating realistic material property profiles using Generative Adversarial Networks (GANs), integrated with physics-based forward/inverse eddy-current modeling for non-destructive testing (NDT) applications.

## Overview

This project combines:
- **AI-driven generation**: Dual-head WGANs for creating correlated conductivity (σ) and permeability (μ) profiles
- **Physics-based validation**: Dodd-Deeds analytical solver for eddy-current impedance
- **Inverse problem solving**: Recovering material properties from impedance measurements
- **Quality assurance**: Comprehensive metrics for physics consistency and statistical realism

## Key Features

- **Multiple GAN Architectures**:
  - Dual-head WGAN for correlated profile generation
  - Improved WGAN v2 with spectral normalization and physics-informed loss
  - Support for variable latent dimensions

- **Complete EDC Workflow**:
  - Forward solver: Dodd-Deeds analytical solution
  - Inverse solver: Multi-start optimization with regularization
  - Profile normalization and denormalization
  - EDC database generation

- **Quality Validation**:
  - Statistical metrics (MSE, KL divergence, Frechet distance)
  - Physics consistency checks (bounds, smoothness, monotonicity)
  - Latent space analysis
  - Automated report generation

## Project Structure

```
├── docs/                   # Documentation (guides, technical specs, reports)
├── eddy_current_workflow/  # Main pipeline library (forward/inverse EDC)
├── models/                 # GAN models organized by type
│   ├── dual_wgan/         # Dual-head WGAN
│   ├── improved_wgan_v2/  # Enhanced WGAN with stability
│   └── legacy/            # Older implementations
├── data/                   # Training, testing, and raw data
├── results/                # Training results and visualizations
├── scripts/                # Utility scripts (data gen, viz, reports)
├── tests/                  # Test suite
├── agents/                 # AI agent configurations
└── .claude/               # Claude Code skills
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed organization.

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/GANs-for-1D-Signal.git
cd GANs-for-1D-Signal

# Install dependencies
pip install torch numpy matplotlib scipy pytest
```

### 2. Generate Training Data

```bash
cd scripts/data_generation
python generate_training_data.py --n_samples 2000 --K 51
```

This creates:
- `data/training/X_raw.npy`: Training profiles (2000 × 102)
- `data/training/normalization_params.json`: Normalization parameters

### 3. Train a Model

```bash
cd models/improved_wgan_v2
python train.py --epochs 500 --nz 100
```

Results saved to `./results/improved_wgan_v2_TIMESTAMP/`

### 4. Validate Quality

```bash
cd scripts/reports
python run_quality_check.py \
    --model improved_wgan_v2 \
    --model_dir ../../models/improved_wgan_v2/results/improved_wgan_v2_TIMESTAMP \
    --n_generated 1000
```

## Documentation

- **[Complete Guide](docs/guides/COMPLETE_GUIDE.md)**: Comprehensive technical overview
- **[Project Structure](PROJECT_STRUCTURE.md)**: Detailed directory organization
- **[Training Guide](docs/guides/RUNPOD_TRAINING_GUIDE.md)**: Remote training on RunPod
- **[Quality Guide](docs/technical/GAN_Quality_Guide.md)**: Quality metrics and validation
- **[Latent Analysis](docs/technical/LATENT_DIMENSION_ANALYSIS.md)**: Latent dimension optimization

## Usage Examples

### Train with Custom Configuration

```bash
cd models/improved_wgan_v2
python train.py \
    --epochs 500 \
    --nz 100 \
    --batch_size 32 \
    --lr_g 5e-5 \
    --lr_c 2e-4 \
    --lambda_physics 0.1
```

### Compare Multiple Models

```bash
cd scripts/comparison
python compare_wgan_approaches.py \
    --model1 ../../models/dual_wgan/results/dual_wgan_20260131_213107 \
    --model2 ../../models/improved_wgan_v2/results/improved_wgan_v2_20260131_213112
```

### Generate EDC Database

```python
from eddy_current_workflow.forward import edc_forward, ProbeSettings
import torch
import numpy as np

# Load trained generator
generator = load_generator('models/improved_wgan_v2/results/TIMESTAMP')

# Generate profiles
noise = torch.randn(1000, 100)
profiles, sigma, mu = generator(noise)

# Compute EDC responses
probe = ProbeSettings(frequency=1e6)
for i in range(len(sigma)):
    edc = edc_forward(sigma[i].numpy(), mu[i].numpy(), probe)
    print(f"Sample {i}: {edc}")
```

## Claude Code Skills

Integrated skills for streamlined workflow:

```bash
/train-model       # Model training guidance
/quality-check     # Quality validation workflow
/generate-report   # Comprehensive reporting
```

## Testing

```bash
cd tests
python test_dodd_deeds_solver.py  # Validate forward solver
python test_inverse_solver.py     # Validate inverse problem
python test_phase1_foundation.py  # Foundation tests
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy 1.20+
- Matplotlib 3.3+
- SciPy 1.7+
- pytest (for testing)

## Key Improvements Over Original

This project extends the original [GANs-for-1D-Signal](https://github.com/LixiangHan/GANs-for-1D-Signal) with:

1. **Physics Integration**: Forward/inverse EDC modeling (Dodd-Deeds)
2. **Dual Profiles**: Simultaneous generation of correlated σ and μ
3. **Enhanced Stability**: Spectral normalization, physics-informed loss
4. **Quality Assurance**: Comprehensive validation metrics
5. **Production Ready**: Organized structure, documentation, tests
6. **Reproducibility**: Checkpointing, configuration management

## Results

### Training Convergence

- **Dual WGAN**: 500 epochs, stable Wasserstein distance
- **Improved WGAN V2**: Better quality, fewer physics violations

### Quality Metrics (Improved WGAN V2)

- MSE: 0.023 (excellent)
- Physics violations: 2.3% (good)
- EDC response accuracy: 6.8% deviation (good)
- Latent interpolation: Smooth

See [docs/reports/](docs/reports/) for detailed analysis.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## Citation

If you use this code, please cite:

```bibtex
@software{gans_eddy_current_ndt,
  title = {GANs for Eddy-Current NDT with Physics-Informed Deep Learning},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/GANs-for-1D-Signal}
}
```

Original GANs-for-1D-Signal:
```bibtex
@software{gans_1d_signal,
  title = {GANs-for-1D-Signal},
  author = {Lixiang Han},
  url = {https://github.com/LixiangHan/GANs-for-1D-Signal}
}
```

## References

1. **Dodd & Deeds (1968)**: "Analytical Solutions to Eddy-Current Probe-Coil Problems", J. Appl. Phys.
2. **Arjovsky et al. (2017)**: "Wasserstein GAN", ICML
3. **Miyato et al. (2018)**: "Spectral Normalization for GANs", ICLR

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Original GANs-for-1D-Signal framework by Lixiang Han
- Eddy-current physics implementation based on Dodd-Deeds analytical solution
- Physics-informed loss inspired by PINNs literature

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Status**: Active Development | Last Updated: February 2026
