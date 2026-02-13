# Project Structure

This document describes the organization of the GANs-for-1D-Signal project.

## Directory Overview

```
GANs-for-1D-Signal/
├── docs/                          # All documentation
├── eddy_current_workflow/         # Main pipeline library
├── models/                        # GAN models organized by type
├── data/                          # Training, testing, and raw data
├── results/                       # Training results and visualizations
├── scripts/                       # Utility scripts
├── tests/                         # Test files
├── agents/                        # AI agents
└── .claude/                       # Claude Code skills
```

## Detailed Structure

### `/docs` - Documentation

All project documentation organized by category:

```
docs/
├── architecture/          # System architecture docs
├── guides/               # User guides (COMPLETE_GUIDE.md, etc.)
│   ├── COMPLETE_GUIDE.md        # Main comprehensive guide
│   ├── RUNPOD_TRAINING_GUIDE.md # Remote training guide
│   └── ...
├── technical/            # Technical specifications
│   ├── GAN_Quality_Guide.md
│   ├── LATENT_DIMENSION_ANALYSIS.md
│   └── ...
├── reports/              # Generated analysis reports
├── setup/                # Setup and installation guides
├── restoration/          # Code restoration documentation
└── i18n/                 # Internationalization (Ukrainian docs)
```

### `/eddy_current_workflow` - Main Pipeline Library

The core library implementing the complete eddy-current NDT workflow:

```
eddy_current_workflow/
├── config/               # Configuration management
│   └── global_config.py
├── forward/              # Forward EDC solver (Dodd-Deeds)
│   └── edc_solver.py
├── inverse/              # Inverse problem solving
│   ├── objective.py      # Objective functions
│   ├── optimizers.py     # Optimization methods
│   └── recovery.py       # High-level recovery API
├── profiles/             # Profile handling and normalization
│   └── normalization.py
├── quality/              # Quality metrics and validation
│   ├── metrics.py
│   ├── physics_consistency.py
│   ├── latent_analysis.py
│   ├── quality_checker.py
│   └── report_generator.py
├── database/             # Database management
├── models/               # Model interfaces
└── pipelines/            # End-to-end pipelines
```

**Key Purpose**: This module should remain untouched as it implements the complete workflow described in [COMPLETE_GUIDE.md](docs/guides/COMPLETE_GUIDE.md).

### `/models` - GAN Models

Different GAN architectures organized by model type:

```
models/
├── dual_wgan/                    # Dual-head WGAN
│   ├── model.py                  # Network architecture
│   ├── train.py                  # Training script
│   ├── train_small_latent.py    # Smaller latent dimension variant
│   └── results/                  # Training results
│       └── dual_wgan_TIMESTAMP/
│           ├── models/           # Saved model checkpoints
│           ├── training_images/  # Generated samples
│           └── training_history.json
│
├── improved_wgan_v2/             # Enhanced WGAN with stability features
│   ├── model.py                  # Spectral norm Conv1D architecture
│   ├── train.py                  # Training script with physics loss
│   └── results/                  # Training results
│       └── improved_wgan_v2_TIMESTAMP/
│           ├── models/           # Saved model checkpoints
│           ├── checkpoints/      # Resumable checkpoints
│           ├── training_images/
│           └── training_history.json
│
├── legacy/                       # Old/experimental models
│   ├── wgan.py
│   ├── dcgan.py
│   ├── wgan_improved.py
│   └── ...
│
└── runpod/                       # Remote training configurations
    └── ...
```

**Training a Model**:
```bash
cd models/improved_wgan_v2
python train.py --epochs 500 --nz 100
```

### `/data` - Data Organization

All datasets organized by purpose:

```
data/
├── training/                     # Training datasets
│   ├── X_raw.npy                # Raw training profiles (N x 2K)
│   ├── normalization_params.json # Normalization parameters
│   └── archive_TIMESTAMP/       # Archived datasets
│
├── testing/                      # Testing datasets
│   ├── test_profiles.npy
│   └── ...
│
└── raw/                          # Raw experimental data
    ├── brilliant_blue/
    ├── quartz/
    └── mddect_*/
```

**Data Format**:
- **X_raw.npy**: Shape (N, 2K) - First K columns are σ profiles, last K columns are μ profiles
- **normalization_params.json**: Contains sigma_min, sigma_max, mu_min, mu_max, K

### `/results` - Training Results

Organized experiment results:

```
results/
├── experiments/                  # Timestamped experiment results
│   ├── comparison_analysis_TIMESTAMP/
│   ├── dual_wgan_training.log
│   └── improved_wgan_v2_training.log
│
├── visualizations/               # Generated plots and figures
│   └── training_data_visualization/
│
└── quality_reports/              # Quality assessment reports
    └── model_TIMESTAMP_quality/
```

**Note**: Most model-specific results are now stored in `models/{model_type}/results/`

### `/scripts` - Utility Scripts

Organized by function:

```
scripts/
├── data_generation/              # Data generation utilities
│   ├── generate_training_data.py
│   ├── generate_single_metal_data.py
│   └── ...
│
├── visualization/                # Visualization scripts
│   ├── visualize_training_data.py
│   ├── preview_training_data.py
│   └── generate_latent_display.py
│
├── comparison/                   # Model comparison tools
│   ├── compare_wgan_approaches.py
│   ├── compare_real_vs_generated_overlay.py
│   └── ...
│
└── reports/                      # Report generation
    ├── generate_dual_wgan_report.py
    ├── generate_improved_wgan_v2_report.py
    └── run_quality_check.py
```

**Usage Example**:
```bash
cd scripts/reports
python run_quality_check.py --model improved_wgan_v2 --model_dir ../../models/improved_wgan_v2/results/TIMESTAMP
```

### `/tests` - Test Suite

All test files:

```
tests/
├── test_phase1_foundation.py    # Foundation component tests
├── test_dodd_deeds_solver.py    # Forward solver validation
├── test_inverse_solver.py       # Inverse problem tests
├── test_config.json
└── test_normalizer.json
```

**Run Tests**:
```bash
cd tests
python test_dodd_deeds_solver.py
python test_inverse_solver.py
```

### `/.claude` - Claude Code Skills

Custom skills for Claude Code:

```
.claude/
└── commands/
    ├── train-model.md           # Model training skill
    ├── quality-check.md         # Quality validation skill
    └── generate-report.md       # Report generation skill
```

**Usage in Claude Code**:
- Type `/train-model` to get training guidance
- Type `/quality-check` to run quality validation
- Type `/generate-report` to create comprehensive reports

### `/agents` - AI Agents

Directory for AI agent configurations (reserved for future use).

## Workflow Examples

### 1. Train a New Model

```bash
# Ensure data is prepared
ls data/training/X_raw.npy

# Train improved WGAN v2
cd models/improved_wgan_v2
python train.py --epochs 500

# Results saved to: ./results/improved_wgan_v2_TIMESTAMP/
```

### 2. Validate Model Quality

```bash
cd scripts/reports
python run_quality_check.py \
    --model improved_wgan_v2 \
    --model_dir ../../models/improved_wgan_v2/results/improved_wgan_v2_TIMESTAMP \
    --n_generated 1000
```

### 3. Compare Models

```bash
cd scripts/comparison
python compare_wgan_approaches.py \
    --model1 ../../models/dual_wgan/results/dual_wgan_TIMESTAMP1 \
    --model2 ../../models/improved_wgan_v2/results/improved_wgan_v2_TIMESTAMP2
```

### 4. Generate Training Data

```bash
cd scripts/data_generation
python generate_training_data.py --n_samples 2000 --K 51
```

## Key Files

| File | Purpose |
|------|---------|
| [README.md](README.md) | Project overview |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | This file - project organization |
| [docs/guides/COMPLETE_GUIDE.md](docs/guides/COMPLETE_GUIDE.md) | Comprehensive technical guide |
| [docs/reports/ROADMAP.md](docs/reports/ROADMAP.md) | Project roadmap |
| [eddy_current_workflow/](eddy_current_workflow/) | Main pipeline library |

## Path Updates

After reorganization, import paths should use:

```python
# Model imports
from models.dual_wgan.model import DualHeadGenerator
from models.improved_wgan_v2.model import ConditionalConv1DGenerator

# Workflow imports
from eddy_current_workflow.forward import edc_forward, ProbeSettings
from eddy_current_workflow.inverse import recover_profiles
from eddy_current_workflow.quality import GANQualityChecker

# Data generation imports
from eddy_current_data_generator.core.dataset_builder import build_dataset
```

## Git Ignore

The `.gitignore` file excludes:
- Python cache files (`__pycache__/`, `*.pyc`)
- Training results (`models/*/results/`)
- Data files (`data/training/*.npy`, `data/testing/*.npy`)
- Checkpoints and large model files
- System files (`.DS_Store`)

## Migration Notes

If you have old scripts referencing previous paths:

**Old paths** → **New paths**:
- `./training_data/` → `../../data/training/` (from model dirs) or `./data/training/` (from root)
- `./testing_data/` → `../../data/testing/` or `./data/testing/`
- `./results/` → Depends on context (model results vs general results)
- `from wgan_dual_profiles import` → `from models.dual_wgan.model import`
- `from wgan_improved_v2 import` → `from models.improved_wgan_v2.model import`

## Questions?

For more information, see:
- [Complete Guide](docs/guides/COMPLETE_GUIDE.md) - Technical overview
- [Training Guide](docs/guides/RUNPOD_TRAINING_GUIDE.md) - Remote training
- [Quality Guide](docs/technical/GAN_Quality_Guide.md) - Quality metrics

Or use Claude Code skills:
- `/train-model` - Training guidance
- `/quality-check` - Quality validation
- `/generate-report` - Report generation
