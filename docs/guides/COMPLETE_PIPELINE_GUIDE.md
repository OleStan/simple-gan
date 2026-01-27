# COMPLETE_PIPELINE_GUIDE

End-to-end guide for generating **σ/μ depth profiles** for eddy current testing, training GANs on them, resuming training, comparing approaches, and producing a final report.

---

## 0) Repository structure (relevant parts)

- `eddy_current_data_generator/`
  - `core/material_profiles.py` — profile family functions (linear/exponential/power/sigmoid)
  - `core/dataset_builder.py` — dataset generation orchestration
  - `core/discretization.py` — discretize continuous profiles into `K` layers
- `training_data/`
  - `X_raw.npy` — raw dataset (physical units)
  - `normalization_params.json` — σ/μ normalization bounds and metadata
  - `metadata.json` — generator bounds and profile type info
- Baseline GAN (original approach)
  - `wgan_dual_profiles.py`
  - `train_dual_wgan.py`
- Improved GAN (current recommended)
  - `wgan_improved.py`
  - `train_improved_wgan.py`
  - `train_improved_wgan_resumable.py`
- Evaluation
  - `compare_wgan_approaches.py`
- Report packaging
  - `generate_final_report.py`

---

## 1) Data generation (eddy_current_data_generator)

### 1.1 What is generated
A sample is two 1D profiles sampled over depth:
- `σ(z)` conductivity profile
- `μᵣ(z)` relative permeability profile

They are discretized into **K layers** (default `K=50`).

### 1.2 Profile families
See `eddy_current_data_generator/core/material_profiles.py`.

Common families used:
- linear
- exponential
- power
- sigmoid

These provide variability in gradient, curvature and transition depth.

### 1.3 Output dataset format
Produced as:
- `training_data/X_raw.npy`

Shape:
- `(N, 2*K)`

Layout:
- `X[i, :K]` → σ
- `X[i, K:2*K]` → μ

### 1.4 Normalization parameters
`training_data/normalization_params.json` includes:
- `sigma_min`, `sigma_max`
- `mu_min`, `mu_max`
- `K`

Training uses normalized values in `[-1, 1]`.

---

## 2) Baseline approach: original dual-head WGAN

### 2.1 Files
- `wgan_dual_profiles.py`
- `train_dual_wgan.py`

### 2.2 Model
- MLP generator with shared trunk and σ/μ heads
- MLP critic over concatenated vector

### 2.3 Training
WGAN-GP:
- critic updates `n_critic` times per generator step
- gradient penalty weight `lambda_gp`

Output is stored under `results/dual_wgan_.../`.

---

## 3) Improved approach: Conv1D WGAN + physics-informed loss

### 3.1 Files
- `wgan_improved.py`
- `train_improved_wgan.py`
- `train_improved_wgan_resumable.py`

### 3.2 Architecture

#### Generator: `Conv1DGenerator`
- `z ∈ R^{nz}`
- FC → reshape → ConvTranspose1d upsampling
- Two output heads:
  - σ head: `tanh` → normalized σ
  - μ head: `tanh` → normalized μ
- Concatenates `[σ, μ]` into `(B, 2K)`

#### Critic: `Conv1DCritic`
- Splits input into σ and μ
- Two Conv encoders → concatenate features → final scalar

### 3.3 Physics-informed penalty
`PhysicsInformedLoss` adds:
- smoothness penalty (first-difference L2)
- bounds penalty (soft constraint)

Generator loss:
- `L_G = L_adv + λ_physics * L_physics`

### 3.4 Quality metrics
`ProfileQualityMetrics`:
- smoothness
- monotonicity (strict)
- diversity

---

## 4) Training the improved model

### 4.1 Run training (simple)
```bash
python train_improved_wgan.py
```

### 4.2 Run training (recommended: resumable)
```bash
python train_improved_wgan_resumable.py --epochs 500 --checkpoint_freq 50
```

Artifacts saved into:
- `results/improved_wgan_YYYYMMDD_HHMMSS/`

Includes:
- `models/netG_final.pth`, `models/netC_final.pth`
- `checkpoints/checkpoint_latest.pth`
- `training_history.json`

---

## 5) How to stop and resume training

### Stop
- Press `Ctrl+C` in the terminal.

### Resume
```bash
python train_improved_wgan_resumable.py \
  --resume results/improved_wgan_YYYYMMDD_HHMMSS/checkpoints/checkpoint_latest.pth \
  --output_dir results/improved_wgan_YYYYMMDD_HHMMSS \
  --epochs 500
```

Notes:
- Resume restores optimizer state (important).
- `--output_dir` must point to the same run directory.

---

## 6) Compare baseline vs improved

### Command
```bash
python compare_wgan_approaches.py \
  results/dual_wgan_20260125_222145 \
  results/improved_wgan_20260125_224041
```

### Output
- `results/comparison_analysis/comparison_results.png`
- `results/comparison_analysis/comparison_metrics.json`

---

## 7) Generate a final report package (recommended)

We adapted `generate_final_report.py` into a reusable “report generator” for σ/μ WGAN runs.

### Command
```bash
python generate_final_report.py \
  --run_dir results/improved_wgan_20260125_224041 \
  --real_data training_data/X_raw.npy
```

### Output
Creates:
- `results/improved_wgan_report_YYYYMMDD_HHMMSS/`

Contains:
- plots under `visualizations/`
- copied model artifacts
- generated `.npy` sample arrays
- `SUMMARY.png`

---

## 8) Future improvements

- Condition on profile type / parameterization used in data generator
- Add stronger shape constraints (TV, curvature, monotonic segments)
- Add forward-model-based evaluation (simulate eddy current response)
- Consider diffusion or normalizing flows for improved coverage/stability

### Recommendations
#### Short-term (After Current Training)
1. Run comparison script to quantify improvements
2. Generate larger sample sets (5000+) for statistical significance
3. Visualize specific profile types (linear, exponential, etc.)

#### Medium-term Enhancements
1. Conditional Generation: Add profile type labels
2. Attention Mechanisms: Focus on important depth regions
3. Curriculum Learning: Start with simple profiles, increase complexity
4. Wasserstein Distance Tracking: Monitor convergence quality

#### Long-term Research Directions
1. VAE-GAN Hybrid: Add reconstruction loss for stability
2. Multi-scale Generation: Hierarchical coarse-to-fine
3. Physical Forward Model: Incorporate actual eddy current equations
4. Active Learning: Generate profiles that improve downstream tasks