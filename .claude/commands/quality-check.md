# Quality Check Skill

Evaluate GAN model quality using comprehensive metrics and physics consistency checks.

## Purpose

Assess whether generated material profiles are:
1. **Statistically realistic**: Match distribution of real data
2. **Physically plausible**: Obey material property constraints
3. **EDC-consistent**: Produce realistic impedance responses

## Quality Metrics

### 1. Statistical Metrics

- **Mean Squared Error (MSE)**: Distribution similarity
- **Kullback-Leibler Divergence**: Distribution difference
- **Inception Score**: Diversity and quality
- **Frechet Distance**: Feature space similarity

### 2. Physics Metrics

- **Bounds Compliance**: σ ∈ [10^6, 6×10^7] S/m, μ ∈ [1, 100]
- **Smoothness**: Gradient continuity (Σ|∆σ|^2 < threshold)
- **Monotonicity**: Penalty for non-physical reversals
- **EDC Response**: Impedance within realistic range

### 3. Latent Space Analysis

- **Interpolation Smoothness**: Linear interpolation produces realistic profiles
- **Latent Coverage**: Entire latent space maps to valid profiles
- **Dimension Sensitivity**: Each latent dimension controls distinct features

## Usage

### Run Quality Check

```bash
cd scripts/reports
python run_quality_check.py \
    --model improved_wgan_v2 \
    --model_dir ../../models/improved_wgan_v2/results/improved_wgan_v2_TIMESTAMP \
    --n_generated 1000
```

### Output

Quality report saved to `{model_dir}/quality_report/`:
- `quality_metrics.json`: Numerical metrics
- `distribution_comparison.png`: Real vs Generated distributions
- `sample_profiles.png`: Visual inspection of generated profiles
- `edc_response_analysis.png`: EDC impedance analysis
- `latent_analysis.png`: Latent space visualization
- `quality_summary.md`: Comprehensive report

## Target Quality Standards

### Excellent (Production-Ready)
- MSE < 0.01
- Physics violations < 1%
- EDC response deviation < 5%
- Latent interpolation smooth

### Good (Research Quality)
- MSE < 0.05
- Physics violations < 5%
- EDC response deviation < 10%
- Some latent artifacts acceptable

### Needs Improvement
- MSE > 0.1
- Physics violations > 10%
- EDC response unrealistic
- Latent space discontinuities

## Improvement Strategies

If quality is insufficient:

1. **Statistical Issues**:
   - Increase training data diversity
   - Extend training epochs
   - Adjust learning rate

2. **Physics Violations**:
   - Increase physics loss weight (λ_physics)
   - Add hard constraints during generation
   - Use physics-guided initialization

3. **EDC Inconsistencies**:
   - Incorporate EDC loss in training
   - Filter generated samples by EDC response
   - Refine forward model accuracy

4. **Latent Space Problems**:
   - Reduce latent dimension (nz)
   - Add latent regularization
   - Use disentangled latent space
