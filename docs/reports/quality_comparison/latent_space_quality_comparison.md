# Latent Space Dimension (nz) Quality Comparison

**Date:** 2026-02-14
**Model:** Dual-Head WGAN (K=51, 500 epochs, lr=1e-4, batch_size=256)
**Training data:** 2000 samples, output_dim=102 (51 σ + 51 μ)

## Overall Results

| nz | Epochs | Overall | Moment Match | Dist. Distance | Latent Trav. | Physics | Noise Robust. |
|----|--------|---------|--------------|----------------|--------------|---------|---------------|
| 6  | 200*   | **PASS** | PASS | INFO | PASS | PASS | PASS |
| 8  | 500    | **PASS** | PASS | INFO | PASS | PASS | PASS |
| 10 | 500    | **PASS** | PASS | INFO | PASS | PASS | PASS |
| 12 | 500    | **PASS** | PASS | INFO | PASS | PASS | PASS |
| 16 | 500    | **FAIL** | FAIL (noise amplification) | INFO | PASS | PASS | PASS |
| 24 | 500    | **PASS** | PASS | INFO | PASS | PASS | PASS |
| 32 | 500    | **PASS** | PASS | INFO | PASS | PASS | PASS |
| 48 | 500    | **PASS** | PASS | INFO | PASS | PASS | PASS |
| 64 | 500    | **FAIL** | FAIL (noise amplification) | INFO | PASS | PASS | PASS |
| 96 | 500    | **PASS** | PASS | INFO | PASS | PASS | PASS |
| 128| 500    | **PASS** | PASS | INFO | PASS | PASS | PASS |

*nz=6 only trained 200/500 epochs (incomplete run)

## Detailed Metrics

### 1. Moment Matching (mean & variance)

| nz | Mean Rel. Diff ↓ | Variance Ratio (ideal=1.0) | Mode Collapse | Noise Amplification |
|----|-------------------|---------------------------|---------------|---------------------|
| 6  | 0.1217 | 1.232 | No | No |
| 8  | 0.0724 | 1.136 | No | No |
| 10 | 0.0775 | 1.119 | No | No |
| 12 | 0.0981 | 1.163 | No | No |
| 16 | 0.0674 | **1.920** | No | **Yes** |
| 24 | 0.0664 | 0.966 | No | No |
| 32 | 0.0594 | 0.900 | No | No |
| 48 | 0.0729 | 1.013 | No | No |
| 64 | **0.3558** | **7.085** | No | **Yes** |
| 96 | 0.0620 | 1.135 | No | No |
| 128| 0.0783 | 0.917 | No | No |

### 2. Distribution Distance

| nz | Wasserstein ↓ | MMD Total ↓ | MMD σ ↓ | MMD μ ↓ |
|----|---------------|-------------|---------|---------|
| 6  | 0.0351 | 0.2498 | 0.0957 | 0.1541 |
| 8  | 0.0306 | 0.2311 | 0.0446 | 0.1865 |
| 10 | 0.0297 | 0.1825 | 0.0619 | 0.1206 |
| 12 | 0.0322 | 0.2006 | 0.0698 | 0.1309 |
| 16 | 0.0277 | 0.1545 | 0.0497 | 0.1048 |
| 24 | 0.0267 | 0.1521 | 0.0568 | 0.0953 |
| 32 | 0.0270 | 0.1517 | 0.0473 | 0.1044 |
| 48 | 0.0304 | 0.1774 | 0.0683 | 0.1091 |
| 64 | **0.1031** | **1.2555** | **0.6734** | **0.5821** |
| 96 | **0.0262** | **0.1248** | **0.0522** | **0.0727** |
| 128| 0.0310 | 0.1982 | 0.0839 | 0.1143 |

### 3. Latent Space Traversal

| nz | Active Dims | Total Tested | Inactive Ratio | Smoothness |
|----|-------------|--------------|----------------|------------|
| 6  | 6  | 6  | 0.0 | 0.833 |
| 8  | 8  | 8  | 0.0 | 1.000 |
| 10 | 10 | 10 | 0.0 | 1.000 |
| 12 | 12 | 12 | 0.0 | 1.000 |
| 16 | 16 | 16 | 0.0 | 1.000 |
| 24 | 20 | 20 | 0.0 | 1.000 |
| 32 | 20 | 20 | 0.0 | 1.000 |
| 48 | 20 | 20 | 0.0 | 1.000 |
| 64 | 20 | 20 | 0.0 | 1.000 |
| 96 | 20 | 20 | 0.0 | 1.000 |
| 128| 20 | 20 | 0.0 | 1.000 |

### 4. Physics Consistency

All models achieved **100% physics validity** (bounds_sigma=1.0, bounds_mu=1.0, forward_valid=20/20).

### 5. Noise Robustness (Lipschitz Constant)

| nz | Mean Lipschitz ↓ |
|----|------------------|
| 6  | **5.996** |
| 8  | 2.913 |
| 10 | 2.215 |
| 12 | 2.149 |
| 16 | 1.647 |
| 24 | 1.186 |
| 32 | 1.173 |
| 48 | 0.570 |
| 64 | 1.223 |
| 96 | **0.387** |
| 128| 0.390 |

## Key Findings

### Best Performers
1. **nz=96** — Best overall: lowest Wasserstein distance (0.0262), lowest MMD (0.1248), good moment matching, low Lipschitz (0.387). Passed all criteria.
2. **nz=32** — Best mean relative difference (0.0594), solid distribution metrics, passed all criteria.
3. **nz=48** — Near-perfect variance ratio (1.013), lowest Lipschitz among mid-range (0.570), passed all criteria.
4. **nz=24** — Excellent variance ratio (0.966), good distribution metrics, passed all criteria.

### Failed Models
- **nz=16** — Noise amplification detected (variance_ratio=1.920). Distribution metrics are decent but moment matching failed.
- **nz=64** — Severe noise amplification (variance_ratio=7.085), very high MMD (1.2555). Worst performer by far.

### Trends
- **Lipschitz constant decreases with nz** (6→128: 5.996→0.390), meaning larger latent spaces produce smoother generators. Exception: nz=64 is an outlier.
- **Distribution quality** generally improves from nz=6 to nz=96, with nz=96 being the sweet spot. nz=128 shows slight degradation.
- **nz=64 is an anomalous failure** — likely a training instability issue rather than a fundamental limitation of the latent dimension. Retraining may resolve this.
- **Very small nz (6-12)** still pass quality checks but have higher MMD scores and Lipschitz constants, indicating less diverse and less smooth generation.
- **nz=6** achieved reasonable results despite only 200/500 epochs, suggesting small latent spaces converge faster but with lower quality ceiling.

### Recommendation
**nz=32 is the recommended latent dimension** for this architecture (K=51, output_dim=102):
- Ratio nz/output_dim = 0.31 — efficient compression
- Passes all quality criteria
- Best mean relative difference (0.0594)
- Good balance of distribution quality and noise robustness
- All 20 tested latent dimensions are active with perfect smoothness

For maximum distribution fidelity, **nz=96** achieves the best raw metrics but at 3x the latent space cost.
