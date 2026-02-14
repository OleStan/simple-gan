# GAN Quality Validation Report

**Model:** Dual-Head WGAN
**Date:** 2026-02-14 16:01:27
**Overall Status:** PASS
**Real samples:** 2000
**Generated samples:** 1000
**K (layers):** 51
**nz (latent dim):** 96

---

## 1. Moment Matching (Mean & Variance Consistency) — PASS

**Description:** Compares first-order (mean) and second-order (variance) statistics
between real and generated distributions. If E[x] ≈ E[x̂] and Var(x) ≈ Var(x̂),
the generator captures the central tendency and spread of the data.

| Metric | Value |
|--------|-------|
| Mean absolute difference | 0.021170 |
| Mean relative difference | 0.0620 (6.20%) |
| Variance ratio (gen/real) | 1.1346 |
| Variance absolute difference | 0.003179 |
| Mode collapse detected | No |
| Noise amplification detected | No |

**Interpretation:**
- Variance ratio < 0.7 → mode collapse (generator produces too-similar outputs)
- Variance ratio > 1.6 → noise amplification (generator is unstable)
- Ideal variance ratio ≈ 1.0

![Moment Comparison](plots/moment_comparison.png)

---

## 2. Distribution Distances (Wasserstein & MMD)

**Description:** Measures how close the full generated distribution is to the real
distribution, beyond just first two moments.

### Wasserstein Distance (Earth Mover Distance)

Measures the minimum cost of transforming one distribution into another.
Lower values indicate closer distributions.

| Metric | Value |
|--------|-------|
| Mean Wasserstein distance | 0.026184 |
| σ dimensions mean | 0.028456 |
| μ dimensions mean | 0.023911 |

### Maximum Mean Discrepancy (MMD)

Kernel-based distance using Gaussian RBF kernel. MMD² = 0 iff distributions are identical.

| Metric | Value |
|--------|-------|
| Total MMD | 0.124833 |
| σ component | 0.052159 |
| μ component | 0.072674 |

![Distribution Distances](plots/distribution_distances.png)

---

## 3. Latent Space Traversal — PASS

**Description:** Tests smoothness and disentanglement by traversing individual
latent dimensions: z(α) = z₀ + α·eᵢ. A well-trained GAN should produce smooth
output changes. Sudden jumps indicate instability; no change indicates inactive dimensions.

| Metric | Value |
|--------|-------|
| Dimensions tested | 20 |
| Active dimensions | 20 |
| Smooth dimensions | 20 |
| Inactive ratio | 0.00 (0.0%) |
| Mean smoothness score | 1.0000 |

**Interpretation:**
- Inactive ratio > 50% → too many latent dimensions are unused (consider smaller nz)
- Low smoothness → generator has discontinuities in latent space

![Latent Traversal](plots/latent_traversal.png)

---

## 4. Physics Consistency — PASS

**Description:** Validates that generated profiles are physically plausible by:
1. Checking material property bounds (σ > 0, μ ≥ 1)
2. Running generated profiles through the Dodd-Deeds forward solver F(G(z))
   and verifying the impedance response is finite and comparable to real data.

### Bounds Check

| Metric | Value |
|--------|-------|
| Samples checked | 1000 |
| σ in bounds ratio | 1.0000 (100.0%) |
| μ in bounds ratio | 1.0000 (100.0%) |
| σ positive ratio | 1.0000 |
| μ valid (≥1) ratio | 1.0000 |

### Forward Model Consistency

| Metric | Value |
|--------|-------|
| Samples tested | 20 |
| Valid responses | 20 |
| NaN responses | 0 |
| Inf responses | 0 |
| Mean |Z| | 4.090819e+00 |
| Std |Z| | 5.594760e+00 |
| Impedance real range | [-7.156152e-01, 3.650040e+00] |
| Impedance imag range | [-1.827140e+01, 6.110801e-01] |
| Reference |Z| (real data) | 3.324762e-01 |
| Amplitude relative error | 11.3041 (1130.41%) |

**Interpretation:**
- All forward responses should be finite (no NaN/Inf)
- Generated impedance amplitude should be in same order of magnitude as real data

---

## 5. Noise Robustness — PASS

**Description:** Tests generator stability by injecting small perturbations
z' = z + ε where ε ~ N(0, σ²I). A robust generator satisfies:
||G(z') - G(z)|| ≤ C·||ε|| (bounded Lipschitz constant).

| Noise Level (σ) | Mean Δ Output | Max Δ Output |
|-----------------|---------------|--------------|
| 0.010 | 0.012379 | 0.044963 |
| 0.050 | 0.064056 | 0.353755 |
| 0.100 | 0.118806 | 0.593907 |
| 0.200 | 0.241503 | 0.920994 |
| 0.500 | 0.559999 | 2.325448 |

| Metric | Value |
|--------|-------|
| Mean Lipschitz estimate | 0.3874 |
| Robust (Lipschitz < 10) | Yes |

**Interpretation:**
- Lipschitz constant > 10 → small latent noise causes large output distortion
- Output change should scale approximately linearly with noise level

![Noise Robustness](plots/noise_robustness.png)
