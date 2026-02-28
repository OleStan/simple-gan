# GAN Quality Validation Report

**Model:** Improved WGAN v2
**Date:** 2026-02-28 20:02:52
**Overall Status:** FAIL
**Real samples:** 2000
**Generated samples:** 500
**K (layers):** 51
**nz (latent dim):** 12

---

## 1. Moment Matching (Mean & Variance Consistency) — FAIL

**Description:** Compares first-order (mean) and second-order (variance) statistics
between real and generated distributions. If E[x] ≈ E[x̂] and Var(x) ≈ Var(x̂),
the generator captures the central tendency and spread of the data.

| Metric | Value |
|--------|-------|
| Mean absolute difference | 0.021576 |
| Mean relative difference | 0.0580 (5.80%) |
| Variance ratio (gen/real) | 0.6026 |
| Variance absolute difference | 0.003476 |
| Mode collapse detected | Yes |
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
| Mean Wasserstein distance | 0.033720 |
| σ dimensions mean | 0.038154 |
| μ dimensions mean | 0.029286 |

### Maximum Mean Discrepancy (MMD)

Kernel-based distance using Gaussian RBF kernel. MMD² = 0 iff distributions are identical.

| Metric | Value |
|--------|-------|
| Total MMD | 0.190386 |
| σ component | 0.070221 |
| μ component | 0.120165 |

![Distribution Distances](plots/distribution_distances.png)

---

## 3. Latent Space Traversal — PASS

**Description:** Tests smoothness and disentanglement by traversing individual
latent dimensions: z(α) = z₀ + α·eᵢ. A well-trained GAN should produce smooth
output changes. Sudden jumps indicate instability; no change indicates inactive dimensions.

| Metric | Value |
|--------|-------|
| Dimensions tested | 12 |
| Active dimensions | 12 |
| Smooth dimensions | 12 |
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
| Samples checked | 500 |
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
| Mean |Z| | 2.009273e+00 |
| Std |Z| | 2.578176e-01 |
| Impedance real range | [-5.387237e-01, 2.646145e-01] |
| Impedance imag range | [1.489366e+00, 2.536954e+00] |
| Reference |Z| (real data) | 3.324762e-01 |
| Amplitude relative error | 5.0434 (504.34%) |

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
| 0.010 | 0.011196 | 0.037414 |
| 0.050 | 0.051888 | 0.173332 |
| 0.100 | 0.103420 | 0.300474 |
| 0.200 | 0.204097 | 0.657294 |
| 0.500 | 0.488657 | 2.030936 |

| Metric | Value |
|--------|-------|
| Mean Lipschitz estimate | 0.7023 |
| Robust (Lipschitz < 10) | Yes |

**Interpretation:**
- Lipschitz constant > 10 → small latent noise causes large output distortion
- Output change should scale approximately linearly with noise level

![Noise Robustness](plots/noise_robustness.png)
