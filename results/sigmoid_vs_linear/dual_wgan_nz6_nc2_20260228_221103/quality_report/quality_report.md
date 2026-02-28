# GAN Quality Validation Report

**Model:** Dual-Head WGAN
**Date:** 2026-02-28 22:29:17
**Overall Status:** FAIL
**Real samples:** 4000
**Generated samples:** 200
**K (layers):** 50
**nz (latent dim):** 6

---

## 1. Moment Matching (Mean & Variance Consistency) — PASS

**Description:** Compares first-order (mean) and second-order (variance) statistics
between real and generated distributions. If E[x] ≈ E[x̂] and Var(x) ≈ Var(x̂),
the generator captures the central tendency and spread of the data.

| Metric | Value |
|--------|-------|
| Mean absolute difference | 0.108801 |
| Mean relative difference | 0.5943 (59.43%) |
| Variance ratio (gen/real) | 1.2119 |
| Variance absolute difference | 0.019725 |
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
| Mean Wasserstein distance | 0.121594 |
| σ dimensions mean | 0.115143 |
| μ dimensions mean | 0.128046 |

### Maximum Mean Discrepancy (MMD)

Kernel-based distance using Gaussian RBF kernel. MMD² = 0 iff distributions are identical.

| Metric | Value |
|--------|-------|
| Total MMD | 0.324038 |
| σ component | 0.118678 |
| μ component | 0.205360 |

![Distribution Distances](plots/distribution_distances.png)

---

## 3. Latent Space Traversal — PASS

**Description:** Tests smoothness and disentanglement by traversing individual
latent dimensions: z(α) = z₀ + α·eᵢ. A well-trained GAN should produce smooth
output changes. Sudden jumps indicate instability; no change indicates inactive dimensions.

| Metric | Value |
|--------|-------|
| Dimensions tested | 6 |
| Active dimensions | 6 |
| Smooth dimensions | 6 |
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
| Samples checked | 200 |
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
| Mean |Z| | 5.577042e+01 |
| Std |Z| | 2.819603e+01 |
| Impedance real range | [-2.908886e+01, 6.305800e+01] |
| Impedance imag range | [-1.271053e+02, 6.273252e+01] |
| Reference |Z| (real data) | 2.741033e+01 |
| Amplitude relative error | 1.0346 (103.46%) |

**Interpretation:**
- All forward responses should be finite (no NaN/Inf)
- Generated impedance amplitude should be in same order of magnitude as real data

---

## 5. Noise Robustness — FAIL

**Description:** Tests generator stability by injecting small perturbations
z' = z + ε where ε ~ N(0, σ²I). A robust generator satisfies:
||G(z') - G(z)|| ≤ C·||ε|| (bounded Lipschitz constant).

| Noise Level (σ) | Mean Δ Output | Max Δ Output |
|-----------------|---------------|--------------|
| 0.010 | 0.020961 | 0.062071 |
| 0.050 | 0.106340 | 0.781679 |
| 0.100 | 0.243233 | 0.823891 |
| 0.200 | 0.450325 | 1.391010 |
| 0.500 | 0.886727 | 3.317765 |

| Metric | Value |
|--------|-------|
| Mean Lipschitz estimate | 2.1662 |
| Robust (Lipschitz < 10) | No |

**Interpretation:**
- Lipschitz constant > 10 → small latent noise causes large output distortion
- Output change should scale approximately linearly with noise level

![Noise Robustness](plots/noise_robustness.png)
