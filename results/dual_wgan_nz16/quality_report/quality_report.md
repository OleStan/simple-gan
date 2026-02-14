# GAN Quality Validation Report

**Model:** Dual-Head WGAN
**Date:** 2026-02-14 16:00:59
**Overall Status:** FAIL
**Real samples:** 2000
**Generated samples:** 1000
**K (layers):** 51
**nz (latent dim):** 16

---

## 1. Moment Matching (Mean & Variance Consistency) — FAIL

**Description:** Compares first-order (mean) and second-order (variance) statistics
between real and generated distributions. If E[x] ≈ E[x̂] and Var(x) ≈ Var(x̂),
the generator captures the central tendency and spread of the data.

| Metric | Value |
|--------|-------|
| Mean absolute difference | 0.022374 |
| Mean relative difference | 0.0674 (6.74%) |
| Variance ratio (gen/real) | 1.9204 |
| Variance absolute difference | 0.003178 |
| Mode collapse detected | No |
| Noise amplification detected | Yes |

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
| Mean Wasserstein distance | 0.027671 |
| σ dimensions mean | 0.028993 |
| μ dimensions mean | 0.026348 |

### Maximum Mean Discrepancy (MMD)

Kernel-based distance using Gaussian RBF kernel. MMD² = 0 iff distributions are identical.

| Metric | Value |
|--------|-------|
| Total MMD | 0.154518 |
| σ component | 0.049683 |
| μ component | 0.104835 |

![Distribution Distances](plots/distribution_distances.png)

---

## 3. Latent Space Traversal — PASS

**Description:** Tests smoothness and disentanglement by traversing individual
latent dimensions: z(α) = z₀ + α·eᵢ. A well-trained GAN should produce smooth
output changes. Sudden jumps indicate instability; no change indicates inactive dimensions.

| Metric | Value |
|--------|-------|
| Dimensions tested | 16 |
| Active dimensions | 16 |
| Smooth dimensions | 16 |
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
| Mean |Z| | 2.998664e+00 |
| Std |Z| | 2.770407e+00 |
| Impedance real range | [-2.049678e-01, 4.644801e+00] |
| Impedance imag range | [-8.345963e+00, 4.653544e+00] |
| Reference |Z| (real data) | 3.324762e-01 |
| Amplitude relative error | 8.0192 (801.92%) |

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
| 0.010 | 0.019420 | 0.133615 |
| 0.050 | 0.096117 | 0.530217 |
| 0.100 | 0.176618 | 0.835167 |
| 0.200 | 0.344090 | 1.675258 |
| 0.500 | 0.754450 | 2.213912 |

| Metric | Value |
|--------|-------|
| Mean Lipschitz estimate | 1.6470 |
| Robust (Lipschitz < 10) | Yes |

**Interpretation:**
- Lipschitz constant > 10 → small latent noise causes large output distortion
- Output change should scale approximately linearly with noise level

![Noise Robustness](plots/noise_robustness.png)
