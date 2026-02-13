# GAN Quality Evaluation Guide

## Overview
Generative Adversarial Networks (GANs) consist of two neural networks trained together:
- **Generator (G):** Produces synthetic data (e.g., images, text, audio).
- **Discriminator (D):** Evaluates whether data is real or generated.

GAN *quality* refers to how well the generator produces outputs that are:
- Realistic
- Diverse
- Useful for downstream tasks
- Stable across training

This document explains how GAN quality should work, how it is measured, and what good vs. poor quality looks like.

---

## What “Good GAN Quality” Means

High-quality GANs should satisfy four core properties:

### 1. Visual / Output Realism
Generated samples should be indistinguishable from real data by humans or strong classifiers.

Indicators:
- Sharp details
- Correct structure and proportions
- No obvious artifacts (noise, distortions, broken geometry)

### 2. Diversity (No Mode Collapse)
The generator should produce **varied outputs**, not the same few samples repeatedly.

Indicators:
- Wide coverage of the real data distribution
- Different styles, poses, colors, or structures
- No repetition patterns

### 3. Distribution Alignment
The generated data distribution should closely match the real data distribution.

Indicators:
- Similar feature statistics
- Balanced class representation (if conditional)
- No missing modes

### 4. Training Stability
Training should converge smoothly without oscillation or collapse.

Indicators:
- Gradual improvement in sample quality
- No sudden drops in generator or discriminator performance
- Loss curves that remain bounded

---

## Common GAN Quality Metrics

### Quantitative Metrics

#### Fréchet Inception Distance (FID)
Measures distance between real and generated feature distributions.

- Lower is better
- Sensitive to both quality and diversity
- Most commonly used metric

#### Inception Score (IS)
Evaluates confidence and diversity of generated samples.

- Higher is better
- Does not compare directly to real data
- Can be misleading if over-optimized

#### Precision & Recall for GANs
Separates:
- **Precision:** Sample realism
- **Recall:** Sample diversity

Useful for diagnosing mode collapse.

#### Kernel Inception Distance (KID)
Alternative to FID with unbiased estimation.

---

### Qualitative Evaluation

Human evaluation is still critical:
- Visual inspection
- A/B comparisons
- Domain-expert review

Questions to ask:
- Do samples look real?
- Are important features preserved?
- Are rare cases represented?

---

## Failure Modes That Reduce Quality

### Mode Collapse
Generator produces very similar outputs.

Symptoms:
- High visual quality but low diversity
- Good discriminator loss but poor recall

### Overfitting
Generator memorizes training data.

Symptoms:
- Near-duplicate samples
- Very low FID but poor generalization

### Discriminator Dominance
Discriminator learns too quickly.

Symptoms:
- Generator loss explodes
- Samples remain noisy or random

### Training Instability
GAN oscillates or collapses.

Symptoms:
- Sudden quality drops
- Rapid loss fluctuations


---

## Acceptance Criteria for “High-Quality” GAN Output

A GAN can be considered high quality when:
- FID consistently improves and stabilizes
- Samples are visually convincing to humans
- No obvious repetition or collapse
- Performance holds across multiple random seeds

---

---

# Additional GAN Quality Validation Criteria

## 2.1 Comparison of First- and Second-Order Statistics

For real samples $begin:math:text$ x \\sim p\_\{data\}\(x\) $end:math:text$ and generated samples $begin:math:text$ \\hat\{x\} \= G\(z\) \\sim p\_G\(x\) $end:math:text$:

### Mean Consistency

$begin:math:display$
\\mathbb\{E\}\[x\] \\approx \\mathbb\{E\}\[\\hat\{x\}\]
$end:math:display$

Empirical estimate:

$begin:math:display$
\\frac\{1\}\{N\}\\sum\_\{i\=1\}\^\{N\} x\_i \\\;\\approx\\\; \\frac\{1\}\{M\}\\sum\_\{j\=1\}\^\{M\} \\hat\{x\}\_j
$end:math:display$

This ensures no systematic bias shift in generated outputs.

---

### Variance Consistency

$begin:math:display$
\\mathrm\{Var\}\(x\) \\approx \\mathrm\{Var\}\(\\hat\{x\}\)
$end:math:display$

Empirical covariance comparison:

$begin:math:display$
\\Sigma\_\{real\} \\approx \\Sigma\_\{gen\}
$end:math:display$

If variance is significantly smaller → **mode collapse**.  
If significantly larger → **noise amplification / instability**.

---

## 2.2 Distance Between Distributions

Beyond first two moments, full distribution similarity must be evaluated.

### Wasserstein Distance (Earth Mover Distance)

For 1-D signals:

$begin:math:display$
W\(p\_\{data\}\, p\_G\) \= \\inf\_\{\\gamma \\in \\Pi\(p\_\{data\}\, p\_G\)\} 
\\mathbb\{E\}\_\{\(x\,y\)\\sim \\gamma\}\[\\\|x \- y\\\|\]
$end:math:display$

Where $begin:math:text$ \\Pi\(p\_\{data\}\, p\_G\) $end:math:text$ is the set of all joint distributions with marginals $begin:math:text$ p\_\{data\} $end:math:text$ and $begin:math:text$ p\_G $end:math:text$.

Lower $begin:math:text$ W $end:math:text$ → closer distributions.

---

### Maximum Mean Discrepancy (MMD)

$begin:math:display$
\\mathrm\{MMD\}\^2\(p\_\{data\}\, p\_G\)
\=
\\mathbb\{E\}\_\{x\,x\'\}\[k\(x\,x\'\)\]
\+
\\mathbb\{E\}\_\{\\hat\{x\}\,\\hat\{x\}\'\}\[k\(\\hat\{x\}\,\\hat\{x\}\'\)\]
\-
2\\mathbb\{E\}\_\{x\,\\hat\{x\}\}\[k\(x\,\\hat\{x\}\)\]
$end:math:display$

Where $begin:math:text$ k\(\\cdot\,\\cdot\) $end:math:text$ is a kernel (e.g., Gaussian RBF).

Lower MMD → better distribution alignment.

---

## 3.2 Latent Space Traversal

To test smoothness and disentanglement:

$begin:math:display$
z\(\\alpha\) \= z\_0 \+ \\alpha e\_i
$end:math:display$

Where:
- $begin:math:text$ z\_0 $end:math:text$ — base latent vector  
- $begin:math:text$ e\_i $end:math:text$ — unit vector in dimension $begin:math:text$ i $end:math:text$  
- $begin:math:text$ \\alpha $end:math:text$ — traversal scalar  

Generated output:

$begin:math:display$
x\(\\alpha\) \= G\(z\(\\alpha\)\)
$end:math:display$

### Expected Behavior

✔ Signal changes **smoothly** with respect to $begin:math:text$ \\alpha $end:math:text$

✘ Sudden jumps or discontinuities → poorly trained GAN  
✘ No visible change → inactive latent dimension  

Smoothness can be evaluated via:

$begin:math:display$
\\left\\\| \\frac\{\\partial G\(z\)\}\{\\partial z\_i\} \\right\\\|
$end:math:display$

Large spikes indicate instability.

---

## 4.2 Consistency with Forward (Physical) Model

If a known forward model $begin:math:text$ F $end:math:text$ exists (e.g., physics-based simulator):

$begin:math:display$
F\(G\(z\)\) \\approx y\_\{sim\}
$end:math:display$

Where:
- $begin:math:text$ G\(z\) $end:math:text$ — generated signal
- $begin:math:text$ F $end:math:text$ — physical model
- $begin:math:text$ y\_\{sim\} $end:math:text$ — expected simulation output

Consistency error:

$begin:math:display$
\\mathcal\{L\}\_\{phys\} \= 
\\\| F\(G\(z\)\) \- y\_\{sim\} \\\|\^2
$end:math:display$

Low physical loss → physically plausible generations.

---

## 5. Robustness to Noise and Extrapolation

Test generator stability by injecting small perturbations:

$begin:math:display$
z\' \= z \+ \\epsilon\, \\quad \\epsilon \\sim \\mathcal\{N\}\(0\, \\sigma\^2 I\)
$end:math:display$

Stability criterion:

$begin:math:display$
\\\| G\(z\'\) \- G\(z\) \\\| \\le C \\\|\\epsilon\\\|
$end:math:display$

If small latent noise produces large output distortion → poor generalization.

---

## Summary of Extended GAN Quality Criteria

A high-quality GAN should satisfy:

1. Moment matching (mean & variance)
2. Low Wasserstein / MMD distance
3. Smooth latent traversals
4. Physical consistency (if model available)
5. Robustness to latent noise

GAN quality must be evaluated using **both statistical and structural validation tests**, not visual inspection alone.
---

## Summary

GAN quality is **not a single number** but a balance of realism, diversity, and stability.
The best evaluations combine:
- Quantitative metrics
- Qualitative inspection
- Training behavior analysis

Strong GANs look real, stay diverse, and improve predictably over time.
