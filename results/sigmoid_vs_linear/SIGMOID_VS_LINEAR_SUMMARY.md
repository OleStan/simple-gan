# Sigmoid vs Linear Profile GAN Experiments Summary

This document summarizes the performance of two GAN architectures trained on a dual-class dataset consisting of **Sigmoid** and **Linear** conductivity/permeability profiles.

## Experiment Configuration
- **Dataset**: 4000 samples (2000 Sigmoid, 2000 Linear)
- **Latent Sizes (nz)**: 6, 10, 20, 30
- **Architectures**: 
  - `dual_wgan`: Standard WGAN with two output heads.
  - `improved_wgan_v2`: Enhanced WGAN with spectral normalization and physics-informed loss components.
- **Epochs**: 500

## Key Findings

### 1. Model Comparison (nz=6 case)
| Metric | Dual-Head WGAN | Improved WGAN v2 | Winner |
| :--- | :--- | :--- | :--- |
| **MMD Score** (Lower is better) | 0.324 | **0.116** | Improved v2 |
| **Wasserstein Distance** | 0.121 | **0.075** | Improved v2 |
| **Noise Robustness** (Lipschitz) | 2.16 (Failed) | **1.80 (Passed)** | Improved v2 |
| **Moment Matching** | Passed | Passed | Tie |
| **Physical Consistency** | 100% Valid | 100% Valid | Tie |
| **Class Separation** | Passed | Passed | Tie |

### 2. Architecture Performance
- **Improved WGAN v2** significantly outperforms the standard Dual-Head model in distribution accuracy (MMD) and stability.
- **Physics Loss**: The inclusion of physics-informed constraints in v2 ensures that generated profiles remain strictly within physical bounds and exhibit smoother transitions.
- **Class Conditioning**: Both models successfully learned to distinguish between sigmoid and linear profiles, with clear separation in the generated distributions.

### 3. Latent Space Sensitivity
- **nz=6** proved sufficient for capturing the primary modes of both classes.
- Higher **nz** (20, 30) allows for more variation but increases the risk of capturing noise if not regularized.

## Results Location
All models, training histories, and visualizations are available locally in:
`results/sigmoid_vs_linear/`

### Visualizations available for each run:
- `sample_profiles.png`: Grid of generated σ and μ profiles.
- `distribution_comparison.png`: Real vs Generated histogram comparison.
- `per_class_distributions.png`: Mean profiles for Sigmoid vs Linear classes.
- `quality_report/plots/`: Detailed analysis of latent traversal and noise sensitivity.

## Conclusion
The **Improved WGAN v2** is the recommended model for inverse solver applications due to its superior distributional accuracy and robustness to latent space perturbations.
