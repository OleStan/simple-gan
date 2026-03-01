# Improved WGAN v3: Inverse-Solver-Ready (ISR)

## Overview
Improved WGAN v3 is a specialized evolution of the architecture designed specifically for high-fidelity physical profile recovery using gradient-based inverse solvers.

## Key ISR Features

### 1. Differentiable Class Steering
- **Evolution**: Replaced `nn.Embedding` (lookup table) with a **Continuous Class MLP**.
- **Inversion Benefit**: The inverse solver can now optimize class labels as continuous vectors. It can "fade" between different material behaviors (e.g., 70% Sigmoid, 30% Linear) if the real signal demands it.

### 2. Latent Space Smoothness (Jacobian Regularization)
- **Constraint**: Added a penalty on the Frobenius norm of the Generator Jacobian: $\| 
abla_z G(z) \|_2$.
- **Inversion Benefit**: Ensures that small steps in the latent space result in smooth, predictable changes in the physical profiles ($\sigma, \mu$). This prevents the inverse solver from getting stuck in "jagged" local minima.

### 3. Generator EMA (Exponential Moving Average)
- **Technique**: Maintains a shadow copy of the generator weights updated via: $	heta_{ema} = \beta 	heta_{ema} + (1-\beta)	heta_{current}$.
- **Inversion Benefit**: Produces significantly smoother 1D signals by filtering out stochastic training noise. The EMA generator is the one promoted to the Model Registry for production use.

### 4. Convergence Optimization
- **Cosine Annealing**: The learning rate is reduced toward the end of training to "settle" the weights into a high-precision state.
- **Aggressive Physics**: Physics-informed loss becomes the primary driver in the final 10% of training to ensure 100% boundary compliance.

## Directory Structure
```text
models/improved_wgan_v3/
├── model.py            # ISR Architecture (Jacobian, EMA, Continuous Class)
├── train.py            # Optimized training loop with LR Scheduler
└── DOCUMENTATION_V3.md # This file
```

## Integration with Solver
Once trained, the Generator is used as a prior:
$$ \min_{z, c} \| 	ext{DoddDeeds}(G(z, c)) - 	ext{MeasuredSignal} \|^2 + \alpha \|z\|^2 $$
Where $z$ is the latent vector and $c$ is the continuous class steering vector.
