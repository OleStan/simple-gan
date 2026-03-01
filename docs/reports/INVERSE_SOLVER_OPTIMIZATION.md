# Inverse Solver Optimization (Phase 3)

## Goal
To transform the GAN from a "Data Generator" into a "Differentiable Physical Prior" suitable for high-accuracy inverse solving.

## Key Optimizations for Inversion

### 1. Latent Space Smoothness (Jacobian Regularization)
For the inverse solver to work, the function $f(z) = 	ext{Profile}$ must be "well-behaved." 
- **Implementation**: Add a penalty term: $\| 
abla_z G(z) \|_2$.
- **Benefit**: Removes "sharp cliffs" in the latent space. The inverse solver's gradient descent will encounter fewer local minima.

### 2. Weight Smoothing (EMA)
- **Implementation**: Maintain an Exponential Moving Average of Generator weights.
- **Benefit**: Eliminates high-frequency noise in the generated $\sigma$ and $\mu$ profiles. This leads to a smoother loss surface when comparing the GAN output to the Dodd-Deeds forward model.

### 3. Latent Manifold Coverage
- **Problem**: Standard GANs can have "dead zones" in the latent space where the output is non-physical.
- **Optimization**: Use **Latent Space Normalization** and aggressive **Physics Informed Loss**.
- **Benefit**: Ensures that *any* $z$ the inverse solver picks during optimization results in a physically valid material profile.

### 4. Differentiable Class Steering
- **Optimization**: Use continuous embeddings for the class labels.
- **Benefit**: The inverse solver can treat the class ID as a continuous variable, allowing it to "fade" between different material behaviors (e.g., a "mostly sigmoid" profile with some linear characteristics).

## Integration with Inverse Solver
The optimized Generator will be frozen and used as the "Prior" in the following objective function:
$$ \min_{z, c} \| 	ext{DoddDeeds}(G(z, c)) - 	ext{MeasuredSignal} \|^2 + \lambda \|z\|^2 $$
Where $z$ is the latent vector and $c$ is the class steering vector.
