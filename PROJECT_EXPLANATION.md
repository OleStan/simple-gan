# GANs for 1D Signal Generation

## Overview

This project implements **Generative Adversarial Networks (GANs)** specifically designed for **1-dimensional signal generation** using PyTorch. Unlike typical GAN applications that work with 2D images, this implementation focuses on generating synthetic 1D time-series or spectral data.

**Original Repository**: [LixiangHan/GANs-for-1D-Signal](https://github.com/LixiangHan/GANs-for-1D-Signal)

---

## Project Purpose

The primary application is generating **synthetic Raman spectra** - chemical fingerprints used in materials science and chemometrics. However, the architecture can be adapted for any 1D signal generation task:

- Time-series data (sensor readings, stock prices)
- Audio waveforms
- Spectroscopic data (IR, UV-Vis, NMR)
- Physiological signals (ECG, EEG)

---

## Implemented GAN Variants

### 1. **DCGAN (Deep Convolutional GAN)**

- **Architecture**: Uses 1D convolutional layers instead of 2D
- **Stability**: Moderate - standard GAN training dynamics
- **Best for**: Quick prototyping and baseline results

**Key characteristics**:
- Binary cross-entropy loss
- Adam optimizer
- BatchNorm for stabilization

### 2. **WGAN (Wasserstein GAN)**

- **Architecture**: Replaces discriminator with a critic
- **Loss function**: Wasserstein distance (Earth Mover's Distance)
- **Stability**: Improved over vanilla GAN - more stable training
- **Best for**: When mode collapse is an issue

**Key characteristics**:
- No sigmoid in critic output
- Weight clipping to enforce Lipschitz constraint
- RMSprop optimizer recommended

### 3. **WGAN-GP (Wasserstein GAN with Gradient Penalty)**

- **Architecture**: WGAN + gradient penalty instead of weight clipping
- **Loss function**: Wasserstein distance + gradient penalty term
- **Stability**: Most stable of the three variants
- **Best for**: Production-quality synthetic data generation

**Key characteristics**:
- Gradient penalty enforces 1-Lipschitz constraint
- No weight clipping (smoother optimization)
- RMSprop used instead of Adam (project-specific finding)

---

## Technical Details

### Signal Specifications

- **Default signal length**: 1824 data points
- **Input format**: Column vectors in `.txt` files
- **Data type**: Continuous 1D arrays (e.g., spectral intensities)

### Network Architecture Considerations

The network dimensions must match your signal length. Key layers to modify:

```python
# Generator: noise_dim → signal_length
# Discriminator: signal_length → 1 (real/fake score)
```

For signals of different lengths, you need to adjust:
- Number of Conv1d/ConvTranspose1d layers
- Kernel sizes and strides
- Output dimensions at each layer

---

## Key Differences from 2D GANs

| Aspect | 2D Image GANs | 1D Signal GANs |
|--------|---------------|----------------|
| **Convolution** | Conv2d | Conv1d |
| **Input shape** | (batch, channels, H, W) | (batch, channels, length) |
| **Spatial structure** | 2D grid with locality | 1D sequence with temporal/spectral ordering |
| **Typical applications** | Face generation, style transfer | Time-series, spectroscopy, audio |
| **Data augmentation** | Rotation, flip, crop | Time-shift, noise injection, scaling |

---

## Training Workflow

### 1. Data Preparation

```
data/
├── sample_001.txt
├── sample_002.txt
└── sample_N.txt
```

Each file contains a single column of signal values.

### 2. Network Configuration

Modify generator and discriminator architectures in:
- `dcgan.py` - for DCGAN
- `wgan.py` - for WGAN
- `wgan_gp.py` - for WGAN-GP

Adjust layers to match your signal length.

### 3. Training

Run the corresponding training script:

```bash
python dcgan_train.py
python wgan_train.py
python wgan_gp_train.py
```

### 4. Monitoring

Training progress is visualized through:
- Loss curves (Generator vs Discriminator/Critic)
- Generated signal samples at intervals
- Comparison with real data distribution

---

## Comparison of Variants

Based on the repository results:

| Metric | DCGAN | WGAN | WGAN-GP |
|--------|-------|------|---------|
| **Training stability** | Moderate | Good | Excellent |
| **Mode collapse risk** | High | Low | Very Low |
| **Sample quality** | Good | Better | Best |
| **Training speed** | Fast | Moderate | Moderate |
| **Hyperparameter sensitivity** | High | Moderate | Low |

**Recommendation**: Start with DCGAN for quick experiments, move to WGAN-GP for production.

---

## Implementation Notes

### Critical Findings from the Project

1. **Optimizer choice matters**: For WGAN-GP, RMSprop worked better than Adam in this specific application (Raman spectra). This differs from the original WGAN-GP paper.

2. **Signal length dependency**: Network architecture is tightly coupled to signal length. You cannot simply load weights trained on length-1824 signals and use them for length-512 signals.

3. **Data format**: All training data must be in the same folder, in `.txt` format, as column vectors.

---

## Use Cases

### When to use this project:

✅ You have limited real data and need synthetic samples  
✅ You want to augment training data for downstream ML tasks  
✅ You need to explore the data distribution space  
✅ You want to generate privacy-preserving synthetic data  

### When NOT to use:

❌ You need exact reproduction of specific signals  
❌ Your signal has complex multi-scale structure (consider WaveGAN or other specialized architectures)  
❌ You need interpretable generation (consider VAEs instead)  

---

## Requirements

```
python 3.7.8
pytorch 1.6.0
numpy 1.19.2
matplotlib 3.3.0
```

**Note**: These are the original requirements. Modern PyTorch versions (1.13+, 2.x) should work with minor adjustments.

---

## Extending the Project

### Potential improvements:

1. **Conditional generation**: Add class labels to generate specific types of signals
2. **Progressive training**: Start with low-resolution signals, gradually increase length
3. **Self-attention**: Add attention mechanisms for long-range dependencies
4. **Evaluation metrics**: Implement FID, IS, or domain-specific metrics
5. **Data augmentation**: Add noise, scaling, or time-warping during training

---

## Comparison with Your MNIST GAN Project

| Aspect | MNIST GAN (Your Project) | 1D Signal GAN (This Project) |
|--------|--------------------------|------------------------------|
| **Framework** | TensorFlow/Keras | PyTorch |
| **Data type** | 2D images (28×28) | 1D signals (1824 points) |
| **Architecture** | Dense layers + Reshape | Conv1d/ConvTranspose1d |
| **Application** | Digit generation | Spectral data generation |
| **Variants** | Basic GAN | DCGAN, WGAN, WGAN-GP |

---

## References

1. Nathan Inkawhich. [DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
2. Yangyangji. [GAN-Tutorial](https://github.com/Yangyangii/GAN-Tutorial)
3. mcclow12. [wgan-gp-pytorch](https://github.com/mcclow12/wgan-gp-pytorch)
4. Original Paper: Goodfellow et al. "Generative Adversarial Networks" (2014)
5. WGAN Paper: Arjovsky et al. "Wasserstein GAN" (2017)
6. WGAN-GP Paper: Gulrajani et al. "Improved Training of Wasserstein GANs" (2017)

---

## Next Steps

1. Clone the repository
2. Prepare your 1D signal data in the required format
3. Modify network architectures for your signal length
4. Start with DCGAN for quick validation
5. Move to WGAN-GP for best results
6. Evaluate synthetic samples against real data distribution
