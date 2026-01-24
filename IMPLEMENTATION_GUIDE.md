# GANs for 1D Signal - Implementation Deep Dive

## Repository Analysis

**Source**: [LixiangHan/GANs-for-1D-Signal](https://github.com/LixiangHan/GANs-for-1D-Signal)

This document provides a detailed analysis of the actual implementation after cloning and examining the codebase.

---

## Project Structure

```
GANs-for-1D-Signal/
├── dcgan.py              # DCGAN architecture (Generator + Discriminator)
├── wgan.py               # WGAN architecture (same as DCGAN but no sigmoid)
├── dcgan_train.py        # DCGAN training loop
├── wgan_train.py         # WGAN training loop
├── wgan_gp_train.py      # WGAN-GP training loop
├── preprocessing.py      # Custom Dataset class for loading 1D signals
├── nets/                 # Pre-trained model weights (.pkl files)
│   ├── dcgan_netG.pkl
│   ├── dcgan_netD.pkl
│   ├── wgan_netG.pkl
│   ├── wgan_netD.pkl
│   ├── wgan_gp_netG.pkl
│   └── wgan_gp_netD.pkl
└── img/                  # Training visualization outputs
```

---

## Architecture Deep Dive

### Signal Dimension Flow

The architecture is designed for **1824-length signals**. Here's the complete dimension transformation:

#### Generator (Noise → Signal)

```python
Input:  (batch_size, nz=100, 1)         # Latent noise vector

Layer 1: ConvTranspose1d(100, 512, kernel=114, stride=1, padding=0)
Output: (batch_size, 512, 114)

Layer 2: ConvTranspose1d(512, 256, kernel=4, stride=2, padding=1)
Output: (batch_size, 256, 228)          # 114 * 2 = 228

Layer 3: ConvTranspose1d(256, 128, kernel=4, stride=2, padding=1)
Output: (batch_size, 128, 456)          # 228 * 2 = 456

Layer 4: ConvTranspose1d(128, 64, kernel=4, stride=2, padding=1)
Output: (batch_size, 64, 912)           # 456 * 2 = 912

Layer 5: ConvTranspose1d(64, 1, kernel=4, stride=2, padding=1)
Output: (batch_size, 1, 1824)           # 912 * 2 = 1824 ✓
```

**Key insight**: The first layer uses a large kernel (114) to bootstrap from 1D noise to a 114-length feature map, then progressively upsamples by 2× four times: 114 → 228 → 456 → 912 → 1824.

#### Discriminator (Signal → Score)

```python
Input:  (batch_size, 1, 1824)           # Real or fake signal

Layer 1: Conv1d(1, 64, kernel=4, stride=2, padding=1)
Output: (batch_size, 64, 912)           # 1824 / 2 = 912

Layer 2: Conv1d(64, 128, kernel=4, stride=2, padding=1)
Output: (batch_size, 128, 456)          # 912 / 2 = 456

Layer 3: Conv1d(128, 256, kernel=4, stride=2, padding=1)
Output: (batch_size, 256, 228)          # 456 / 2 = 228

Layer 4: Conv1d(256, 512, kernel=4, stride=2, padding=1)
Output: (batch_size, 512, 114)          # 228 / 2 = 114

Layer 5: Conv1d(512, 1, kernel=114, stride=1, padding=0)
Output: (batch_size, 1, 1)              # Collapse to single score
```

**Key insight**: Mirror of the generator - downsamples by 2× four times, then collapses the 114-length feature map to a single scalar using kernel=114.

---

## Implementation Differences Between Variants

### 1. DCGAN vs WGAN Architecture

The **only** difference in network architecture:

```python
# DCGAN Discriminator (dcgan.py:37)
nn.Conv1d(512, 1, kernel_size=114, stride=1, padding=0, bias=False),
nn.Sigmoid()  # ← Output in [0, 1]

# WGAN Discriminator (wgan.py:35)
nn.Conv1d(512, 1, kernel_size=114, stride=1, padding=0, bias=False),
# No Sigmoid! ← Output is unbounded (Wasserstein distance)
```

Generator is **identical** in both files.

### 2. Training Loop Differences

#### DCGAN Training (`dcgan_train.py`)

```python
# Loss function
criterion = nn.BCELoss()  # Binary Cross Entropy

# Discriminator training
errD_real = criterion(netD(real), ones)
errD_fake = criterion(netD(fake.detach()), zeros)
errD = errD_real + errD_fake
errD.backward()
optimizerD.step()

# Generator training
errG = criterion(netD(fake), ones)  # Fool discriminator
errG.backward()
optimizerG.step()

# Optimizer
Adam(lr=2e-4, betas=(0.5, 0.999))
```

**Key characteristics**:
- Standard GAN loss with BCE
- Labels: real=1, fake=0
- Generator tries to make discriminator output 1 for fakes
- Single update per batch for both D and G

#### WGAN Training (`wgan_train.py`)

```python
# No criterion! Direct Wasserstein distance

# Discriminator training
loss_D = -torch.mean(netD(real)) + torch.mean(netD(fake))
loss_D.backward()
optimizerD.step()

# Weight clipping (enforce Lipschitz constraint)
for p in netD.parameters():
    p.data.clamp_(-0.01, 0.01)

# Generator training (every n_critic=5 steps)
if step % n_critic == 0:
    loss_G = -torch.mean(netD(fake))
    loss_G.backward()
    optimizerG.step()

# Optimizer
RMSprop(lr=1e-4)
```

**Key characteristics**:
- Wasserstein distance: maximize D(real) - D(fake)
- Weight clipping to [-0.01, 0.01] after each D update
- Train discriminator 5× more than generator (n_critic=5)
- RMSprop instead of Adam

#### WGAN-GP Training (`wgan_gp_train.py`)

```python
# Gradient penalty computation
eps = torch.Tensor(b_size, 1, 1).uniform_(0, 1)
x_p = eps * real + (1 - eps) * fake  # Interpolated samples
grad = autograd.grad(netD(x_p).mean(), x_p, create_graph=True)[0]
grad_norm = torch.norm(grad.view(b_size, -1), 2, 1)
grad_penalty = 10 * torch.pow(grad_norm - 1, 2)  # p_coeff=10

# Discriminator loss with gradient penalty
loss_D = torch.mean(netD(fake) - netD(real)) + grad_penalty.mean()
loss_D.backward()
optimizerD.step()

# Still has weight clipping (unusual - typically not needed with GP)
for p in netD.parameters():
    p.data.clamp_(-0.01, 0.01)

# Generator training (every n_critic=5 steps)
if step % n_critic == 0:
    loss_G = -torch.mean(netD(fake))
    loss_G.backward()
    optimizerG.step()

# Optimizer
RMSprop(lr=1e-4)  # Note: Adam commented out
```

**Key characteristics**:
- Gradient penalty enforces 1-Lipschitz constraint
- Penalty coefficient = 10 (standard)
- Interpolates between real and fake samples
- **Unusual**: Still uses weight clipping (typically removed with GP)
- RMSprop used instead of Adam (lines 43-44 show Adam commented out)

---

## Data Loading Implementation

### Custom Dataset Class (`preprocessing.py`)

```python
class Dataset():
    def __init__(self, root):
        self.root = root
        self.dataset = self.build_dataset()
        self.length = self.dataset.shape[1]
        self.minmax_normalize()
    
    def build_dataset(self):
        dataset = []
        for _file in os.listdir(self.root):
            sample = np.loadtxt(os.path.join(self.root, _file)).T
            dataset.append(sample)
        dataset = np.vstack(dataset).T  # Shape: (signal_length, n_samples)
        return torch.from_numpy(dataset).float()
    
    def minmax_normalize(self):
        for index in range(self.length):
            self.dataset[:, index] = (
                (self.dataset[:, index] - self.dataset[:, index].min()) / 
                (self.dataset[:, index].max() - self.dataset[:, index].min())
            )
    
    def __getitem__(self, idx):
        step = self.dataset[:, idx]
        step = torch.unsqueeze(step, 0)  # Add channel dimension
        target = 0  # Single class (unconditional GAN)
        return step, target
```

**Key points**:
- Loads all `.txt` files from a directory
- Each file contains a column vector (signal values)
- Transposes and stacks into shape `(signal_length, n_samples)`
- **Min-max normalization** to [0, 1] per sample
- Returns shape `(1, signal_length)` for Conv1d compatibility

**Data format requirement**:
```
data/brilliant_blue/
├── sample_001.txt    # Column: 1824 rows × 1 column
├── sample_002.txt
└── sample_N.txt
```

---

## Hyperparameters Comparison

| Parameter | DCGAN | WGAN | WGAN-GP |
|-----------|-------|------|---------|
| **Learning rate** | 2e-4 | 1e-4 | 1e-4 |
| **Optimizer** | Adam | RMSprop | RMSprop |
| **Beta1 (Adam)** | 0.5 | - | 0 (commented) |
| **Batch size** | 8 | 8 | 8 |
| **Epochs** | 32 | 64 | 64 |
| **Noise dim (nz)** | 100 | 100 | 100 |
| **n_critic** | 1 | 5 | 5 |
| **Weight clip** | - | 0.01 | 0.01 |
| **Gradient penalty** | - | - | 10 |

**Observations**:
- WGAN variants train 2× longer (64 vs 32 epochs)
- WGAN variants use lower learning rate (1e-4 vs 2e-4)
- Small batch size (8) due to limited data
- Discriminator trained 5× more in WGAN/WGAN-GP

---

## Weight Initialization

All models use the same initialization strategy:

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # Mean=0, Std=0.02
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)  # Mean=1, Std=0.02
        nn.init.constant_(m.bias.data, 0)
```

Applied via:
```python
netD.apply(weights_init)
netG.apply(weights_init)
```

This is the **DCGAN standard initialization** from the original paper.

---

## Training Visualization

All training scripts save visualizations every epoch:

```python
fixed_noise = torch.randn(16, nz, 1, device=device)  # Fixed for consistency

# After each epoch
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
    f, a = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        for j in range(4):
            a[i][j].plot(fake[i * 4 + j].view(-1))  # Plot 1D signal
            a[i][j].set_xticks(())
            a[i][j].set_yticks(())
    plt.savefig(f'./img/{variant}_epoch_{epoch}.png')
```

**Purpose**: Track generator improvement over time using the same noise vector.

---

## Critical Implementation Details

### 1. Noise Shape

```python
noise = torch.randn(batch_size, nz, 1, device=device)
```

Shape is `(batch, 100, 1)` not `(batch, 100)` because:
- Conv1d expects 3D input: `(batch, channels, length)`
- Here: channels=100 (latent dimensions), length=1 (single point)
- First ConvTranspose1d expands length from 1 → 114

### 2. Signal Range

```python
# Generator output
nn.Tanh()  # Output in [-1, 1]

# But data is normalized to [0, 1]
```

**Mismatch**: Generator outputs [-1, 1] but data is in [0, 1]. This works because:
- Discriminator learns to handle the range
- In practice, tanh outputs will shift to match data distribution
- Could be improved by normalizing data to [-1, 1] instead

### 3. WGAN-GP Gradient Penalty Implementation

```python
eps = torch.Tensor(b_size, 1, 1).uniform_(0, 1)
x_p = eps * data + (1 - eps) * fake
```

**Issue**: `eps` should be on the same device as `data`. Correct version:

```python
eps = torch.rand(b_size, 1, 1, device=device)
x_p = eps * data + (1 - eps) * fake.detach()
```

Also, `fake` should be detached to avoid backprop through generator.

### 4. Redundant Weight Clipping in WGAN-GP

```python
# In wgan_gp_train.py line 69-70
for p in netD.parameters():
    p.data.clamp_(-0.01, 0.01)
```

This is **redundant** when using gradient penalty. The GP already enforces the Lipschitz constraint, so weight clipping is not needed (and can hurt performance).

---

## How to Adapt for Different Signal Lengths

To modify for a signal of length `L`:

### Step 1: Calculate Layer Dimensions

Work backwards from target length:
```
L = 1824 (target)
L/2 = 912
L/4 = 456
L/8 = 228
L/16 = 114  ← This becomes the first ConvTranspose1d output size
```

### Step 2: Modify Generator

```python
class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        first_size = L // 16  # Calculate based on your L
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz, 512, first_size, 1, 0, bias=False),
            # ... rest stays the same
        )
```

### Step 3: Modify Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        final_size = L // 16  # Same calculation
        self.main = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # ... middle layers stay the same
            nn.Conv1d(512, 1, kernel_size=final_size, stride=1, padding=0, bias=False),
        )
```

### Example: For L=512

```
512 / 16 = 32

Generator: ConvTranspose1d(nz, 512, 32, 1, 0)
Discriminator: Conv1d(512, 1, kernel_size=32, ...)
```

---

## Comparison with Your MNIST GAN

| Aspect | MNIST GAN (Yours) | 1D Signal GAN |
|--------|-------------------|---------------|
| **Framework** | TensorFlow/Keras | PyTorch |
| **Architecture** | Dense → Reshape | Conv1d/ConvTranspose1d |
| **Data shape** | (28, 28, 1) | (1, 1824) |
| **Normalization** | [-1, 1] (tanh) | [0, 1] (minmax) |
| **Loss** | Binary crossentropy | BCE / Wasserstein |
| **Batch norm** | After dense layers | After conv layers |
| **Activation** | LeakyReLU | LeakyReLU (D), ReLU (G) |
| **Output** | Tanh | Tanh |
| **Training ratio** | 1:1 (D:G) | 1:1 (DCGAN), 5:1 (WGAN) |

**Key architectural difference**:
- Your MNIST: `Dense(784) → Reshape(28,28,1)` - no spatial structure learned
- 1D Signal: `ConvTranspose1d` chain - learns hierarchical features

---

## Practical Usage Guide

### Training a New Model

```bash
# 1. Prepare data
mkdir -p data/my_signals
# Add .txt files (column vectors, 1824 rows each)

# 2. Choose variant and train
python dcgan_train.py      # Fast, baseline
python wgan_train.py       # More stable
python wgan_gp_train.py    # Best quality

# 3. Monitor training
ls img/  # Check generated samples per epoch
```

### Loading Pre-trained Models

```python
import torch

# Load generator
netG = torch.load('./nets/dcgan_netG.pkl')
netG.eval()

# Generate samples
with torch.no_grad():
    noise = torch.randn(16, 100, 1)
    fake_signals = netG(noise)
    
# fake_signals shape: (16, 1, 1824)
```

### Generating New Samples

```python
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = torch.load('./nets/wgan_gp_netG.pkl', map_location=device)
netG.eval()

n_samples = 10
noise = torch.randn(n_samples, 100, 1, device=device)

with torch.no_grad():
    generated = netG(noise).cpu().numpy()

for i in range(n_samples):
    plt.figure()
    plt.plot(generated[i, 0, :])
    plt.title(f'Generated Signal {i+1}')
    plt.savefig(f'generated_{i}.png')
    plt.close()
```

---

## Potential Improvements

### 1. Fix Data Normalization Mismatch

```python
# In preprocessing.py, change minmax_normalize to:
def normalize_tanh(self):
    for index in range(self.length):
        self.dataset[:, index] = 2 * (
            (self.dataset[:, index] - self.dataset[:, index].min()) / 
            (self.dataset[:, index].max() - self.dataset[:, index].min())
        ) - 1  # Now in [-1, 1] to match tanh output
```

### 2. Remove Weight Clipping from WGAN-GP

```python
# In wgan_gp_train.py, remove lines 69-70
# for p in netD.parameters():
#     p.data.clamp_(-0.01, 0.01)
```

### 3. Fix Gradient Penalty Device Issue

```python
# In wgan_gp_train.py line 59
eps = torch.rand(b_size, 1, 1, device=device)  # Add device
x_p = eps * data + (1 - eps) * fake.detach()   # Add detach
```

### 4. Add Spectral Normalization

For even better stability:

```python
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv1d(1, 64, 4, 2, 1, bias=False)),
            # ... apply to all Conv1d layers
        )
```

### 5. Implement Progressive Growing

For very long signals (>10k points):

```python
# Start with low resolution (e.g., 228)
# Gradually add layers to reach 1824
# Helps training stability and quality
```

### 6. Add Evaluation Metrics

```python
from scipy.stats import wasserstein_distance

def evaluate_quality(real_signals, fake_signals):
    # Compute Wasserstein distance between distributions
    distances = []
    for i in range(real_signals.shape[0]):
        dist = wasserstein_distance(real_signals[i], fake_signals[i])
        distances.append(dist)
    return np.mean(distances)
```

---

## Common Issues and Solutions

### Issue 1: Mode Collapse

**Symptom**: Generator produces same/similar signals regardless of noise

**Solutions**:
- Use WGAN-GP instead of DCGAN
- Increase n_critic (train D more)
- Add noise to discriminator inputs
- Use minibatch discrimination

### Issue 2: Training Instability

**Symptom**: Loss oscillates wildly, no convergence

**Solutions**:
- Lower learning rate (try 5e-5)
- Use RMSprop instead of Adam for WGAN variants
- Increase batch size if possible
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_()`

### Issue 3: Poor Sample Quality

**Symptom**: Generated signals don't look realistic

**Solutions**:
- Train longer (more epochs)
- Ensure data normalization matches generator output range
- Check if signal length matches architecture
- Increase model capacity (more channels)

### Issue 4: Out of Memory

**Symptom**: CUDA out of memory errors

**Solutions**:
- Reduce batch size (try 4 or 2)
- Use gradient accumulation
- Enable mixed precision training
- Use CPU if GPU memory insufficient

---

## Key Takeaways

1. **Architecture is tightly coupled to signal length** - you must recalculate layer dimensions for different lengths

2. **WGAN-GP is most stable** but the implementation has minor issues (redundant clipping, device mismatch)

3. **RMSprop works better than Adam** for WGAN variants in this specific application (Raman spectra)

4. **Small batch size (8)** suggests limited training data - data augmentation could help

5. **Pre-trained models included** - you can immediately generate samples without training

6. **Visualization is built-in** - easy to monitor training progress

7. **Data format is simple** - just column vectors in .txt files, no complex preprocessing

8. **Code is concise** (~100 lines per training script) - easy to understand and modify

---

## Next Steps for Your Learning

1. **Run the pre-trained models** to see generated Raman spectra
2. **Train DCGAN on your own 1D data** (e.g., time-series, audio)
3. **Compare DCGAN vs WGAN vs WGAN-GP** on the same dataset
4. **Modify architecture** for different signal lengths (practice dimension calculations)
5. **Implement the suggested improvements** (fix normalization, remove redundant clipping)
6. **Add conditional generation** (class labels) for multi-class signal generation
7. **Port to TensorFlow** to compare with your MNIST GAN implementation

---

## References

- Original DCGAN Paper: Radford et al. (2015)
- WGAN Paper: Arjovsky et al. (2017)
- WGAN-GP Paper: Gulrajani et al. (2017)
- PyTorch DCGAN Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
