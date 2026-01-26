# MDDECT Dataset Integration - Setup Complete ✅

## Overview

Successfully integrated the **MDDECT (Multi-Dimensional Defect Eddy Current Testing)** dataset with your GAN training pipeline. The MDDECT dataset contains eddy current testing signals for defect detection.

---

## 📊 Dataset Information

**MDDECT Test Set:**
- **Total samples**: 4,800 (3 experiments × 8 angles × 2 directions × 5 repeats × 20 classes)
- **Signal length**: 1250 temporal points
- **Channels**: 2 (dual-channel sensor data)
- **Classes**: 20 total
  - 18 defect sizes: 2.0mm, 1.9mm, ..., 0.4mm, 0.3mm
  - 1 lift-off class
  - 1 normal signal class

**Dataset Tensor Shape**: `(3, 8, 2, 5, 20, 1250, 2)`
- Dimension 0: Experiments (3)
- Dimension 1: Scanning angles (8)
- Dimension 2: Forward/Backward movements (2)
- Dimension 3: Repeats (5)
- Dimension 4: Classes (20)
- Dimension 5: Temporal points (1250)
- Dimension 6: Channels (2)

---

## ✅ Files Created

### 1. Data Extraction
- **`convert_mddect_to_txt.py`** - Extracts signals from .npy to individual .txt files
  - Supports class filtering
  - Supports channel selection (0 or 1)
  - Supports sampling (max samples per class)

### 2. Network Architectures
- **`dcgan_mddect.py`** - DCGAN for 1024-length signals
- **`wgan_mddect.py`** - WGAN for 1024-length signals
  - Architecture: 1024 → 512 → 256 → 128 → 64 → 1

### 3. Data Loading
- **`preprocessing_mddect.py`** - Dataset loader with automatic truncation
  - Truncates 1250 → 1024 for optimal conv layer performance
  - Min-max normalization per sample

### 4. Training Script
- **`wgan_train_mddect.py`** - Complete WGAN training script for MDDECT

### 5. Documentation
- **`MDDECT_GUIDE.md`** - Comprehensive usage guide

---

## 🚀 Quick Start Guide

### Step 1: Extract MDDECT Signals

**Extract all classes (all 4,800 samples):**
```bash
python convert_mddect_to_txt.py \
    --input training_data/MDDECT_v1_test.npy \
    --output data/mddect \
    --channel 0
```

**Extract specific classes only:**
```bash
# Only defects (exclude lift-off and normal)
python convert_mddect_to_txt.py \
    --input training_data/MDDECT_v1_test.npy \
    --output data/mddect_defects \
    --channel 0 \
    --classes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
```

**Extract with sampling:**
```bash
# Limit to 50 samples per class
python convert_mddect_to_txt.py \
    --input training_data/MDDECT_v1_test.npy \
    --output data/mddect_sampled \
    --channel 0 \
    --max-per-class 50
```

### Step 2: Train GAN

```bash
python wgan_train_mddect.py
```

Training will:
- Load signals from `data/mddect/`
- Automatically truncate 1250 → 1024
- Train for 64 epochs
- Save models to `nets_mddect/`
- Save training images to `img_mddect/`

---

## 📈 Architecture Details

### Signal Processing
- **Original length**: 1250 points
- **Network length**: 1024 points (2^10)
- **Truncation**: Automatic in preprocessing
- **Normalization**: Min-max per sample to [0, 1]

### Network Architecture
```
Discriminator: 1024 → 512 → 256 → 128 → 64 → 1
Generator:     1 → 64 → 128 → 256 → 512 → 1024
```

### Training Parameters
- **Batch size**: 8
- **Latent dimension (nz)**: 100
- **Learning rate**: 0.00005
- **Optimizer**: RMSprop
- **Epochs**: 64
- **Clip value**: 0.01 (WGAN weight clipping)
- **n_critic**: 5 (discriminator updates per generator update)

---

## 🧪 Verified Test Extraction

Successfully extracted sample data:
```
✅ Dataset shape: (3, 8, 2, 5, 20, 1250, 2)
✅ Extracted 80 signals (4 classes × 20 samples)
✅ Classes: 2.0mm, 1.9mm, lift-off, normal
✅ Signal length: 1250 points
✅ Output: data/mddect_sample/
```

Sample filenames:
- `mddect_2.0mm_ch0_0000.txt`
- `mddect_1.9mm_ch0_0000.txt`
- `mddect_lift-off_ch0_0000.txt`
- `mddect_normal_ch0_0000.txt`

---

## 💡 Usage Examples

### Extract Both Channels Separately

```bash
# Channel 0
python convert_mddect_to_txt.py \
    --output data/mddect_ch0 \
    --channel 0

# Channel 1
python convert_mddect_to_txt.py \
    --output data/mddect_ch1 \
    --channel 1
```

### Train on Specific Defect Sizes

```bash
# Large defects only (2.0mm - 1.0mm)
python convert_mddect_to_txt.py \
    --output data/mddect_large \
    --classes 0 1 2 3 4 5 6 7 8 9 10

# Small defects only (0.9mm - 0.3mm)
python convert_mddect_to_txt.py \
    --output data/mddect_small \
    --classes 11 12 13 14 15 16 17
```

### Load Specific Scanning Angle (Python)

```python
import numpy as np

data = np.load('training_data/MDDECT_v1_test.npy')

# Get all 1.9mm defect signals at scanning angle 2
signals = data[:, 2, :, :, 1, :, 0]  # shape: (3, 2, 5, 1250)

# Flatten to get all samples
signals_flat = signals.reshape(-1, 1250)
print(f"Extracted {signals_flat.shape[0]} signals")
```

---

## 📊 Comparison: Raman vs MDDECT

| Feature | Raman Spectra | MDDECT |
|---------|---------------|--------|
| **Signal Type** | Optical spectra | Eddy current |
| **Original Length** | 2185 | 1250 |
| **Network Length** | 2048 | 1024 |
| **Classes** | 1 (quartz) | 20 (defects + normal) |
| **Training Samples** | 1599 | 4800 |
| **Channels** | 1 | 2 |
| **Application** | Material identification | Defect detection |

---

## 🎯 Next Steps

1. **Extract full dataset** (all 4,800 samples):
   ```bash
   python convert_mddect_to_txt.py \
       --input training_data/MDDECT_v1_test.npy \
       --output data/mddect_full \
       --channel 0
   ```

2. **Train WGAN**:
   ```bash
   python wgan_train_mddect.py
   ```

3. **Generate report** (after training):
   - Modify `generate_final_report.py` to use MDDECT models
   - Or create new report script for MDDECT results

4. **Experiment with**:
   - Different class combinations
   - Both channels (train separate models)
   - Different scanning angles
   - Class-conditional GAN (future enhancement)

---

## 📝 Key Differences from Raman Training

1. **Signal length**: 1024 instead of 2048
2. **Architecture files**: Use `*_mddect.py` versions
3. **Output directories**: `img_mddect/` and `nets_mddect/`
4. **Multi-class data**: 20 classes available (can train on subsets)
5. **Dual channels**: Can train separate models for each channel

---

## ⚠️ Important Notes

- You currently have the **test set only** (4,800 samples)
- This is sufficient for GAN training
- Signals are automatically truncated from 1250 → 1024
- MDDECT signals are noisier than Raman spectra (expected for eddy current data)
- Consider training on defect classes only (exclude lift-off/normal) for better results

---

## 📚 Documentation

For detailed information, see:
- **`MDDECT_GUIDE.md`** - Complete usage guide
- **`convert_mddect_to_txt.py --help`** - Command-line options

---

**Setup Status**: ✅ Complete and tested  
**Ready for training**: ✅ Yes  
**Sample extraction verified**: ✅ 80 signals extracted successfully


 python generate_mddect_report.py \
  --data-root data/mddect_train_1p9mm_angle3_ch0_v2 \
  --img-dir img_mddect_train_1p9mm_angle3_ch0_v2 \
  --nets-dir nets_mddect_train_1p9mm_angle3_ch0_v2 \
  --out-root results_mddect_train_1p9mm_angle3_ch0_v2