# MDDECT Dataset Integration Guide

## Dataset Overview

**MDDECT** (Multi-Dimensional Defect Eddy Current Testing) is an eddy current testing dataset containing:
- **48,000 samples** total (test set: 4,800 samples available)
- **20 classes**: 18 defect sizes (2.0mm to 0.3mm) + lift-off + normal signal
- **Signal length**: 1250 temporal points
- **2 channels**: Dual-channel sensor data

### Dataset Structure
```
Shape: (N, 8, 2, 5, 20, 1250, 2)
  - N: Number of experiments (3 in test set)
  - 8: Scanning angles
  - 2: Forward/Backward movements
  - 5: Repeats
  - 20: Classes
  - 1250: Temporal points (signal length)
  - 2: Channels
```

### Class Labels
```
0-17: Defect sizes (2.0mm, 1.9mm, ..., 0.4mm, 0.3mm)
18: Lift-off
19: Normal signal
```

---

## Quick Start

### Step 1: Extract Signals from MDDECT

Extract all classes, channel 0:
```bash
python convert_mddect_to_txt.py \
    --input ./training_data/MDDECT_v1_test.npy \
    --output ./data/mddect \
    --channel 0
```

Extract specific classes (e.g., only defects, no lift-off/normal):
```bash
python convert_mddect_to_txt.py \
    --input ./training_data/MDDECT_v1_test.npy \
    --output ./data/mddect_defects \
    --channel 0 \
    --classes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
```

Extract with sampling (limit samples per class):
```bash
python convert_mddect_to_txt.py \
    --input ./training_data/MDDECT_v1_test.npy \
    --output ./data/mddect_sampled \
    --channel 0 \
    --max-per-class 50
```

### Step 2: Train GAN

Use the MDDECT-specific architecture (optimized for 1024-length signals):

**WGAN Training:**
```bash
python wgan_train_mddect.py
```

**DCGAN Training:**
```bash
python dcgan_train_mddect.py
```

---

## Architecture Details

### Signal Length Handling
- **Original MDDECT**: 1250 points
- **Network target**: 1024 points (2^10, optimal for conv layers)
- **Preprocessing**: Automatically truncates 1250 → 1024

### Network Architecture
```
Discriminator: 1024 → 512 → 256 → 128 → 64 → 1
Generator:     1 → 64 → 128 → 256 → 512 → 1024
```

---

## File Structure

### New Files Created

1. **`convert_mddect_to_txt.py`**
   - Extracts signals from .npy to individual .txt files
   - Supports class filtering and sampling
   - Handles both channels

2. **`dcgan_mddect.py`**
   - DCGAN architecture for 1024-length signals
   - Optimized for MDDECT data

3. **`wgan_mddect.py`**
   - WGAN architecture for 1024-length signals
   - Optimized for MDDECT data

4. **`preprocessing_mddect.py`**
   - Dataset loader with automatic 1250→1024 truncation
   - Min-max normalization per sample

5. **`wgan_train_mddect.py`** (to be created)
   - Training script using MDDECT architecture

---

## Advanced Usage

### Extract Specific Scanning Angle

If you want signals from a specific scanning angle (e.g., angle 2):
```python
import numpy as np

data = np.load('./training_data/MDDECT_v1_test.npy')
# Get all data for 1.9mm defect (class 1) at angle 2
signals = data[:, 2, :, :, 1, :, 0]  # shape: (3, 2, 5, 1250)
```

### Use Both Channels

Extract both channels and train separate models:
```bash
# Channel 0
python convert_mddect_to_txt.py --output ./data/mddect_ch0 --channel 0

# Channel 1
python convert_mddect_to_txt.py --output ./data/mddect_ch1 --channel 1
```

### Class-Specific Training

Train GAN on specific defect types:
```bash
# Only large defects (2.0mm - 1.0mm)
python convert_mddect_to_txt.py \
    --output ./data/mddect_large_defects \
    --classes 0 1 2 3 4 5 6 7 8 9 10

# Only small defects (0.9mm - 0.3mm)
python convert_mddect_to_txt.py \
    --output ./data/mddect_small_defects \
    --classes 11 12 13 14 15 16 17
```

---

## Expected Results

After extraction, you should see:
```
data/mddect/
├── mddect_2.0mm_ch0_0000.txt
├── mddect_2.0mm_ch0_0001.txt
├── ...
├── mddect_normal_ch0_0239.txt
└── (Total: 4,800 files for test set)
```

Each file contains 1250 values (one per line).

---

## Training Recommendations

### For Best Results:

1. **Start with all classes** to learn general signal patterns
2. **Use WGAN** for more stable training
3. **Batch size**: 8-16 (depending on GPU memory)
4. **Epochs**: 64-128
5. **Learning rate**: 0.00005 (WGAN) or 0.0002 (DCGAN)

### Data Augmentation Options:

- Train on both channels separately
- Mix different scanning angles
- Focus on specific defect size ranges

---

## Comparison: MDDECT vs Raman Spectra

| Feature | Raman (Previous) | MDDECT (New) |
|---------|------------------|--------------|
| **Signal Type** | Optical spectra | Eddy current |
| **Original Length** | 2185 | 1250 |
| **Network Length** | 2048 | 1024 |
| **Classes** | 1 (single mineral) | 20 (defects + normal) |
| **Samples** | 1599 | 4800 (test set) |
| **Channels** | 1 | 2 |

---

## Troubleshooting

### Issue: "FileNotFoundError: MDDECT_v1_train.npy"
**Solution**: You only have the test set. This is fine for GAN training. The test set has 4,800 samples which is sufficient.

### Issue: "Out of memory"
**Solution**: Reduce batch size in training script or extract fewer samples per class.

### Issue: "Signals look noisy"
**Solution**: MDDECT signals are naturally noisier than Raman spectra. This is expected for eddy current testing data.

---

## Next Steps

1. ✅ Extract MDDECT signals to txt format
2. ✅ Verify data loading with preprocessing_mddect.py
3. ⏳ Train WGAN on MDDECT data
4. ⏳ Generate synthetic eddy current signals
5. ⏳ Compare with real signals and evaluate quality

---

## References

- MDDECT paper: Multi-dimensional defect detection using eddy current testing
- Original dataset: Test set (4,800 samples across 20 classes)
- Signal processing: Truncation to 1024 for optimal conv layer performance
