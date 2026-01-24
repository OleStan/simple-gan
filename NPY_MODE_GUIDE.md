# NPY Mode Guide - Direct .npy Training

## Overview

You can now train WGANs directly on `.npy` files **without converting to txt first**. This is more efficient and eliminates the intermediate extraction step.

---

## Quick Comparison

| Mode | Data Source | Pros | Cons |
|------|-------------|------|------|
| **txt** | Extracted .txt files | Simple, pre-filtered | Requires extraction step |
| **npy** | Direct .npy file | No extraction needed, flexible filtering | Loads entire .npy into memory |

---

## Usage Examples

### Example 1: Train on 1.9mm defect, 3rd angle, channel 0 (test set)

**Old way (txt mode):**
```bash
# Step 1: Extract
python convert_mddect_to_txt.py \
  --input ./training_data/MDDECT_v1_test.npy \
  --output ./data/mddect_1p9mm_angle3_ch0 \
  --channel 0 \
  --classes 1 \
  --angles 2

# Step 2: Train
python wgan_train_mddect.py \
  --data-root ./data/mddect_1p9mm_angle3_ch0 \
  --img-dir ./img_mddect_1p9mm_angle3_ch0 \
  --nets-dir ./nets_mddect_1p9mm_angle3_ch0
```

**New way (npy mode):**
```bash
# Single step - train directly
python wgan_train_mddect.py \
  --dataset-type npy \
  --data-root ./training_data/MDDECT_v1_test.npy \
  --classes 1 \
  --angles 2 \
  --channel 0 \
  --img-dir ./img_mddect_1p9mm_angle3_ch0_npy \
  --nets-dir ./nets_mddect_1p9mm_angle3_ch0_npy
```

---

### Example 2: Train on all classes, all angles (train set)

```bash
python wgan_train_mddect.py \
  --dataset-type npy \
  --data-root ./training_data/MDDECT_v1_train.npy \
  --channel 0 \
  --img-dir ./img_mddect_train_all \
  --nets-dir ./nets_mddect_train_all \
  --epochs 100
```

---

### Example 3: Train on multiple specific classes

```bash
# Large defects only (2.0mm - 1.0mm = classes 0-10)
python wgan_train_mddect.py \
  --dataset-type npy \
  --data-root ./training_data/MDDECT_v1_train.npy \
  --classes 0 1 2 3 4 5 6 7 8 9 10 \
  --channel 0 \
  --img-dir ./img_mddect_large_defects \
  --nets-dir ./nets_mddect_large_defects
```

---

### Example 4: Train on specific angles only

```bash
# Angles 0, 2, 4 (1st, 3rd, 5th scanning angles)
python wgan_train_mddect.py \
  --dataset-type npy \
  --data-root ./training_data/MDDECT_v1_test.npy \
  --classes 1 \
  --angles 0 2 4 \
  --channel 0 \
  --img-dir ./img_mddect_multi_angle \
  --nets-dir ./nets_mddect_multi_angle
```

---

### Example 5: Train on channel 1 instead of channel 0

```bash
python wgan_train_mddect.py \
  --dataset-type npy \
  --data-root ./training_data/MDDECT_v1_train.npy \
  --classes 1 \
  --angles 2 \
  --channel 1 \
  --img-dir ./img_mddect_ch1 \
  --nets-dir ./nets_mddect_ch1
```

---

## Parameters Reference

### NPY Mode Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset-type` | str | `txt` | `txt` or `npy` |
| `--data-root` | str | required | Path to .npy file (npy mode) or folder (txt mode) |
| `--classes` | int list | all | Class indices 0-19 (npy mode only) |
| `--angles` | int list | all | Angle indices 0-7 (npy mode only) |
| `--channel` | int | 0 | Channel 0 or 1 (npy mode only) |

### Class Index Reference

```
0  = 2.0mm    6  = 1.4mm    12 = 0.8mm    18 = lift-off
1  = 1.9mm    7  = 1.3mm    13 = 0.7mm    19 = normal
2  = 1.8mm    8  = 1.2mm    14 = 0.6mm
3  = 1.7mm    9  = 1.1mm    15 = 0.5mm
4  = 1.6mm    10 = 1.0mm    16 = 0.4mm
5  = 1.5mm    11 = 0.9mm    17 = 0.3mm
```

### Angle Index Reference

```
0 = 1st scanning angle
1 = 2nd scanning angle
2 = 3rd scanning angle
3 = 4th scanning angle
4 = 5th scanning angle
5 = 6th scanning angle
6 = 7th scanning angle
7 = 8th scanning angle
```

---

## Expected Sample Counts

### Test Set (MDDECT_v1_test.npy)
- **Shape**: `(3, 8, 2, 5, 20, 1250, 2)`
- **Total samples**: 4,800
- **Per class**: 240 samples
- **Per class + angle**: 30 samples (3 exp × 2 dir × 5 repeats)

### Train Set (MDDECT_v1_train.npy)
- **Shape**: `(27, 8, 2, 5, 20, 1250, 2)`
- **Total samples**: 43,200
- **Per class**: 2,160 samples
- **Per class + angle**: 270 samples (27 exp × 2 dir × 5 repeats)

---

## Generating Reports (NPY Mode)

After training in npy mode, you still need to provide a `--data-root` folder with txt files for report generation.

**Option 1: Extract txt files for report only**
```bash
# Extract the same subset you trained on
python convert_mddect_to_txt.py \
  --input ./training_data/MDDECT_v1_test.npy \
  --output ./data/mddect_1p9mm_angle3_ch0 \
  --channel 0 \
  --classes 1 \
  --angles 2

# Generate report
python generate_mddect_report.py \
  --data-root data/mddect_1p9mm_angle3_ch0 \
  --img-dir img_mddect_1p9mm_angle3_ch0_npy \
  --nets-dir nets_mddect_1p9mm_angle3_ch0_npy \
  --out-root results_mddect_npy
```

**Option 2: Use existing txt folder (if you already extracted it)**
```bash
python generate_mddect_report.py \
  --data-root data/mddect_1p9mm_angle3_ch0 \
  --img-dir img_mddect_1p9mm_angle3_ch0_npy \
  --nets-dir nets_mddect_1p9mm_angle3_ch0_npy \
  --out-root results_mddect_npy
```

---

## When to Use NPY Mode vs TXT Mode

### Use NPY Mode When:
- ✅ You want to quickly experiment with different class/angle combinations
- ✅ You don't want to manage multiple extracted data folders
- ✅ You have enough RAM to load the .npy file
- ✅ You want a single-command workflow

### Use TXT Mode When:
- ✅ You want to train on the same subset multiple times (faster loading)
- ✅ You have limited RAM
- ✅ You want to inspect/modify individual signal files
- ✅ You're sharing datasets with others (txt files are human-readable)

---

## Performance Notes

- **NPY mode** loads the entire .npy file into memory, then filters
- **TXT mode** only loads the pre-extracted signals
- For large datasets, txt mode may be faster after the initial extraction
- For experimentation, npy mode is more convenient

---

## Troubleshooting

### "FileNotFoundError: [Errno 2] No such file or directory"
- Check that `--data-root` points to the correct .npy file path
- Use absolute paths if relative paths don't work

### "IndexError: index X is out of bounds"
- Check that class indices are in range 0-19
- Check that angle indices are in range 0-7

### "Dataset loaded: 0 samples"
- Check that your class/angle filters aren't too restrictive
- Verify the .npy file has the expected shape

---

**Created**: 2026-01-18  
**Compatible with**: `wgan_train_mddect.py`, `preprocessing_mddect.py`
