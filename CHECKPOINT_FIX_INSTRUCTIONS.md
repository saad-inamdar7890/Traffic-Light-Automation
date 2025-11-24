# Checkpoint Shape Mismatch Fix - Upload Instructions

## Problem
Your old checkpoint was trained with:
- **State dimension**: 17 (included emergency vehicle flag)
- **Action dimension**: 4 (included emergency override action)

Your new code has:
- **State dimension**: 16 (emergency removed)
- **Action dimension**: 3 (emergency removed)

This causes `RuntimeError: size mismatch` when loading checkpoints.

## Solution Applied
The `load_checkpoint()` method in `s1/mappo_k1_implementation.py` now includes:
1. **Shape-tolerant loading**: Copies overlapping tensor slices
2. **Automatic adaptation**: Handles mismatched dimensions gracefully
3. **Logging**: Shows which parameters were adapted

## ✅ What You Need To Do

### 1. Upload Updated File to Kaggle
Replace your Kaggle version with the local file:

**Option A - Direct Upload:**
1. Go to your Kaggle notebook
2. In the file browser, delete or rename old `s1/mappo_k1_implementation.py`
3. Upload the new version from: `d:\Codes\Projects\Traffic-Light-Automation\s1\mappo_k1_implementation.py`

**Option B - Dataset Method (Recommended):**
1. Create a new dataset version with the updated file:
   - Go to Kaggle Datasets → Create New Dataset
   - Upload `s1/mappo_k1_implementation.py`
   - Name it: "traffic-light-code-phase1-fixed"
2. In your notebook, add this dataset as input
3. Copy files in first cell:
   ```python
   !cp -r /kaggle/input/traffic-light-code-phase1-fixed/* .
   ```

### 2. Verify Upload
Run this in a Kaggle notebook cell:
```python
# Check if the fix is present
import os
with open('s1/mappo_k1_implementation.py', 'r') as f:
    content = f.read()
    if '_adapt_and_copy' in content:
        print("✅ Shape-tolerant loader is present")
    else:
        print("❌ Old version detected - please re-upload")
```

### 3. Resume Training
```bash
python s1/mappo_k1_implementation.py \
  --resume-checkpoint /kaggle/input/traffic5 \
  --max-hours 2 \
  --device cuda
```

## Expected Output
When loading the mismatched checkpoint, you should see:
```
[3/4] Resuming training from checkpoint: /kaggle/input/traffic5
Adapted parameter 'network.0.weight' for actor_0: torch.Size([128, 17]) -> torch.Size([128, 16])
Adapted parameter 'network.4.weight' for actor_0: torch.Size([4, 64]) -> torch.Size([3, 64])
Adapted parameter 'network.4.bias' for actor_0: torch.Size([4]) -> torch.Size([3])
✓ Checkpoint loaded - resuming from episode XX
```

## Alternative: Start Fresh Training
If you prefer not to resume from the old checkpoint:

```bash
# Train from scratch with Phase 1 improvements
python s1/mappo_k1_implementation.py \
  --num-episodes 50 \
  --device cuda
```

This will use the improved reward structure and hyperparameters without legacy weights.

## File Locations
- **Local fixed file**: `d:\Codes\Projects\Traffic-Light-Automation\s1\mappo_k1_implementation.py`
- **Checkpoint manager**: `d:\Codes\Projects\Traffic-Light-Automation\kaggle_auto_checkpoint.py`
- **Notebook template**: `d:\Codes\Projects\Traffic-Light-Automation\kaggle_automated_training.ipynb`

## Quick Verification Commands
```python
# In Kaggle notebook, check file size/timestamp
!ls -lh s1/mappo_k1_implementation.py

# Check for the fix
!grep -n "_adapt_and_copy" s1/mappo_k1_implementation.py

# If found, you'll see line numbers where the function appears
```

---

**Status**: Fix is ready in your local file. Just upload to Kaggle and retry!
