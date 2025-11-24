# Kaggle Manual Training Workflow

## Problem: Cached File Version
Kaggle is running an OLD cached version of `mappo_k1_implementation.py` that doesn't have the shape-tolerant checkpoint loader. The error at line 1098 shows it's using the old code.

## Solution: Force Fresh File Usage

### Option 1: Upload as Kaggle Dataset (RECOMMENDED)
1. **Create a new Kaggle dataset with your training code:**
   - Go to Kaggle → Datasets → New Dataset
   - Upload `s1/mappo_k1_implementation.py`
   - Name it: `traffic-mappo-training-fixed`
   - Make it public or private

2. **In your Kaggle notebook, add the dataset:**
   - Add Dataset → Search for your `traffic-mappo-training-fixed`
   - It will mount at `/kaggle/input/traffic-mappo-training-fixed/`

3. **Copy the file to working directory:**
   ```python
   !mkdir -p s1
   !cp /kaggle/input/traffic-mappo-training-fixed/mappo_k1_implementation.py s1/
   ```

4. **Run training:**
   ```bash
   !python s1/mappo_k1_implementation.py --resume-checkpoint /kaggle/input/traffic5 --max-hours 2 --device cuda
   ```

### Option 2: Upload via Notebook UI
1. **In Kaggle notebook, click "Add Data" → "File" → Upload `mappo_k1_implementation.py`**
2. **Move to correct location:**
   ```python
   !mkdir -p s1
   !cp /kaggle/input/<uploaded-name>/mappo_k1_implementation.py s1/
   ```
3. **Run training**

### Option 3: Paste Code Directly (Quick Test)
If you just want to test, you can paste the fixed `load_checkpoint()` method directly in a notebook cell before running training.

---

## Manual Checkpoint Workflow

### 1. Start Training
```bash
# Fresh training (no checkpoint)
!python s1/mappo_k1_implementation.py --max-hours 2 --device cuda

# Resume from checkpoint
!python s1/mappo_k1_implementation.py --resume-checkpoint /kaggle/input/traffic5 --max-hours 2 --device cuda
```

### 2. Checkpoints Auto-Save
- **Every 100 episodes:** `mappo_models/checkpoint_<episode>`
- **When time limit hits:** `mappo_models/checkpoint_time_YYYYMMDD_HHMMSS`

### 3. Monitor Progress
```bash
# Check what checkpoints exist
!ls -lh mappo_models/

# View training logs
# (logs will print to notebook cell output)
```

### 4. Manually Download Checkpoints
```bash
# When training finishes or you want to save progress
!zip -r my_checkpoint.zip mappo_models/checkpoint_time_*

# Download the zip file from Kaggle's output panel
# Or if you have Kaggle API configured:
# !kaggle kernels output <your-kernel> -p .
```

### 5. Resume in Next Session
1. Upload your `my_checkpoint.zip` as a new dataset
2. Unzip it: `!unzip /kaggle/input/my-checkpoint-dataset/my_checkpoint.zip -d .`
3. Resume: `!python s1/mappo_k1_implementation.py --resume-checkpoint mappo_models/checkpoint_time_YYYYMMDD_HHMMSS --max-hours 2`

---

## Key Points

✅ **No automation scripts needed** - Training handles everything automatically
✅ **Checkpoints save every 100 episodes** - No manual intervention required
✅ **Time-limited mode** - Gracefully exits and saves when time runs out
✅ **Shape-tolerant loading** - Can resume from old 17-dim/4-action checkpoints
✅ **Manual download** - Simple zip and download when ready

❌ **Don't use:**
- `kaggle_auto_checkpoint.py` (delete this)
- `kaggle_automated_training.ipynb` (delete this)
- Any background monitoring scripts

---

## Verification

To verify you're using the fixed version, check that line ~1098-1124 contains the `_adapt_and_copy` function:

```python
def _adapt_and_copy(source_dict, target_model, strict=False):
    """Adapt state dict shapes if they don't match."""
    # ... function implementation ...
```

If you see `actor.load_state_dict(torch.load(...))` at line 1098, you're using the OLD version.

---

## Expected Output

```
🖥️  DEVICE INFORMATION:
  Device: cuda
  GPU: Tesla T4
  ...

[3/4] Resuming training from checkpoint: /kaggle/input/traffic5
  ✓ Loaded actor 0 (adapted 3 parameters due to shape mismatch)
  ✓ Loaded actor 1 (adapted 3 parameters due to shape mismatch)
  ...
  ✓ Checkpoint loaded successfully from episode 80
  
📊 STARTING TRAINING (Episode 81 → 1000)
...
```

The key difference: **"adapted X parameters"** messages show the shape-tolerant loader is working.
