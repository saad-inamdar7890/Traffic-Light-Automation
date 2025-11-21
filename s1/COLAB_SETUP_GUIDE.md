# Running MAPPO on Google Colab - Complete Guide

## Quick Fix for Your Current Error

The error you're seeing happens because:
1. SUMO can't find the config files (path issue)
2. Using `sumo-gui` on Colab (no display)

**I've fixed the code** - the script now:
- Auto-detects headless environment and uses `sumo` instead of `sumo-gui`
- Makes config paths absolute
- Shows helpful error messages

## Complete Colab Setup (Step-by-Step)

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Clone/Copy Repository
```bash
# Option A: Clone from GitHub
!git clone https://github.com/saad-inamdar7890/Traffic-Light-Automation.git
%cd /content/Traffic-Light-Automation/s1

# Option B: Copy from your Drive (if you uploaded it)
!cp -r /content/drive/MyDrive/Traffic-Light-Automation /content/
%cd /content/Traffic-Light-Automation/s1
```

### Cell 3: Install SUMO
```bash
# Update package lists
!sudo apt-get update -qq

# Install SUMO (headless version)
!sudo apt-get install -y sumo sumo-tools sumo-doc

# Set SUMO_HOME environment variable
import os
os.environ['SUMO_HOME'] = '/usr/share/sumo'

# Verify installation
!sumo --version
```

### Cell 4: Install Python Dependencies
```bash
# Install PyTorch with CUDA support (for GPU acceleration)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!pip install numpy tensorboard traci

# Verify installations
import torch
import numpy as np
import traci
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy: {np.__version__}")
print("All packages installed successfully!")
```

### Cell 5: Verify Files
```python
import os
print("Current directory:", os.getcwd())
print("\nFiles in s1 folder:")
!ls -lh

print("\nChecking critical files:")
print("k1.sumocfg exists:", os.path.exists('k1.sumocfg'))
print("k1.net.xml exists:", os.path.exists('k1.net.xml'))
print("k1_routes_24h.rou.xml exists:", os.path.exists('k1_routes_24h.rou.xml'))
print("mappo_k1_implementation.py exists:", os.path.exists('mappo_k1_implementation.py'))
```

### Cell 6: Configure Model Directory to Save to Drive
```python
# IMPORTANT: Save checkpoints to Drive so they persist across sessions
import os
os.makedirs('/content/drive/MyDrive/MAPPO_Checkpoints', exist_ok=True)

# Set environment variable (optional - can also modify config)
os.environ['MODEL_DIR'] = '/content/drive/MyDrive/MAPPO_Checkpoints'
```

### Cell 7: Start Training (3-hour session)
```bash
# Run training for 3 hours, then auto-save checkpoint
!python mappo_k1_implementation.py --max-hours 3
```

### Cell 8: Resume Training from Checkpoint
```bash
# Find latest checkpoint
!ls -lht /content/drive/MyDrive/MAPPO_Checkpoints/

# Resume from specific checkpoint (replace with your actual checkpoint name)
!python mappo_k1_implementation.py \
  --resume-checkpoint /content/drive/MyDrive/MAPPO_Checkpoints/checkpoint_time_20251121_123000 \
  --max-hours 3
```

## Important Notes

### 1. Working Directory
Always make sure you're in the `s1` folder before running:
```python
import os
os.chdir('/content/Traffic-Light-Automation/s1')
print("Current directory:", os.getcwd())
```

### 2. SUMO Config Files
The config needs these files in the same directory:
- `k1.sumocfg` (main config)
- `k1.net.xml` (network definition)
- `k1_routes_24h.rou.xml` (traffic routes)

### 3. Headless Mode
Colab has no display, so:
- Never use `sumo-gui` (use `sumo` binary)
- The code now auto-detects this
- No visualization during training (use TensorBoard instead)

### 4. Monitor Training with TensorBoard
```python
# Load TensorBoard extension
%load_ext tensorboard

# Start TensorBoard (point to your log directory)
%tensorboard --logdir mappo_logs
```

### 5. Session Management
- **Colab Free**: ~12 hours max, can disconnect anytime
- **Strategy**: Run 2-3 hour sessions with `--max-hours`
- **Always** save checkpoints to Drive
- Resume when disconnected

### 6. GPU Usage
```python
# Check GPU allocation
!nvidia-smi

# PyTorch will use GPU automatically if available
# Note: SUMO simulation is CPU-bound, PyTorch training uses GPU
```

## Troubleshooting

### Error: "Connection closed by SUMO"
**Cause**: Config file path issue or missing files

**Fix**:
```python
import os
os.chdir('/content/Traffic-Light-Automation/s1')
print(os.getcwd())
!ls k1.sumocfg k1.net.xml k1_routes_24h.rou.xml
```

### Error: "sumo-gui: command not found" or display issues
**Fix**: The updated code now auto-detects headless environment. If you still see this, manually edit the config:
```python
# In mappo_k1_implementation.py, line ~373:
# Change: sumo_binary = "sumo-gui"
# To:     sumo_binary = "sumo"
```
(Already fixed in the latest version!)

### Error: "SUMO_HOME not set"
**Fix**:
```python
import os
os.environ['SUMO_HOME'] = '/usr/share/sumo'
```

### Slow Training
- SUMO simulation is CPU-intensive
- Consider reducing `STEPS_PER_EPISODE` for faster episodes:
```bash
!python mappo_k1_implementation.py --max-hours 3 --num-episodes 1000
```

### Out of Memory
- Reduce batch size or network size in config
- Use CPU-only PyTorch if GPU memory is limited:
```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Best Practices for Colab Training

1. **Save Often**: Use `--max-hours 2` or `--max-hours 3` for frequent checkpoints
2. **Use Drive**: Always save to `/content/drive/MyDrive/...` paths
3. **Monitor Progress**: Use TensorBoard in separate cell
4. **Verify Setup**: Run verification cells before starting long training
5. **Keep Notebook Open**: Colab may disconnect if browser tab closes (especially on Free tier)
6. **Use Pro if Available**: Colab Pro gives longer sessions and better GPUs

## Example: Complete Training Workflow

```python
# 1. Setup (run once per session)
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/saad-inamdar7890/Traffic-Light-Automation.git
%cd /content/Traffic-Light-Automation/s1

!sudo apt-get update -qq && sudo apt-get install -y sumo sumo-tools
!pip install torch numpy tensorboard traci

import os
os.environ['SUMO_HOME'] = '/usr/share/sumo'

# 2. Create checkpoint directory
!mkdir -p /content/drive/MyDrive/MAPPO_Checkpoints

# 3. Start training (first time)
!python mappo_k1_implementation.py --max-hours 3

# 4. When session ends, start new session and resume:
!python mappo_k1_implementation.py \
  --resume-checkpoint /content/drive/MyDrive/MAPPO_Checkpoints/checkpoint_time_XXXXXX \
  --max-hours 3

# 5. Repeat step 4 as many times as needed
```

## Summary

✅ **Now Fixed**: Auto-detects headless mode, uses `sumo` instead of `sumo-gui`  
✅ **Checkpointing**: Works seamlessly with Drive  
✅ **Resume**: Full state restoration (models, optimizers, RNG)  
✅ **Time-Limited**: Run in 2-3 hour chunks  
✅ **Monitoring**: TensorBoard integration  

You can now train your MAPPO model on Colab successfully! 🎉
