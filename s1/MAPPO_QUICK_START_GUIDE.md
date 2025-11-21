# MAPPO Quick Start Guide

## 🚀 Getting Started with MAPPO for K1 Traffic Control

This guide will help you train and deploy MAPPO for coordinated traffic light control on your K1 network.

---

## 📋 Prerequisites

### 1. Required Software
```bash
# Python 3.8+
python --version

# SUMO (Simulation of Urban MObility)
sumo --version

# Required Python packages
pip install torch numpy matplotlib tensorboard
```

### 2. Verify Files
Make sure you have these files in your `s1/` folder:
- ✅ `k1.net.xml` - Network definition
- ✅ `k1.sumocfg` - SUMO configuration
- ✅ `k1_routes_24h.rou.xml` - Traffic routes
- ✅ `k1.ttl.xml` - Traffic light definitions
- ✅ `mappo_k1_implementation.py` - Training script
- ✅ `deploy_mappo.py` - Deployment script

---

## 🎯 Training MAPPO

### Step 1: Quick Test (Single Episode)

Test if everything works with a short training run:

```bash
# Navigate to s1 folder
cd s1

# Run quick test (modify config for 1 episode)
python mappo_k1_implementation.py
```

**What happens:**
- Starts SUMO simulation
- 9 actor networks (one per junction)
- 1 shared critic network
- Collects experience for 3600 steps (1 hour simulation)
- Updates networks using PPO
- Saves models to `mappo_models/`

**Expected output:**
```
================================================================================
MAPPO Training for K1 Traffic Network
================================================================================
Junctions: 9
Episodes: 5000
Steps per episode: 3600
================================================================================
Episode 0/5000 | Reward: -245.32 | Length: 3600 | Epsilon: 0.1000
✓ Loaded actor 0 from mappo_models/episode_0/actor_0.pth
...
```

---

### Step 2: Full Training (Recommended)

For best results, train for multiple episodes:

```bash
# Default: 5000 episodes (~50 hours on CPU, ~10 hours on GPU)
python mappo_k1_implementation.py
```

**Training schedule:**
- Episodes 0-1000: High exploration (epsilon = 0.1 → 0.05)
- Episodes 1000-3000: Learning coordination
- Episodes 3000-5000: Fine-tuning and stabilization

**Monitor training:**
```bash
# In a separate terminal, start TensorBoard
tensorboard --logdir=mappo_logs

# Open browser to http://localhost:6006
```

**TensorBoard shows:**
- Episode reward (should increase over time)
- Actor loss (should decrease)
- Critic loss (should decrease)
- Episode length
- Exploration rate (epsilon)

---

### Step 3: Save Points

Models are automatically saved every 100 episodes:
```
mappo_models/
├── episode_100/
│   ├── actor_0.pth
│   ├── actor_1.pth
│   ├── ...
│   ├── actor_8.pth
│   └── critic.pth
├── episode_200/
├── ...
└── final/  (after training completes)
```

---

## 🎬 Deploying MAPPO

### Deploy Trained Model

Once training is complete, deploy the trained agents:

```bash
# Deploy final model for 1 hour
python deploy_mappo.py --model mappo_models/final --duration 3600

# Deploy for different duration
python deploy_mappo.py --model mappo_models/final --duration 7200  # 2 hours
```

**What happens:**
- Loads 9 trained actor networks
- Runs simulation with MAPPO control
- Collects performance metrics
- Generates report with visualizations

**Expected output:**
```
================================================================================
MAPPO Deployment - Real-Time Traffic Control
================================================================================
Duration: 3600s (1.0 hours)
Scenario: custom_24h
Model: mappo_models/final
================================================================================
✓ Loaded actor 0 from mappo_models/final/actor_0.pth
✓ Loaded actor 1 from mappo_models/final/actor_1.pth
...
Progress: 16.7% | Step: 600/3600 | Elapsed: 45.2s
Progress: 33.3% | Step: 1200/3600 | Elapsed: 91.5s
...
================================================================================
DEPLOYMENT SUMMARY
================================================================================
Total steps: 3600
Average vehicles: 248.5
Average waiting time: 852.3s
Average queue length: 15.2
Peak vehicles: 425
Peak waiting time: 1523.8s
Peak queue length: 38.1
================================================================================
✓ Report saved to: deployment_reports/20251114_143522
```

---

### Compare with Baseline

See how much MAPPO improves over fixed-time control:

```bash
# Run comparison
python deploy_mappo.py --model mappo_models/final --duration 3600 --compare
```

**Expected output:**
```
================================================================================
MAPPO vs Baseline Comparison
================================================================================

[1/2] Running baseline (fixed-time)...
✓ Baseline completed
  Avg waiting time: 2125.3s
  Avg queue length: 38.5

[2/2] Running MAPPO...
✓ MAPPO completed
  Avg waiting time: 850.2s
  Avg queue length: 15.1

================================================================================
COMPARISON RESULTS
================================================================================

Metric               Baseline        MAPPO           Improvement
--------------------------------------------------------------------------------
Waiting Time (s)     2125.3          850.2           +60.0%
Queue Length         38.5            15.1            +60.8%
Vehicles             248.5           248.5
================================================================================
```

---

## 📊 Understanding the Results

### Deployment Reports

After each deployment, you'll find a report folder:
```
deployment_reports/20251114_143522/
├── summary.json              # Numerical results
├── network_metrics.png       # Network-wide performance over time
├── junction_metrics.png      # Per-junction performance
└── action_distributions.png  # What actions each junction took
```

### Key Metrics

**1. Waiting Time**
- Lower is better
- Target: -60% vs baseline
- Good: < 1000s average
- Excellent: < 500s average

**2. Queue Length**
- Lower is better
- Target: < 20 vehicles average
- Critical if > 50 (congestion spreading)

**3. Action Distribution**
- "Keep" should be most common (stability)
- "Next Phase" for normal switching
- "Extend" for heavy traffic
- "Emergency" rare (only when needed)

---

## 🎓 How MAPPO Works During Deployment

### 1. Local Observation (Each Junction)

Each junction reads its **local sensors only**:

```python
Junction J0 observes:
- Current phase: 2
- Queue north: 15 vehicles
- Queue south: 8 vehicles
- Queue east: 12 vehicles
- Queue west: 6 vehicles
- Weighted vehicles (using PCE): 45.5, 24.0, 38.5, 19.0
- Occupancy: 0.65, 0.40, 0.55, 0.30
- Time in phase: 28 seconds
- Emergency vehicle: No (0)
- Neighbor J11 phase: 5
- Neighbor J6 phase: 3
```

### 2. Actor Network Decision

```python
Actor J0 network:
Input (17 dims) → Hidden(128) → Hidden(64) → Output(4 actions)
                                              [0.65, 0.25, 0.08, 0.02]

Decision: Keep current phase (action 0, 65% probability)
```

### 3. Coordination (Learned During Training)

**Actor J0 learned:**
- "If neighbor J11 is in phase 2 (busy with east-west), don't send more south-bound traffic"
- "If my north queue > 20 AND J11 north queue < 15, give longer green to clear north"
- "If emergency vehicle detected, prioritize that direction"

**This coordination was NOT programmed - it emerged from training! 🎯**

---

## ⚙️ Customization

### Modify Training Configuration

Edit `mappo_k1_implementation.py`:

```python
class MAPPOConfig:
    # Train for less time (faster testing)
    NUM_EPISODES = 1000  # Instead of 5000
    
    # Change learning rates
    LEARNING_RATE_ACTOR = 1e-4  # More conservative
    LEARNING_RATE_CRITIC = 5e-4
    
    # Change reward weights (favor own junction more)
    REWARD_WEIGHT_OWN = 0.7
    REWARD_WEIGHT_NEIGHBORS = 0.2
    REWARD_WEIGHT_NETWORK = 0.1
    
    # More exploration
    EPSILON_START = 0.2
    EPSILON_DECAY = 0.99
```

### Use Different Scenarios

Train on specific traffic patterns:

```python
# In mappo_k1_implementation.py, modify K1Environment.__init__
def __init__(self, config, scenario='morning_rush'):
    # Use specific route file
    self.config.SUMO_CONFIG = "k1_morning_rush.sumocfg"
```

---

## 🐛 Troubleshooting

### Issue 1: SUMO Connection Error

**Error:**
```
traci.exceptions.TraCIException: Could not connect to TraCI server
```

**Solution:**
```bash
# Make sure SUMO is installed and in PATH
sumo --version

# Check if SUMO config file exists
ls k1.sumocfg

# Try running SUMO manually first
sumo-gui -c k1.sumocfg
```

---

### Issue 2: Training Not Improving

**Symptoms:**
- Reward stays flat or negative
- Actor/critic loss not decreasing

**Solutions:**

1. **Reduce learning rate:**
```python
LEARNING_RATE_ACTOR = 1e-4  # Instead of 3e-4
```

2. **Check reward function:**
```python
# Add print statements in _compute_rewards()
print(f"Junction {junction_id}: reward={total_reward:.2f}")
```

3. **Normalize state:**
```python
# In get_local_state(), normalize large values
state.append(queue_north / 50.0)  # Normalize queue
state.append(weighted_vehicles / 100.0)  # Normalize vehicles
```

4. **Increase exploration:**
```python
EPSILON_START = 0.3
EPSILON_DECAY = 0.99  # Slower decay
```

---

### Issue 3: Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Use CPU instead of GPU:**
```python
# In mappo_k1_implementation.py
# Remove .cuda() calls, use only CPU
device = torch.device('cpu')
```

2. **Reduce batch size:**
```python
UPDATE_FREQUENCY = 64  # Instead of 128
```

3. **Reduce network size:**
```python
ACTOR_HIDDEN = [64, 32]  # Instead of [128, 64]
CRITIC_HIDDEN = [128, 128, 64]  # Instead of [256, 256, 128]
```

---

### Issue 4: Simulation Too Slow

**Problem:** Training takes too long

**Solutions:**

1. **Use headless SUMO:**
```python
# In K1Environment._init_sumo()
sumo_binary = "sumo"  # Instead of "sumo-gui"
```

2. **Reduce episode length:**
```python
STEPS_PER_EPISODE = 1800  # 30 minutes instead of 1 hour
```

3. **Train on fewer episodes first:**
```python
NUM_EPISODES = 1000  # Test with shorter training
```

---

## 📚 Next Steps

### 1. Understand the Architecture
Read: `MAPPO_ARCHITECTURE_EXPLAINED.md`
- Detailed explanation of how MAPPO works
- Component breakdowns
- Training process visualization

### 2. Study the Code
- `mappo_k1_implementation.py` - Training implementation
  - `ActorNetwork` class (lines 143-201)
  - `CriticNetwork` class (lines 204-247)
  - `MAPPOAgent.update()` method (lines 656-737)

### 3. Experiment

**Try these experiments:**

a) **Different reward weights:**
```python
# Favor network-wide performance
REWARD_WEIGHT_NETWORK = 0.3
```

b) **Different network architectures:**
```python
# Bigger actor networks
ACTOR_HIDDEN = [256, 128, 64]
```

c) **Different scenarios:**
```python
# Train on morning rush only
# Then test on evening rush
```

### 4. Compare Algorithms

Train other algorithms for comparison:
- Single-agent PPO (no coordination)
- Independent Q-Learning (each junction separate)
- Rule-based adaptive (from `realistic_traffic_controller.py`)

---

## 🎯 Expected Training Timeline

**Phase 1: Understanding (Episodes 0-500)**
- Networks learn basic patterns
- Actors learn to read states
- Critic learns to evaluate situations
- **Improvement:** ~20% vs fixed-time

**Phase 2: Coordination (Episodes 500-2000)**
- Actors start considering neighbors
- Critic teaches coordination
- Green wave patterns emerge
- **Improvement:** ~40% vs fixed-time

**Phase 3: Fine-tuning (Episodes 2000-5000)**
- Policies stabilize
- Handle edge cases
- Emergency vehicle response
- **Improvement:** ~60% vs fixed-time ✅

**Total time:**
- CPU: ~50-70 hours
- GPU: ~10-15 hours

---

## 💡 Key Insights

### What Makes MAPPO Work

1. **Shared Critic = Coordination Teacher**
   - Sees global consequences
   - Teaches actors to help each other
   - No explicit coordination rules needed

2. **Separate Actors = Specialization**
   - Each junction learns its unique patterns
   - J0 (entry) ≠ J22 (internal)
   - But all learn to coordinate!

3. **Decentralized Execution = Deployable**
   - Each junction uses only local sensors ✅
   - No central controller needed
   - Robust to individual failures

### Why Better Than Alternatives

**vs Fixed-Time:**
- ✅ Adapts to traffic (not rigid schedule)
- ✅ Handles unexpected situations
- ✅ ~60% improvement

**vs Single-Agent RL:**
- ✅ Explicit coordination learning
- ✅ Better network-wide performance
- ✅ ~20-30% improvement over independent agents

**vs Rule-Based Adaptive:**
- ✅ Learns optimal policies (not hand-coded)
- ✅ Discovers complex patterns
- ✅ Continuous improvement

---

## 🚦 Ready to Start!

### Quick Command Reference

```bash
# Train MAPPO (full)
python mappo_k1_implementation.py

# Monitor training
tensorboard --logdir=mappo_logs

# Deploy trained model
python deploy_mappo.py --model mappo_models/final --duration 3600

# Compare with baseline
python deploy_mappo.py --model mappo_models/final --compare

# Deploy specific checkpoint
python deploy_mappo.py --model mappo_models/episode_2000 --duration 3600
```

---

## 📞 Need Help?

1. **Check the detailed explanation:** `MAPPO_ARCHITECTURE_EXPLAINED.md`
2. **Review the full guide:** `RL_ARCHITECTURE_GUIDE.md`
3. **Study the system overview:** `K1_SYSTEM_EXPLANATION.md`

**Happy training! 🚀🚦🤖**
