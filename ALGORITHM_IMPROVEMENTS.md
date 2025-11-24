# MAPPO Algorithm Improvement Analysis

## Current Algorithm Assessment (80 Episodes Trained)

### ✅ What's Working Well
1. **Positive rewards achieved** - Model learning basic coordination
2. **MAPPO architecture** - CTDE framework is sound
3. **GPU acceleration** - Training infrastructure solid
4. **State representation** - Includes queues, occupancy, weighted vehicles
5. **Neighbor coordination** - Local communication embedded

### 🔍 Identified Weaknesses

#### 1. **Reward Structure Issues**
```python
# Current reward:
own_reward = 0.01 * waiting_change + 0.1 * throughput
neighbor_reward = mean(0.005 * neighbor_waiting_change)
network_reward = -0.001 * total_vehicles
```

**Problems:**
- **Throughput scaling is off**: +0.1 per vehicle is too large vs waiting change (0.01)
- **Imbalanced scales**: Own (0.6×large) dominates neighbors (0.3×small) and network (0.1×tiny)
- **Network penalty too weak**: -0.001 per vehicle barely impacts decisions
- **No pressure normalization**: Raw waiting times vary wildly (0-500s)
- **Missing green wave incentive**: No reward for coordinated phase timing

#### 2. **State Representation Gaps**
```python
# Missing critical information:
state = [phase, queues(4), weighted_veh(4), occupancy(4), time_in_phase, neighbor_phases(2)]
```

**Missing:**
- **Phase duration history**: Can't learn min/max green times
- **Incoming traffic rate**: No prediction capability
- **Queue growth rate**: Only snapshot, not trend
- **Downstream capacity**: Can't anticipate blockage
- **Time of day**: No awareness of rush hour patterns (for 3h scenario)

#### 3. **Action Space Limitations**
```python
ACTION_DIM = 3  # [keep, next_phase, extend]
```

**Problems:**
- **Binary extension**: Only +5s extension, can't fine-tune
- **No early termination**: Can't cut short bad phases
- **Fixed phase sequence**: Can't skip to critical phase

#### 4. **Training Hyperparameters**
```python
UPDATE_FREQUENCY = 128  # Too infrequent for 10,800 step episodes
ENTROPY_COEF = 0.01     # Very low - poor exploration
EPSILON_START = 0.1     # Low initial exploration
```

**Issues:**
- **Updates only ~84 times per episode** (10800/128) - slow learning
- **Low entropy** → premature convergence to suboptimal policies
- **Epsilon decay too fast** → exploration dies quickly

#### 5. **Network Architecture**
```python
ACTOR_HIDDEN = [128, 64]      # Small for 16-dim state
CRITIC_HIDDEN = [256, 256, 128]  # Reasonable but no attention
```

**Limitations:**
- **No attention mechanism**: Can't focus on critical junctions
- **No temporal processing**: No LSTM/GRU for phase timing patterns
- **Fixed neighbors**: Can't dynamically weight neighbor importance

---

## 🚀 Proposed Improvements (Prioritized)

### **Priority 1: Fix Reward Structure (High Impact, Low Risk)**

#### A. Normalize and Balance Scales
```python
# Improved reward calculation:
def _compute_improved_reward(self, junction_id):
    # 1. Waiting time change (normalized by max capacity)
    max_waiting = 500.0  # Deadlock threshold
    current_waiting = traci.lane.getWaitingTime(...)
    prev_waiting = self.prev_waiting_times.get(junction_id, current_waiting)
    waiting_reduction = (prev_waiting - current_waiting) / max_waiting
    
    # 2. Throughput (normalized per step)
    throughput = traci.lane.getLastStepVehicleNumber(...)
    throughput_normalized = throughput / 10.0  # ~10 veh/step is high
    
    # 3. Queue pressure difference (promotes balance)
    queue_balance = -std_dev(queues_in_4_directions) / 10.0
    
    # 4. Green wave bonus (if neighbor just switched, reward keeping)
    green_wave_bonus = 0.0
    if neighbor_just_switched and self_action == KEEP:
        green_wave_bonus = 0.2
    
    # Balanced combination:
    own_reward = (
        0.5 * waiting_reduction +    # Primary objective
        0.3 * throughput_normalized + # Efficiency
        0.2 * queue_balance          # Fairness
    )
    
    # Neighbor impact (stronger weight)
    neighbor_reward = mean(neighbor_waiting_reductions) * 0.5
    
    # Network (stronger penalty)
    network_reward = -0.01 * (total_vehicles / 100.0)  # 10x stronger
    
    return 0.5*own + 0.35*neighbor + 0.15*network + green_wave_bonus
```

**Expected Impact:** +30-50% improvement in total reward

---

### **Priority 2: Enhanced State Representation (High Impact, Medium Risk)**

#### Add Critical Features:
```python
LOCAL_STATE_DIM = 16 → 22  # Add 6 features

# New features:
state.extend([
    time_in_phase / 60.0,              # Normalized phase duration
    min(phase_changes_last_10_steps, 3) / 3.0,  # Phase stability
    queue_growth_rate,                  # (current_queue - prev_queue) / 10.0
    avg_speed_incoming / 13.9,          # Normalized by speed limit
    time_of_episode / 10800.0,          # For 3h scenario awareness
    downstream_available_capacity       # Space for released vehicles
])
```

**Benefits:**
- Better temporal awareness (phase timing patterns)
- Predictive capability (queue trends)
- Traffic pattern adaptation (time of day)
- Congestion anticipation (downstream capacity)

**Risk:** Increases state space → may need more training episodes

---

### **Priority 3: Training Hyperparameter Tuning (Medium Impact, Low Risk)**

```python
# Improved hyperparameters:
UPDATE_FREQUENCY = 64         # 128 → 64 (2x more updates)
ENTROPY_COEF = 0.02           # 0.01 → 0.02 (better exploration)
EPSILON_START = 0.2           # 0.1 → 0.2 (more initial exploration)
EPSILON_DECAY = 0.998         # 0.995 → 0.998 (slower decay)
LEARNING_RATE_ACTOR = 5e-4    # 3e-4 → 5e-4 (faster learning)
PPO_EPOCHS = 15               # 10 → 15 (more thorough updates)
CLIP_EPSILON = 0.25           # 0.2 → 0.25 (less conservative)
```

**Expected Impact:** Faster convergence, better exploration

---

### **Priority 4: Curriculum Learning (Medium Impact, Medium Risk)**

Start with easier scenarios, gradually increase difficulty:

```python
# Episode-based curriculum:
if episode < 100:
    traffic_multiplier = 0.7  # Light traffic
elif episode < 300:
    traffic_multiplier = 1.0  # Normal traffic  
else:
    traffic_multiplier = 1.3  # Heavy traffic (challenge)
```

**Benefits:**
- Faster early learning (easier patterns)
- Progressive skill building
- Better final performance

---

### **Priority 5: Reward Shaping Enhancements (Low Impact, Low Risk)**

Add sparse milestone rewards:

```python
# Additional rewards:
if avg_waiting_time < 30:  # Good performance threshold
    bonus += 1.0
    
if no_phase_changes_last_30_steps and queue_growing:  # Punish inaction
    bonus -= 0.5
    
if all_queues_below_5:  # Excellent state
    bonus += 2.0
```

---

## 📋 Implementation Plan

### **Phase 1: Quick Wins (Implement Now)**
1. ✅ Fix reward normalization and balancing
2. ✅ Increase update frequency (128→64)
3. ✅ Boost exploration (entropy 0.01→0.02, epsilon 0.1→0.2)
4. ✅ Add green wave coordination bonus

**Estimated time:** 30 minutes  
**Expected improvement:** +40% cumulative reward

### **Phase 2: State Enhancement (After Phase 1 shows improvement)**
1. Add 6 critical state features (22-dim state)
2. Update network dimensions
3. Test on 50 episodes

**Estimated time:** 1 hour  
**Expected improvement:** +25% additional

### **Phase 3: Advanced (If needed)**
1. Implement curriculum learning
2. Add attention mechanism to critic
3. Experiment with action space refinement

---

## 🎯 Success Metrics

### Current Performance (80 episodes, 1-hour training):
- Average episode reward: ~positive (exact value unknown)
- Convergence: Achieved positive rewards

### Target Performance (After improvements):
- **Phase 1 goal:** +40% reward improvement within 50 episodes
- **Phase 2 goal:** +60% total improvement within 150 episodes
- **Final goal:** Consistent positive rewards on 3-hour scenarios with 2x throughput vs baseline

---

## 🔧 Testing Strategy

1. **Baseline**: Run 10 episodes with current algorithm (save metrics)
2. **Phase 1**: Implement reward fixes, run 50 episodes
3. **Compare**: If >30% improvement, proceed to Phase 2
4. **Phase 2**: Add state features, run 50 episodes
5. **Validation**: Test on unseen traffic patterns

---

## 📊 Quick Implementation Priority

| Improvement | Impact | Risk | Effort | Priority |
|------------|--------|------|--------|----------|
| Reward normalization | ⭐⭐⭐⭐⭐ | 🟢 Low | 15m | **DO NOW** |
| Update frequency | ⭐⭐⭐⭐ | 🟢 Low | 5m | **DO NOW** |
| Exploration boost | ⭐⭐⭐⭐ | 🟢 Low | 5m | **DO NOW** |
| Green wave bonus | ⭐⭐⭐⭐ | 🟢 Low | 10m | **DO NOW** |
| State enhancement | ⭐⭐⭐⭐ | 🟡 Medium | 45m | Phase 2 |
| Curriculum learning | ⭐⭐⭐ | 🟡 Medium | 30m | Phase 3 |
| Attention network | ⭐⭐ | 🔴 High | 2h | Optional |

**Next Action:** Implement Priority 1 improvements (reward structure) now!
