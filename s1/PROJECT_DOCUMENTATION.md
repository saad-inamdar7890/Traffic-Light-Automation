# Traffic Light Automation with MAPPO
## Comprehensive Project Documentation

**Version:** 1.0  
**Last Updated:** December 2025  
**Authors:** Traffic Light Automation Team

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why Multi-Agent Reinforcement Learning?](#2-why-multi-agent-reinforcement-learning)
3. [Algorithm: MAPPO](#3-algorithm-mappo)
4. [Network Architecture](#4-network-architecture)
5. [State Space Design](#5-state-space-design)
6. [Action Space](#6-action-space)
7. [Reward Function Design](#7-reward-function-design)
8. [Training Pipeline](#8-training-pipeline)
9. [Traffic Scenarios](#9-traffic-scenarios)
10. [SUMO Integration](#10-sumo-integration)
11. [Deployment](#11-deployment)
12. [Performance & Results](#12-performance--results)
13. [File Structure](#13-file-structure)
14. [How to Run](#14-how-to-run)

---

## 1. Project Overview

This project implements an intelligent traffic light control system using **Multi-Agent Proximal Policy Optimization (MAPPO)** for coordinated traffic signal control across a network of 9 intersections (the K1 network).

### Key Features

- **Multi-Agent Coordination**: 9 independent agents (one per junction) learning to coordinate
- **Centralized Training, Decentralized Execution (CTDE)**: Agents share a critic during training but act independently at deployment
- **Realistic Sensors**: Uses only data available from real-world traffic sensors (induction loops)
- **Time-Varying Traffic**: Supports multiple traffic patterns (rush hours, weekday/weekend, events)
- **GPU Acceleration**: Neural network training runs on GPU; SUMO simulation runs on CPU

### Problem Statement

Traditional traffic light systems use fixed timing cycles, which are inefficient for varying traffic conditions. This project trains AI agents to:

1. **Minimize waiting time** at each intersection
2. **Maximize throughput** (vehicles passing through)
3. **Coordinate across intersections** (green wave effect)
4. **Adapt to different traffic patterns** (rush hour, night, events)

---

## 2. Why Multi-Agent Reinforcement Learning?

### Why Not Single-Agent RL?

| Approach | Pros | Cons |
|----------|------|------|
| **Single Agent** | Simple, centralized control | Doesn't scale; O(A^N) action space for N junctions |
| **Independent Learners** | Scalable | Non-stationary environment; agents interfere |
| **Multi-Agent (MAPPO)** | Scalable + coordinated | More complex training |

For 9 junctions with 3 actions each:
- Single agent: 3^9 = 19,683 possible joint actions per step
- Multi-agent: 9 × 3 = 27 decisions (much simpler)

### Why Reinforcement Learning Over Traditional Methods?

| Method | Adaptability | Learning | Coordination |
|--------|--------------|----------|--------------|
| Fixed Timing | ❌ None | ❌ None | ❌ Manual |
| Actuated Control | ⚠️ Limited | ❌ None | ❌ Limited |
| Adaptive (SCOOT/SCATS) | ✅ Good | ⚠️ Rule-based | ⚠️ Predefined |
| **RL (MAPPO)** | ✅ Excellent | ✅ Continuous | ✅ Learned |

---

## 3. Algorithm: MAPPO

### What is MAPPO?

**Multi-Agent Proximal Policy Optimization (MAPPO)** is a state-of-the-art multi-agent reinforcement learning algorithm. It extends PPO (Proximal Policy Optimization) to multi-agent settings.

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MAPPO Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐    │
│   │ Actor 0 │ │ Actor 1 │ │ Actor 2 │  ...  │ Actor 8 │    │
│   │  (J0)   │ │  (J1)   │ │  (J5)   │       │  (J22)  │    │
│   └────┬────┘ └────┬────┘ └────┬────┘       └────┬────┘    │
│        │           │           │                 │          │
│        ▼           ▼           ▼                 ▼          │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐    │
│   │ Local   │ │ Local   │ │ Local   │       │ Local   │    │
│   │ State   │ │ State   │ │ State   │       │ State   │    │
│   │ (16-dim)│ │ (16-dim)│ │ (16-dim)│       │ (16-dim)│    │
│   └─────────┘ └─────────┘ └─────────┘       └─────────┘    │
│                                                              │
│                    ┌──────────────────┐                     │
│                    │  Shared Critic   │                     │
│                    │  (Global State)  │                     │
│                    │   (146-dim)      │                     │
│                    └──────────────────┘                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why PPO? (vs Other RL Algorithms)

| Algorithm | Sample Efficiency | Stability | Hyperparameter Sensitivity |
|-----------|-------------------|-----------|---------------------------|
| DQN | Medium | Medium | High |
| A2C/A3C | Low | Low | High |
| DDPG | High | Low | Very High |
| **PPO** | Medium-High | **High** | **Low** |
| SAC | High | High | Medium |

PPO's **clipped objective** prevents catastrophically large policy updates:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio
- $\hat{A}_t$ is the advantage estimate
- $\epsilon = 0.25$ (our clip range)

### CTDE: Centralized Training, Decentralized Execution

```
TRAINING PHASE                          DEPLOYMENT PHASE
┌─────────────────────┐                ┌─────────────────────┐
│                     │                │                     │
│  Global State ──────┼──► Critic      │  Local State ──────┼──► Actor
│  (all junctions)    │      │         │  (own junction)     │      │
│                     │      ▼         │                     │      ▼
│                     │   Value        │                     │   Action
│  Local States ──────┼──► Actors      │                     │
│                     │      │         │                     │
│                     │      ▼         │                     │
│                     │   Actions      │                     │
│                     │                │                     │
└─────────────────────┘                └─────────────────────┘

✓ Critic sees everything              ✓ Actors use only local sensors
✓ Actors learn coordination           ✓ No communication needed
✓ Shared value baseline               ✓ Fault-tolerant
```

---

## 4. Network Architecture

### Actor Network (Per Junction)

Each of the 9 junctions has its own actor network:

```
Input: Local State (16 dimensions)
         │
         ▼
┌─────────────────────┐
│   Linear(16, 128)   │
│      + ReLU         │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Linear(128, 64)   │
│      + ReLU         │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Linear(64, 3)     │
│    + Softmax        │
└─────────────────────┘
         │
         ▼
Output: Action Probabilities (3 actions)
```

**Total Actor Parameters:** ~10,500 per junction × 9 = ~94,500

### Critic Network (Shared)

Single critic network shared by all agents:

```
Input: Global State (146 dimensions)
         │
         ▼
┌─────────────────────┐
│   Linear(146, 256)  │
│      + ReLU         │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Linear(256, 256)  │
│      + ReLU         │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Linear(256, 128)  │
│      + ReLU         │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Linear(128, 1)    │
└─────────────────────┘
         │
         ▼
Output: State Value (scalar)
```

**Total Critic Parameters:** ~135,000

---

## 5. State Space Design

### Local State (Per Junction) — 16 Dimensions

Each agent observes only what real-world sensors can detect:

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | `current_phase` | Current traffic light phase | 0-7 |
| 1-4 | `queue_N/S/E/W` | Halting vehicles per direction | 0-50+ |
| 5-8 | `weighted_veh_N/S/E/W` | PCE-weighted vehicle count | 0-100+ |
| 9-12 | `occupancy_N/S/E/W` | Lane occupancy (0-1) | 0.0-1.0 |
| 13 | `time_in_phase` | Seconds in current phase | 0-300+ |
| 14-15 | `neighbor_phases` | Phases of 2 nearest neighbors | 0-7 |

### Vehicle Type Weighting (PCE - Passenger Car Equivalents)

Different vehicle types have different impacts on traffic:

```python
VEHICLE_WEIGHTS = {
    'passenger': 1.0,   # Base unit
    'delivery': 2.5,    # Small trucks
    'truck': 5.0,       # Large trucks
    'bus': 4.5          # Buses
}
```

This means 1 truck = 5 cars in terms of traffic impact.

### Global State (Critic Only) — 146 Dimensions

```
┌────────────────────────────────────────────────────────┐
│ Dim 0-15:   Junction J0  local state  (16 features)   │
│ Dim 16-31:  Junction J1  local state  (16 features)   │
│ Dim 32-47:  Junction J5  local state  (16 features)   │
│ Dim 48-63:  Junction J6  local state  (16 features)   │
│ Dim 64-79:  Junction J7  local state  (16 features)   │
│ Dim 80-95:  Junction J10 local state  (16 features)   │
│ Dim 96-111: Junction J11 local state  (16 features)   │
│ Dim 112-127: Junction J12 local state (16 features)   │
│ Dim 128-143: Junction J22 local state (16 features)   │
│ Dim 144:    Total vehicles in network                  │
│ Dim 145:    Total waiting time (normalized)            │
└────────────────────────────────────────────────────────┘
```

---

## 6. Action Space

Each junction can take one of **3 discrete actions** at each timestep:

| Action | Name | Effect |
|--------|------|--------|
| 0 | **Keep** | Stay in current phase |
| 1 | **Next** | Switch to next phase in cycle |
| 2 | **Extend** | Extend current phase duration |

### Why These Actions?

- **Simple**: Only 3 choices per step (not selecting specific phases)
- **Safe**: Always follows valid phase transitions
- **Realistic**: Mimics how real adaptive systems work
- **Learnable**: Small action space = faster learning

### Phase Cycle Example

```
Phase 0: NS Green, EW Red
    │
    ▼ (action=1)
Phase 1: NS Yellow, EW Red
    │
    ▼ (action=1)
Phase 2: NS Red, EW Green
    │
    ▼ (action=1)
Phase 3: NS Red, EW Yellow
    │
    ▼ (action=1)
Phase 0: (cycle repeats)
```

---

## 7. Reward Function Design

### The Challenge

Traffic optimization has multiple competing objectives:
- Minimize waiting time (primary)
- Maximize throughput (secondary)
- Balance queues across directions
- Coordinate with neighbors (green waves)
- Avoid gridlock

### Reward Components

```python
total_reward = (
    0.50 × own_reward +           # Own junction performance
    0.35 × neighbor_reward +      # Neighbor coordination
    0.15 × network_reward +       # Network-wide efficiency
    green_wave_bonus +            # Coordination bonus
    deadlock_penalty              # Emergency penalty
)
```

### 1. Own Junction Performance (Weight: 50%)

```python
own_reward = (
    0.5 × waiting_reduction +     # Primary: reduce waiting
    0.3 × throughput_normalized + # Secondary: increase flow
    0.2 × queue_balance           # Tertiary: balance queues
)
```

**Waiting Reduction** (most important):
$$\text{waiting\_reduction} = \frac{\text{prev\_waiting} - \text{current\_waiting}}{\text{MAX\_THRESHOLD}}$$

Clipped to [-1, 1] to prevent extreme values.

**Throughput**:
$$\text{throughput\_normalized} = \frac{\text{vehicles\_passed}}{\text{MAX\_THROUGHPUT\_PER\_STEP}}$$

**Queue Balance** (penalize imbalanced queues):
$$\text{queue\_balance} = -\frac{\sigma(\text{queues})}{10}$$

### 2. Neighbor Coordination (Weight: 35%)

Agents care about their neighbors' performance:

```python
neighbor_reward = mean(
    waiting_reduction for each neighbor
)
```

This encourages agents to make decisions that help nearby junctions.

### 3. Network-Wide Efficiency (Weight: 15%)

Small penalty proportional to total vehicles in network:

$$\text{network\_reward} = -0.01 \times \frac{\text{total\_vehicles}}{100}$$

This encourages clearing vehicles from the network.

### 4. Green Wave Bonus (+0.3)

Rewards coordinated timing between adjacent junctions:

```python
if neighbor_changed_phase_recently AND I_kept_my_phase:
    green_wave_bonus += 0.3 / num_neighbors
```

This teaches agents to create "green waves" where vehicles can pass through multiple intersections without stopping.

### 5. Deadlock Penalty (-10.0)

Emergency penalty when waiting time exceeds threshold:

```python
if current_waiting > MAX_WAITING_THRESHOLD:
    deadlock_penalty = -10.0
```

### Reward Summary Diagram

```
                    Total Reward
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Own (50%)      Neighbor (35%)    Network (15%)
        │                │                │
   ┌────┼────┐           │                │
   │    │    │           │                │
   ▼    ▼    ▼           ▼                ▼
Wait  Thru  Bal      Wait Red         Congestion
 Red  put   ance                       Penalty
(50%) (30%) (20%)
                         │
                         ▼
               + Green Wave Bonus (+0.3)
                         │
                         ▼
               + Deadlock Penalty (-10)
```

---

## 8. Training Pipeline

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LEARNING_RATE_ACTOR` | 5e-4 | Actor learning rate |
| `LEARNING_RATE_CRITIC` | 1e-3 | Critic learning rate |
| `GAMMA` | 0.99 | Discount factor |
| `LAMBDA` | 0.95 | GAE parameter |
| `CLIP_EPSILON` | 0.25 | PPO clip range |
| `ENTROPY_COEF` | 0.02 | Entropy bonus coefficient |
| `GRAD_CLIP` | 0.5 | Gradient clipping |
| `UPDATE_FREQUENCY` | 64 | Steps between updates |
| `PPO_EPOCHS` | 15 | Epochs per update |
| `EPSILON_START` | 0.2 | Initial exploration rate |
| `EPSILON_END` | 0.01 | Final exploration rate |
| `EPSILON_DECAY` | 0.998 | Decay rate per episode |

### Training Loop

```
for episode in range(NUM_EPISODES):
    │
    ├──► Reset SUMO environment
    │
    ├──► for step in range(STEPS_PER_EPISODE):
    │        │
    │        ├──► Get local states for all 9 junctions
    │        │
    │        ├──► Select actions (with exploration)
    │        │
    │        ├──► Execute actions in SUMO
    │        │
    │        ├──► Compute rewards
    │        │
    │        ├──► Store experience in buffer
    │        │
    │        └──► if step % UPDATE_FREQUENCY == 0:
    │                 │
    │                 ├──► Compute GAE advantages
    │                 │
    │                 ├──► Update actors (PPO)
    │                 │
    │                 └──► Update critic (MSE loss)
    │
    ├──► Decay exploration rate
    │
    └──► Save checkpoint (every 100 episodes)
```

### Generalized Advantage Estimation (GAE)

We use GAE for variance reduction in advantage estimation:

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

---

## 9. Traffic Scenarios

### Available Scenarios

| Scenario | Duration | Description | File |
|----------|----------|-------------|------|
| 3-Hour Varying | 3h | Base training scenario | `k1_routes_3h_varying.rou.xml` |
| 6-Hour Weekday | 6h | Morning/evening rush | `k1_routes_6h_weekday.rou.xml` |
| 6-Hour Weekend | 6h | Relaxed midday peak | `k1_routes_6h_weekend.rou.xml` |
| 6-Hour Event | 6h | Heavy event traffic | `k1_routes_6h_event.rou.xml` |
| 24-Hour Realistic | 24h | Full day cycle | `k1_routes_24h_realistic.rou.xml` |

### Time Periods (6-Hour Scenarios)

| Period | Time | Multiplier (Weekday) | Multiplier (Event) |
|--------|------|---------------------|-------------------|
| Early Morning | 0-1h | 0.5× | 0.4× |
| Morning Peak | 1-2h | 1.6× | 1.0× |
| Midday | 2-3h | 1.0× | 1.8× |
| Afternoon | 3-4h | 0.9× | 1.5× |
| Evening Peak | 4-5h | 1.7× | 2.0× |
| Night | 5-6h | 0.4× | 0.6× |

### Traffic Flow Types

```
Bus Routes (b_1, b_2):
  - Lower frequency (8-20 veh/hour)
  - Higher PCE weight (4.5)

Passenger Routes (f_0 to f_27):
  - High frequency (80-350 veh/hour)
  - Standard PCE weight (1.0)

Truck Routes (t_1, t_2):
  - Low frequency (6-17 veh/hour)
  - Highest PCE weight (5.0)
```

---

## 10. SUMO Integration

### What is SUMO?

**SUMO** (Simulation of Urban MObility) is an open-source, microscopic traffic simulation tool. We use it to:

1. Simulate realistic vehicle movements
2. Control traffic lights via TraCI (Traffic Control Interface)
3. Measure traffic metrics (waiting time, queues, throughput)

### TraCI Communication

```
┌─────────────────┐     TraCI      ┌─────────────────┐
│                 │ ◄────────────► │                 │
│  Python Agent   │                │  SUMO Simulator │
│  (MAPPO)        │                │                 │
│                 │   Commands:    │  - k1.net.xml   │
│  - Get states   │   - getState   │  - routes.xml   │
│  - Set phases   │   - setPhase   │  - sumocfg      │
│  - Get rewards  │   - simStep    │                 │
│                 │                │                 │
└─────────────────┘                └─────────────────┘
```

### Key SUMO Files

| File | Purpose |
|------|---------|
| `k1.net.xml` | Network topology (roads, junctions, lanes) |
| `k1_routes_*.rou.xml` | Vehicle routes and traffic flows |
| `k1_*.sumocfg` | Simulation configuration |

### Route Precomputation (duarouter)

To avoid runtime routing errors, we precompute routes:

```bash
python generate_6h_routes.py --duarouter
```

This converts `<flow>` definitions to explicit `<vehicle>` entries with precomputed paths.

---

## 11. Deployment

### Deployment Modes

1. **Training Mode**: Full MAPPO with exploration
2. **Evaluation Mode**: Trained actors, no exploration
3. **Adaptive Mode**: Fine-tune on new traffic patterns

### Using Trained Models

```python
from mappo_k1_implementation import MAPPOAgent, MAPPOConfig

# Load trained model
config = MAPPOConfig()
agent = MAPPOAgent(config)
agent.load_models("mappo_models/final")

# Deploy (actors only, no critic needed)
actions, _, _ = agent.select_actions(local_states)
```

### Real-World Deployment Considerations

| Aspect | Simulation | Real World |
|--------|------------|------------|
| Sensors | Perfect data | Noise, failures |
| Latency | Instant | Network delays |
| Safety | Can crash | Must be safe |
| Updates | Any time | Scheduled windows |

---

## 12. Performance & Results

### Metrics Tracked

1. **Average Waiting Time**: Seconds vehicles wait at red lights
2. **Total Throughput**: Vehicles cleared per episode
3. **Queue Lengths**: Average halting vehicles per lane
4. **Episode Reward**: Total reward accumulated

### Expected Training Progress

| Episodes | Avg Reward | Avg Wait Time | Status |
|----------|------------|---------------|--------|
| 0-100 | -50 to -20 | High | Exploring |
| 100-500 | -20 to 0 | Decreasing | Learning basics |
| 500-1000 | 0 to +20 | Moderate | Coordinating |
| 1000-3000 | +20 to +50 | Low | Optimizing |
| 3000+ | +50+ | Minimal | Converged |

### Monitoring with TensorBoard

```bash
tensorboard --logdir=mappo_logs
```

Key metrics:
- `Episode/Reward` - Should increase over time
- `Episode/Length` - Steps before done (higher = better)
- `Loss/Actor` - Should decrease then stabilize
- `Loss/Critic` - Should decrease

---

## 13. File Structure

```
s1/
├── mappo_k1_implementation.py   # Main training script
├── deploy_mappo.py              # Deployment utilities
├── generate_6h_routes.py        # Route generation + duarouter
├── validate_routes.py           # Route validation tool
│
├── k1.net.xml                   # SUMO network
├── k1_routes_*.rou.xml          # Traffic scenarios
├── k1_*.sumocfg                 # SUMO configurations
│
├── mappo_models/                # Saved model checkpoints
│   ├── actor_0.pth ... actor_8.pth
│   ├── critic.pth
│   └── train_state.pkl
│
├── mappo_logs/                  # TensorBoard logs
│
├── requirements.txt             # Python dependencies
└── *.md                         # Documentation files
```

---

## 14. How to Run

### Prerequisites

```bash
# Install SUMO
# Windows: Download from https://sumo.dlr.de/docs/Downloads.php
# Linux: sudo apt install sumo sumo-tools

# Set SUMO_HOME environment variable
# Windows: set SUMO_HOME=C:\Program Files\Eclipse\Sumo
# Linux: export SUMO_HOME=/usr/share/sumo

# Install Python dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic 3-hour training
python mappo_k1_implementation.py

# 6-hour weekday scenario
python mappo_k1_implementation.py --scenario weekday

# Resume from checkpoint with time limit
python mappo_k1_implementation.py --resume-checkpoint mappo_models/checkpoint_500 --max-hours 3

# Custom episode count
python mappo_k1_implementation.py --num-episodes 1000
```

### Generate Route Files

```bash
# Generate flow-based routes
python generate_6h_routes.py

# Generate + precompute static routes (recommended)
python generate_6h_routes.py --duarouter
```

### Validate Routes

```bash
python validate_routes.py
```

### Monitor Training

```bash
tensorboard --logdir=mappo_logs
```

---

## Summary

This project demonstrates a complete pipeline for training multi-agent RL for traffic control:

1. **Algorithm**: MAPPO with CTDE paradigm
2. **State**: Realistic 16-dim local states per junction
3. **Actions**: 3 discrete actions (keep/next/extend)
4. **Rewards**: Multi-component reward encouraging coordination
5. **Training**: PPO with GAE, gradient clipping, entropy bonus
6. **Simulation**: SUMO with TraCI for realistic traffic

The system learns to coordinate 9 traffic lights to minimize waiting times and maximize throughput, adapting to varying traffic patterns throughout the day.

---

## References

1. Yu, C. et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS.
2. Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv.
3. SUMO Documentation: https://sumo.dlr.de/docs/
4. Liang, X. et al. (2019). "Deep Reinforcement Learning for Traffic Signal Control." IEEE ITSC.
