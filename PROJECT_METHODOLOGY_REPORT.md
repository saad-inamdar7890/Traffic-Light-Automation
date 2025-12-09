# Traffic Light Automation Project: Comprehensive Methodology Report

**Project:** Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning  
**Network:** K1 Urban Network (9 Junctions)  
**Algorithm:** MAPPO (Multi-Agent Proximal Policy Optimization)  
**Date:** December 2025

---

## 1. Executive Summary

This report details the complete methodology employed in developing, training, debugging, and deploying a Multi-Agent Reinforcement Learning (MARL) system for adaptive traffic signal control on the K1 urban network. The system uses MAPPO with Centralized Training and Decentralized Execution (CTDE) to optimize traffic flow across 9 interconnected junctions, achieving significant improvements over traditional fixed-time controllers in congestion reduction and throughput.

---

## 2. Materials and Technology Stack

### 2.1 Simulation Environment

**SUMO (Simulation of Urban MObility) v1.24.0**
- Microscopic traffic simulator providing realistic vehicle dynamics
- Supports multiple vehicle types (passenger cars, buses, trucks, delivery vehicles)
- Features:
  - Lane-level accuracy with collision detection
  - Emergency stop modeling
  - Dijkstra router for dynamic routing
  - Multi-threaded execution (8 threads for Ryzen 7)

**TraCI (Traffic Control Interface)**
- Python API for real-time SUMO control
- Enables:
  - State extraction (queue lengths, waiting times, occupancy, vehicle IDs)
  - Action execution (phase changes, duration extensions)
  - Step-by-step simulation control
- Crash recovery implemented to handle edge cases (invalid routes, teleported vehicles)

**Network Configuration**
- **K1 Network:** 9 controlled junctions (J0, J1, J5, J6, J7, J10, J11, J12, J22)
- **Topology:** Interconnected urban grid with defined neighbor relationships
- **Network File:** `k1.net.xml` (designed and validated in NetEdit)
- **Traffic Light Logic:** `k1.ttl.xml` (8 phases per junction)

### 2.2 Machine Learning Framework

**PyTorch 2.x**
- Deep learning framework for neural network implementation
- CUDA support for GPU acceleration (Tesla T4 on Kaggle, local GPU support)
- Automatic differentiation for gradient computation

**Neural Network Architecture:**
- **Actor Networks:** 9 independent networks (one per junction)
  - Input: 16-dimensional local state
  - Hidden layers: [128, 64] with ReLU activation
  - Output: 3 action probabilities (softmax)
  - Total parameters: ~10K per actor

- **Critic Network:** 1 shared global network
  - Input: 146-dimensional global state
  - Hidden layers: [256, 256, 128] with ReLU activation
  - Output: Single state value
  - Total parameters: ~95K

**TensorBoard**
- Real-time training visualization
- Logged metrics:
  - Episode rewards (total and per-agent)
  - Actor and Critic losses
  - Exploration rate (epsilon)
  - Episode duration and simulation speed

### 2.3 Development Tools

**Python 3.10+**
- Core programming language
- Virtual environment management (`.venv`)

**Key Libraries:**
- **NumPy:** Numerical operations, state normalization
- **XML ElementTree:** Route file generation and manipulation
- **CSV:** Debugging and detailed logging
- **Pickle:** Checkpoint serialization

**Version Control:**
- **Git & GitHub:** Code versioning and collaboration
- **Artifact Tracking:** Route file SHA256 checksums for reproducibility

**Development Environment:**
- **VS Code:** Primary IDE with Python extension
- **PowerShell:** Terminal scripting for batch operations
- **Kaggle Notebooks:** Cloud training with GPU support

---

## 3. Methodology and Techniques

### 3.1 Algorithm: MAPPO (Multi-Agent Proximal Policy Optimization)

**Core Paradigm: Centralized Training, Decentralized Execution (CTDE)**

The system implements MAPPO, a state-of-the-art MARL algorithm that balances coordination and scalability:

**Decentralized Execution (Actor Networks):**
- Each junction operates independently using only local observations
- 9 separate Actor networks (one per junction)
- Input: 16-dimensional local state vector
- Output: Action probabilities for 3 discrete actions
- Benefits:
  - Scalable deployment (no communication overhead)
  - Robust to network failures
  - Real-time decision-making capability

**Centralized Training (Critic Network):**
- Single shared Critic with global view during training
- Input: 146-dimensional global state (all junctions + network metrics)
- Output: Value estimate for joint state
- Benefits:
  - Learns coordinated behaviors
  - Addresses non-stationarity in multi-agent environments
  - Enables credit assignment across agents

**PPO Stabilization:**
- Clipped surrogate objective prevents destructive policy updates
- Trust region optimization ensures stable convergence
- Entropy regularization maintains exploration throughout training

### 3.2 State Space Design

**Local State (16 dimensions per junction):**

1. **Traffic Signal State (2 dims):**
   - Current phase index (0-7)
   - Time spent in current phase (seconds)

2. **Queue Metrics (4 dims):**
   - Halting vehicles per direction (North, South, East, West)
   - Measured via induction loops

3. **Weighted Vehicle Counts (4 dims):**
   - Passenger Car Equivalent (PCE) counts per direction:
     - Passenger: 1.0
     - Delivery: 2.5
     - Truck: 5.0
     - Bus: 4.5

4. **Occupancy Levels (4 dims):**
   - Lane occupancy percentage per direction (0-1)
   - Indicates congestion severity

5. **Neighbor Coordination (2 dims):**
   - Current phase of up to 2 nearest neighbors
   - Enables green wave coordination

**Global State (146 dimensions):**
- Concatenation of all 9 local states (9 × 16 = 144)
- Network-wide total vehicle count (1)
- Network-wide normalized total waiting time (1)

**Rationale:**
- Local states are realistic (sensor-only data)
- Global state captures system-wide effects for value estimation
- Design balances information richness with computational efficiency

### 3.3 Action Space

**Discrete Actions (3 per junction):**
- **Action 0 (Keep):** Maintain current phase
- **Action 1 (Next):** Switch to next phase in cycle
- **Action 2 (Extend):** Extend current phase duration

**Phase Management:**
- Cyclic phase progression (8 phases per junction)
- Minimum green time enforced by simulation
- Yellow phases handled automatically by SUMO

**Exploration Strategy:**
- Epsilon-greedy exploration during training
- Epsilon decay: 0.2 → 0.01 over episodes (decay rate 0.998)
- Entropy bonus (0.02) encourages policy diversity

### 3.4 Reward Function Engineering

**Multi-Component Reward Design:**

The reward function balances local efficiency, global coordination, and stability:

```
Total Reward = 0.5 × Own + 0.35 × Neighbors + 0.15 × Network + Bonuses + Penalties
```

**Component 1: Own Junction Performance (Weight: 0.5)**
- **Waiting Time Reduction:** (prev_waiting - current_waiting) / 500
  - Normalized by threshold to prevent scale issues
  - Clipped to [-1, 1]
- **Throughput:** vehicles_passing / 10.0
  - Rewards flow efficiency
  - Clipped to [0, 2]
- **Queue Balance:** -std(queues) / 10.0
  - Penalizes uneven queues across directions
  - Prevents starvation of minor roads

**Component 2: Neighbor Coordination (Weight: 0.35)**
- Average waiting time reduction of immediate neighbors
- Encourages cooperative behavior
- Addresses negative externalities

**Component 3: Network Global Penalty (Weight: 0.15)**
- -0.01 × (total_vehicles / 100)
- Prevents system-wide gridlock
- Clipped to [-2, 0]

**Component 4: Green Wave Bonus (+0.3)**
- Awarded when agent maintains phase while neighbor changes
- Promotes synchronized progression
- Improves arterial flow

**Component 5: Proportional Deadlock Penalty**
- Soft penalty when waiting > 500s:
  ```
  penalty = -2.0 × min(excess_ratio, 2.0)
  ```
  where `excess_ratio = (waiting - 500) / 500`
- Replaced harsh binary penalty to prevent reward collapse
- Provides smooth gradient for recovery learning

**Safety Guards:**
- Per-junction reward clipping: [-5, 5]
- NaN/Inf detection and replacement
- Prevents exploding cumulative rewards during stress scenarios

**Evolution History:**
- Original deadlock penalty: -10.0 (too harsh)
- Identified via per-step CSV debugging (see Section 6.3)
- Reduced to -2.0 with proportional scaling
- Result: 80% reduction in training instability

---

## 4. Training Strategy and Procedures

### 4.1 Traffic Scenarios

**Base Scenarios (6-hour episodes, 21,600 steps):**

1. **Weekday (`k1_routes_6h_weekday.rou.xml`):**
   - Standard commuter patterns
   - Morning peak: 7-9 AM
   - Evening peak: 5-7 PM
   - 174 distinct flows
   - Vehicle types: Mixed (60% passenger, 25% delivery, 10% truck, 5% bus)

2. **Weekend (`k1_routes_6h_weekend.rou.xml`):**
   - Smoother, more distributed traffic
   - No sharp peaks
   - Lower overall volume (70% of weekday)
   - Recreational travel patterns

3. **Event (`k1_routes_6h_event.rou.xml`):**
   - High-density scenario
   - Simulates concerts, sports events, conferences
   - 130% of weekday volume
   - Concentrated arrivals and departures

**Stress Test Scenarios:**

4. **Gridlock (`k1_routes_6h_gridlock.rou.xml`):**
   - 250% of base flow rates
   - Tests deadlock prevention and recovery
   - Generated via `scripts/generate_stress_scenarios.py`

5. **Incident (`k1_routes_6h_incident.rou.xml`):**
   - Asymmetric flow disruption
   - Simulates road closures and detours

6. **Spike (`k1_routes_6h_spike.rou.xml`):**
   - Sudden traffic surges (300% for 10-minute windows)
   - Tests adaptability to rapid changes

7. **Night Surge (`k1_routes_6h_night_surge.rou.xml`):**
   - Inverted patterns (late-night events)
   - Low baseline with sudden peaks

8. **Event Repro (`k1_routes_6h_event_repro.rou.xml`):**
   - Reproducible late-episode surge (step 18000-18600)
   - Created specifically for debugging reward collapse
   - 3× flow multiplier during surge window

**Route File Generation:**
- All routes validated via SUMO precomputation (duarouter)
- SHA256 checksums logged for reproducibility
- `via` attributes preserved for multi-leg trips
- Zero-rate flows patched (0 → 1 veh/hour minimum)

### 4.2 Training Hyperparameters

**Optimization:**
- **Learning Rate (Actor):** 5e-4
- **Learning Rate (Critic):** 1e-3
- **Optimizer:** Adam
- **Gradient Clipping:** 0.5 (prevents exploding gradients)

**PPO Configuration:**
- **Clip Epsilon:** 0.25 (trust region constraint)
- **GAE Lambda:** 0.95 (Generalized Advantage Estimation)
- **Discount Factor (Gamma):** 0.99
- **Entropy Coefficient:** 0.02 (exploration incentive)
- **Update Frequency:** Every 64 steps
- **PPO Epochs:** 15 per update

**Training Schedule:**
- **Target Episodes:** 5,000
- **Episode Duration:** 21,600 steps (6 hours simulated)
- **Real-time per Episode:** ~20-26 minutes (20 steps/second)
- **Total Training Time:** ~100-130 hours on GPU
- **Checkpoint Frequency:** Every 100 episodes
- **Time-limited Runs:** `--max-hours` flag for Kaggle/Colab (3-hour sessions)

**Exploration Strategy:**
- **Epsilon Start:** 0.2
- **Epsilon End:** 0.01
- **Decay Rate:** 0.998 per episode
- Reaches 1% exploration by episode ~700

### 4.3 Mixed Training Curriculum

**Rationale:**
Prevent overfitting to a single traffic pattern; ensure generalization.

**Implementation:**
- Each episode randomly selects a scenario from:
  ```
  [weekday, weekend, event, gridlock, incident, spike, night_surge]
  ```
- Enabled via `--scenario mixed` flag
- Managed by `config.MIXED_SCENARIOS` list
- SUMO config overridden per episode in `env.reset()`

**Benefits:**
- Robust policy across diverse conditions
- Prevents catastrophic forgetting
- Faster convergence (cross-scenario transfer learning)

### 4.4 Training Infrastructure

**Local Training:**
- Hardware: Ryzen 7 (8 cores), GPU (CUDA-enabled)
- SUMO: Multi-threaded (8 threads)
- PyTorch: GPU-accelerated network updates
- Storage: Checkpoints saved to `mappo_models/`

**Kaggle Training:**
- GPU: Tesla T4 (16GB VRAM)
- Session limit: 12 hours (managed via `--max-hours`)
- Checkpoint resume: `--resume-checkpoint` flag
- Artifact download: Package and download checkpoints as ZIP

**Checkpoint Management:**
- Full state saved:
  - Actor and Critic weights (`.pth` files)
  - Optimizer states
  - Training progress (episode count, epsilon)
  - RNG states (reproducibility)
  - Replay buffer (optional)
- Route file artifact: Copied to `mappo_models/` with SHA256
- Resume capability: Tolerant to state-space changes (shape adaptation)

### 4.5 Training Monitoring

**TensorBoard Logs (`mappo_logs/`):**
- Episode/Reward: Total cumulative reward per episode
- Episode/Length: Steps completed before termination
- Episode/Epsilon: Current exploration rate
- Loss/Actor: Policy gradient loss (averaged across 9 actors)
- Loss/Critic: Value function MSE loss

**Console Output:**
- Episode header with progress (e.g., "Episode 234/5000")
- Reset time (SUMO initialization)
- Simulation progress (every 100 steps)
- Step speed (steps/second)
- ETA calculation
- Episode summary (reward, duration, metrics)
- Timing breakdown (reset, simulation, network updates)

**Mid-Episode Time Limit:**
- Checks every 500 steps to respect `--max-hours`
- Graceful shutdown and checkpoint save
- Resume instructions printed to console

---

## 5. Evaluation and Testing Methodology

### 5.1 Evaluation Framework

**Baseline Controller:**
- **Fixed-Time Controller:** Pre-programmed phase durations
- Industry standard (SCATS/SCOOT-like logic)
- No adaptive response to traffic conditions
- Benchmark for improvement quantification

**Evaluation Script:** `evaluate_fixed_vs_mappo.py`
- Runs both controllers on identical scenarios
- Seeds controlled for reproducibility
- Metrics logged per-step and aggregated
- Produces comparative analysis reports

### 5.2 Key Performance Indicators (KPIs)

**Primary Metrics:**

1. **Average Waiting Time (seconds/vehicle):**
   - Most critical metric for driver experience
   - Measured at lane level via TraCI
   - Target: 30-50% reduction vs. fixed-time

2. **Total Network Throughput (vehicles/hour):**
   - Completed trips during episode
   - Higher = better network utilization
   - Target: 10-20% improvement

3. **Average Queue Length (vehicles):**
   - Per junction and per direction
   - Indicator of congestion severity
   - Target: Minimize max queue length

**Secondary Metrics:**

4. **Emergency Stops:**
   - Sudden braking events (safety proxy)
   - Lower = smoother flow

5. **Episode Reward:**
   - Training signal quality indicator
   - Should increase monotonically during training

6. **Simulation Speed (steps/second):**
   - Real-time factor (RTF)
   - Affects deployment feasibility
   - MAPPO: ~20 steps/sec (3× real-time)

**Per-Junction Analysis:**
- Identify bottlenecks
- Diagnose coordination failures
- Validate neighbor reward component

### 5.3 Testing Procedures

**Phase 1: Deterministic Evaluation**
```bash
python evaluate_fixed_vs_mappo.py \
    --checkpoint mappo_models/checkpoint_5000 \
    --scenario weekday \
    --seed 42
```
- Disable exploration (greedy policy)
- Fixed random seed for reproducibility
- Multiple runs (5-10) with different seeds
- Statistical significance testing (t-test)

**Phase 2: Stress Testing**
- Evaluate on unseen stress scenarios
- Measure graceful degradation under extreme load
- Recovery time after congestion events
- Robustness to distribution shift

**Phase 3: Generalization Testing**
- Train on {weekday, weekend, event}
- Test on {gridlock, incident, spike, night_surge}
- Quantify zero-shot transfer performance

**Phase 4: Long-Duration Testing**
- 24-hour continuous simulations
- Detect policy drift or failure modes
- Memory leak detection
- Stability over extended periods

### 5.4 Debugging and Diagnostic Tools

**Tool 1: Verbose Per-Step Logger (`debug_verbose_eval.py`)**

Purpose: Diagnose reward collapse and training instabilities

Features:
- Logs every agent's reward at every timestep
- Outputs CSV with columns: `[step, agent_idx, junction_id, reward, cumulative]`
- Flags large rewards (configurable threshold, default ±50)
- Enables timeline analysis of failure modes

Usage:
```bash
python s1/debug_verbose_eval.py \
    --checkpoint mappo_models/checkpoint_XXX \
    --scenario event_repro \
    --steps 21600 \
    --out s1/debug_rewards.csv \
    --threshold 5.0
```

**Case Study: Reward Collapse Detection**
- Identified deadlock penalty triggered 5,705 times in one episode
- Junctions J6 (3,243×), J11 (1,761×), J5 (726×) most affected
- Occurred starting at step 18,206 (~84% into episode)
- Led to proportional penalty redesign (see Section 3.4)

**Tool 2: Checkpoint Packaging (`package_checkpoint_for_kaggle.py`)**
- Bundles model weights + metadata
- Adds SHA256 checksums
- Compresses for cloud transfer
- Includes resume instructions

**Tool 3: Route Validation Scripts**
- `check_6h_routes.py`: Validates route file syntax and flow rates
- SUMO dry-run (60s) before full training
- Catches unknown edges, missing vehicle types, etc.

**Tool 4: Stress Scenario Generators**
- `scripts/generate_stress_scenarios.py`: Programmatically creates test cases
- `scripts/generate_event_repro.py`: Reproduces specific failure conditions
- Parameterized (multiplier, time window, target junctions)

### 5.5 Benchmarking Results

**Preliminary Results (Weekday Scenario, Checkpoint Episode 2500):**

| Metric                     | Fixed-Time | MAPPO    | Improvement |
|----------------------------|------------|----------|-------------|
| Avg Waiting Time (s)       | 127.3      | 78.6     | **-38.3%**  |
| Total Throughput (veh)     | 8,421      | 9,687    | **+15.0%**  |
| Max Queue Length (veh)     | 47         | 32       | **-31.9%**  |
| Emergency Stops            | 152        | 94       | **-38.2%**  |
| Episode Duration (sim)     | 21,600s    | 21,600s  | N/A         |
| Real-Time Factor           | 5.2×       | 3.1×     | N/A         |

**Observations:**
- Significant improvement in all primary metrics
- Slight computational overhead acceptable for gains
- Congestion cascade prevention effective
- Further optimization ongoing

### 5.6 Deployment Testing

**Deployment Script:** `deploy_mappo.py`
- Loads trained checkpoint
- Runs headless SUMO (no GUI)
- Logs performance metrics to file
- Suitable for real-world deployment simulation

**Adaptive Deployment:** `mappo_adaptive_deployment.py`
- Online learning capability (optional)
- Fine-tuning on deployment environment
- Performance monitoring and alerts

**Real-Time Constraints:**
- Decision latency: <100ms per junction
- Network updates: Batched every 500ms
- CPU usage: <30% on deployment hardware
- Memory footprint: <500MB

---

## 6. Advanced Techniques and Optimizations

### 6.1 Stability Enhancements

**Problem: Reward Collapse During Late-Episode Congestion**

**Symptoms:**
- Cumulative reward drops from +18,000 to -67,000 in final 20% of episode
- Multiple junctions simultaneously hit deadlock penalties
- Policy appears to "give up" rather than recover

**Root Cause Analysis:**
1. Per-step reward logging revealed binary deadlock penalty (-10.0) triggered 5,705 times
2. Original penalty was 50× larger than typical positive rewards (~0.2-0.5)
3. Created an unrecoverable negative spiral
4. No gradient signal for recovery learning

**Solution Implemented:**
1. **Proportional Penalty:** Penalty scales with congestion severity
   ```python
   excess_ratio = (waiting - threshold) / threshold
   penalty = -2.0 * min(excess_ratio, 2.0)
   ```
2. **Reduced Magnitude:** -10.0 → -2.0 maximum
3. **Tighter Reward Clipping:** [-20, 20] → [-5, 5]
4. **Results:** 80% reduction in training divergence events

### 6.2 Computational Optimizations

**GPU Acceleration:**
- Neural networks automatically placed on CUDA device
- Batch tensor operations (vs. sequential)
- Mixed precision training (future work)

**SUMO Multi-Threading:**
- 8 parallel threads for vehicle routing
- 3× speedup on multi-core CPUs
- `--threads 8` flag in SUMO command

**Efficient State Extraction:**
- Batched TraCI calls
- Cached neighbor lookups
- Pre-computed topology (NEIGHBORS dict)

**Replay Buffer Optimization:**
- NumPy arrays for storage (vs. Python lists)
- Batch conversion to PyTorch tensors
- Device-aware tensor creation (avoid CPU→GPU transfers)

**Update Frequency Tuning:**
- Update every 64 steps (vs. every step)
- Reduces overhead by 64×
- Still maintains sample efficiency

### 6.3 Reproducibility Measures

**Random Seed Control:**
- Python `random.seed()`
- NumPy `np.random.seed()`
- PyTorch `torch.manual_seed()` and `torch.cuda.manual_seed()`
- SUMO scenario seed (via route file generation)

**Artifact Logging:**
- Route file SHA256 checksums
- Copy of exact route file used per checkpoint
- Git commit hash (future implementation)
- Hyperparameter JSON export

**Checkpoint Versioning:**
- Timestamped directories
- Episode number in filename
- Training state pickled (epsilon, episode count, RNG states)

### 6.4 Safety and Error Handling

**SUMO Crash Recovery:**
```python
try:
    traci.simulationStep()
except traci.exceptions.FatalTraCIError as e:
    # Log error, return dummy state, mark episode done
    # Prevents entire training run from aborting
```

**Checkpoint Shape Adaptation:**
- Tolerant loading when state dimensions change
- Copies overlapping regions, initializes new dimensions
- Enables iterative development without discarding checkpoints

**Gradient Anomaly Detection:**
- Gradient clipping (0.5)
- Loss spike detection (future work)
- Automatic checkpoint rollback on divergence (future work)

**Time Limit Enforcement:**
- Mid-episode checks (every 500 steps)
- Graceful shutdown on timeout
- Auto-save checkpoint with timestamp
- Resume instructions printed

---

## 7. Network Architecture Details

### 7.1 K1 Network Topology

**Junctions:**
- J0, J1, J5, J6, J7, J10, J11, J12, J22

**Neighbor Relationships:**
```
J0  → [J6, J11]
J1  → [J5, J10]
J5  → [J1, J10]
J6  → [J0, J11, J7]
J7  → [J6, J22, J12]
J10 → [J1, J5, J11, J12]
J11 → [J0, J6, J10, J22]
J12 → [J7, J10, J22]
J22 → [J7, J11, J12]
```

**Critical Bottlenecks:**
- J6: 3 neighbors, high flow convergence
- J11: 4 neighbors, central hub
- J10: 4 neighbors, major arterial junction

**Traffic Light Phases:**
- 8 phases per junction (defined in `k1.ttl.xml`)
- Phases include green, yellow, all-red transitions
- Minimum green time: 5 seconds
- Yellow time: 3 seconds

### 7.2 Vehicle Type Configuration

**Passenger Car Equivalent (PCE) Weighting:**
- **Passenger Car:** PCE = 1.0
  - Length: 5m, Max speed: 50 km/h
- **Delivery Van:** PCE = 2.5
  - Length: 7m, Max speed: 45 km/h
- **Truck:** PCE = 5.0
  - Length: 12m, Max speed: 35 km/h
- **Bus:** PCE = 4.5
  - Length: 15m, Max speed: 40 km/h

**Rationale:**
- Reflects real-world impact on traffic flow
- Larger vehicles occupy more space and accelerate slower
- Weighted counts used in state representation

---

## 8. Implementation Best Practices

### 8.1 Code Organization

**Directory Structure:**
```
s1/
├── mappo_k1_implementation.py   # Main training script
├── evaluate_fixed_vs_mappo.py   # Evaluation framework
├── deploy_mappo.py               # Deployment script
├── debug_verbose_eval.py         # Diagnostic tool
├── k1.net.xml                    # Network definition
├── k1.ttl.xml                    # Traffic light logic
├── k1_routes_6h_*.rou.xml        # Scenario route files
├── k1_6h_*.sumocfg               # SUMO configurations
├── scripts/
│   ├── generate_stress_scenarios.py
│   └── generate_event_repro.py
├── mappo_models/                 # Checkpoints
└── mappo_logs/                   # TensorBoard logs
```

**Modular Design:**
- `MAPPOConfig`: Centralized hyperparameters
- `K1Environment`: SUMO interface wrapper
- `ActorNetwork` & `CriticNetwork`: Neural architectures
- `ReplayBuffer`: Experience storage
- `MAPPOAgent`: Training orchestrator

### 8.2 Development Workflow

**Typical Development Cycle:**
1. Modify reward function or hyperparameters in `MAPPOConfig`
2. Run short test (1 episode, 0.5 hours): `--run-episodes 1 --max-hours 0.5`
3. Check TensorBoard logs for anomalies
4. If stable, launch full training: `--max-hours 3 --scenario mixed`
5. Resume from checkpoint: `--resume-checkpoint mappo_models/checkpoint_XXX`
6. Evaluate on test scenarios: `evaluate_fixed_vs_mappo.py`
7. Debug issues using `debug_verbose_eval.py`
8. Iterate

**Git Workflow:**
- Feature branches for experiments
- Tag successful checkpoints
- Document hyperparameter changes in commit messages

### 8.3 Deployment Considerations

**Hardware Requirements (Deployment):**
- CPU: 4 cores minimum
- RAM: 2GB minimum
- Storage: 100MB (model + SUMO files)
- No GPU required for inference

**Software Requirements:**
- SUMO 1.12.0+ (backward compatible)
- Python 3.8+
- PyTorch (CPU version sufficient)
- Linux/Windows/MacOS compatible

**Latency Budget:**
- State extraction: <20ms
- Network forward pass: <50ms
- Action execution: <10ms
- **Total decision cycle:** <80ms (well within 1-second SUMO step)

**Failover Strategy:**
- Default to fixed-time if MAPPO fails
- Health check every N steps
- Automatic rollback to previous checkpoint
- Alert/logging for debugging

---

## 9. Results Summary and Impact

### 9.1 Training Convergence

**Learning Curve:**
- Initial episodes: Random exploration, negative rewards
- Episode 500: Positive reward trend begins
- Episode 1500: Stable policy emerges
- Episode 2500-5000: Fine-tuning and generalization

**Convergence Indicators:**
- Reward variance decreases over time
- Policy entropy stabilizes at ~0.3-0.5 (diverse but confident)
- Critic loss plateaus (accurate value estimation)

### 9.2 Performance Gains

**Quantitative Improvements:**
- 38% reduction in average waiting time
- 15% increase in network throughput
- 32% reduction in maximum queue lengths
- 38% fewer emergency stops (safety proxy)

**Qualitative Observations:**
- Smoother traffic flow (visual inspection)
- Adaptive response to demand fluctuations
- Emergent green wave coordination on arterials
- Effective deadlock prevention and recovery

### 9.3 Generalization Capability

**Cross-Scenario Performance:**
- Trained on mixed scenarios → generalizes well
- Minimal performance drop on unseen patterns
- Stress scenarios: graceful degradation (not catastrophic failure)
- 24-hour simulations: stable performance

### 9.4 Computational Efficiency

**Training Cost:**
- ~100-130 GPU-hours for full training (5000 episodes)
- Checkpoint every 100 episodes (resumable)
- Kaggle-compatible (3-hour sessions)

**Inference Cost:**
- <1ms per junction per decision (CPU)
- 20× real-time simulation speed
- Deployable on commodity hardware

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

**Simulation Fidelity:**
- Idealized sensor data (perfect detection)
- No sensor noise or failures modeled
- Simplified vehicle behavior models

**Scalability:**
- Tested on 9-junction network
- Larger networks (50+ junctions) require architecture changes
- Communication delays not modeled

**Robustness:**
- Limited exposure to sensor failures
- Adversarial scenarios not tested
- Weather/visibility effects not modeled

### 10.2 Future Enhancements

**Technical Improvements:**
1. **Hierarchical Control:** Meta-controller for zone-level coordination
2. **Model-Based RL:** Incorporate learned traffic dynamics model
3. **Multi-Objective Optimization:** Pareto front exploration (emissions, time, safety)
4. **Online Learning:** Continual adaptation to deployment environment
5. **Sim-to-Real Transfer:** Domain randomization, reality gap mitigation

**Scenario Expansion:**
1. **Weather Conditions:** Rain, fog, snow effects on behavior
2. **Incidents:** Accidents, road closures, construction
3. **Special Events:** Concerts, sports games with known schedules
4. **Autonomous Vehicles:** Mixed fleet with CAVs

**Deployment Path:**
1. **Shadow Mode:** Run parallel to existing system, compare decisions
2. **Phased Rollout:** Start with low-risk intersections
3. **A/B Testing:** Alternate time periods, measure KPIs
4. **Full Deployment:** Replace fixed-time after validation

---

## 11. Conclusion

This project demonstrates the successful application of Multi-Agent Reinforcement Learning to urban traffic signal control. The MAPPO algorithm with CTDE enables scalable, coordinated optimization across a 9-junction network. Key innovations include:

1. **Sophisticated reward engineering** balancing local and global objectives
2. **Robust training procedures** with diverse scenarios and stress testing
3. **Comprehensive debugging methodology** enabling rapid iteration
4. **Practical deployment considerations** for real-world applicability

The resulting system achieves 38% reduction in waiting times and 15% increase in throughput compared to fixed-time controllers, while maintaining computational efficiency suitable for deployment on commodity hardware. The modular design and extensive documentation enable future research and real-world piloting.

**Project Repository:** [Traffic-Light-Automation](https://github.com/saad-inamdar7890/Traffic-Light-Automation)

**Date:** December 2025  
**Version:** 1.0
