# 24-Hour Traffic Scenario Training and Evaluation Guide

This guide explains how to train and evaluate the MAPPO model on full 24-hour traffic scenarios.

## Overview

24-hour scenarios present unique challenges compared to shorter (3h, 6h) scenarios:

| Challenge | Description | Solution |
|-----------|-------------|----------|
| Multiple Rush Hours | Morning (6-9am) and Evening (5-8pm) peaks | Time-adaptive reward normalization |
| Long Episodes | 86,400 steps per episode | Periodic checkpointing, memory-efficient updates |
| Day-Night Transitions | Low to high traffic changes | Gradual exploration decay |
| Memory Usage | Large replay buffers | Less frequent updates (every 128 steps) |

## Available Scenarios

| Scenario | Description | Peak Times |
|----------|-------------|------------|
| `weekday` | Standard workday pattern | 8-9am, 5-8pm |
| `weekend` | Relaxed pattern, later peaks | 11am-2pm, 7-9pm |
| `friday` | Extended evening rush | 8-9am, 5-10pm |
| `event` | Major event causing surge | 2-4pm, 5-8pm (extreme) |

## Quick Start

### Step 1: Generate Route Files

```powershell
cd s1

# Generate a specific scenario
python generate_24h_routes.py --scenario weekday

# Or generate all scenarios
python generate_24h_routes.py --scenario all
```

This creates:
- `k1_routes_24h_{scenario}.rou.xml` - Route definitions
- `k1_24h_{scenario}.sumocfg` - SUMO configuration

### Step 2: Train the Model

```powershell
# Basic training (100 episodes)
python train_24h_scenario.py --scenario weekday

# Train for limited time (e.g., 6 real-world hours)
python train_24h_scenario.py --scenario weekday --max-hours 6

# Resume from checkpoint
python train_24h_scenario.py --scenario weekday --resume-checkpoint checkpoint_24h_weekday_ep10

# Generate routes automatically before training
python train_24h_scenario.py --scenario weekday --generate-routes
```

**Expected Training Time:**
- 1 episode (24h simulation): ~45-90 minutes on CPU, ~15-30 min on GPU
- 100 episodes: ~75-150 hours on CPU, ~25-50 hours on GPU

### Step 3: Evaluate Performance

```powershell
# Full 24-hour evaluation
python evaluate_24h_performance.py --checkpoint checkpoint_24h_weekday_final --scenario weekday

# Quick 9-hour evaluation (includes morning rush)
python evaluate_24h_performance.py --checkpoint checkpoint_24h_weekday_final --scenario weekday --quick

# Compare with existing 6h model on 24h scenario
python evaluate_fixed_vs_mappo.py --checkpoint checkpoint_time_20251129_103621 --scenario 24h_weekday --duration 86400
```

## Output Files

### Training Outputs
```
mappo_models_24h/
├── checkpoint_24h_weekday_ep0/
│   ├── actors_*.pt          # Actor network weights
│   ├── critic.pt            # Critic network weights
│   └── metadata.json        # Training info
├── checkpoint_24h_weekday_ep1/
└── checkpoint_24h_weekday_final/
```

### Evaluation Outputs
```
evaluation_24h_results/
├── metrics_24h.csv           # Step-by-step metrics
├── summary_24h.json          # Aggregate statistics
└── evaluation_24h_comparison.png  # Visualization
```

## Key Metrics Explained

### Overall Metrics
- **Avg Waiting Time**: Mean total waiting time across all junctions
- **Avg Queue Length**: Mean vehicles stopped at intersections
- **Total Throughput**: Vehicles that completed their journey

### Time Period Analysis
The evaluation breaks down performance by time period:
1. Night (12am-6am): Low traffic baseline
2. Morning Rush (6am-9am): Critical stress test
3. Midday (9am-2pm): Moderate traffic
4. Afternoon (2pm-5pm): Building toward rush
5. Evening Rush (5pm-8pm): Maximum stress test
6. Late Night (8pm-12am): Declining traffic

### Rush Hour Performance
Special focus on morning and evening rush hours where:
- Traffic volume is 3-5× higher than off-peak
- Fixed-time controllers typically struggle
- MAPPO adaptation benefits are most visible

## Configuration Details

### Training Parameters (24H-Specific)

```python
# From train_24h_scenario.py
STEPS_PER_EPISODE = 86400    # 24 hours
UPDATE_FREQUENCY = 128        # Less frequent (memory efficiency)
PPO_EPOCHS = 10               # Fewer epochs per update
EPSILON_START = 0.25          # More initial exploration
EPSILON_DECAY = 0.999         # Slower decay
LEARNING_RATE_ACTOR = 3e-4    # More conservative
MAX_WAITING_THRESHOLD = 750.0 # Higher threshold for 24h variation
```

### Fixed-Time Controller (Time-Adaptive)

The 24h fixed-time controller uses time-adaptive phase durations:

| Time Period | Phase Duration | Rationale |
|-------------|---------------|-----------|
| 12am-6am (Night) | 45s | Less traffic, longer cycles OK |
| 6am-9am (Rush) | 25s | Responsive to high demand |
| 9am-5pm (Day) | 30s | Standard timing |
| 5pm-8pm (Rush) | 25s | Responsive to high demand |
| 8pm-12am | 30s | Standard timing |

This is a **strong baseline** - not a naive fixed-time controller.

## Tips for Best Results

### Training Tips

1. **Start with shorter scenarios** (6h) to validate model before 24h
2. **Use GPU** if available (10-20× faster)
3. **Monitor TensorBoard** for early signs of issues:
   ```powershell
   tensorboard --logdir mappo_logs_24h
   ```
4. **Use mixed scenario training** for robust generalization:
   ```powershell
   # Train on all scenarios
   for scenario in weekday weekend friday event; do
       python train_24h_scenario.py --scenario $scenario --episodes 25
   done
   ```

### Evaluation Tips

1. **Always use same random seed** for fair comparison (`--seed 42`)
2. **Run multiple seeds** for statistical significance:
   ```powershell
   for seed in 42 123 456 789 1000; do
       python evaluate_24h_performance.py --checkpoint <cp> --scenario weekday --seed $seed
   done
   ```
3. **Focus on rush hours** - that's where MAPPO shines

### Debugging

If training seems stuck:
1. Check TensorBoard for reward trends
2. Verify SUMO simulation runs: `sumo-gui -c k1_24h_weekday.sumocfg`
3. Review hourly reward breakdown in console output
4. Consider reducing episode length for initial debugging

## File Structure Summary

```
s1/
├── generate_24h_routes.py      # Creates route files
├── train_24h_scenario.py       # 24h training script
├── evaluate_24h_performance.py # 24h evaluation script
├── k1_routes_24h_weekday.rou.xml   # Generated routes
├── k1_24h_weekday.sumocfg          # SUMO config
├── mappo_models_24h/               # Checkpoints
└── evaluation_24h_results/         # Results
```

## Expected Performance

Based on 6h training results, expected 24h improvements:

| Metric | Expected Improvement |
|--------|---------------------|
| Overall Waiting Time | -30% to -40% |
| Rush Hour Waiting Time | -35% to -50% |
| Total Throughput | +10% to +20% |
| Peak Queue Length | -25% to -35% |

**Note:** Actual results depend on training duration and scenario complexity.
