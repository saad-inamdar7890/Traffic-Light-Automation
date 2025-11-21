# Dynamic Traffic Flow System - Quick Start Guide

## Overview
The dynamic flow system allows you to easily change traffic scenarios without manually editing XML files.

## Quick Start

### 1. List Available Scenarios
```powershell
cd s1
python dynamic_flow_generator.py --list
```

### 2. Generate a Scenario
```powershell
# Generate light traffic (for testing)
python dynamic_flow_generator.py --scenario uniform_light

# Generate heavy traffic
python dynamic_flow_generator.py --scenario uniform_heavy

# Generate morning rush hour
python dynamic_flow_generator.py --scenario morning_rush

# Generate evening rush hour
python dynamic_flow_generator.py --scenario evening_rush

# Generate full 24-hour simulation
python dynamic_flow_generator.py --scenario custom_24h
```

### 3. Run the Simulation
```powershell
sumo-gui k1.sumocfg
```

## Available Scenarios

### Test Scenarios (1 hour each)
- **uniform_light** - Light traffic, 100 veh/hour per route
- **uniform_medium** - Medium traffic, 200 veh/hour per route
- **uniform_heavy** - Heavy traffic, 300 veh/hour per route

### Rush Hour Scenarios (2 hours each)
- **morning_rush** - Simulates morning commute (residential → commercial)
  - High flow from Southwest/North to East/Center
  - Low flow in reverse direction
  
- **evening_rush** - Simulates evening commute (commercial → residential)
  - High flow from East/Center to Southwest
  - Low flow in reverse direction

### Full Day Scenario (24 hours)
- **custom_24h** - Complete day with varying traffic
  - Night (00:00-06:00): 20% of normal traffic
  - Morning Rush (06:00-09:00): 300% of normal traffic
  - Midday (09:00-17:00): 100% of normal traffic
  - Evening Rush (17:00-19:00): 300% of normal traffic
  - Evening (19:00-22:00): 60% of normal traffic
  - Late Night (22:00-24:00): 20% of normal traffic

## Customizing Scenarios

### Method 1: Modify the Python Script
Edit `dynamic_flow_generator.py` and add your scenario in the `_define_scenarios()` method:

```python
"my_custom_scenario": {
    "name": "My Custom Scenario",
    "description": "Description of what this tests",
    "duration": 3600,  # in seconds
    "default_flow": 200,  # vehicles per hour
    "flow_multipliers": {
        "sw_to_e": 2.0,    # Double the flow for this route
        "n_to_center": 1.5, # 1.5x flow for this route
        "e_center_to_sw": 0.5,  # Half the flow
    }
},
```

### Method 2: Python Script
```python
from dynamic_flow_generator import TrafficFlowGenerator

gen = TrafficFlowGenerator()

# Add custom scenario
gen.add_custom_scenario(
    name="weekend_light",
    description="Light weekend traffic",
    duration=7200,
    default_flow=80,
    flow_multipliers={
        "sw_to_e": 1.5,
        "w_to_e": 1.5
    }
)

# Generate routes
gen.generate_routes_file("weekend_light")
```

## Understanding Route Names

Routes are named as `{origin}_to_{destination}`:

### Origins
- **sw** = Southwest (J19, entry -E15)
- **w** = West (J13, entry -E9)
- **n** = North (J14, entry -E10)
- **nw** = Northwest (J15, entry -E11)
- **e_center** = East Center (J11, entry -E12)
- **e_north** = East North (J17, entry -E13)
- **se** = Southeast (J23, entry -E24)
- **e_edge** = East Edge (J22, entry -E21)
- **s** = South (J8, entry -E6)

### Destinations
- Same as origins (where the flow exits)

### Examples
- `sw_to_e` = Traffic from Southwest (J19) to East (J22)
- `n_to_center` = Traffic from North (J14) to Center (J11)
- `e_center_to_sw` = Traffic from East Center (J11) to Southwest (J19)

## Flow Multipliers

The `flow_multipliers` dictionary lets you adjust specific routes:
- `1.0` = Normal flow (default)
- `2.0` = Double the flow
- `0.5` = Half the flow
- `3.0` = Triple the flow

Example:
```python
"flow_multipliers": {
    "sw_to_e": 3.0,     # Very heavy traffic from southwest to east
    "e_center_to_sw": 0.3,  # Very light reverse flow
}
```

## Testing Your Scenarios

### Quick Validation
```powershell
# Generate scenario
python dynamic_flow_generator.py --scenario uniform_light

# Run for 10 minutes to verify
sumo-gui k1.sumocfg
# In SUMO GUI: Simulation > Run, observe for ~600 seconds
```

### Check Flow Statistics
After running simulation, check:
1. Vehicle spawn rates at entry points
2. Queue formation at junctions
3. Total vehicles in simulation
4. Waiting times at traffic lights

## Tips

1. **Start Small**: Test with `uniform_light` first
2. **Gradual Increase**: Move from light → medium → heavy
3. **Rush Hours**: Use rush scenarios to stress-test your RL algorithm
4. **24-Hour**: Use for final comprehensive testing

## Troubleshooting

### "No connection between edges" error
- The route definitions in the script use validated paths
- If you see this, check if you modified the network structure

### Too many/few vehicles
- Adjust `default_flow` in scenario (vehicles per hour)
- Modify `flow_multipliers` for specific routes

### Simulation too slow
- Reduce `default_flow`
- Use shorter `duration`
- Test with `uniform_light` first

## Example Workflow

```powershell
# 1. Test with light traffic (5 minutes)
python dynamic_flow_generator.py --scenario uniform_light
sumo-gui k1.sumocfg

# 2. If OK, try medium traffic
python dynamic_flow_generator.py --scenario uniform_medium
sumo-gui k1.sumocfg

# 3. Test morning rush pattern
python dynamic_flow_generator.py --scenario morning_rush
sumo-gui k1.sumocfg

# 4. Run full 24-hour simulation for RL training
python dynamic_flow_generator.py --scenario custom_24h
# Use this for baseline metrics and RL training
```

## Next Steps

After you're comfortable with the scenarios:
1. Create custom scenarios for your specific tests
2. Use different scenarios for RL training phases
3. Compare baseline vs RL performance across scenarios
4. Document which scenarios show best improvement
