# Dynamic Traffic Light Automation System

A modular traffic light simulation system with dynamic flow management and adaptive control.

## 📁 File Structure

### Core Modules

1. **`main_simulation.py`** - Main entry point that orchestrates all components
2. **`dynamic_flow_manager.py`** - Handles random traffic flow generation and time-based patterns
3. **`dynamic_traffic_light.py`** - Manages adaptive traffic light control based on real-time conditions
4. **`results_analyzer.py`** - Collects data, performs analysis, and generates reports

### Legacy Files

- **`dynamic_flow.py`** - Original comprehensive implementation
- **`dynamic_flow_simple.py`** - Simplified version for testing

## 🚀 Quick Start

### Run the Modular Simulation
```bash
python main_simulation.py
```

### Run Individual Components (for testing)
```bash
# Test dynamic flow manager
python -c "from dynamic_flow_manager import DynamicFlowManager; dfm = DynamicFlowManager(); print('Flow manager loaded successfully')"

# Test traffic light controller
python -c "from dynamic_traffic_light import AdaptiveTrafficController; atc = AdaptiveTrafficController(); print('Traffic controller loaded successfully')"

# Test results analyzer
python -c "from results_analyzer import TrafficAnalyzer; ta = TrafficAnalyzer(); print('Results analyzer loaded successfully')"
```

## 🔧 Components Overview

### 1. Dynamic Flow Manager (`dynamic_flow_manager.py`)
**Purpose**: Generate realistic, time-varying traffic flows
**Features**:
- Time-based traffic patterns (rush hour, night, etc.)
- Random variations (±40%)
- Flow rate constraints (50-2000 vehicles/hour)
- Updates every 2 minutes

**Key Methods**:
- `update_flow_rates(step)` - Updates flow rates based on time and randomness
- `get_flow_summary()` - Returns current flow statistics
- `apply_flow_changes()` - Applies changes to simulation

### 2. Adaptive Traffic Light Controller (`dynamic_traffic_light.py`)
**Purpose**: Optimize traffic light timing based on real-time conditions
**Features**:
- Traffic pressure calculation
- Adaptive phase duration (10-60 seconds)
- Direction-based optimization
- Historical data tracking

**Key Methods**:
- `calculate_traffic_pressure(data, directions)` - Calculates traffic pressure
- `analyze_traffic_state(data)` - Provides timing recommendations
- `apply_adaptive_control(data, step)` - Main control logic

### 3. Results Analyzer (`results_analyzer.py`)
**Purpose**: Collect data, analyze performance, and generate reports
**Features**:
- Real-time metrics collection
- Performance rating system
- Edge efficiency analysis
- Comprehensive reporting

**Key Methods**:
- `collect_traffic_metrics(step, traci)` - Collects data from SUMO
- `display_real_time_analysis(...)` - Shows live analysis
- `generate_comprehensive_report(...)` - Creates final report

### 4. Main Simulation (`main_simulation.py`)
**Purpose**: Orchestrate all components and manage simulation lifecycle
**Features**:
- Component integration
- Simulation timing control
- Error handling
- Final insights generation

## 📊 Output Examples

### Real-time Status (every minute)
```
Step  300 | Time:  0.1h | Vehicles: 157 | NS:  89.3 | EW: 112.7 | Phase 3 (EW_Green)
```

### Detailed Analysis (every 5 minutes)
```
🚦 REAL-TIME TRAFFIC ANALYSIS - Step 300 (Hour: 0.1)
📊 Network Status:
   Total Vehicles: 157
   Vehicles Waiting: 89
   Average Waiting Time: 24.5s
🎯 Traffic Pressure Analysis:
   North-South Pressure: 89.3
   East-West Pressure: 112.7
💡 Adaptive Recommendations:
   Suggested NS Time: 22s
   Suggested EW Time: 28s
   Priority: 🔴 PRIORITIZE EAST-WEST
```

### Final Report
```
🎯 COMPREHENSIVE TRAFFIC SIMULATION REPORT
⏱️  WAITING TIME PERFORMANCE:
   Average: 28.5s | Median: 26.0s | Max: 78.0s | Min: 2.0s
   Overall Rating: 🟡 GOOD
📊 TRAFFIC PRESSURE ANALYSIS:
   North-South: Avg 95.2 | Peak 156.7
   East-West: Avg 108.4 | Peak 189.3
   Recommendation: 🔴 Consider longer EW green phases
```

## 🎯 Key Features

### Dynamic Behavior
- **Flow Rates**: Change every 2 minutes based on time of day and random factors
- **Traffic Lights**: Adapt timing based on real-time traffic pressure
- **Analysis**: Continuous monitoring with recommendations

### Performance Metrics
- Average/median/max waiting times
- Traffic pressure by direction
- Edge efficiency analysis
- Flow rate variations
- Balance recommendations

### Time-based Patterns
- **Night** (22:00-06:00): 30% of base traffic
- **Morning Rush** (07:00-09:00): 180% of base traffic
- **Day** (10:00-16:00): 100% of base traffic
- **Evening Rush** (17:00-19:00): 160% of base traffic
- **Evening** (20:00-21:00): 70% of base traffic

## 🛠️ Customization

### Modify Traffic Patterns
Edit `time_patterns` in `dynamic_flow_manager.py`:
```python
self.time_patterns = {
    'custom_period': {'factor': 1.5, 'hours': [14, 15, 16]},
    # Add your patterns here
}
```

### Adjust Traffic Light Parameters
Edit parameters in `dynamic_traffic_light.py`:
```python
self.min_green_time = 15  # Minimum green duration
self.max_green_time = 90  # Maximum green duration
```

### Change Analysis Intervals
Edit intervals in `main_simulation.py`:
```python
self.analysis_interval = 600    # Detailed analysis every 10 minutes
self.status_interval = 30       # Brief status every 30 seconds
```

## 📋 Requirements

- Python 3.7+
- SUMO (Simulation of Urban Mobility)
- SUMO_HOME environment variable set
- TraCI (included with SUMO)

## 🔍 Troubleshooting

1. **"SUMO_HOME not set"**: Ensure SUMO is installed and environment variable is configured
2. **"Connection error"**: Check if SUMO GUI is already running
3. **"Config file not found"**: Ensure `demo.sumocfg` and related files are in the directory
4. **Import errors**: Make sure all module files are in the same directory

## 📈 Future Enhancements

- Real traffic data integration
- Machine learning-based optimization
- Multi-intersection coordination
- Real-time web dashboard
- Historical data persistence
