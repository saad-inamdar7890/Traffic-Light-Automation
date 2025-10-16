"""
Traffic Light Test Scenarios Simulation
Tests adaptive traffic control under various real-world traffic conditions

Scenarios:
1. Heavy Traffic in One Direction (North-South congested, East-West free)
2. Light Traffic in Three Lanes (Only East lane free)
3. Sudden Traffic Spike (Normal flow with sudden surge in one direction)
4. Low Traffic Overall (Light traffic in all directions)

Each scenario runs: 10 min Normal Mode → 10 min Adaptive Mode
Total simulation time: 80 minutes (4 scenarios × 20 minutes each)
"""

import os
import sys
import traci
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
import time
import random

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

# Import our modular components
from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class ScenarioFlowManager:
    """Specialized flow manager for test scenarios"""
    
    def __init__(self):
        """Initialize scenario-based flow manager"""
        self.current_scenario = None
        self.scenario_start_time = 0
        self.flow_configs = {}
        self.base_flows = self._create_base_flows()
        
    def _create_base_flows(self):
        """Create base flow configurations"""
        return {
            # Car flows - base rates for normal conditions
            'f_0_cars': {'from': 'E0', 'to': 'E0.319', 'base_rate': 100, 'current_rate': 100, 'vtype': 'car'},
            'f_1_cars': {'from': 'E0', 'to': '-E1.238', 'base_rate': 50, 'current_rate': 50, 'vtype': 'car'},
            'f_2_cars': {'from': 'E0', 'to': 'E1.200', 'base_rate': 50, 'current_rate': 50, 'vtype': 'car'},
            'f_3_cars': {'from': '-E1', 'to': '-E1.238', 'base_rate': 100, 'current_rate': 100, 'vtype': 'car'},
            'f_4_cars': {'from': '-E1', 'to': '-E0.254', 'base_rate': 50, 'current_rate': 50, 'vtype': 'car'},
            'f_5_cars': {'from': '-E1', 'to': 'E0.319', 'base_rate': 50, 'current_rate': 50, 'vtype': 'car'},
            'f_6_cars': {'from': '-E0', 'to': '-E1.238', 'base_rate': 50, 'current_rate': 50, 'vtype': 'car'},
            'f_7_cars': {'from': '-E0', 'to': 'E1.200', 'base_rate': 50, 'current_rate': 50, 'vtype': 'car'},
            'f_8_cars': {'from': 'E1', 'to': '-E0.254', 'base_rate': 50, 'current_rate': 50, 'vtype': 'car'},
            'f_9_cars': {'from': 'E1', 'to': 'E0.319', 'base_rate': 50, 'current_rate': 50, 'vtype': 'car'},
            'f_10_cars': {'from': 'E1', 'to': 'E1.200', 'base_rate': 100, 'current_rate': 100, 'vtype': 'car'},
            'f_11_cars': {'from': '-E0', 'to': '-E0.254', 'base_rate': 100, 'current_rate': 100, 'vtype': 'car'},
            
            # Motorcycle flows
            'f_0_bikes': {'from': 'E0', 'to': 'E0.319', 'base_rate': 60, 'current_rate': 60, 'vtype': 'motorcycle'},
            'f_1_bikes': {'from': 'E0', 'to': '-E1.238', 'base_rate': 30, 'current_rate': 30, 'vtype': 'motorcycle'},
            'f_2_bikes': {'from': 'E0', 'to': 'E1.200', 'base_rate': 30, 'current_rate': 30, 'vtype': 'motorcycle'},
            'f_3_bikes': {'from': '-E1', 'to': '-E1.238', 'base_rate': 60, 'current_rate': 60, 'vtype': 'motorcycle'},
            'f_4_bikes': {'from': '-E1', 'to': '-E0.254', 'base_rate': 30, 'current_rate': 30, 'vtype': 'motorcycle'},
            'f_5_bikes': {'from': '-E1', 'to': 'E0.319', 'base_rate': 30, 'current_rate': 30, 'vtype': 'motorcycle'},
            'f_6_bikes': {'from': '-E0', 'to': '-E1.238', 'base_rate': 30, 'current_rate': 30, 'vtype': 'motorcycle'},
            'f_7_bikes': {'from': '-E0', 'to': 'E1.200', 'base_rate': 30, 'current_rate': 30, 'vtype': 'motorcycle'},
            'f_8_bikes': {'from': 'E1', 'to': '-E0.254', 'base_rate': 30, 'current_rate': 30, 'vtype': 'motorcycle'},
            'f_9_bikes': {'from': 'E1', 'to': 'E0.319', 'base_rate': 30, 'current_rate': 30, 'vtype': 'motorcycle'},
            'f_10_bikes': {'from': 'E1', 'to': 'E1.200', 'base_rate': 60, 'current_rate': 60, 'vtype': 'motorcycle'},
            'f_11_bikes': {'from': '-E0', 'to': '-E0.254', 'base_rate': 60, 'current_rate': 60, 'vtype': 'motorcycle'}
        }
    
    def set_scenario(self, scenario_name, step):
        """Set current traffic scenario"""
        self.current_scenario = scenario_name
        self.scenario_start_time = step
        self.flow_configs = self.base_flows.copy()
        self._apply_scenario_modifications()
        
    def _apply_scenario_modifications(self):
        """Apply scenario-specific traffic modifications"""
        if self.current_scenario == "heavy_one_direction":
            self._setup_heavy_one_direction()
        elif self.current_scenario == "light_three_lanes":
            self._setup_light_three_lanes()
        elif self.current_scenario == "sudden_spike":
            self._setup_sudden_spike()
        elif self.current_scenario == "low_traffic_all":
            self._setup_low_traffic_all()
    
    def _setup_heavy_one_direction(self):
        """Scenario 1: Heavy traffic North-South, light East-West"""
        print("🚦 SCENARIO 1: Heavy North-South Traffic, Light East-West")
        
        # Increase North-South flows significantly
        ns_multiplier = 4.0
        ew_multiplier = 0.3
        
        for flow_id, config in self.flow_configs.items():
            if config['from'] in ['E1', '-E1']:  # North-South traffic
                config['current_rate'] = int(config['base_rate'] * ns_multiplier)
            elif config['from'] in ['E0', '-E0']:  # East-West traffic
                config['current_rate'] = int(config['base_rate'] * ew_multiplier)
    
    def _setup_light_three_lanes(self):
        """Scenario 2: Light traffic in three directions, one direction free"""
        print("🚦 SCENARIO 2: Light Traffic in Three Lanes, East Lane Free")
        
        for flow_id, config in self.flow_configs.items():
            if config['from'] == 'E0':  # East lane - make it free
                config['current_rate'] = int(config['base_rate'] * 0.1)
            else:  # Other three directions - light traffic
                config['current_rate'] = int(config['base_rate'] * 0.6)
    
    def _setup_sudden_spike(self):
        """Scenario 3: Normal flow with sudden spikes"""
        print("🚦 SCENARIO 3: Normal Flow with Sudden Traffic Spikes")
        
        # Start with normal flows
        for flow_id, config in self.flow_configs.items():
            config['current_rate'] = config['base_rate']
    
    def _setup_low_traffic_all(self):
        """Scenario 4: Low traffic in all directions"""
        print("🚦 SCENARIO 4: Low Traffic in All Directions")
        
        low_multiplier = 0.4
        for flow_id, config in self.flow_configs.items():
            config['current_rate'] = int(config['base_rate'] * low_multiplier)
    
    def update_flows(self, step):
        """Update flows based on current scenario and time"""
        if self.current_scenario == "sudden_spike":
            self._handle_sudden_spike(step)
        
        # Add some random variation (±15%)
        for flow_id, config in self.flow_configs.items():
            base_rate = config['current_rate']
            variation = random.uniform(0.85, 1.15)
            config['current_rate'] = max(10, int(base_rate * variation))
    
    def _handle_sudden_spike(self, step):
        """Handle sudden traffic spikes for scenario 3"""
        elapsed = step - self.scenario_start_time
        
        # Create spikes at minutes 3, 6, and 8
        spike_times = [180, 360, 480]  # 3, 6, 8 minutes
        spike_duration = 60  # 1 minute spike
        
        for spike_time in spike_times:
            if spike_time <= elapsed < spike_time + spike_duration:
                # Create sudden spike in random direction
                if spike_time == 180:  # 3 min - North spike
                    direction = 'E1'
                elif spike_time == 360:  # 6 min - East spike
                    direction = 'E0'
                else:  # 8 min - South spike
                    direction = '-E1'
                
                for flow_id, config in self.flow_configs.items():
                    if config['from'] == direction:
                        config['current_rate'] = int(config['base_rate'] * 3.0)
                break
    
    def get_flow_summary(self):
        """Get current flow summary"""
        total_cars = sum(config['current_rate'] for config in self.flow_configs.values() 
                        if config['vtype'] == 'car')
        total_bikes = sum(config['current_rate'] for config in self.flow_configs.values() 
                         if config['vtype'] == 'motorcycle')
        
        return {
            'scenario': self.current_scenario,
            'total_cars': total_cars,
            'total_bikes': total_bikes,
            'total_vehicles': total_cars + total_bikes
        }

class TrafficScenarioSimulation:
    """Main simulation class for testing traffic scenarios"""
    
    def __init__(self, config_file="demo.sumocfg", junction_id="J4"):
        """Initialize scenario simulation"""
        self.config_file = config_file
        self.junction_id = junction_id
        self.sumo_cmd = ["sumo-gui", "-c", config_file]
        
        # Simulation timing
        self.scenario_duration = 600  # 10 minutes per mode
        self.total_scenario_time = 1200  # 20 minutes per scenario
        
        # Test scenarios
        self.scenarios = [
            "heavy_one_direction",
            "light_three_lanes", 
            "sudden_spike",
            "low_traffic_all"
        ]
        
        # Data collection
        self.scenario_results = {}
        
        # Initialize components
        self.flow_manager = ScenarioFlowManager()
        self.traffic_controller = AdaptiveTrafficController(junction_id)
        self.analyzer = TrafficAnalyzer()
    
    def run_all_scenarios(self):
        """Run all test scenarios with continuous phases"""
        print("🚦 TRAFFIC LIGHT TEST SCENARIOS SIMULATION")
        print("=" * 80)
        print("🔬 Testing 4 different traffic patterns:")
        print("   1. Heavy One Direction (NS heavy, EW light)")
        print("   2. Light Three Lanes (3 lanes light, 1 free)")
        print("   3. Sudden Traffic Spikes (Normal + random spikes)")
        print("   4. Low Traffic Overall (Light traffic everywhere)")
        print("📋 EXECUTION PLAN:")
        print("   🔴 Phase 1: All scenarios in NORMAL mode (40 min continuous)")
        print("   🧹 Clear Phase: Complete flow stop and network reset")
        print("   🟢 Phase 2: All scenarios in ADAPTIVE mode (40 min continuous)")
        print("📊 Total simulation time: 80 minutes")
        print("=" * 80)
        
        total_start_time = time.time()
        
        # Start SUMO once for the entire simulation
        traci.start(self.sumo_cmd)
        
        try:
            # Phase 1: All scenarios in NORMAL mode
            print(f"\n🔴 PHASE 1: NORMAL MODE - ALL SCENARIOS CONTINUOUS")
            print("=" * 70)
            normal_results = self._run_continuous_phase(adaptive=False)
            
            # Clear Phase: Complete network and flow reset
            print(f"\n🧹 CLEARING PHASE: Complete Flow Stop and Network Reset")
            print("=" * 70)
            self._complete_network_reset()
            
            # Phase 2: All scenarios in ADAPTIVE mode  
            print(f"\n🟢 PHASE 2: ADAPTIVE MODE - ALL SCENARIOS CONTINUOUS")
            print("=" * 70)
            adaptive_results = self._run_continuous_phase(adaptive=True)
            
            # Combine results
            self._combine_phase_results(normal_results, adaptive_results)
            
        except Exception as e:
            print(f"❌ Error in continuous simulation: {e}")
        
        finally:
            traci.close()
        
        total_end_time = time.time()
        print(f"\n🎉 ALL SCENARIOS COMPLETED!")
        print(f"⏱️  Total time: {(total_end_time - total_start_time)/60:.1f} minutes")
        
        # Generate comprehensive analysis
        self._generate_scenario_analysis()
        self._create_scenario_graphs()
        
    def _run_continuous_phase(self, adaptive=False):
        """Run all scenarios continuously in one phase"""
        mode_name = "ADAPTIVE" if adaptive else "NORMAL"
        phase_results = {}
        
        current_step = 0
        
        for i, scenario in enumerate(self.scenarios, 1):
            scenario_start_step = current_step
            scenario_end_step = current_step + self.scenario_duration
            
            print(f"\n🎯 {mode_name} - SCENARIO {i}/4: {scenario.upper().replace('_', ' ')}")
            print(f"   Time: {current_step/60:.1f}-{scenario_end_step/60:.1f} minutes")
            print("-" * 50)
            
            # Set scenario without clearing network
            self.flow_manager.set_scenario(scenario, current_step)
            
            # Initialize scenario data collection
            scenario_data = {
                'waiting_times': [], 'pressures_ns': [], 'pressures_ew': [],
                'vehicle_counts': [], 'avg_speeds': [], 'timestamps': []
            }
            
            # Run scenario for 10 minutes
            for step in range(scenario_start_step, scenario_end_step):
                traci.simulationStep()
                
                # Update flows based on scenario
                self.flow_manager.update_flows(step)
                
                # Collect traffic data
                step_data = self.analyzer.collect_traffic_metrics(step, traci)
                
                if step_data:
                    # Calculate pressure
                    ns_pressure = self.traffic_controller.calculate_traffic_pressure(
                        step_data, ['E1', '-E1']
                    )
                    ew_pressure = self.traffic_controller.calculate_traffic_pressure(
                        step_data, ['E0', '-E0']
                    )
                    
                    # Apply adaptive control if in adaptive phase
                    if adaptive:
                        self.traffic_controller.apply_adaptive_control(step_data, step)
                    
                    # Store data every 30 seconds
                    if step % 30 == 0:
                        scenario_data['waiting_times'].append(step_data['avg_waiting_time'])
                        scenario_data['pressures_ns'].append(ns_pressure)
                        scenario_data['pressures_ew'].append(ew_pressure)
                        scenario_data['vehicle_counts'].append(step_data['total_vehicles'])
                        scenario_data['timestamps'].append(step - scenario_start_step)  # Relative to scenario start
                        
                        # Calculate average speed
                        total_speed = 0
                        vehicle_count = 0
                        for edge_data in step_data['edge_data'].values():
                            if edge_data['vehicle_count'] > 0:
                                total_speed += edge_data['avg_speed'] * edge_data['vehicle_count']
                                vehicle_count += edge_data['vehicle_count']
                        
                        avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0
                        scenario_data['avg_speeds'].append(avg_speed)
                    
                    # Display progress every 2 minutes
                    if step % 120 == 0:
                        scenario_minute = (step - scenario_start_step) / 60
                        total_minute = step / 60
                        flow_summary = self.flow_manager.get_flow_summary()
                        print(f"   {mode_name} - S{i} {scenario_minute:4.1f}min (Total: {total_minute:5.1f}min) | "
                              f"Vehicles: {step_data['total_vehicles']:3d} | "
                              f"Wait: {step_data['avg_waiting_time']:5.1f}s | "
                              f"NS: {ns_pressure:5.1f} | EW: {ew_pressure:5.1f} | "
                              f"Flow: {flow_summary['total_vehicles']} veh/h")
            
            phase_results[scenario] = scenario_data
            current_step = scenario_end_step
            
            print(f"✅ {mode_name} Scenario {i} completed")
        
        return phase_results
    
    def _complete_network_reset(self):
        """Complete network and flow reset between phases"""
        print("🛑 Stopping all traffic flows...")
        
        # Stop all flows for 60 seconds to clear the network
        reset_start_step = traci.simulation.getTime()
        reset_end_step = reset_start_step + 60
        
        for step in range(int(reset_start_step), int(reset_end_step)):
            traci.simulationStep()
            
            # Remove all vehicles every 10 seconds during reset
            if step % 10 == 0:
                self._clear_network_vehicles()
                remaining_vehicles = len(traci.vehicle.getIDList())
                print(f"   Reset step {step - reset_start_step + 1}/60 - Remaining vehicles: {remaining_vehicles}")
        
        # Final vehicle clearance
        self._clear_network_vehicles()
        final_count = len(traci.vehicle.getIDList())
        print(f"✅ Network reset completed - Final vehicle count: {final_count}")
        
        # Brief pause
        time.sleep(1)
    
    def _combine_phase_results(self, normal_results, adaptive_results):
        """Combine results from both phases"""
        self.scenario_results = {}
        
        for scenario in self.scenarios:
            if scenario in normal_results and scenario in adaptive_results:
                self.scenario_results[scenario] = {
                    'normal': normal_results[scenario],
                    'adaptive': adaptive_results[scenario]
                }
                print(f"✅ Combined results for scenario: {scenario}")
        
        print(f"📊 Total scenarios with complete data: {len(self.scenario_results)}")
    
    def _clear_network(self):
        """Clear network between phases (not between scenarios in continuous mode)"""
        print("🧹 Brief pause between phases...")
        time.sleep(0.5)  # Brief pause only
    
    def _clear_network_vehicles(self):
        """Remove all vehicles from network"""
        try:
            vehicle_ids = traci.vehicle.getIDList()
            for veh_id in vehicle_ids:
                try:
                    traci.vehicle.remove(veh_id)
                except:
                    pass
        except:
            pass
    
    def _generate_scenario_analysis(self):
        """Generate comprehensive analysis of all scenarios"""
        print(f"\n📊 GENERATING COMPREHENSIVE SCENARIO ANALYSIS")
        print("=" * 70)
        
        summary_data = {}
        
        for scenario, data in self.scenario_results.items():
            if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
                continue
                
            # Calculate metrics (excluding first 2 minutes)
            warmup = 4  # 2 minutes * 2 samples per minute
            
            normal_metrics = self._calculate_metrics(data['normal'], warmup)
            adaptive_metrics = self._calculate_metrics(data['adaptive'], warmup)
            
            # Calculate improvements
            waiting_improvement = ((normal_metrics['avg_waiting'] - adaptive_metrics['avg_waiting']) / 
                                 normal_metrics['avg_waiting'] * 100) if normal_metrics['avg_waiting'] > 0 else 0
            
            speed_improvement = ((adaptive_metrics['avg_speed'] - normal_metrics['avg_speed']) / 
                               normal_metrics['avg_speed'] * 100) if normal_metrics['avg_speed'] > 0 else 0
            
            pressure_balance_normal = abs(normal_metrics['avg_ns_pressure'] - normal_metrics['avg_ew_pressure'])
            pressure_balance_adaptive = abs(adaptive_metrics['avg_ns_pressure'] - adaptive_metrics['avg_ew_pressure'])
            balance_improvement = ((pressure_balance_normal - pressure_balance_adaptive) / 
                                 pressure_balance_normal * 100) if pressure_balance_normal > 0 else 0
            
            summary_data[scenario] = {
                'normal': normal_metrics,
                'adaptive': adaptive_metrics,
                'waiting_improvement': waiting_improvement,
                'speed_improvement': speed_improvement,
                'balance_improvement': balance_improvement
            }
        
        # Display results table
        print(f"\n📈 SCENARIO COMPARISON RESULTS:")
        print(f"{'Scenario':<20} {'Wait Improve':<12} {'Speed Improve':<13} {'Balance Improve':<15} {'Rating':<10}")
        print("-" * 80)
        
        for scenario, metrics in summary_data.items():
            scenario_name = scenario.replace('_', ' ').title()
            rating = self._get_scenario_rating(metrics)
            
            print(f"{scenario_name:<20} {metrics['waiting_improvement']:>10.1f}% "
                  f"{metrics['speed_improvement']:>11.1f}% "
                  f"{metrics['balance_improvement']:>13.1f}% "
                  f"{rating:<10}")
        
        self.summary_data = summary_data
    
    def _calculate_metrics(self, data, warmup=0):
        """Calculate metrics for a dataset"""
        waiting_times = data['waiting_times'][warmup:]
        pressures_ns = data['pressures_ns'][warmup:]
        pressures_ew = data['pressures_ew'][warmup:]
        vehicle_counts = data['vehicle_counts'][warmup:]
        avg_speeds = data['avg_speeds'][warmup:]
        
        return {
            'avg_waiting': statistics.mean(waiting_times) if waiting_times else 0,
            'max_waiting': max(waiting_times) if waiting_times else 0,
            'avg_ns_pressure': statistics.mean(pressures_ns) if pressures_ns else 0,
            'avg_ew_pressure': statistics.mean(pressures_ew) if pressures_ew else 0,
            'avg_vehicles': statistics.mean(vehicle_counts) if vehicle_counts else 0,
            'avg_speed': statistics.mean(avg_speeds) if avg_speeds else 0
        }
    
    def _get_scenario_rating(self, metrics):
        """Get overall rating for scenario performance"""
        improvements = [
            metrics['waiting_improvement'],
            metrics['speed_improvement'],
            metrics['balance_improvement']
        ]
        
        avg_improvement = statistics.mean(improvements)
        
        if avg_improvement > 15:
            return "🟢 EXCELLENT"
        elif avg_improvement > 5:
            return "🟡 GOOD"
        elif avg_improvement > -5:
            return "🟠 FAIR"
        else:
            return "🔴 POOR"
    
    def _create_scenario_graphs(self):
        """Create comprehensive scenario comparison graphs"""
        print(f"\n📊 CREATING SCENARIO COMPARISON GRAPHS")
        
        # Create comprehensive comparison figure with better layout
        fig = plt.figure(figsize=(20, 12))
        
        # Main comparison chart (top row, spanning all columns)
        ax1 = plt.subplot(2, 1, 1)
        self._plot_scenario_summary(ax1)
        
        # Individual scenario plots (bottom row - 4 subplots)
        scenarios_short = ['Heavy 1-Dir', 'Light 3-Lane', 'Sudden Spike', 'Low Traffic']
        
        # Create a separate figure for individual scenarios
        fig2 = plt.figure(figsize=(20, 10))
        
        for i, (scenario, short_name) in enumerate(zip(self.scenarios, scenarios_short)):
            if scenario in self.scenario_results:
                ax = plt.subplot(2, 2, i + 1)  # 2x2 grid for 4 scenarios
                self._plot_single_scenario(ax, scenario, short_name)
        
        plt.figure(fig.number)  # Switch back to first figure
        plt.suptitle('Traffic Light Test Scenarios: Summary Analysis\n'
                    'Normal Mode vs Adaptive Control Performance Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('scenario_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(fig2.number)  # Switch to second figure
        plt.suptitle('Individual Scenario Analysis\n'
                    'Waiting Time Trends: Normal vs Adaptive Control', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('scenario_individual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed metrics comparison
        self._create_detailed_metrics_chart()
    
    def _plot_scenario_summary(self, ax):
        """Plot summary comparison of all scenarios"""
        if not hasattr(self, 'summary_data'):
            return
        
        scenarios = list(self.summary_data.keys())
        scenario_names = [s.replace('_', ' ').title() for s in scenarios]
        
        waiting_improvements = [self.summary_data[s]['waiting_improvement'] for s in scenarios]
        speed_improvements = [self.summary_data[s]['speed_improvement'] for s in scenarios]
        balance_improvements = [self.summary_data[s]['balance_improvement'] for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        bars1 = ax.bar(x - width, waiting_improvements, width, label='Waiting Time Improvement', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x, speed_improvements, width, label='Speed Improvement', 
                      color='lightgreen', alpha=0.8)
        bars3 = ax.bar(x + width, balance_improvements, width, label='Balance Improvement', 
                      color='salmon', alpha=0.8)
        
        ax.set_xlabel('Test Scenarios')
        ax.set_ylabel('Improvement Percentage (%)')
        ax.set_title('Adaptive Control Performance Across Different Traffic Scenarios')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3 if height >= 0 else -15),
                          textcoords="offset points",
                          ha='center', va='bottom' if height >= 0 else 'top',
                          fontsize=8)
    
    def _plot_single_scenario(self, ax, scenario, short_name):
        """Plot individual scenario results"""
        if scenario not in self.scenario_results:
            return
        
        data = self.scenario_results[scenario]
        
        # Convert timestamps to minutes
        normal_time = [t/60 for t in data['normal']['timestamps']]
        adaptive_time = [t/60 for t in data['adaptive']['timestamps']]
        
        ax.plot(normal_time, data['normal']['waiting_times'], 'r-', 
               label='Normal Mode', linewidth=2, alpha=0.7)
        ax.plot(adaptive_time, data['adaptive']['waiting_times'], 'g-', 
               label='Adaptive Mode', linewidth=2, alpha=0.7)
        
        ax.set_title(f'{short_name}\nWaiting Time Comparison')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Waiting Time (s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _create_detailed_metrics_chart(self):
        """Create detailed metrics comparison chart"""
        if not hasattr(self, 'summary_data'):
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        scenarios = list(self.summary_data.keys())
        scenario_names = [s.replace('_', ' ').title() for s in scenarios]
        
        # Waiting Times
        normal_waiting = [self.summary_data[s]['normal']['avg_waiting'] for s in scenarios]
        adaptive_waiting = [self.summary_data[s]['adaptive']['avg_waiting'] for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax1.bar(x - width/2, normal_waiting, width, label='Normal Mode', color='red', alpha=0.7)
        ax1.bar(x + width/2, adaptive_waiting, width, label='Adaptive Mode', color='green', alpha=0.7)
        ax1.set_title('Average Waiting Time by Scenario')
        ax1.set_ylabel('Waiting Time (seconds)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenario_names, rotation=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average Speeds
        normal_speed = [self.summary_data[s]['normal']['avg_speed'] for s in scenarios]
        adaptive_speed = [self.summary_data[s]['adaptive']['avg_speed'] for s in scenarios]
        
        ax2.bar(x - width/2, normal_speed, width, label='Normal Mode', color='red', alpha=0.7)
        ax2.bar(x + width/2, adaptive_speed, width, label='Adaptive Mode', color='green', alpha=0.7)
        ax2.set_title('Average Speed by Scenario')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenario_names, rotation=15)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Traffic Pressure NS
        normal_ns = [self.summary_data[s]['normal']['avg_ns_pressure'] for s in scenarios]
        adaptive_ns = [self.summary_data[s]['adaptive']['avg_ns_pressure'] for s in scenarios]
        
        ax3.bar(x - width/2, normal_ns, width, label='Normal Mode', color='red', alpha=0.7)
        ax3.bar(x + width/2, adaptive_ns, width, label='Adaptive Mode', color='green', alpha=0.7)
        ax3.set_title('North-South Traffic Pressure')
        ax3.set_ylabel('Traffic Pressure')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenario_names, rotation=15)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Traffic Pressure EW
        normal_ew = [self.summary_data[s]['normal']['avg_ew_pressure'] for s in scenarios]
        adaptive_ew = [self.summary_data[s]['adaptive']['avg_ew_pressure'] for s in scenarios]
        
        ax4.bar(x - width/2, normal_ew, width, label='Normal Mode', color='red', alpha=0.7)
        ax4.bar(x + width/2, adaptive_ew, width, label='Adaptive Mode', color='green', alpha=0.7)
        ax4.set_title('East-West Traffic Pressure')
        ax4.set_ylabel('Traffic Pressure')
        ax4.set_xticks(x)
        ax4.set_xticklabels(scenario_names, rotation=15)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Detailed Performance Metrics Across Test Scenarios', fontsize=14)
        plt.tight_layout()
        plt.savefig('scenario_detailed_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run the traffic scenarios simulation"""
    try:
        simulation = TrafficScenarioSimulation()
        simulation.run_all_scenarios()
        print(f"\n🎉 SCENARIO TESTING COMPLETE!")
        print(f"📊 Check generated graphs: scenario_analysis_comprehensive.png, scenario_detailed_metrics.png")
    except Exception as e:
        print(f"❌ Error running scenarios: {e}")

if __name__ == "__main__":
    main()