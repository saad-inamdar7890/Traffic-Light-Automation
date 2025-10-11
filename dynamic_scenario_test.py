"""
Dynamic Traffic Scenario Test - Comprehensive Edge Algorithm Evaluation
======================================================================

This module creates a realistic dynamic traffic scenario that tests the edge algorithm
through various traffic patterns over time:

1. Phase 1 (30 min): Low traffic in all lanes
2. Phase 2 (30 min): Heavy traffic in one lane (North), others low
3. Phase 3 (30 min): Heavy traffic in another lane (East), others low  
4. Phase 4 (30 min): Reduce traffic on all lanes
5. Phase 5 (30 min): Sudden increase in all lanes (rush hour)
6. Phase 6 (30 min): Gradually reduce traffic on all lanes

Total simulation time: 3 hours (10,800 seconds)
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from edge_traffic_controller import EdgeTrafficController
from utils import RouteGenerator, SUMOConfigManager, FileManager
from analyzer import TrafficAnalyzer
import traci

class DynamicScenarioTest:
    def __init__(self):
        self.base_directory = "dynamic_scenario_results"
        self.simulation_duration = 10800  # 3 hours in seconds
        self.phase_duration = 1800  # 30 minutes per phase
        
        # Traffic flow patterns for each phase
        self.traffic_phases = {
            'phase_1': {
                'name': 'Low Traffic All Lanes',
                'duration': 1800,  # 30 minutes
                'flows': {
                    'north': {'vehicles_per_hour': 120, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 100, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 110, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 90, 'pattern': 'constant'}
                }
            },
            'phase_2': {
                'name': 'Heavy North, Light Others',
                'duration': 1800,
                'flows': {
                    'north': {'vehicles_per_hour': 800, 'pattern': 'gradual_increase'},
                    'south': {'vehicles_per_hour': 100, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 80, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 70, 'pattern': 'constant'}
                }
            },
            'phase_3': {
                'name': 'Heavy East, Light Others',
                'duration': 1800,
                'flows': {
                    'north': {'vehicles_per_hour': 100, 'pattern': 'gradual_decrease'},
                    'south': {'vehicles_per_hour': 90, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 900, 'pattern': 'gradual_increase'},
                    'west': {'vehicles_per_hour': 80, 'pattern': 'constant'}
                }
            },
            'phase_4': {
                'name': 'Reduced Traffic All Lanes',
                'duration': 1800,
                'flows': {
                    'north': {'vehicles_per_hour': 80, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 70, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 90, 'pattern': 'gradual_decrease'},
                    'west': {'vehicles_per_hour': 60, 'pattern': 'constant'}
                }
            },
            'phase_5': {
                'name': 'Rush Hour - Heavy All Lanes',
                'duration': 1800,
                'flows': {
                    'north': {'vehicles_per_hour': 1200, 'pattern': 'sudden_increase'},
                    'south': {'vehicles_per_hour': 1100, 'pattern': 'sudden_increase'},
                    'east': {'vehicles_per_hour': 1000, 'pattern': 'sudden_increase'},
                    'west': {'vehicles_per_hour': 1150, 'pattern': 'sudden_increase'}
                }
            },
            'phase_6': {
                'name': 'Gradual Reduction All Lanes',
                'duration': 1800,
                'flows': {
                    'north': {'vehicles_per_hour': 150, 'pattern': 'gradual_decrease'},
                    'south': {'vehicles_per_hour': 140, 'pattern': 'gradual_decrease'},
                    'east': {'vehicles_per_hour': 160, 'pattern': 'gradual_decrease'},
                    'west': {'vehicles_per_hour': 130, 'pattern': 'gradual_decrease'}
                }
            }
        }
        
        # Data collection storage
        self.normal_mode_data = []
        self.adaptive_mode_data = []
        self.phase_boundaries = []
        
        # Create results directory
        os.makedirs(self.base_directory, exist_ok=True)
        
    def generate_dynamic_route_file(self, mode_suffix=""):
        """Generate route file with dynamic traffic patterns"""
        route_file = f"{self.base_directory}/dynamic_traffic_{mode_suffix}.rou.xml"
        
        with open(route_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')
            
            # Define vehicle types
            f.write('    <vType id="normal_car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" />\n')
            
            # Define routes for each direction (based on demo.rou.xml)
            routes = {
                'north_to_south': ['E1', 'E1.200'],      # North to South
                'south_to_north': ['-E1', '-E1.238'],    # South to North  
                'east_to_west': ['E0', 'E0.319'],        # East to West
                'west_to_east': ['-E0', '-E0.254']       # West to East
            }
            
            for route_id, edges in routes.items():
                f.write(f'    <route id="{route_id}" edges="{" ".join(edges)}" />\n')
            
            # Generate flows for each phase
            current_time = 0
            vehicle_id = 0
            
            for phase_key, phase_data in self.traffic_phases.items():
                phase_start = current_time
                phase_end = current_time + phase_data['duration']
                
                print(f"Generating {phase_data['name']} ({phase_start}s - {phase_end}s)")
                
                # Calculate flow rates for this phase
                for direction, flow_data in phase_data['flows'].items():
                    base_rate = flow_data['vehicles_per_hour']
                    pattern = flow_data['pattern']
                    
                    # Generate vehicles for this direction and phase
                    vehicles_generated = self._generate_vehicles_for_phase(
                        f, direction, base_rate, pattern, phase_start, phase_end, vehicle_id
                    )
                    vehicle_id += vehicles_generated
                
                current_time = phase_end
                self.phase_boundaries.append(phase_end)
            
            f.write('</routes>\n')
        
        print(f"✅ Dynamic route file generated: {route_file}")
        return route_file
    
    def _generate_vehicles_for_phase(self, file_obj, direction, base_rate, pattern, start_time, end_time, start_id):
        """Generate vehicles for a specific phase and direction"""
        duration = end_time - start_time
        vehicles_generated = 0
        
        # Route mapping (corrected based on demo.rou.xml)
        route_map = {
            'north': 'north_to_south',   # E1 -> E1.200
            'south': 'south_to_north',   # -E1 -> -E1.238
            'east': 'east_to_west',      # E0 -> E0.319
            'west': 'west_to_east'       # -E0 -> -E0.254
        }
        
        route_id = route_map[direction]
        
        if pattern == 'constant':
            # Constant flow throughout the phase
            vehicles_per_second = base_rate / 3600
            interval = 1.0 / vehicles_per_second if vehicles_per_second > 0 else 60
            
            current_time = start_time
            while current_time < end_time:
                file_obj.write(f'    <vehicle id="veh_{start_id + vehicles_generated}" '
                             f'type="normal_car" route="{route_id}" depart="{current_time:.1f}" />\n')
                vehicles_generated += 1
                current_time += interval
        
        elif pattern == 'gradual_increase':
            # Gradually increase from 10% to 100% of base rate
            time_steps = duration // 60  # Every minute
            for step in range(int(time_steps)):
                step_start = start_time + (step * 60)
                step_end = min(step_start + 60, end_time)
                
                # Calculate flow rate for this step (10% to 100%)
                progress = step / max(time_steps - 1, 1)
                current_rate = base_rate * (0.1 + 0.9 * progress)
                vehicles_per_second = current_rate / 3600
                interval = 1.0 / vehicles_per_second if vehicles_per_second > 0 else 60
                
                current_time = step_start
                while current_time < step_end:
                    file_obj.write(f'    <vehicle id="veh_{start_id + vehicles_generated}" '
                                 f'type="normal_car" route="{route_id}" depart="{current_time:.1f}" />\n')
                    vehicles_generated += 1
                    current_time += interval
        
        elif pattern == 'gradual_decrease':
            # Gradually decrease from 100% to 10% of base rate
            time_steps = duration // 60
            for step in range(int(time_steps)):
                step_start = start_time + (step * 60)
                step_end = min(step_start + 60, end_time)
                
                # Calculate flow rate for this step (100% to 10%)
                progress = step / max(time_steps - 1, 1)
                current_rate = base_rate * (1.0 - 0.9 * progress)
                vehicles_per_second = current_rate / 3600
                interval = 1.0 / vehicles_per_second if vehicles_per_second > 0 else 60
                
                current_time = step_start
                while current_time < step_end:
                    file_obj.write(f'    <vehicle id="veh_{start_id + vehicles_generated}" '
                                 f'type="normal_car" route="{route_id}" depart="{current_time:.1f}" />\n')
                    vehicles_generated += 1
                    current_time += interval
        
        elif pattern == 'sudden_increase':
            # Immediate jump to high traffic rate
            vehicles_per_second = base_rate / 3600
            interval = 1.0 / vehicles_per_second if vehicles_per_second > 0 else 60
            
            current_time = start_time
            while current_time < end_time:
                file_obj.write(f'    <vehicle id="veh_{start_id + vehicles_generated}" '
                             f'type="normal_car" route="{route_id}" depart="{current_time:.1f}" />\n')
                vehicles_generated += 1
                current_time += interval
        
        return vehicles_generated
    
    def create_sumo_config(self, route_file, mode_suffix=""):
        """Create SUMO configuration file"""
        config_file = f"{self.base_directory}/dynamic_config_{mode_suffix}.sumocfg"
        
        config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="../demo.net.xml"/>
        <route-files value="{os.path.basename(route_file)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{self.simulation_duration}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
    </processing>
    <output>
        <tripinfo-output value="tripinfo_{mode_suffix}.xml"/>
    </output>
</configuration>'''
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        return config_file
    
    def run_normal_mode_simulation(self):
        """Run simulation with normal (fixed-time) traffic controller"""
        print("\n🚦 RUNNING NORMAL MODE SIMULATION")
        print("=" * 60)
        
        # Generate route file
        route_file = self.generate_dynamic_route_file("normal")
        config_file = self.create_sumo_config(route_file, "normal")
        
        # Start SUMO
        sumo_cmd = ['sumo', '-c', config_file, '--no-warnings', '--no-step-log']
        traci.start(sumo_cmd)
        
        # Initialize normal controller (fixed 72s cycle)
        analyzer = TrafficAnalyzer()
        cycle_time = 72  # Fixed cycle time
        
        print(f"🚦 Normal controller: Fixed {cycle_time}s cycle")
        
        # Run simulation
        step = 0
        while step < self.simulation_duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Collect data every 30 seconds
            if step % 30 == 0:
                try:
                    # Get current traffic metrics
                    traffic_data = analyzer.collect_traffic_metrics(step, traci)
                    if traffic_data:
                        data_point = {
                            'time': step,
                            'avg_waiting_time': traffic_data.get('avg_waiting_time', 0),
                            'avg_speed': traffic_data.get('avg_speed', 0),
                            'total_vehicles': traffic_data.get('total_vehicles', 0),
                            'throughput': traffic_data.get('throughput', 0),
                            'phase': self._get_current_phase(step),
                            'mode': 'normal'
                        }
                        self.normal_mode_data.append(data_point)
                        
                        if step % 300 == 0:  # Print every 5 minutes
                            phase_name = self._get_current_phase_name(step)
                            print(f"   Time: {step:4d}s | Phase: {phase_name} | "
                                  f"Wait: {data_point['avg_waiting_time']:5.1f}s | "
                                  f"Speed: {data_point['avg_speed']:5.1f} m/s | "
                                  f"Vehicles: {data_point['total_vehicles']:3d}")
                
                except Exception as e:
                    print(f"⚠️  Error collecting data at step {step}: {e}")
            
            step += 1
        
        traci.close()
        print("✅ Normal mode simulation completed")
        
        # Save normal mode data
        with open(f"{self.base_directory}/normal_mode_data.json", 'w') as f:
            json.dump(self.normal_mode_data, f, indent=2)
    
    def run_adaptive_mode_simulation(self):
        """Run simulation with adaptive edge traffic controller"""
        print("\n🚦 RUNNING ADAPTIVE MODE SIMULATION")
        print("=" * 60)
        
        # Generate route file
        route_file = self.generate_dynamic_route_file("adaptive")
        config_file = self.create_sumo_config(route_file, "adaptive")
        
        # Start SUMO
        sumo_cmd = ['sumo', '-c', config_file, '--no-warnings', '--no-step-log']
        traci.start(sumo_cmd)
        
        # Initialize edge controller
        edge_controller = EdgeTrafficController(junction_id="J4", base_green_time=30)
        analyzer = TrafficAnalyzer()
        
        print(f"🚦 Edge controller: Base {edge_controller.base_green_time}s, Range {edge_controller.min_green_time}s-{edge_controller.max_green_time}s")
        
        # Run simulation
        step = 0
        adaptations_count = 0
        
        while step < self.simulation_duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Apply edge algorithm
            if step > 10:  # Start after initialization
                adaptation_result = edge_controller.apply_edge_algorithm(step)
                if adaptation_result:
                    adaptations_count += 1
            
            # Collect data every 30 seconds
            if step % 30 == 0:
                try:
                    # Get current traffic metrics
                    traffic_data = analyzer.collect_traffic_metrics(step, traci)
                    if traffic_data:
                        data_point = {
                            'time': step,
                            'avg_waiting_time': traffic_data.get('avg_waiting_time', 0),
                            'avg_speed': traffic_data.get('avg_speed', 0),
                            'total_vehicles': traffic_data.get('total_vehicles', 0),
                            'throughput': traffic_data.get('throughput', 0),
                            'phase': self._get_current_phase(step),
                            'mode': 'adaptive',
                            'adaptations': adaptations_count
                        }
                        self.adaptive_mode_data.append(data_point)
                        
                        if step % 300 == 0:  # Print every 5 minutes
                            phase_name = self._get_current_phase_name(step)
                            print(f"   Time: {step:4d}s | Phase: {phase_name} | "
                                  f"Wait: {data_point['avg_waiting_time']:5.1f}s | "
                                  f"Speed: {data_point['avg_speed']:5.1f} m/s | "
                                  f"Vehicles: {data_point['total_vehicles']:3d} | "
                                  f"Adaptations: {adaptations_count}")
                
                except Exception as e:
                    print(f"⚠️  Error collecting data at step {step}: {e}")
            
            step += 1
        
        traci.close()
        print(f"✅ Adaptive mode simulation completed with {adaptations_count} adaptations")
        
        # Save adaptive mode data
        with open(f"{self.base_directory}/adaptive_mode_data.json", 'w') as f:
            json.dump(self.adaptive_mode_data, f, indent=2)
    
    def _get_current_phase(self, time_step):
        """Get current phase number based on time"""
        phase_duration = 1800  # 30 minutes
        return min(int(time_step // phase_duration) + 1, 6)
    
    def _get_current_phase_name(self, time_step):
        """Get current phase name based on time"""
        phase_num = self._get_current_phase(time_step)
        phase_names = {
            1: "Low All", 2: "Heavy N", 3: "Heavy E", 
            4: "Reduced", 5: "Rush Hour", 6: "Gradual Down"
        }
        return phase_names.get(phase_num, "Unknown")
    
    def analyze_and_plot_results(self):
        """Analyze results and create comprehensive comparison plots"""
        print("\n📊 ANALYZING RESULTS AND GENERATING PLOTS")
        print("=" * 60)
        
        if not self.normal_mode_data or not self.adaptive_mode_data:
            print("❌ Missing simulation data. Run simulations first.")
            return
        
        # Convert to DataFrames for easier analysis
        normal_df = pd.DataFrame(self.normal_mode_data)
        adaptive_df = pd.DataFrame(self.adaptive_mode_data)
        
        # Create comprehensive comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Dynamic Traffic Scenario Analysis - Normal vs Adaptive Edge Algorithm', fontsize=16)
        
        # Plot 1: Average Waiting Time
        axes[0, 0].plot(normal_df['time'] / 60, normal_df['avg_waiting_time'], 
                       label='Normal (Fixed)', color='red', linewidth=2)
        axes[0, 0].plot(adaptive_df['time'] / 60, adaptive_df['avg_waiting_time'], 
                       label='Adaptive (Edge)', color='blue', linewidth=2)
        axes[0, 0].set_title('Average Waiting Time Over Time')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Waiting Time (seconds)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add phase boundaries
        for boundary in self.phase_boundaries:
            axes[0, 0].axvline(x=boundary/60, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 2: Average Speed
        axes[0, 1].plot(normal_df['time'] / 60, normal_df['avg_speed'], 
                       label='Normal (Fixed)', color='red', linewidth=2)
        axes[0, 1].plot(adaptive_df['time'] / 60, adaptive_df['avg_speed'], 
                       label='Adaptive (Edge)', color='blue', linewidth=2)
        axes[0, 1].set_title('Average Speed Over Time')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Speed (m/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        for boundary in self.phase_boundaries:
            axes[0, 1].axvline(x=boundary/60, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 3: Total Vehicles
        axes[0, 2].plot(normal_df['time'] / 60, normal_df['total_vehicles'], 
                       label='Normal (Fixed)', color='red', linewidth=2)
        axes[0, 2].plot(adaptive_df['time'] / 60, adaptive_df['total_vehicles'], 
                       label='Adaptive (Edge)', color='blue', linewidth=2)
        axes[0, 2].set_title('Total Vehicles in Network')
        axes[0, 2].set_xlabel('Time (minutes)')
        axes[0, 2].set_ylabel('Number of Vehicles')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        for boundary in self.phase_boundaries:
            axes[0, 2].axvline(x=boundary/60, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 4: Phase-wise Average Waiting Time
        phase_analysis = self._analyze_by_phases(normal_df, adaptive_df)
        phases = list(phase_analysis.keys())
        normal_wait_avg = [phase_analysis[p]['normal']['avg_waiting'] for p in phases]
        adaptive_wait_avg = [phase_analysis[p]['adaptive']['avg_waiting'] for p in phases]
        
        x_pos = np.arange(len(phases))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, normal_wait_avg, width, label='Normal', color='red', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, adaptive_wait_avg, width, label='Adaptive', color='blue', alpha=0.7)
        axes[1, 0].set_title('Average Waiting Time by Phase')
        axes[1, 0].set_xlabel('Traffic Phase')
        axes[1, 0].set_ylabel('Avg Waiting Time (s)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f'Phase {p}' for p in phases])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Phase-wise Average Speed
        normal_speed_avg = [phase_analysis[p]['normal']['avg_speed'] for p in phases]
        adaptive_speed_avg = [phase_analysis[p]['adaptive']['avg_speed'] for p in phases]
        
        axes[1, 1].bar(x_pos - width/2, normal_speed_avg, width, label='Normal', color='red', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, adaptive_speed_avg, width, label='Adaptive', color='blue', alpha=0.7)
        axes[1, 1].set_title('Average Speed by Phase')
        axes[1, 1].set_xlabel('Traffic Phase')
        axes[1, 1].set_ylabel('Avg Speed (m/s)')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([f'Phase {p}' for p in phases])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Improvement Percentage by Phase
        improvements = []
        for p in phases:
            normal_wait = phase_analysis[p]['normal']['avg_waiting']
            adaptive_wait = phase_analysis[p]['adaptive']['avg_waiting']
            if normal_wait > 0:
                improvement = ((normal_wait - adaptive_wait) / normal_wait) * 100
            else:
                improvement = 0
            improvements.append(improvement)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[1, 2].bar(x_pos, improvements, color=colors, alpha=0.7)
        axes[1, 2].set_title('Waiting Time Improvement by Phase')
        axes[1, 2].set_xlabel('Traffic Phase')
        axes[1, 2].set_ylabel('Improvement (%)')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels([f'Phase {p}' for p in phases])
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add phase labels
        phase_labels = [
            "Low All", "Heavy N", "Heavy E", 
            "Reduced", "Rush Hour", "Gradual Down"
        ]
        for i, (ax_row, ax_col) in enumerate([(0, 0), (0, 1), (0, 2)]):
            for j, (boundary, label) in enumerate(zip(self.phase_boundaries, phase_labels)):
                if j < len(phase_labels) - 1:  # Don't add label for last boundary
                    mid_point = (boundary + (self.phase_boundaries[j+1] if j+1 < len(self.phase_boundaries) else self.simulation_duration)) / 2
                    axes[ax_row, ax_col].text(mid_point/60, axes[ax_row, ax_col].get_ylim()[1] * 0.95, 
                                            label, ha='center', va='top', fontsize=8, 
                                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plot_file = f"{self.base_directory}/dynamic_scenario_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comprehensive comparison plot saved: {plot_file}")
        
        # Generate summary report
        self._generate_summary_report(phase_analysis)
    
    def _analyze_by_phases(self, normal_df, adaptive_df):
        """Analyze performance by traffic phases"""
        phase_analysis = {}
        
        for phase in range(1, 7):
            normal_phase_data = normal_df[normal_df['phase'] == phase]
            adaptive_phase_data = adaptive_df[adaptive_df['phase'] == phase]
            
            phase_analysis[phase] = {
                'normal': {
                    'avg_waiting': normal_phase_data['avg_waiting_time'].mean(),
                    'avg_speed': normal_phase_data['avg_speed'].mean(),
                    'avg_vehicles': normal_phase_data['total_vehicles'].mean()
                },
                'adaptive': {
                    'avg_waiting': adaptive_phase_data['avg_waiting_time'].mean(),
                    'avg_speed': adaptive_phase_data['avg_speed'].mean(),
                    'avg_vehicles': adaptive_phase_data['total_vehicles'].mean()
                }
            }
        
        return phase_analysis
    
    def _generate_summary_report(self, phase_analysis):
        """Generate comprehensive summary report"""
        report_file = f"{self.base_directory}/summary_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("DYNAMIC TRAFFIC SCENARIO ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Simulation Time: {self.simulation_duration / 3600:.1f} hours\n")
            f.write(f"Phase Duration: {self.phase_duration / 60:.0f} minutes each\n\n")
            
            # Phase-by-phase analysis
            f.write("PHASE-BY-PHASE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            phase_names = [
                "Low Traffic All Lanes",
                "Heavy North, Light Others", 
                "Heavy East, Light Others",
                "Reduced Traffic All Lanes",
                "Rush Hour - Heavy All Lanes",
                "Gradual Reduction All Lanes"
            ]
            
            total_improvement = 0
            valid_phases = 0
            
            for i, (phase, data) in enumerate(phase_analysis.items()):
                f.write(f"\nPhase {phase}: {phase_names[i-1] if i > 0 else phase_names[0]}\n")
                
                normal_wait = data['normal']['avg_waiting']
                adaptive_wait = data['adaptive']['avg_waiting']
                
                if normal_wait > 0:
                    improvement = ((normal_wait - adaptive_wait) / normal_wait) * 100
                    total_improvement += improvement
                    valid_phases += 1
                else:
                    improvement = 0
                
                f.write(f"  Normal Mode:\n")
                f.write(f"    Avg Waiting Time: {normal_wait:.2f}s\n")
                f.write(f"    Avg Speed: {data['normal']['avg_speed']:.2f} m/s\n")
                f.write(f"    Avg Vehicles: {data['normal']['avg_vehicles']:.1f}\n")
                
                f.write(f"  Adaptive Mode:\n")
                f.write(f"    Avg Waiting Time: {adaptive_wait:.2f}s\n")
                f.write(f"    Avg Speed: {data['adaptive']['avg_speed']:.2f} m/s\n")
                f.write(f"    Avg Vehicles: {data['adaptive']['avg_vehicles']:.1f}\n")
                
                f.write(f"  Improvement: {improvement:+.1f}%\n")
            
            # Overall summary
            f.write(f"\nOVERALL PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            avg_improvement = total_improvement / valid_phases if valid_phases > 0 else 0
            f.write(f"Average Improvement: {avg_improvement:+.1f}%\n")
            
            if avg_improvement > 0:
                f.write("✅ Adaptive Edge Algorithm shows POSITIVE improvement\n")
            else:
                f.write("❌ Adaptive Edge Algorithm needs optimization\n")
        
        print(f"📄 Summary report generated: {report_file}")

def main():
    """Main function to run the dynamic scenario test"""
    print("🚦 DYNAMIC TRAFFIC SCENARIO TEST")
    print("=" * 60)
    print("Testing Edge Algorithm through realistic traffic patterns:")
    print("  Phase 1: Low traffic (30 min)")
    print("  Phase 2: Heavy North traffic (30 min)")
    print("  Phase 3: Heavy East traffic (30 min)")
    print("  Phase 4: Reduced traffic (30 min)")
    print("  Phase 5: Rush hour - all heavy (30 min)")
    print("  Phase 6: Gradual reduction (30 min)")
    print("  Total: 3 hours simulation")
    print()
    
    # Create test instance
    scenario_test = DynamicScenarioTest()
    
    try:
        # Run normal mode simulation
        scenario_test.run_normal_mode_simulation()
        
        # Run adaptive mode simulation
        scenario_test.run_adaptive_mode_simulation()
        
        # Analyze and plot results
        scenario_test.analyze_and_plot_results()
        
        print("\n🎉 DYNAMIC SCENARIO TEST COMPLETED!")
        print("=" * 60)
        print(f"📁 Results saved in: {scenario_test.base_directory}/")
        print("📊 Check the comparison plots and summary report")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()