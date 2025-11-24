"""
Dynamic Traffic Scenario Test - Enhanced with Motorcycles
==========================================================

Comprehensive edge algorithm evaluation with realistic mixed vehicle traffic:
- Cars (70% of traffic)
- Motorcycles (30% of traffic)
- 6 dynamic traffic phases over 3 hours
- Comparative analysis between normal and adaptive modes

This module integrates with the src project structure and uses the enhanced
edge traffic controller for intelligent traffic management.
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Import project modules
from edge_traffic_controller import EdgeTrafficController
from improved_edge_traffic_controller import ImprovedEdgeTrafficController
from analyzer import TrafficAnalyzer
from utils import RouteGenerator, SUMOConfigManager, FileManager
from visualizer import TrafficVisualizer
import traci

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicScenarioTestSrc:
    def __init__(self):
        self.base_directory = "scenario_results"
        self.simulation_duration = 10800  # 3 hours in seconds
        self.phase_duration = 1800  # 30 minutes per phase
        
        # Vehicle mix configuration
        self.vehicle_types = {
            'car': {
                'probability': 0.70,  # 70% cars
                'vtype_id': 'car',
                'length': 5.0,
                'max_speed': 50.0,
                'accel': 2.6,
                'decel': 4.5,
                'sigma': 0.5,
                'color': '1,1,0'  # Yellow
            },
            'motorcycle': {
                'probability': 0.30,  # 30% motorcycles
                'vtype_id': 'motorcycle',
                'length': 2.2,
                'max_speed': 60.0,
                'accel': 3.5,
                'decel': 5.0,
                'sigma': 0.3,
                'color': '0,0,1'  # Blue
            }
        }
        
        # Traffic flow patterns for each phase (OPTIMIZED FOR CLEAR ALGORITHM DEMONSTRATION)
        self.traffic_phases = {
            'phase_1': {
                'name': 'Low Mixed Traffic All Lanes',
                'duration': 1800,
                'description': 'Very light traffic with cars and motorcycles',
                'flows': {
                    'north': {'vehicles_per_hour': 120, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 100, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 110, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 90, 'pattern': 'constant'}
                }
            },
            'phase_2': {
                'name': 'Heavy North Traffic (Mixed Vehicles)',
                'duration': 1800,
                'description': 'Very heavy north traffic vs very light others - EXTREME CONTRAST',
                'flows': {
                    'north': {'vehicles_per_hour': 2400, 'pattern': 'gradual_increase'},
                    'south': {'vehicles_per_hour': 80, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 60, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 50, 'pattern': 'constant'}
                }
            },
            'phase_3': {
                'name': 'Heavy East Traffic (Mixed Vehicles)',
                'duration': 1800,
                'description': 'Very heavy east traffic vs very light others - EXTREME CONTRAST',
                'flows': {
                    'north': {'vehicles_per_hour': 70, 'pattern': 'gradual_decrease'},
                    'south': {'vehicles_per_hour': 60, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 2500, 'pattern': 'gradual_increase'},
                    'west': {'vehicles_per_hour': 50, 'pattern': 'constant'}
                }
            },
            'phase_4': {
                'name': 'Reduced Mixed Traffic All Lanes',
                'duration': 1800,
                'description': 'Very minimal traffic on all lanes - ALGORITHM EFFICIENCY TEST',
                'flows': {
                    'north': {'vehicles_per_hour': 40, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 35, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 45, 'pattern': 'gradual_decrease'},
                    'west': {'vehicles_per_hour': 30, 'pattern': 'constant'}
                }
            },
            'phase_5': {
                'name': 'Rush Hour - Heavy Mixed Traffic All Lanes',
                'duration': 1800,
                'description': 'Maximum rush hour with very high car and motorcycle traffic',
                'flows': {
                    'north': {'vehicles_per_hour': 2800, 'pattern': 'sudden_increase'},
                    'south': {'vehicles_per_hour': 2700, 'pattern': 'sudden_increase'},
                    'east': {'vehicles_per_hour': 2600, 'pattern': 'sudden_increase'},
                    'west': {'vehicles_per_hour': 2750, 'pattern': 'sudden_increase'}
                }
            },
            'phase_6': {
                'name': 'Gradual Reduction Mixed Traffic',
                'duration': 1800,
                'description': 'Evening traffic reduction from heavy to light',
                'flows': {
                    'north': {'vehicles_per_hour': 200, 'pattern': 'gradual_decrease'},
                    'south': {'vehicles_per_hour': 180, 'pattern': 'gradual_decrease'},
                    'east': {'vehicles_per_hour': 220, 'pattern': 'gradual_decrease'},
                    'west': {'vehicles_per_hour': 160, 'pattern': 'gradual_decrease'}
                }
            }
        }
        
        # Data collection storage
        self.normal_mode_data = []
        self.adaptive_mode_data = []
        self.phase_boundaries = []
        
        # Create results directory
        os.makedirs(self.base_directory, exist_ok=True)
        
        # Initialize components
        self.analyzer = TrafficAnalyzer()
        self.visualizer = TrafficVisualizer()
        
        logger.info(f"Dynamic scenario test initialized - {len(self.traffic_phases)} phases")
    
    def generate_dynamic_route_file(self, mode_suffix=""):
        """Generate route file with dynamic mixed vehicle traffic patterns"""
        route_file = f"{self.base_directory}/dynamic_mixed_traffic_{mode_suffix}.rou.xml"
        
        logger.info(f"Generating route file with mixed vehicles: {route_file}")
        
        with open(route_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')
            
            # Define vehicle types (cars and motorcycles)
            for vtype_name, vtype_config in self.vehicle_types.items():
                f.write(f'    <vType id="{vtype_config["vtype_id"]}" '
                       f'accel="{vtype_config["accel"]}" '
                       f'decel="{vtype_config["decel"]}" '
                       f'sigma="{vtype_config["sigma"]}" '
                       f'length="{vtype_config["length"]}" '
                       f'maxSpeed="{vtype_config["max_speed"]}" '
                       f'color="{vtype_config["color"]}" />\n')
            
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
                
                logger.info(f"Generating {phase_data['name']} ({phase_start}s - {phase_end}s)")
                
                # Calculate flow rates for this phase
                for direction, flow_data in phase_data['flows'].items():
                    base_rate = flow_data['vehicles_per_hour']
                    pattern = flow_data['pattern']
                    
                    # Generate mixed vehicles for this direction and phase
                    vehicles_generated = self._generate_mixed_vehicles_for_phase(
                        f, direction, base_rate, pattern, phase_start, phase_end, vehicle_id
                    )
                    vehicle_id += vehicles_generated
                
                current_time = phase_end
                self.phase_boundaries.append(phase_end)
            
            f.write('</routes>\n')
        
        logger.info(f"Route file generated with mixed traffic: {route_file}")
        return route_file
    
    def _generate_mixed_vehicles_for_phase(self, file_obj, direction, base_rate, pattern, start_time, end_time, start_id):
        """Generate mixed vehicles (cars + motorcycles) for a specific phase and direction"""
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
                # Determine vehicle type (car or motorcycle)
                vehicle_type = self._get_random_vehicle_type()
                
                file_obj.write(f'    <vehicle id="veh_{start_id + vehicles_generated}" '
                             f'type="{vehicle_type}" route="{route_id}" depart="{current_time:.1f}" />\n')
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
                    vehicle_type = self._get_random_vehicle_type()
                    file_obj.write(f'    <vehicle id="veh_{start_id + vehicles_generated}" '
                                 f'type="{vehicle_type}" route="{route_id}" depart="{current_time:.1f}" />\n')
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
                    vehicle_type = self._get_random_vehicle_type()
                    file_obj.write(f'    <vehicle id="veh_{start_id + vehicles_generated}" '
                                 f'type="{vehicle_type}" route="{route_id}" depart="{current_time:.1f}" />\n')
                    vehicles_generated += 1
                    current_time += interval
        
        elif pattern == 'sudden_increase':
            # Immediate jump to high traffic rate
            vehicles_per_second = base_rate / 3600
            interval = 1.0 / vehicles_per_second if vehicles_per_second > 0 else 60
            
            current_time = start_time
            while current_time < end_time:
                vehicle_type = self._get_random_vehicle_type()
                file_obj.write(f'    <vehicle id="veh_{start_id + vehicles_generated}" '
                             f'type="{vehicle_type}" route="{route_id}" depart="{current_time:.1f}" />\n')
                vehicles_generated += 1
                current_time += interval
        
        return vehicles_generated
    
    def _get_random_vehicle_type(self):
        """Randomly select vehicle type based on probability distribution"""
        import random
        rand_val = random.random()
        
        if rand_val < self.vehicle_types['car']['probability']:
            return self.vehicle_types['car']['vtype_id']
        else:
            return self.vehicle_types['motorcycle']['vtype_id']
    
    def create_sumo_config(self, route_file, mode_suffix=""):
        """Create SUMO configuration file"""
        config_file = f"{self.base_directory}/dynamic_config_{mode_suffix}.sumocfg"
        
        config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="../../demo.net.xml"/>
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
        logger.info("Starting Normal Mode Simulation")
        print("\n🚦 RUNNING NORMAL MODE SIMULATION (Mixed Vehicles)")
        print("=" * 70)
        
        # Generate route file
        route_file = self.generate_dynamic_route_file("normal")
        config_file = self.create_sumo_config(route_file, "normal")
        
        # Start SUMO
        sumo_cmd = ['sumo', '-c', config_file, '--no-warnings', '--no-step-log']
        traci.start(sumo_cmd)
        
        # Initialize normal controller (fixed 30s green for each direction)
        green_time = 30  # Fixed green time per direction
        cycle_time = (green_time * 2) + 8  # 30s NS + 30s EW + 4s yellow each = 68s total cycle
        
        print(f"🚦 Normal controller: Fixed {green_time}s green per direction ({cycle_time}s total cycle) with mixed vehicles")
        
        # Track vehicle types
        car_count = 0
        motorcycle_count = 0
        
        # Traffic light control variables for normal mode
        current_phase = 0  # 0 = NS green, 1 = NS yellow, 2 = EW green, 3 = EW yellow
        phase_start_time = 0
        phase_durations = [green_time, 4, green_time, 4]  # NS green, NS yellow, EW green, EW yellow
        
        # Run simulation
        step = 0
        
        # Initialize traffic lights at start
        try:
            # Set initial phase (North-South green)
            traci.trafficlight.setPhase("J4", 0)
            current_phase = 0
            phase_start_time = 0
            logger.info(f"Normal mode: Initialized with NS green phase, {green_time}s timing")
        except Exception as e:
            logger.warning(f"Could not initialize traffic lights: {e}")
        
        while step < self.simulation_duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Normal mode traffic light control (fixed timing)
            if step > 0:
                phase_elapsed = step - phase_start_time
                current_phase_duration = phase_durations[current_phase]
                
                if phase_elapsed >= current_phase_duration:
                    # Switch to next phase
                    current_phase = (current_phase + 1) % 4
                    phase_start_time = step
                    
                    try:
                        if current_phase == 0:  # NS green
                            traci.trafficlight.setPhase("J4", 0)
                        elif current_phase == 1:  # NS yellow
                            traci.trafficlight.setPhase("J4", 1)
                        elif current_phase == 2:  # EW green
                            traci.trafficlight.setPhase("J4", 2)
                        elif current_phase == 3:  # EW yellow
                            traci.trafficlight.setPhase("J4", 3)
                    except Exception as e:
                        logger.warning(f"Could not set traffic light phase: {e}")
            
            # Count vehicle types
            if step % 300 == 0:  # Every 5 minutes
                vehicle_ids = traci.vehicle.getIDList()
                current_cars = sum(1 for vid in vehicle_ids if traci.vehicle.getTypeID(vid) == 'car')
                current_motorcycles = sum(1 for vid in vehicle_ids if traci.vehicle.getTypeID(vid) == 'motorcycle')
                
                if current_cars > 0 or current_motorcycles > 0:
                    car_count = current_cars
                    motorcycle_count = current_motorcycles
            
            # Collect data every 30 seconds
            if step % 30 == 0:
                try:
                    # Get current traffic metrics
                    traffic_data = self.analyzer.collect_traffic_metrics(step, traci)
                    if traffic_data:
                        data_point = {
                            'time': step,
                            'avg_waiting_time': traffic_data.get('avg_waiting_time', 0),
                            'avg_speed': traffic_data.get('avg_speed', 0),
                            'total_vehicles': traffic_data.get('total_vehicles', 0),
                            'throughput': traffic_data.get('throughput', 0),
                            'phase': self._get_current_phase(step),
                            'mode': 'normal',
                            'cars': car_count,
                            'motorcycles': motorcycle_count,
                            'vehicle_mix_ratio': motorcycle_count / (car_count + motorcycle_count) if (car_count + motorcycle_count) > 0 else 0
                        }
                        self.normal_mode_data.append(data_point)
                        
                        if step % 300 == 0:  # Print every 5 minutes
                            phase_name = self._get_current_phase_name(step)
                            total_vehicles = car_count + motorcycle_count
                            motorcycle_percentage = (motorcycle_count / total_vehicles * 100) if total_vehicles > 0 else 0
                            
                            print(f"   Time: {step:4d}s | Phase: {phase_name:12s} | "
                                  f"Wait: {data_point['avg_waiting_time']:5.1f}s | "
                                  f"Speed: {data_point['avg_speed']:5.1f} m/s | "
                                  f"Cars: {car_count:3d} | Bikes: {motorcycle_count:3d} ({motorcycle_percentage:.0f}%)")
                
                except Exception as e:
                    logger.error(f"Error collecting data at step {step}: {e}")
            
            step += 1
        
        traci.close()
        logger.info("Normal mode simulation completed")
        
        # Save normal mode data
        with open(f"{self.base_directory}/normal_mode_data.json", 'w') as f:
            json.dump(self.normal_mode_data, f, indent=2)
    
    def run_adaptive_mode_simulation(self):
        """Run simulation with adaptive edge traffic controller"""
        logger.info("Starting Adaptive Mode Simulation")
        print("\n🚦 RUNNING ADAPTIVE MODE SIMULATION (Mixed Vehicles)")
        print("=" * 70)
        
        # Generate route file
        route_file = self.generate_dynamic_route_file("adaptive")
        config_file = self.create_sumo_config(route_file, "adaptive")
        
        # Start SUMO
        sumo_cmd = ['sumo', '-c', config_file, '--no-warnings', '--no-step-log']
        traci.start(sumo_cmd)
        
        # Initialize improved edge controller
        edge_controller = ImprovedEdgeTrafficController(junction_id="J4", base_green_time=30)
        
        print(f"� Improved Edge Controller: Base {edge_controller.base_green_time}s")
        print(f"   Dynamic intervals: 15-50s, Change limits: 6-35%")
        print(f"   Enhanced categorization with urgency assessment enabled")
        print(f"   vs Normal: Fixed 30s green per direction (no adaptation)")
        
        # Track vehicle types and adaptations
        car_count = 0
        motorcycle_count = 0
        adaptations_count = 0
        
        # Run simulation
        step = 0
        
        while step < self.simulation_duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Apply edge algorithm
            if step > 10:  # Start after initialization
                adaptation_result = edge_controller.apply_edge_algorithm(step)
                if adaptation_result:
                    adaptations_count += 1
            
            # Count vehicle types
            if step % 300 == 0:  # Every 5 minutes
                vehicle_ids = traci.vehicle.getIDList()
                current_cars = sum(1 for vid in vehicle_ids if traci.vehicle.getTypeID(vid) == 'car')
                current_motorcycles = sum(1 for vid in vehicle_ids if traci.vehicle.getTypeID(vid) == 'motorcycle')
                
                if current_cars > 0 or current_motorcycles > 0:
                    car_count = current_cars
                    motorcycle_count = current_motorcycles
            
            # Collect data every 30 seconds
            if step % 30 == 0:
                try:
                    # Get current traffic metrics
                    traffic_data = self.analyzer.collect_traffic_metrics(step, traci)
                    if traffic_data:
                        data_point = {
                            'time': step,
                            'avg_waiting_time': traffic_data.get('avg_waiting_time', 0),
                            'avg_speed': traffic_data.get('avg_speed', 0),
                            'total_vehicles': traffic_data.get('total_vehicles', 0),
                            'throughput': traffic_data.get('throughput', 0),
                            'phase': self._get_current_phase(step),
                            'mode': 'adaptive',
                            'adaptations': adaptations_count,
                            'cars': car_count,
                            'motorcycles': motorcycle_count,
                            'vehicle_mix_ratio': motorcycle_count / (car_count + motorcycle_count) if (car_count + motorcycle_count) > 0 else 0
                        }
                        self.adaptive_mode_data.append(data_point)
                        
                        if step % 300 == 0:  # Print every 5 minutes
                            phase_name = self._get_current_phase_name(step)
                            total_vehicles = car_count + motorcycle_count
                            motorcycle_percentage = (motorcycle_count / total_vehicles * 100) if total_vehicles > 0 else 0
                            
                            print(f"   Time: {step:4d}s | Phase: {phase_name:12s} | "
                                  f"Wait: {data_point['avg_waiting_time']:5.1f}s | "
                                  f"Speed: {data_point['avg_speed']:5.1f} m/s | "
                                  f"Cars: {car_count:3d} | Bikes: {motorcycle_count:3d} ({motorcycle_percentage:.0f}%) | "
                                  f"Adaptations: {adaptations_count}")
                
                except Exception as e:
                    logger.error(f"Error collecting data at step {step}: {e}")
            
            step += 1
        
        traci.close()
        logger.info(f"Adaptive mode simulation completed with {adaptations_count} adaptations")
        
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
        """Analyze results and create comprehensive comparison plots with vehicle mix analysis"""
        logger.info("Analyzing results and generating comprehensive plots")
        print("\n📊 ANALYZING RESULTS AND GENERATING COMPREHENSIVE PLOTS")
        print("=" * 70)
        
        if not self.normal_mode_data or not self.adaptive_mode_data:
            logger.error("Missing simulation data. Run simulations first.")
            return
        
        # Use visualizer to create comprehensive plots
        plot_file = f"{self.base_directory}/dynamic_mixed_vehicle_comparison.png"
        
        # Create enhanced comparison plots
        self.visualizer.create_mixed_vehicle_comparison_plots(
            self.normal_mode_data, 
            self.adaptive_mode_data, 
            self.phase_boundaries,
            plot_file
        )
        
        # Generate detailed summary report
        self._generate_mixed_vehicle_summary_report()
        
        logger.info(f"Comprehensive analysis complete. Results saved in {self.base_directory}/")
    
    def _generate_mixed_vehicle_summary_report(self):
        """Generate comprehensive summary report with vehicle mix analysis"""
        report_file = f"{self.base_directory}/mixed_vehicle_summary_report.txt"
        
        logger.info(f"Generating summary report: {report_file}")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("DYNAMIC MIXED VEHICLE TRAFFIC SCENARIO ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Simulation Time: {self.simulation_duration / 3600:.1f} hours\n")
            f.write(f"Phase Duration: {self.phase_duration / 60:.0f} minutes each\n")
            f.write(f"Vehicle Mix: 70% Cars + 30% Motorcycles\n\n")
            
            # Vehicle type analysis
            if self.adaptive_mode_data:
                avg_cars = sum(d.get('cars', 0) for d in self.adaptive_mode_data) / len(self.adaptive_mode_data)
                avg_motorcycles = sum(d.get('motorcycles', 0) for d in self.adaptive_mode_data) / len(self.adaptive_mode_data)
                avg_total = avg_cars + avg_motorcycles
                actual_motorcycle_ratio = (avg_motorcycles / avg_total * 100) if avg_total > 0 else 0
                
                f.write("VEHICLE MIX ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average Cars in Network: {avg_cars:.1f}\n")
                f.write(f"Average Motorcycles in Network: {avg_motorcycles:.1f}\n")
                f.write(f"Actual Motorcycle Ratio: {actual_motorcycle_ratio:.1f}%\n")
                f.write(f"Target Motorcycle Ratio: 30.0%\n\n")
            
            # Phase-by-phase analysis
            f.write("PHASE-BY-PHASE MIXED VEHICLE ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            
            phase_names = [
                "Low Mixed Traffic All Lanes",
                "Heavy North Traffic (Mixed)", 
                "Heavy East Traffic (Mixed)",
                "Reduced Mixed Traffic All Lanes",
                "Rush Hour - Heavy Mixed Traffic All Lanes",
                "Gradual Reduction Mixed Traffic"
            ]
            
            total_improvement = 0
            valid_phases = 0
            
            for i, phase_name in enumerate(phase_names, 1):
                normal_phase = [d for d in self.normal_mode_data if d['phase'] == i]
                adaptive_phase = [d for d in self.adaptive_mode_data if d['phase'] == i]
                
                if normal_phase and adaptive_phase:
                    normal_wait = sum(d['avg_waiting_time'] for d in normal_phase) / len(normal_phase)
                    adaptive_wait = sum(d['avg_waiting_time'] for d in adaptive_phase) / len(adaptive_phase)
                    
                    if normal_wait > 0:
                        improvement = ((normal_wait - adaptive_wait) / normal_wait) * 100
                        total_improvement += improvement
                        valid_phases += 1
                    else:
                        improvement = 0
                    
                    # Vehicle mix in this phase
                    avg_cars_phase = sum(d.get('cars', 0) for d in adaptive_phase) / len(adaptive_phase)
                    avg_motorcycles_phase = sum(d.get('motorcycles', 0) for d in adaptive_phase) / len(adaptive_phase)
                    
                    f.write(f"\nPhase {i}: {phase_name}\n")
                    f.write(f"  Normal Mode:\n")
                    f.write(f"    Avg Waiting Time: {normal_wait:.2f}s\n")
                    f.write(f"  Adaptive Mode:\n")
                    f.write(f"    Avg Waiting Time: {adaptive_wait:.2f}s\n")
                    f.write(f"    Avg Cars: {avg_cars_phase:.1f}\n")
                    f.write(f"    Avg Motorcycles: {avg_motorcycles_phase:.1f}\n")
                    f.write(f"  Improvement: {improvement:+.1f}%\n")
            
            # Overall summary
            f.write(f"\nOVERALL MIXED VEHICLE PERFORMANCE SUMMARY:\n")
            f.write("-" * 50 + "\n")
            avg_improvement = total_improvement / valid_phases if valid_phases > 0 else 0
            f.write(f"Average Improvement: {avg_improvement:+.1f}%\n")
            
            # Algorithm performance with mixed vehicles
            total_adaptations = max(d.get('adaptations', 0) for d in self.adaptive_mode_data)
            f.write(f"Total Algorithm Adaptations: {total_adaptations}\n")
            f.write(f"Adaptation Rate: {total_adaptations / (self.simulation_duration/60):.1f} adaptations/minute\n")
            
            if avg_improvement > 0:
                f.write("SUCCESS: Edge Algorithm shows positive improvement with mixed vehicles\n")
            else:
                f.write("OPTIMIZATION NEEDED: Edge Algorithm needs tuning for mixed vehicle scenarios\n")
        
        logger.info(f"Summary report generated: {report_file}")

def main():
    """Main function to run the dynamic mixed vehicle scenario test"""
    print("🏍️🚗 DYNAMIC MIXED VEHICLE TRAFFIC SCENARIO TEST")
    print("=" * 70)
    print("Testing Edge Algorithm with Mixed Vehicle Traffic:")
    print("  Vehicle Mix: 70% Cars + 30% Motorcycles")
    print("  Normal Mode: Fixed 30s green per direction (68s cycle)")
    print("  Adaptive Mode: Edge algorithm with 15s-45s adaptive timing")
    print("  Phase 1: Low mixed traffic (30 min)")
    print("  Phase 2: Heavy North mixed traffic (30 min)")
    print("  Phase 3: Heavy East mixed traffic (30 min)")
    print("  Phase 4: Reduced mixed traffic (30 min)")
    print("  Phase 5: Rush hour - all heavy mixed (30 min)")
    print("  Phase 6: Gradual reduction mixed (30 min)")
    print("  Total: 3 hours simulation with cars and motorcycles")
    print()
    
    # Create test instance
    scenario_test = DynamicScenarioTestSrc()
    
    try:
        # Run normal mode simulation
        scenario_test.run_normal_mode_simulation()
        
        # Run adaptive mode simulation
        scenario_test.run_adaptive_mode_simulation()
        
        # Analyze and plot results
        scenario_test.analyze_and_plot_results()
        
        print("\n🎉 DYNAMIC MIXED VEHICLE SCENARIO TEST COMPLETED!")
        print("=" * 70)
        print(f"📁 Results saved in: src/{scenario_test.base_directory}/")
        print("📊 Check the comparison plots and summary report")
        print("🏍️ Mixed vehicle analysis includes cars and motorcycles")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        logger.warning("Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()