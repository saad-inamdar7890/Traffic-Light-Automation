"""
Dynamic Continuous Flow Traffic Scenario Test - Fixed Version
=============================================================

Comprehensive test comparing baseline fixed-time controller vs improved adaptive controller
with continuous mixed vehicle traffic (70% cars, 30% motorcycles) over 6 dynamic phases.
"""

import sys
import os
import time
import json
import random
import logging
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict

# TRACI import
import traci

# Import local controllers
from traci_improved_controller import TraciImprovedController
from baseline_controller import BaselineFixedController

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicContinuousFlowTest:
    """Continuous flow dynamic scenario test with mixed vehicles."""
    
    def __init__(self, simulation_duration: int = 21600):  # 6 hours total = 21600 seconds
        self.simulation_duration = simulation_duration
        self.phase_duration = 3600  # 1 hour per phase = 3600 seconds
        
        # Results directory
        self.base_directory = "continuous_flow_results"
        os.makedirs(self.base_directory, exist_ok=True)
        
        # Vehicle types configuration (70% cars, 30% motorcycles)
        self.vehicle_types = {
            'car': {
                'probability': 0.70,
                'vtype_id': 'car',
                'length': 4.5,
                'max_speed': 50.0,
                'accel': 2.5,
                'decel': 4.5,
                'sigma': 0.5,
                'color': '1,1,0'
            },
            'motorcycle': {
                'probability': 0.30,
                'vtype_id': 'motorcycle',
                'length': 2.2,
                'max_speed': 60.0,
                'accel': 3.5,
                'decel': 5.0,
                'sigma': 0.3,
                'color': '0,0,1'
            }
        }
        
        # Traffic flow patterns for 6 dynamic phases (1 hour each)
        self.traffic_phases = {
            'phase_1': {
                'name': 'Low Mixed Traffic',
                'duration': 3600,  # 1 hour
                'flows': {
                    'north': {'vehicles_per_hour': 240, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 200, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 220, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 180, 'pattern': 'constant'}
                }
            },
            'phase_2': {
                'name': 'Heavy North Traffic',
                'duration': 3600,  # 1 hour
                'flows': {
                    'north': {'vehicles_per_hour': 1200, 'pattern': 'gradual_increase'},
                    'south': {'vehicles_per_hour': 160, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 120, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 100, 'pattern': 'constant'}
                }
            },
            'phase_3': {
                'name': 'Heavy East Traffic',
                'duration': 3600,  # 1 hour
                'flows': {
                    'north': {'vehicles_per_hour': 140, 'pattern': 'gradual_decrease'},
                    'south': {'vehicles_per_hour': 120, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 1300, 'pattern': 'gradual_increase'},
                    'west': {'vehicles_per_hour': 100, 'pattern': 'constant'}
                }
            },
            'phase_4': {
                'name': 'Minimal Traffic',
                'duration': 3600,  # 1 hour
                'flows': {
                    'north': {'vehicles_per_hour': 80, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 70, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 90, 'pattern': 'gradual_decrease'},
                    'west': {'vehicles_per_hour': 60, 'pattern': 'constant'}
                }
            },
            'phase_5': {
                'name': 'Rush Hour',
                'duration': 3600,  # 1 hour
                'flows': {
                    'north': {'vehicles_per_hour': 1400, 'pattern': 'sudden_increase'},
                    'south': {'vehicles_per_hour': 1350, 'pattern': 'sudden_increase'},
                    'east': {'vehicles_per_hour': 1300, 'pattern': 'sudden_increase'},
                    'west': {'vehicles_per_hour': 1375, 'pattern': 'sudden_increase'}
                }
            },
            'phase_6': {
                'name': 'Gradual Reduction',
                'duration': 3600,  # 1 hour
                'flows': {
                    'north': {'vehicles_per_hour': 400, 'pattern': 'gradual_decrease'},
                    'south': {'vehicles_per_hour': 360, 'pattern': 'gradual_decrease'},
                    'east': {'vehicles_per_hour': 440, 'pattern': 'gradual_decrease'},
                    'west': {'vehicles_per_hour': 320, 'pattern': 'gradual_decrease'}
                }
            }
        }
        
        self.baseline_data = []
        self.adaptive_data = []
        
        logger.info(f"Dynamic continuous flow test initialized - {len(self.traffic_phases)} phases")
    
    def generate_route_file(self, mode_suffix: str = "") -> str:
        """Generate route file with continuous mixed vehicle flow."""
        route_file = f"{self.base_directory}/continuous_flow_{mode_suffix}.rou.xml"
        
        logger.info(f"Generating route file: {route_file}")
        
        with open(route_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')
            
            # Define vehicle types
            for vtype_name, vtype_config in self.vehicle_types.items():
                f.write(f'    <vType id="{vtype_config["vtype_id"]}" '
                       f'accel="{vtype_config["accel"]}" '
                       f'decel="{vtype_config["decel"]}" '
                       f'sigma="{vtype_config["sigma"]}" '
                       f'length="{vtype_config["length"]}" '
                       f'maxSpeed="{vtype_config["max_speed"]}" '
                       f'color="{vtype_config["color"]}" />\n')
            
            # Define routes
            routes = {
                'north_to_south': ['E1', 'E1.200'],
                'south_to_north': ['-E1', '-E1.238'],
                'east_to_west': ['E0', 'E0.319'],
                'west_to_east': ['-E0', '-E0.254']
            }
            
            for route_id, edges in routes.items():
                f.write(f'    <route id="{route_id}" edges="{" ".join(edges)}" />\n')
            
            # Generate vehicles for each phase
            current_time = 0
            vehicle_id = 0
            
            for phase_key, phase_data in self.traffic_phases.items():
                phase_start = current_time
                phase_end = current_time + phase_data['duration']
                
                logger.info(f"Generating {phase_data['name']} ({phase_start}s - {phase_end}s)")
                
                for direction, flow_data in phase_data['flows'].items():
                    vehicles_generated = self._generate_continuous_flow(
                        f, direction, flow_data, phase_start, phase_end, vehicle_id
                    )
                    vehicle_id += vehicles_generated
                
                current_time = phase_end
            
            f.write('</routes>\n')
        
        logger.info(f"Route file generated: {route_file}")
        return route_file
    
    def _generate_continuous_flow(self, file_obj, direction: str, flow_data: Dict[str, Any], 
                                start_time: int, end_time: int, start_id: int) -> int:
        """Generate continuous flow of mixed vehicles."""
        base_rate = flow_data['vehicles_per_hour']
        pattern = flow_data['pattern']
        duration = end_time - start_time
        vehicles_generated = 0
        
        route_map = {
            'north': 'north_to_south',
            'south': 'south_to_north',
            'east': 'east_to_west',
            'west': 'west_to_east'
        }
        
        route_id = route_map[direction]
        
        if pattern == 'constant':
            vehicles_per_second = base_rate / 3600
            interval = 1.0 / vehicles_per_second if vehicles_per_second > 0 else 60
            
            current_time = start_time
            while current_time < end_time:
                vehicle_type = self._get_random_vehicle_type()
                file_obj.write(f'    <vehicle id="veh_{start_id + vehicles_generated}" '
                             f'type="{vehicle_type}" route="{route_id}" depart="{current_time:.1f}" />\n')
                vehicles_generated += 1
                current_time += interval
        
        elif pattern == 'gradual_increase':
            time_steps = duration // 60
            for step in range(int(time_steps)):
                step_start = start_time + (step * 60)
                step_end = min(step_start + 60, end_time)
                
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
            time_steps = duration // 60
            for step in range(int(time_steps)):
                step_start = start_time + (step * 60)
                step_end = min(step_start + 60, end_time)
                
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
    
    def _get_random_vehicle_type(self) -> str:
        """Randomly select vehicle type (70% cars, 30% motorcycles)."""
        rand_val = random.random()
        return 'car' if rand_val < 0.70 else 'motorcycle'
    
    def create_sumo_config(self, route_file: str, mode_suffix: str = "") -> str:
        """Create SUMO configuration file."""
        config_file = f"{self.base_directory}/continuous_flow_config_{mode_suffix}.sumocfg"
        
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
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return config_file
    
    def run_baseline_simulation(self):
        """Run simulation with baseline fixed-time controller."""
        logger.info("Starting Baseline Simulation")
        print("\n🚦 RUNNING BASELINE SIMULATION (Fixed-Time Controller)")
        print("=" * 70)
        
        route_file = self.generate_route_file("baseline")
        config_file = self.create_sumo_config(route_file, "baseline")
        
        sumo_cmd = ['sumo', '-c', config_file, '--no-warnings', '--no-step-log']
        traci.start(sumo_cmd)
        
        controller = BaselineFixedController(junction_id="J4", cycle_time=90)
        controller.initialize_traffic_lights()
        
        print(f"🚦 Baseline Controller: Fixed 45s green per direction (90s cycle)")
        print(f"   Vehicle Mix: 70% Cars + 30% Motorcycles")
        
        car_count = 0
        motorcycle_count = 0
        step = 0
        
        while step < self.simulation_duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            controller.apply_fixed_time_control()
            
            if step % 300 == 0:
                vehicle_ids = traci.vehicle.getIDList()
                car_count = sum(1 for vid in vehicle_ids if traci.vehicle.getTypeID(vid) == 'car')
                motorcycle_count = sum(1 for vid in vehicle_ids if traci.vehicle.getTypeID(vid) == 'motorcycle')
            
            if step % 30 == 0:
                try:
                    traffic_data = controller.collect_traffic_data()
                    if traffic_data:
                        data_point = {
                            'time': step,
                            'avg_waiting_time': traffic_data.get('avg_waiting_time', 0),
                            'avg_speed': traffic_data.get('avg_speed', 0),
                            'total_vehicles': traffic_data.get('total_vehicles', 0),
                            'phase': self._get_current_phase(step),
                            'mode': 'baseline',
                            'cars': car_count,
                            'motorcycles': motorcycle_count
                        }
                        self.baseline_data.append(data_point)
                        
                        if step % 300 == 0:
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
        logger.info("Baseline simulation completed")
        
        with open(f"{self.base_directory}/baseline_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.baseline_data, f, indent=2)
    
    def run_adaptive_simulation(self):
        """Run simulation with improved adaptive controller."""
        logger.info("Starting Adaptive Simulation")
        print("\n🚦 RUNNING ADAPTIVE SIMULATION (Improved Controller)")
        print("=" * 70)
        
        route_file = self.generate_route_file("adaptive")
        config_file = self.create_sumo_config(route_file, "adaptive")
        
        sumo_cmd = ['sumo', '-c', config_file, '--no-warnings', '--no-step-log']
        traci.start(sumo_cmd)
        
        controller = TraciImprovedController(junction_id="J4")
        controller.initialize_traffic_lights()
        
        print(f"🚀 Improved Adaptive Controller:")
        print(f"   Dynamic intervals: 15-50s based on traffic urgency")
        print(f"   Enhanced categorization with urgency assessment")
        print(f"   Vehicle Mix: 70% Cars + 30% Motorcycles")
        
        car_count = 0
        motorcycle_count = 0
        adaptations_count = 0
        step = 0
        
        while step < self.simulation_duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            if step > 10:
                adaptation_result = controller.apply_adaptive_control(step)
                if adaptation_result and adaptation_result.get('adaptation_made'):
                    adaptations_count += 1
            
            if step % 300 == 0:
                vehicle_ids = traci.vehicle.getIDList()
                car_count = sum(1 for vid in vehicle_ids if traci.vehicle.getTypeID(vid) == 'car')
                motorcycle_count = sum(1 for vid in vehicle_ids if traci.vehicle.getTypeID(vid) == 'motorcycle')
            
            if step % 30 == 0:
                try:
                    traffic_data = controller.collect_traffic_data()
                    if traffic_data:
                        data_point = {
                            'time': step,
                            'avg_waiting_time': traffic_data.get('avg_waiting_time', 0),
                            'avg_speed': traffic_data.get('avg_speed', 0),
                            'total_vehicles': traffic_data.get('total_vehicles', 0),
                            'phase': self._get_current_phase(step),
                            'mode': 'adaptive',
                            'adaptations': adaptations_count,
                            'cars': car_count,
                            'motorcycles': motorcycle_count
                        }
                        self.adaptive_data.append(data_point)
                        
                        if step % 300 == 0:
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
        logger.info(f"Adaptive simulation completed with {adaptations_count} adaptations")
        
        with open(f"{self.base_directory}/adaptive_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.adaptive_data, f, indent=2)
    
    def _get_current_phase(self, time_step: int) -> int:
        """Get current phase number based on time."""
        return min(int(time_step // self.phase_duration) + 1, 6)
    
    def _get_current_phase_name(self, time_step: int) -> str:
        """Get current phase name based on time."""
        phase_num = self._get_current_phase(time_step)
        phase_names = {
            1: "Low", 2: "Heavy N", 3: "Heavy E", 
            4: "Minimal", 5: "Rush Hour", 6: "Gradual"
        }
        return phase_names.get(phase_num, "Unknown")
    
    def analyze_and_generate_report(self):
        """Analyze results and generate report."""
        logger.info("Analyzing results and generating report")
        print("\n📊 ANALYZING RESULTS AND GENERATING REPORT")
        print("=" * 70)
        
        if not self.baseline_data or not self.adaptive_data:
            logger.error("Missing simulation data. Run simulations first.")
            return
        
        report_file = f"{self.base_directory}/continuous_flow_summary_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("DYNAMIC CONTINUOUS FLOW TRAFFIC SCENARIO ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Simulation Time: {self.simulation_duration / 3600:.1f} hours\n")
            f.write(f"Phase Duration: {self.phase_duration / 60:.0f} minutes each\n")
            f.write(f"Vehicle Mix: 70% Cars + 30% Motorcycles\n\n")
            
            # Calculate overall performance
            baseline_avg_wait = sum(d['avg_waiting_time'] for d in self.baseline_data) / len(self.baseline_data)
            adaptive_avg_wait = sum(d['avg_waiting_time'] for d in self.adaptive_data) / len(self.adaptive_data)
            
            overall_improvement = ((baseline_avg_wait - adaptive_avg_wait) / baseline_avg_wait) * 100 if baseline_avg_wait > 0 else 0
            
            f.write("OVERALL PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Baseline Average Waiting Time: {baseline_avg_wait:.2f}s\n")
            f.write(f"Adaptive Average Waiting Time: {adaptive_avg_wait:.2f}s\n")
            f.write(f"Overall Improvement: {overall_improvement:+.1f}%\n\n")
            
            # Phase-by-phase analysis
            f.write("PHASE-BY-PHASE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            phase_names = ["Low Mixed Traffic", "Heavy North Traffic", "Heavy East Traffic",
                          "Minimal Mixed Traffic", "Rush Hour", "Gradual Reduction"]
            
            for i, phase_name in enumerate(phase_names, 1):
                baseline_phase = [d for d in self.baseline_data if d['phase'] == i]
                adaptive_phase = [d for d in self.adaptive_data if d['phase'] == i]
                
                if baseline_phase and adaptive_phase:
                    baseline_wait = sum(d['avg_waiting_time'] for d in baseline_phase) / len(baseline_phase)
                    adaptive_wait = sum(d['avg_waiting_time'] for d in adaptive_phase) / len(adaptive_phase)
                    
                    improvement = ((baseline_wait - adaptive_wait) / baseline_wait) * 100 if baseline_wait > 0 else 0
                    
                    f.write(f"\nPhase {i}: {phase_name}\n")
                    f.write(f"  Baseline: {baseline_wait:.2f}s\n")
                    f.write(f"  Adaptive: {adaptive_wait:.2f}s\n")
                    f.write(f"  Improvement: {improvement:+.1f}%\n")
            
            # Algorithm performance
            total_adaptations = max(d.get('adaptations', 0) for d in self.adaptive_data) if self.adaptive_data else 0
            f.write(f"\nALGORITHM PERFORMANCE:\n")
            f.write(f"Total Adaptations: {total_adaptations}\n")
            f.write(f"Adaptation Rate: {total_adaptations / (self.simulation_duration/60):.1f} adaptations/minute\n")
            
            if overall_improvement > 0:
                f.write(f"\nSUCCESS: Improved algorithm shows {overall_improvement:.1f}% improvement\n")
            else:
                f.write(f"\nOPTIMIZATION NEEDED: Algorithm needs further tuning\n")
        
        logger.info(f"Analysis complete. Report saved: {report_file}")
        print(f"📊 Analysis complete. Report saved: {report_file}")
        
        return overall_improvement

def main():
    """Main function to run the dynamic continuous flow scenario test."""
    print("🚗🏍️ DYNAMIC CONTINUOUS FLOW TRAFFIC SCENARIO TEST")
    print("=" * 70)
    print("Testing Improved Algorithm with Continuous Mixed Vehicle Flow:")
    print("  Baseline Mode: Fixed 45s green per direction (90s cycle)")
    print("  Adaptive Mode: Improved algorithm with dynamic 15-50s timing")
    print("  Vehicle Mix: 70% Cars + 30% Motorcycles")
    print("  6 Dynamic Phases: 10 minutes each (1-hour total)")
    print("  Continuous Flow: Realistic traffic generation patterns")
    print()
    
    continuous_flow_test = DynamicContinuousFlowTest()
    
    try:
        continuous_flow_test.run_baseline_simulation()
        continuous_flow_test.run_adaptive_simulation()
        improvement = continuous_flow_test.analyze_and_generate_report()
        
        print("\n🎉 CONTINUOUS FLOW SCENARIO TEST COMPLETED!")
        print("=" * 70)
        print(f"📁 Results saved in: f1/{continuous_flow_test.base_directory}/")
        print(f"📊 Overall improvement: {improvement:+.1f}%" if improvement else "📊 Check report for detailed analysis")
        print("🚗🏍️ Mixed vehicle continuous flow analysis complete")
        
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