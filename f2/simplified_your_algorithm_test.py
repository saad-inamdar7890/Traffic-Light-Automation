"""
Simplified Your Algorithm Test
=============================

A simplified version that uses the same infrastructure as f1 but tests YOUR algorithm approach.
"""

import os
import sys
import traci
import time
import json
import statistics
from typing import Dict, Any, List
import logging
from datetime import datetime

# Add both f1 and f2 directories to Python path
f1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "f1")
f2_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f1_path)
sys.path.append(f2_path)

from your_throughput_controller import YourThroughputOptimizedController
from baseline_fixed_controller import BaselineFixedTimeController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedYourAlgorithmTest:
    """Simplified test of YOUR algorithm vs baseline using existing infrastructure."""
    
    def __init__(self):
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.results_directory = os.path.join(self.base_directory, "simplified_results")
        
        # Use parent directory's demo files
        self.parent_directory = os.path.dirname(self.base_directory)
        self.sumo_config = os.path.join(self.parent_directory, "demo.sumocfg")
        
        # Ensure results directory exists
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        
        # Shorter test for quick results
        self.simulation_duration = 1800  # 30 minutes
        self.phase_duration = 300  # 5 minutes per phase
        
        # 6 test phases with varying traffic intensities
        self.traffic_phases = {
            'phase_1': {
                'name': 'Light Traffic',
                'duration': 300,
                'flows': {
                    'north': {'vehicles_per_hour': 120},
                    'south': {'vehicles_per_hour': 100},
                    'east': {'vehicles_per_hour': 110},
                    'west': {'vehicles_per_hour': 90}
                }
            },
            'phase_2': {
                'name': 'Heavy North',
                'duration': 300,
                'flows': {
                    'north': {'vehicles_per_hour': 600},
                    'south': {'vehicles_per_hour': 80},
                    'east': {'vehicles_per_hour': 70},
                    'west': {'vehicles_per_hour': 60}
                }
            },
            'phase_3': {
                'name': 'Heavy East',
                'duration': 300,
                'flows': {
                    'north': {'vehicles_per_hour': 80},
                    'south': {'vehicles_per_hour': 70},
                    'east': {'vehicles_per_hour': 600},
                    'west': {'vehicles_per_hour': 60}
                }
            },
            'phase_4': {
                'name': 'Minimal Traffic',
                'duration': 300,
                'flows': {
                    'north': {'vehicles_per_hour': 40},
                    'south': {'vehicles_per_hour': 35},
                    'east': {'vehicles_per_hour': 45},
                    'west': {'vehicles_per_hour': 30}
                }
            },
            'phase_5': {
                'name': 'Rush Hour',
                'duration': 300,
                'flows': {
                    'north': {'vehicles_per_hour': 720},
                    'south': {'vehicles_per_hour': 700},
                    'east': {'vehicles_per_hour': 680},
                    'west': {'vehicles_per_hour': 710}
                }
            },
            'phase_6': {
                'name': 'Evening Reduction',
                'duration': 300,
                'flows': {
                    'north': {'vehicles_per_hour': 200},
                    'south': {'vehicles_per_hour': 180},
                    'east': {'vehicles_per_hour': 220},
                    'west': {'vehicles_per_hour': 160}
                }
            }
        }
        
        logger.info(f"Simplified test initialized - {len(self.traffic_phases)} phases")
    
    def generate_route_file(self, mode_suffix: str = "") -> str:
        """Generate simplified route file."""
        route_file = f"{self.results_directory}/simplified_{mode_suffix}.rou.xml"
        
        logger.info(f"Generating route file: {route_file}")
        
        with open(route_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
            f.write('<routes>\\n')
            
            # Define vehicle types
            f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="50.0"/>\\n')
            f.write('    <vType id="motorcycle" accel="3.0" decel="5.0" sigma="0.3" length="2.5" maxSpeed="60.0"/>\\n')
            
            # Define routes
            f.write('    <route id="north_south" edges="-E2 E3"/>\\n')
            f.write('    <route id="south_north" edges="E2 -E3"/>\\n')
            f.write('    <route id="east_west" edges="-E1 E4"/>\\n')
            f.write('    <route id="west_east" edges="E1 -E4"/>\\n')
            
            # Generate vehicles
            vehicle_id = 0
            current_time = 0
            
            for phase_name, phase_config in self.traffic_phases.items():
                phase_start = current_time
                phase_end = current_time + phase_config['duration']
                
                while current_time < phase_end:
                    for direction, config in phase_config['flows'].items():
                        vehicles_per_hour = config['vehicles_per_hour']
                        interval = 3600 / vehicles_per_hour if vehicles_per_hour > 0 else 3600
                        
                        if current_time % interval < 15:  # Generate vehicle
                            # 70% cars, 30% motorcycles
                            vehicle_type = "car" if vehicle_id % 10 < 7 else "motorcycle"
                            
                            route_map = {
                                'north': "north_south",
                                'south': "south_north", 
                                'east': "east_west",
                                'west': "west_east"
                            }
                            
                            f.write(f'    <vehicle id="{vehicle_type}_{vehicle_id}" type="{vehicle_type}" '
                                   f'route="{route_map[direction]}" depart="{current_time}"/>\\n')
                            vehicle_id += 1
                    
                    current_time += 15  # 15-second intervals
            
            f.write('</routes>\\n')
        
        logger.info(f"Route file generated: {route_file}")
        return route_file
    
    def run_simulation(self, controller_type: str) -> List[Dict[str, Any]]:
        """Run simulation with specified controller."""
        mode_suffix = controller_type.lower().replace(' ', '_')
        route_file = self.generate_route_file(mode_suffix)
        
        # Create config file
        config_file = f"{self.results_directory}/simplified_{mode_suffix}.sumocfg"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
            f.write('<configuration>\\n')
            f.write('    <input>\\n')
            f.write(f'        <net-file value="../demo.net.xml"/>\\n')
            f.write(f'        <route-files value="{os.path.basename(route_file)}"/>\\n')
            f.write('    </input>\\n')
            f.write('    <output>\\n')
            f.write(f'        <tripinfo-output value="tripinfo_{mode_suffix}.xml"/>\\n')
            f.write('    </output>\\n')
            f.write('    <time>\\n')
            f.write(f'        <begin value="0"/>\\n')
            f.write(f'        <end value="{self.simulation_duration}"/>\\n')
            f.write('    </time>\\n')
            f.write('</configuration>\\n')
        
        # Run simulation
        try:
            traci.start(["sumo", "-c", config_file, "--no-warnings"])
            
            # Initialize controller
            if controller_type == "Normal Mode":
                controller = BaselineFixedTimeController()
                print(f"\\n🔄 RUNNING NORMAL MODE SIMULATION")
            else:
                controller = YourThroughputOptimizedController()
                print(f"\\n🚀 RUNNING YOUR ALGORITHM SIMULATION")
            
            print("=" * 60)
            controller.initialize_traffic_lights()
            
            simulation_data = []
            step = 0
            
            while traci.simulation.getMinExpectedNumber() > 0:
                current_time = traci.simulation.getTime()
                
                # Apply control
                if controller_type == "Normal Mode":
                    control_result = controller.apply_fixed_control(current_time)
                    adaptations = 0
                    urgency = "FIXED"
                else:
                    control_result = controller.apply_adaptive_control(current_time)
                    adaptations = control_result['total_adaptations']
                    urgency = control_result['urgency_level']
                
                # Collect data every 30 seconds
                if step % 30 == 0:
                    traffic_data = control_result['traffic_data']
                    
                    # Determine current phase
                    phase_index = int(current_time // self.phase_duration)
                    phase_names = ['Light', 'Heavy N', 'Heavy E', 'Minimal', 'Rush Hour', 'Evening']
                    current_phase_name = phase_names[min(phase_index, len(phase_names) - 1)]
                    
                    # Vehicle mix
                    total_vehicles = traffic_data['total_vehicles']
                    motorcycle_count = int(total_vehicles * 0.3)
                    car_count = total_vehicles - motorcycle_count
                    
                    data_point = {
                        'time': current_time,
                        'avg_waiting_time': traffic_data['avg_waiting_time'],
                        'avg_speed': traffic_data['avg_speed'],
                        'total_vehicles': total_vehicles,
                        'phase': current_phase_name,
                        'mode': mode_suffix,
                        'adaptations': adaptations,
                        'cars': car_count,
                        'motorcycles': motorcycle_count,
                        'urgency': urgency
                    }
                    
                    simulation_data.append(data_point)
                    
                    # Print progress
                    if step % 150 == 0:  # Every 2.5 minutes
                        if controller_type == "Normal Mode":
                            print(f"   Time: {int(current_time):4d}s | Phase: {current_phase_name:10s} | "
                                  f"Wait: {traffic_data['avg_waiting_time']:6.1f}s | "
                                  f"Speed: {traffic_data['avg_speed']:6.1f} m/s | "
                                  f"Vehicles: {total_vehicles:3d}")
                        else:
                            print(f"   Time: {int(current_time):4d}s | Phase: {current_phase_name:10s} | "
                                  f"Wait: {traffic_data['avg_waiting_time']:6.1f}s | "
                                  f"Speed: {traffic_data['avg_speed']:6.1f} m/s | "
                                  f"Vehicles: {total_vehicles:3d} | "
                                  f"Adaptations: {adaptations:3d} | Urgency: {urgency}")
                
                traci.simulationStep()
                step += 1
            
            logger.info(f"{controller_type} simulation completed")
            return simulation_data
            
        except Exception as e:
            logger.error(f"Error in {controller_type} simulation: {e}")
            return []
        
        finally:
            try:
                traci.close()
            except:
                pass
    
    def analyze_and_compare(self, baseline_data: List[Dict], adaptive_data: List[Dict]):
        """Analyze and compare results."""
        if not baseline_data or not adaptive_data:
            print("❌ Insufficient data for analysis")
            return
        
        # Save data
        with open(f"{self.results_directory}/baseline_data.json", 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        with open(f"{self.results_directory}/adaptive_data.json", 'w') as f:
            json.dump(adaptive_data, f, indent=2)
        
        # Overall analysis
        baseline_avg_wait = statistics.mean([d['avg_waiting_time'] for d in baseline_data])
        adaptive_avg_wait = statistics.mean([d['avg_waiting_time'] for d in adaptive_data])
        baseline_avg_speed = statistics.mean([d['avg_speed'] for d in baseline_data])
        adaptive_avg_speed = statistics.mean([d['avg_speed'] for d in adaptive_data])
        
        wait_improvement = ((baseline_avg_wait - adaptive_avg_wait) / baseline_avg_wait) * 100
        speed_improvement = ((adaptive_avg_speed - baseline_avg_speed) / baseline_avg_speed) * 100
        
        total_adaptations = adaptive_data[-1]['adaptations'] if adaptive_data else 0
        adaptation_rate = total_adaptations / (self.simulation_duration / 60)
        
        # Phase analysis
        phase_results = {}
        for phase_name in ['Light', 'Heavy N', 'Heavy E', 'Minimal', 'Rush Hour', 'Evening']:
            baseline_phase = [d for d in baseline_data if d['phase'] == phase_name]
            adaptive_phase = [d for d in adaptive_data if d['phase'] == phase_name]
            
            if baseline_phase and adaptive_phase:
                baseline_wait = statistics.mean([d['avg_waiting_time'] for d in baseline_phase])
                adaptive_wait = statistics.mean([d['avg_waiting_time'] for d in adaptive_phase])
                improvement = ((baseline_wait - adaptive_wait) / baseline_wait) * 100
                phase_results[phase_name] = improvement
        
        # Print results
        print("\\n" + "=" * 60)
        print("🏁 YOUR ALGORITHM TEST RESULTS")
        print("=" * 60)
        
        print(f"\\n📊 OVERALL PERFORMANCE:")
        print(f"   Normal Mode Wait Time: {baseline_avg_wait:.2f}s")
        print(f"   Your Algorithm Wait Time: {adaptive_avg_wait:.2f}s")
        print(f"   Wait Time Improvement: {wait_improvement:+.1f}%")
        
        print(f"\\n🚗 SPEED PERFORMANCE:")
        print(f"   Normal Mode Speed: {baseline_avg_speed:.2f} m/s")
        print(f"   Your Algorithm Speed: {adaptive_avg_speed:.2f} m/s")
        print(f"   Speed Improvement: {speed_improvement:+.1f}%")
        
        print(f"\\n🔄 YOUR ALGORITHM ACTIVITY:")
        print(f"   Total Adaptations: {total_adaptations}")
        print(f"   Adaptation Rate: {adaptation_rate:.1f} per minute")
        
        print(f"\\n📈 PHASE-BY-PHASE RESULTS:")
        phases_won = 0
        for phase_name, improvement in phase_results.items():
            status = "✅ WON" if improvement > 0 else "❌ LOST"
            if improvement > 0:
                phases_won += 1
            print(f"   {phase_name:12s}: {improvement:+6.1f}% {status}")
        
        total_phases = len(phase_results)
        print(f"\\n🏆 FINAL VERDICT:")
        
        if wait_improvement > 10 and phases_won > total_phases * 0.7:
            print("🎉 OUTSTANDING SUCCESS!")
            print("   Your algorithm with shorter phases for light traffic works excellently!")
            print(f"   {wait_improvement:+.1f}% improvement, won {phases_won}/{total_phases} phases")
        elif wait_improvement > 5 and phases_won > total_phases / 2:
            print("✅ SUCCESS!")
            print("   Your hypothesis is validated - shorter phases improve performance!")
            print(f"   {wait_improvement:+.1f}% improvement, won {phases_won}/{total_phases} phases")
        elif wait_improvement > 0:
            print("🤔 MIXED RESULTS")
            print("   Some improvement but not decisive")
            print(f"   {wait_improvement:+.1f}% improvement, won {phases_won}/{total_phases} phases")
        else:
            print("❌ HYPOTHESIS NOT VALIDATED")
            print("   Normal mode performs better overall")
            print(f"   {wait_improvement:+.1f}% change, won {phases_won}/{total_phases} phases")
        
        print(f"\\n💡 KEY INSIGHTS:")
        if adaptation_rate > 3:
            print(f"   ⚠️  Very high adaptation rate ({adaptation_rate:.1f}/min) - may be over-switching")
        elif adaptation_rate > 1.5:
            print(f"   ⚠️  High adaptation rate ({adaptation_rate:.1f}/min) - frequent switching")
        else:
            print(f"   ✅ Reasonable adaptation rate ({adaptation_rate:.1f}/min)")
        
        if speed_improvement > 0:
            print(f"   ✅ Speed improvement indicates better traffic flow")
        else:
            print(f"   ⚠️  Speed reduction may indicate flow disruption from switching")
        
        # Generate simple report
        report_file = f"{self.results_directory}/your_algorithm_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"YOUR ALGORITHM TEST REPORT\\n")
            f.write(f"=" * 40 + "\\n\\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Duration: {self.simulation_duration/60:.0f} minutes\\n\\n")
            f.write(f"Overall Wait Time Improvement: {wait_improvement:+.1f}%\\n")
            f.write(f"Overall Speed Improvement: {speed_improvement:+.1f}%\\n")
            f.write(f"Total Adaptations: {total_adaptations}\\n")
            f.write(f"Adaptation Rate: {adaptation_rate:.1f}/minute\\n")
            f.write(f"Phases Won: {phases_won}/{total_phases}\\n")
        
        print(f"\\n📄 Report saved: {report_file}")
    
    def run_complete_test(self):
        """Run the complete test."""
        print("\\n🔬 YOUR ALGORITHM vs NORMAL MODE - SIMPLIFIED TEST")
        print("=" * 60)
        print("Testing: Shorter phases for light traffic = better throughput")
        print()
        
        # Run both simulations
        baseline_data = self.run_simulation("Normal Mode")
        adaptive_data = self.run_simulation("Your Algorithm")
        
        # Analyze results
        self.analyze_and_compare(baseline_data, adaptive_data)

if __name__ == "__main__":
    test = SimplifiedYourAlgorithmTest()
    test.run_complete_test()