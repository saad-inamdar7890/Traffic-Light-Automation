"""
Your Algorithm vs Normal Mode Dynamic Scenario Test
==================================================

This test implements YOUR throughput-optimized algorithm (shorter phases for light traffic)
and compares it against normal fixed-time mode using the same dynamic traffic scenarios.

Your Hypothesis: "Shorter phases for light traffic = better throughput"
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

# Add the f2 directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from your_throughput_controller import YourThroughputOptimizedController
from baseline_fixed_controller import BaselineFixedTimeController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YourAlgorithmDynamicTest:
    """Test YOUR algorithm vs normal mode in dynamic traffic scenarios."""
    
    def __init__(self):
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.results_directory = os.path.join(self.base_directory, "your_algorithm_results")
        
        # Ensure results directory exists
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        
        # Dynamic traffic scenarios (same as f1 but focused on testing your hypothesis)
        self.simulation_duration = 3600  # 1 hour total
        self.phase_duration = 600  # 10 minutes per phase
        
        # Traffic flow patterns for 6 dynamic phases
        self.traffic_phases = {
            'phase_1': {
                'name': 'Low Mixed Traffic',
                'duration': 600,
                'flows': {
                    'north': {'vehicles_per_hour': 180, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 150, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 165, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 135, 'pattern': 'constant'}
                }
            },
            'phase_2': {
                'name': 'Heavy North Traffic',
                'duration': 600,
                'flows': {
                    'north': {'vehicles_per_hour': 900, 'pattern': 'gradual_increase'},
                    'south': {'vehicles_per_hour': 120, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 90, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 75, 'pattern': 'constant'}
                }
            },
            'phase_3': {
                'name': 'Heavy East Traffic',
                'duration': 600,
                'flows': {
                    'north': {'vehicles_per_hour': 105, 'pattern': 'gradual_decrease'},
                    'south': {'vehicles_per_hour': 90, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 975, 'pattern': 'gradual_increase'},
                    'west': {'vehicles_per_hour': 75, 'pattern': 'constant'}
                }
            },
            'phase_4': {
                'name': 'Minimal Traffic',
                'duration': 600,
                'flows': {
                    'north': {'vehicles_per_hour': 60, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 52, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 67, 'pattern': 'gradual_decrease'},
                    'west': {'vehicles_per_hour': 45, 'pattern': 'constant'}
                }
            },
            'phase_5': {
                'name': 'Rush Hour',
                'duration': 600,
                'flows': {
                    'north': {'vehicles_per_hour': 1050, 'pattern': 'sudden_increase'},
                    'south': {'vehicles_per_hour': 1012, 'pattern': 'sudden_increase'},
                    'east': {'vehicles_per_hour': 975, 'pattern': 'sudden_increase'},
                    'west': {'vehicles_per_hour': 1031, 'pattern': 'sudden_increase'}
                }
            },
            'phase_6': {
                'name': 'Gradual Reduction',
                'duration': 600,
                'flows': {
                    'north': {'vehicles_per_hour': 300, 'pattern': 'gradual_decrease'},
                    'south': {'vehicles_per_hour': 270, 'pattern': 'gradual_decrease'},
                    'east': {'vehicles_per_hour': 330, 'pattern': 'gradual_decrease'},
                    'west': {'vehicles_per_hour': 240, 'pattern': 'gradual_decrease'}
                }
            }
        }
        
        self.baseline_data = []
        self.your_algorithm_data = []
        
        logger.info(f"Your algorithm test initialized - {len(self.traffic_phases)} phases")
    
    def generate_route_file(self, mode_suffix: str = "") -> str:
        """Generate route file with dynamic mixed vehicle flow."""
        route_file = f"{self.results_directory}/your_algorithm_{mode_suffix}.rou.xml"
        
        logger.info(f"Generating route file: {route_file}")
        
        with open(route_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
            f.write('<routes>\\n')
            
            # Define vehicle types (70% cars, 30% motorcycles)
            f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="50.0"/>\\n')
            f.write('    <vType id="motorcycle" accel="3.0" decel="5.0" sigma="0.3" length="2.5" maxSpeed="60.0"/>\\n')
            
            # Define routes
            f.write('    <route id="north_south" edges="-E2 E3"/>\\n')
            f.write('    <route id="south_north" edges="E2 -E3"/>\\n')
            f.write('    <route id="east_west" edges="-E1 E4"/>\\n')
            f.write('    <route id="west_east" edges="E1 -E4"/>\\n')
            
            # Generate vehicles for each phase
            vehicle_id = 0
            current_time = 0
            
            for phase_name, phase_config in self.traffic_phases.items():
                logger.info(f"Generating {phase_config['name']} ({current_time}s - {current_time + phase_config['duration']}s)")
                
                phase_start = current_time
                phase_end = current_time + phase_config['duration']
                
                while current_time < phase_end:
                    for direction, config in phase_config['flows'].items():
                        vehicles_per_hour = config['vehicles_per_hour']
                        
                        # Apply pattern variations
                        pattern = config['pattern']
                        if pattern == 'gradual_increase':
                            progress = (current_time - phase_start) / phase_config['duration']
                            multiplier = 0.5 + (progress * 0.5)  # Start at 50%, end at 100%
                        elif pattern == 'gradual_decrease':
                            progress = (current_time - phase_start) / phase_config['duration']
                            multiplier = 1.0 - (progress * 0.5)  # Start at 100%, end at 50%
                        elif pattern == 'sudden_increase':
                            multiplier = 1.2 if current_time > phase_start + 120 else 0.6  # Jump after 2 minutes
                        else:  # constant
                            multiplier = 1.0
                        
                        adjusted_rate = vehicles_per_hour * multiplier
                        interval = 3600 / adjusted_rate if adjusted_rate > 0 else 3600
                        
                        if current_time % interval < 10:  # Generate vehicle
                            # 70% cars, 30% motorcycles
                            vehicle_type = "car" if vehicle_id % 10 < 7 else "motorcycle"
                            
                            if direction == 'north':
                                route = "north_south"
                            elif direction == 'south':
                                route = "south_north"
                            elif direction == 'east':
                                route = "east_west"
                            else:  # west
                                route = "west_east"
                            
                            f.write(f'    <vehicle id="{vehicle_type}_{vehicle_id}" type="{vehicle_type}" '
                                   f'route="{route}" depart="{current_time}"/>\\n')
                            vehicle_id += 1
                    
                    current_time += 10  # 10-second intervals
            
            f.write('</routes>\\n')
        
        logger.info(f"Route file generated: {route_file}")
        return route_file
    
    def run_simulation(self, controller_type: str) -> List[Dict[str, Any]]:
        """Run simulation with specified controller type."""
        mode_suffix = controller_type.lower().replace(' ', '_')
        route_file = self.generate_route_file(mode_suffix)
        config_file = f"{self.results_directory}/your_algorithm_{mode_suffix}.sumocfg"
        
        # Create SUMO config file
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
            f.write('    <processing>\\n')
            f.write('        <ignore-junction-blocker value="60"/>\\n')
            f.write('        <time-to-teleport value="300"/>\\n')
            f.write('    </processing>\\n')
            f.write('</configuration>\\n')
        
        # Initialize TRACI
        try:
            traci.start(["sumo", "-c", config_file, "--no-warnings"])
            
            # Initialize appropriate controller
            if controller_type == "Normal Mode":
                controller = BaselineFixedTimeController()
                print(f"\\n🔄 RUNNING NORMAL MODE SIMULATION")
                print("=" * 60)
            else:  # Your Algorithm
                controller = YourThroughputOptimizedController()
                print(f"\\n🚀 RUNNING YOUR ALGORITHM SIMULATION")
                print("=" * 60)
            
            controller.initialize_traffic_lights()
            
            simulation_data = []
            step = 0
            
            while traci.simulation.getMinExpectedNumber() > 0:
                current_time = traci.simulation.getTime()
                
                # Apply appropriate control
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
                    current_phase_time = current_time % self.simulation_duration
                    phase_index = int(current_phase_time // self.phase_duration)
                    phase_names = ['Low', 'Heavy N', 'Heavy E', 'Minimal', 'Rush Hour', 'Gradual']
                    current_phase_name = phase_names[min(phase_index, len(phase_names) - 1)]
                    
                    # Calculate vehicle mix
                    total_vehicles = traffic_data['total_vehicles']
                    # Approximate motorcycle count (30% target)
                    motorcycle_count = int(total_vehicles * 0.3)
                    car_count = total_vehicles - motorcycle_count
                    motorcycle_percentage = (motorcycle_count / max(total_vehicles, 1)) * 100
                    
                    data_point = {
                        'time': current_time,
                        'avg_waiting_time': traffic_data['avg_waiting_time'],
                        'avg_speed': traffic_data['avg_speed'],
                        'total_vehicles': total_vehicles,
                        'phase': current_phase_name,
                        'mode': controller_type.lower().replace(' ', '_'),
                        'adaptations': adaptations,
                        'cars': car_count,
                        'motorcycles': motorcycle_count,
                        'urgency': urgency
                    }
                    
                    simulation_data.append(data_point)
                    
                    # Print progress
                    if step % 300 == 0:  # Every 5 minutes
                        if controller_type == "Normal Mode":
                            print(f"   Time: {int(current_time):4d}s | Phase: {current_phase_name:10s} | "
                                  f"Wait: {traffic_data['avg_waiting_time']:6.1f}s | "
                                  f"Speed: {traffic_data['avg_speed']:6.1f} m/s | "
                                  f"Cars: {car_count:3d} | Bikes: {motorcycle_count:3d} ({motorcycle_percentage:.0f}%)")
                        else:
                            print(f"   Time: {int(current_time):4d}s | Phase: {current_phase_name:10s} | "
                                  f"Wait: {traffic_data['avg_waiting_time']:6.1f}s | "
                                  f"Speed: {traffic_data['avg_speed']:6.1f} m/s | "
                                  f"Cars: {car_count:3d} | Bikes: {motorcycle_count:3d} ({motorcycle_percentage:.0f}%) | "
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
    
    def save_data(self, baseline_data: List[Dict], adaptive_data: List[Dict]):
        """Save simulation data to JSON files."""
        baseline_file = os.path.join(self.results_directory, "baseline_data.json")
        adaptive_file = os.path.join(self.results_directory, "your_algorithm_data.json")
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        with open(adaptive_file, 'w') as f:
            json.dump(adaptive_data, f, indent=2)
        
        logger.info(f"Data saved: {baseline_file} and {adaptive_file}")
    
    def analyze_results(self, baseline_data: List[Dict], adaptive_data: List[Dict]) -> Dict[str, Any]:
        """Analyze and compare results between normal mode and your algorithm."""
        
        # Overall performance
        baseline_avg_wait = statistics.mean([d['avg_waiting_time'] for d in baseline_data])
        adaptive_avg_wait = statistics.mean([d['avg_waiting_time'] for d in adaptive_data])
        baseline_avg_speed = statistics.mean([d['avg_speed'] for d in baseline_data])
        adaptive_avg_speed = statistics.mean([d['avg_speed'] for d in adaptive_data])
        
        # Total adaptations
        total_adaptations = adaptive_data[-1]['adaptations'] if adaptive_data else 0
        adaptation_rate = total_adaptations / (self.simulation_duration / 60)  # per minute
        
        # Phase-by-phase analysis
        phases = {}
        for phase_name in ['Low', 'Heavy N', 'Heavy E', 'Minimal', 'Rush Hour', 'Gradual']:
            baseline_phase = [d for d in baseline_data if d['phase'] == phase_name]
            adaptive_phase = [d for d in adaptive_data if d['phase'] == phase_name]
            
            if baseline_phase and adaptive_phase:
                baseline_wait = statistics.mean([d['avg_waiting_time'] for d in baseline_phase])
                adaptive_wait = statistics.mean([d['avg_waiting_time'] for d in adaptive_phase])
                improvement = ((baseline_wait - adaptive_wait) / baseline_wait) * 100
                
                phases[phase_name] = {
                    'baseline_wait': baseline_wait,
                    'adaptive_wait': adaptive_wait,
                    'improvement': improvement
                }
        
        # Overall improvement
        overall_improvement = ((baseline_avg_wait - adaptive_avg_wait) / baseline_avg_wait) * 100
        speed_improvement = ((adaptive_avg_speed - baseline_avg_speed) / baseline_avg_speed) * 100
        
        return {
            'overall': {
                'baseline_avg_wait': baseline_avg_wait,
                'adaptive_avg_wait': adaptive_avg_wait,
                'baseline_avg_speed': baseline_avg_speed,
                'adaptive_avg_speed': adaptive_avg_speed,
                'wait_improvement': overall_improvement,
                'speed_improvement': speed_improvement,
                'total_adaptations': total_adaptations,
                'adaptation_rate': adaptation_rate
            },
            'phases': phases
        }
    
    def generate_report(self, analysis: Dict[str, Any]):
        """Generate comprehensive analysis report."""
        report_file = os.path.join(self.results_directory, "your_algorithm_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("YOUR ALGORITHM vs NORMAL MODE ANALYSIS REPORT\\n")
            f.write("=" * 60 + "\\n\\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Total Simulation Time: {self.simulation_duration / 3600:.1f} hours\\n")
            f.write(f"Phase Duration: {self.phase_duration / 60} minutes each\\n")
            f.write(f"Vehicle Mix: 70% Cars + 30% Motorcycles\\n\\n")
            
            overall = analysis['overall']
            f.write("OVERALL PERFORMANCE SUMMARY:\\n")
            f.write("-" * 40 + "\\n")
            f.write(f"Normal Mode Average Waiting Time: {overall['baseline_avg_wait']:.2f}s\\n")
            f.write(f"Your Algorithm Average Waiting Time: {overall['adaptive_avg_wait']:.2f}s\\n")
            f.write(f"Overall Improvement: {overall['wait_improvement']:+.1f}%\\n\\n")
            
            f.write(f"Normal Mode Average Speed: {overall['baseline_avg_speed']:.2f} m/s\\n")
            f.write(f"Your Algorithm Average Speed: {overall['adaptive_avg_speed']:.2f} m/s\\n")
            f.write(f"Speed Improvement: {overall['speed_improvement']:+.1f}%\\n\\n")
            
            f.write("PHASE-BY-PHASE ANALYSIS:\\n")
            f.write("-" * 40 + "\\n\\n")
            
            for phase_name, phase_data in analysis['phases'].items():
                f.write(f"{phase_name}:\\n")
                f.write(f"  Normal Mode: {phase_data['baseline_wait']:.2f}s\\n")
                f.write(f"  Your Algorithm: {phase_data['adaptive_wait']:.2f}s\\n")
                f.write(f"  Improvement: {phase_data['improvement']:+.1f}%\\n\\n")
            
            f.write("YOUR ALGORITHM PERFORMANCE:\\n")
            f.write("-" * 40 + "\\n")
            f.write(f"Total Adaptations: {overall['total_adaptations']}\\n")
            f.write(f"Adaptation Rate: {overall['adaptation_rate']:.1f} adaptations/minute\\n")
        
        logger.info(f"Report generated: {report_file}")
        return report_file
    
    def run_complete_test(self):
        """Run complete comparison test."""
        print("\\n🔬 YOUR ALGORITHM vs NORMAL MODE COMPARISON TEST")
        print("=" * 60)
        print("Testing YOUR hypothesis: Shorter phases for light traffic = better performance")
        print("Your algorithm: Aggressive switching, 12-20s phases for light traffic")
        print("Normal mode: Fixed 60s phases for all traffic conditions")
        print()
        
        # Run normal mode simulation
        baseline_data = self.run_simulation("Normal Mode")
        
        # Run your algorithm simulation
        adaptive_data = self.run_simulation("Your Algorithm")
        
        if baseline_data and adaptive_data:
            # Save data
            self.save_data(baseline_data, adaptive_data)
            
            # Analyze results
            analysis = self.analyze_results(baseline_data, adaptive_data)
            
            # Generate report
            report_file = self.generate_report(analysis)
            
            # Print summary
            self.print_summary(analysis)
            
            return analysis
        else:
            print("❌ Test failed - insufficient data")
            return None
    
    def print_summary(self, analysis: Dict[str, Any]):
        """Print test summary."""
        print("\\n" + "=" * 60)
        print("🏁 YOUR ALGORITHM TEST RESULTS")
        print("=" * 60)
        
        overall = analysis['overall']
        
        print(f"\\n📊 OVERALL PERFORMANCE:")
        print(f"   Normal Mode Wait Time: {overall['baseline_avg_wait']:.2f}s")
        print(f"   Your Algorithm Wait Time: {overall['adaptive_avg_wait']:.2f}s")
        print(f"   Improvement: {overall['wait_improvement']:+.1f}%")
        
        print(f"\\n🚗 SPEED PERFORMANCE:")
        print(f"   Normal Mode Speed: {overall['baseline_avg_speed']:.2f} m/s")
        print(f"   Your Algorithm Speed: {overall['adaptive_avg_speed']:.2f} m/s")
        print(f"   Improvement: {overall['speed_improvement']:+.1f}%")
        
        print(f"\\n🔄 ADAPTATION ACTIVITY:")
        print(f"   Total Adaptations: {overall['total_adaptations']}")
        print(f"   Adaptation Rate: {overall['adaptation_rate']:.1f} per minute")
        
        # Count winning phases
        phases_won = 0
        total_phases = len(analysis['phases'])
        
        print(f"\\n📈 PHASE-BY-PHASE RESULTS:")
        for phase_name, phase_data in analysis['phases'].items():
            improvement = phase_data['improvement']
            status = "✅ WON" if improvement > 0 else "❌ LOST"
            if improvement > 0:
                phases_won += 1
            print(f"   {phase_name}: {improvement:+.1f}% {status}")
        
        print(f"\\n🏆 FINAL VERDICT:")
        if overall['wait_improvement'] > 5 and phases_won > total_phases / 2:
            print("✅ YOUR HYPOTHESIS IS VALIDATED!")
            print("   Your algorithm with shorter phases for light traffic WORKS!")
            print(f"   Won {phases_won}/{total_phases} phases with {overall['wait_improvement']:+.1f}% overall improvement")
        elif overall['wait_improvement'] > 0:
            print("🤔 MIXED RESULTS")
            print("   Your algorithm shows improvement but not decisive")
            print(f"   {overall['wait_improvement']:+.1f}% improvement, won {phases_won}/{total_phases} phases")
        else:
            print("❌ HYPOTHESIS NOT VALIDATED")
            print("   Normal mode performs better overall")
            print(f"   {overall['wait_improvement']:+.1f}% change, won {phases_won}/{total_phases} phases")
        
        # Insights
        print(f"\\n💡 KEY INSIGHTS:")
        if overall['adaptation_rate'] > 2:
            print(f"   ⚠️  High adaptation rate ({overall['adaptation_rate']:.1f}/min) may indicate over-switching")
        else:
            print(f"   ✅ Reasonable adaptation rate ({overall['adaptation_rate']:.1f}/min)")
        
        if overall['speed_improvement'] > 0:
            print(f"   ✅ Speed improvement shows better traffic flow")
        else:
            print(f"   ⚠️  Speed reduction suggests flow disruption")

if __name__ == "__main__":
    test = YourAlgorithmDynamicTest()
    test.run_complete_test()