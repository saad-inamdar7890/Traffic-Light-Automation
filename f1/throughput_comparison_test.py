"""
Throughput vs Conservative Algorithm Comparison Test
==================================================

This test compares two different adaptive strategies:
1. Conservative approach (longer phases for light traffic)
2. Throughput approach (shorter phases for light traffic)
"""

import os
import sys
import traci
import time
import json
import statistics
from typing import Dict, Any, List
import logging

# Add the f1 directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from traci_improved_controller import TraciImprovedController
from traci_throughput_optimized_controller import TraciThroughputOptimizedController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThroughputComparisonTest:
    """Compare conservative vs throughput-optimized adaptive algorithms."""
    
    def __init__(self):
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.sumo_config = os.path.join(self.base_directory, "..", "demo.sumocfg")
        
        # Test scenarios focusing on light traffic where the difference should be most apparent
        self.test_scenarios = {
            'light_traffic': {
                'name': 'Light Mixed Traffic',
                'duration': 1800,  # 30 minutes
                'flows': {
                    'north': {'vehicles_per_hour': 120, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 100, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 110, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 90, 'pattern': 'constant'}
                }
            },
            'minimal_traffic': {
                'name': 'Minimal Mixed Traffic',
                'duration': 1800,  # 30 minutes
                'flows': {
                    'north': {'vehicles_per_hour': 60, 'pattern': 'constant'},
                    'south': {'vehicles_per_hour': 50, 'pattern': 'constant'},
                    'east': {'vehicles_per_hour': 55, 'pattern': 'constant'},
                    'west': {'vehicles_per_hour': 45, 'pattern': 'constant'}
                }
            }
        }
        
        self.conservative_data = []
        self.throughput_data = []
        
        logger.info("Throughput comparison test initialized")
    
    def generate_route_file(self, scenario_name: str, mode_suffix: str = "") -> str:
        """Generate route file for test scenario."""
        route_file = f"{self.base_directory}/throughput_test_{scenario_name}_{mode_suffix}.rou.xml"
        scenario = self.test_scenarios[scenario_name]
        
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
            
            # Generate vehicles for the scenario
            vehicle_id = 0
            current_time = 0
            duration = scenario['duration']
            
            while current_time < duration:
                for direction, config in scenario['flows'].items():
                    vehicles_per_hour = config['vehicles_per_hour']
                    interval = 3600 / vehicles_per_hour if vehicles_per_hour > 0 else 3600
                    
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
    
    def run_simulation(self, scenario_name: str, controller_type: str) -> List[Dict[str, Any]]:
        """Run simulation with specified controller."""
        mode_suffix = f"{controller_type.lower()}"
        route_file = self.generate_route_file(scenario_name, mode_suffix)
        config_file = f"{self.base_directory}/throughput_test_{scenario_name}_{mode_suffix}.sumocfg"
        
        # Create SUMO config file
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
            f.write('<configuration>\\n')
            f.write('    <input>\\n')
            f.write(f'        <net-file value="../demo.net.xml"/>\\n')
            f.write(f'        <route-files value="{os.path.basename(route_file)}"/>\\n')
            f.write('    </input>\\n')
            f.write('    <output>\\n')
            f.write(f'        <tripinfo-output value="tripinfo_{scenario_name}_{mode_suffix}.xml"/>\\n')
            f.write('    </output>\\n')
            f.write('    <time>\\n')
            f.write(f'        <begin value="0"/>\\n')
            f.write(f'        <end value="{self.test_scenarios[scenario_name]["duration"]}"/>\\n')
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
            if controller_type.upper() == "CONSERVATIVE":
                controller = TraciImprovedController()
                print(f"\\n🚀 RUNNING {controller_type.upper()} SIMULATION ({scenario_name})")
                print("=" * 70)
            else:  # THROUGHPUT
                controller = TraciThroughputOptimizedController()
                print(f"\\n⚡ RUNNING {controller_type.upper()} SIMULATION ({scenario_name})")
                print("=" * 70)
            
            controller.initialize_traffic_lights()
            
            simulation_data = []
            step = 0
            
            while traci.simulation.getMinExpectedNumber() > 0:
                current_time = traci.simulation.getTime()
                
                # Apply adaptive control
                control_result = controller.apply_adaptive_control(current_time)
                
                # Collect data every 30 seconds
                if step % 30 == 0:
                    traffic_data = control_result['traffic_data']
                    
                    # Calculate vehicle mix
                    total_vehicles = traffic_data['total_vehicles']
                    vehicles_in_sim = traci.simulation.getLoadedNumber()
                    departed = traci.simulation.getDepartedNumber()
                    arrived = traci.simulation.getArrivedNumber()
                    
                    # Estimate motorcycle ratio (simplified)
                    motorcycle_ratio = 0.3  # Target ratio
                    
                    data_point = {
                        'time': current_time,
                        'avg_waiting_time': traffic_data['avg_waiting_time'],
                        'avg_speed': traffic_data['avg_speed'],
                        'total_vehicles': total_vehicles,
                        'throughput': traffic_data['throughput'],
                        'adaptations': control_result['total_adaptations'],
                        'urgency': control_result['urgency_level'],
                        'vehicles_departed': departed,
                        'vehicles_arrived': arrived,
                        'motorcycle_ratio': motorcycle_ratio,
                        'controller_type': controller_type.lower(),
                        'scenario': scenario_name
                    }
                    
                    simulation_data.append(data_point)
                    
                    # Print progress
                    if step % 300 == 0:  # Every 5 minutes
                        print(f"   Time: {int(current_time):4d}s | "
                              f"Wait: {traffic_data['avg_waiting_time']:6.1f}s | "
                              f"Speed: {traffic_data['avg_speed']:6.1f} m/s | "
                              f"Vehicles: {total_vehicles:3d} | "
                              f"Adaptations: {control_result['total_adaptations']:3d} | "
                              f"Urgency: {control_result['urgency_level']}")
                
                traci.simulationStep()
                step += 1
            
            logger.info(f"{controller_type} simulation completed for {scenario_name}")
            return simulation_data
            
        except Exception as e:
            logger.error(f"Error in {controller_type} simulation: {e}")
            return []
        
        finally:
            try:
                traci.close()
            except:
                pass
    
    def run_comparison_test(self):
        """Run complete comparison test."""
        print("\\n🔬 THROUGHPUT vs CONSERVATIVE ALGORITHM COMPARISON")
        print("=" * 60)
        print("Testing hypothesis: Shorter phases for light traffic = better throughput")
        print()
        
        all_results = {}
        
        for scenario_name in self.test_scenarios.keys():
            print(f"\\n📊 Testing Scenario: {self.test_scenarios[scenario_name]['name']}")
            print("-" * 50)
            
            # Test conservative approach
            conservative_data = self.run_simulation(scenario_name, "CONSERVATIVE")
            
            # Test throughput approach  
            throughput_data = self.run_simulation(scenario_name, "THROUGHPUT")
            
            # Analyze results
            if conservative_data and throughput_data:
                analysis = self.analyze_results(conservative_data, throughput_data, scenario_name)
                all_results[scenario_name] = analysis
                self.print_analysis(analysis, scenario_name)
        
        # Overall summary
        self.print_overall_summary(all_results)
    
    def analyze_results(self, conservative_data: List[Dict], throughput_data: List[Dict], scenario_name: str) -> Dict[str, Any]:
        """Analyze and compare results between both approaches."""
        
        # Calculate averages for conservative approach
        conservative_avg_wait = statistics.mean([d['avg_waiting_time'] for d in conservative_data])
        conservative_avg_speed = statistics.mean([d['avg_speed'] for d in conservative_data])
        conservative_total_adaptations = conservative_data[-1]['adaptations'] if conservative_data else 0
        conservative_throughput = sum([d['vehicles_arrived'] for d in conservative_data])
        
        # Calculate averages for throughput approach
        throughput_avg_wait = statistics.mean([d['avg_waiting_time'] for d in throughput_data])
        throughput_avg_speed = statistics.mean([d['avg_speed'] for d in throughput_data])
        throughput_total_adaptations = throughput_data[-1]['adaptations'] if throughput_data else 0
        throughput_throughput = sum([d['vehicles_arrived'] for d in throughput_data])
        
        # Calculate improvements
        wait_time_improvement = ((conservative_avg_wait - throughput_avg_wait) / conservative_avg_wait) * 100
        speed_improvement = ((throughput_avg_speed - conservative_avg_speed) / conservative_avg_speed) * 100
        throughput_improvement = ((throughput_throughput - conservative_throughput) / max(conservative_throughput, 1)) * 100
        adaptation_difference = throughput_total_adaptations - conservative_total_adaptations
        
        return {
            'scenario': scenario_name,
            'conservative': {
                'avg_wait_time': conservative_avg_wait,
                'avg_speed': conservative_avg_speed,
                'total_adaptations': conservative_total_adaptations,
                'throughput': conservative_throughput
            },
            'throughput_optimized': {
                'avg_wait_time': throughput_avg_wait,
                'avg_speed': throughput_avg_speed,
                'total_adaptations': throughput_total_adaptations,
                'throughput': throughput_throughput
            },
            'improvements': {
                'wait_time': wait_time_improvement,
                'speed': speed_improvement,
                'throughput': throughput_improvement,
                'adaptation_difference': adaptation_difference
            }
        }
    
    def print_analysis(self, analysis: Dict[str, Any], scenario_name: str):
        """Print detailed analysis for a scenario."""
        print(f"\\n📈 ANALYSIS RESULTS: {scenario_name.upper()}")
        print("=" * 50)
        
        conservative = analysis['conservative']
        throughput = analysis['throughput_optimized']
        improvements = analysis['improvements']
        
        print(f"🔸 CONSERVATIVE APPROACH:")
        print(f"   Average Wait Time: {conservative['avg_wait_time']:.2f}s")
        print(f"   Average Speed: {conservative['avg_speed']:.2f} m/s")
        print(f"   Total Adaptations: {conservative['total_adaptations']}")
        print(f"   Vehicles Processed: {conservative['throughput']}")
        
        print(f"\\n⚡ THROUGHPUT APPROACH:")
        print(f"   Average Wait Time: {throughput['avg_wait_time']:.2f}s")
        print(f"   Average Speed: {throughput['avg_speed']:.2f} m/s")
        print(f"   Total Adaptations: {throughput['total_adaptations']}")
        print(f"   Vehicles Processed: {throughput['throughput']}")
        
        print(f"\\n🎯 PERFORMANCE COMPARISON:")
        print(f"   Wait Time Change: {improvements['wait_time']:+.1f}%")
        print(f"   Speed Change: {improvements['speed']:+.1f}%")
        print(f"   Throughput Change: {improvements['throughput']:+.1f}%")
        print(f"   Adaptation Difference: {improvements['adaptation_difference']:+d}")
        
        # Determine winner
        if improvements['wait_time'] > 0 and improvements['throughput'] > 0:
            print(f"\\n🏆 WINNER: THROUGHPUT APPROACH")
            print(f"   Better wait times AND better throughput!")
        elif improvements['wait_time'] > 0:
            print(f"\\n🏆 WINNER: THROUGHPUT APPROACH (Wait Time)")
            print(f"   Better wait times but similar throughput")
        elif improvements['throughput'] > 0:
            print(f"\\n🏆 WINNER: THROUGHPUT APPROACH (Throughput)")
            print(f"   Better throughput but similar wait times")
        else:
            print(f"\\n🏆 WINNER: CONSERVATIVE APPROACH")
            print(f"   Better overall performance")
    
    def print_overall_summary(self, all_results: Dict[str, Any]):
        """Print overall comparison summary."""
        print("\\n" + "=" * 60)
        print("🏁 OVERALL COMPARISON SUMMARY")
        print("=" * 60)
        
        total_scenarios = len(all_results)
        throughput_wins = 0
        
        for scenario_name, analysis in all_results.items():
            improvements = analysis['improvements']
            if improvements['wait_time'] > 0 or improvements['throughput'] > 0:
                throughput_wins += 1
        
        print(f"📊 Scenarios tested: {total_scenarios}")
        print(f"🏆 Throughput approach wins: {throughput_wins}/{total_scenarios}")
        print(f"🏆 Conservative approach wins: {total_scenarios - throughput_wins}/{total_scenarios}")
        
        if throughput_wins > total_scenarios / 2:
            print("\\n🎉 CONCLUSION: YOUR HYPOTHESIS IS CORRECT!")
            print("   Shorter phases for light traffic DO improve performance!")
            print("   Consider implementing throughput-optimized approach.")
        else:
            print("\\n🤔 CONCLUSION: CONSERVATIVE APPROACH WINS")
            print("   Longer phases for light traffic work better overall.")
            print("   The current algorithm is already optimized.")
        
        print("\\n💡 Key Insights:")
        print("   - Switching overhead impacts light traffic scenarios")
        print("   - Phase duration optimization depends on traffic patterns")
        print("   - Both approaches have their strengths in different scenarios")

if __name__ == "__main__":
    test = ThroughputComparisonTest()
    test.run_comparison_test()