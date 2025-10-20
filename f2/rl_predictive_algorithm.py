"""
RL Predictive Traffic Algorithm
===============================

This algorithm simulates an RL model that predicts traffic flow and pre-allocates
base times for each signal based on anticipated traffic patterns, then adapts 
dynamically when conditions change.
"""

import json
import statistics
from typing import Dict, List, Tuple
from datetime import datetime

class RLPredictiveController:
    """RL-based controller that predicts traffic and pre-allocates base times."""
    
    def __init__(self):
        self.current_phase = 0  # 0=North, 1=East, 2=South, 3=West
        self.phase_start_time = 0
        self.total_adaptations = 0
        self.scenario_base_times = {}
        self.current_base_times = [30, 30, 30, 30]  # Default base times
        self.adaptation_history = []
        
    def set_scenario_base_times(self, scenario: str):
        """Set base times based on RL prediction for each scenario."""
        
        base_time_configs = {
            'initial_low': {
                'description': 'Initial low traffic - equal shorter times',
                'north': 20, 'east': 20, 'south': 20, 'west': 20
            },
            'heavy_north': {
                'description': 'Heavy North traffic predicted - prioritize North',
                'north': 50, 'east': 30, 'south': 30, 'west': 30
            },
            'heavy_east': {
                'description': 'Heavy East traffic predicted - prioritize East',
                'north': 30, 'east': 50, 'south': 30, 'west': 30
            },
            'heavy_north_east': {
                'description': 'Heavy North-East traffic predicted',
                'north': 50, 'east': 50, 'south': 30, 'west': 30
            },
            'all_heavy': {
                'description': 'All lanes heavy - equal longer times',
                'north': 45, 'east': 45, 'south': 45, 'west': 45
            },
            'gradual_slowdown': {
                'description': 'Traffic reducing - moderate times',
                'north': 30, 'east': 30, 'south': 30, 'west': 30
            },
            'evening_light': {
                'description': 'Evening light traffic - shorter times',
                'north': 20, 'east': 20, 'south': 20, 'west': 20
            }
        }
        
        if scenario in base_time_configs:
            config = base_time_configs[scenario]
            self.current_base_times = [
                config['north'], config['east'], 
                config['south'], config['west']
            ]
            self.scenario_base_times[scenario] = config
            print(f"🤖 RL Model Prediction: {config['description']}")
            print(f"   Base times set: N={config['north']}s, E={config['east']}s, S={config['south']}s, W={config['west']}s")
        
    def analyze_traffic_pressure(self, lane_vehicles: Dict[str, int]) -> Dict[str, str]:
        """Analyze current traffic pressure in each lane."""
        
        lane_mapping = {'north': 0, 'east': 1, 'south': 2, 'west': 3}
        pressure_levels = {}
        
        for lane, vehicle_count in lane_vehicles.items():
            if vehicle_count >= 20:
                pressure_levels[lane] = 'CRITICAL'
            elif vehicle_count >= 15:
                pressure_levels[lane] = 'HIGH'
            elif vehicle_count >= 8:
                pressure_levels[lane] = 'MODERATE'
            elif vehicle_count >= 4:
                pressure_levels[lane] = 'LOW'
            else:
                pressure_levels[lane] = 'MINIMAL'
                
        return pressure_levels
    
    def calculate_dynamic_adjustment(self, current_time: float, lane_vehicles: Dict[str, int], 
                                   avg_waiting: float) -> Tuple[bool, int]:
        """Calculate if adjustment needed and new phase duration."""
        
        time_in_current_phase = current_time - self.phase_start_time
        current_lane = ['north', 'east', 'south', 'west'][self.current_phase]
        current_base_time = self.current_base_times[self.current_phase]
        
        # Analyze pressure in all lanes
        pressure_levels = self.analyze_traffic_pressure(lane_vehicles)
        current_pressure = pressure_levels[current_lane]
        
        # Check if adjustment needed
        should_adjust = False
        new_duration = current_base_time
        adjustment_reason = ""
        
        # Rule 1: Current lane has critical pressure - extend time
        if current_pressure in ['CRITICAL', 'HIGH'] and time_in_current_phase < current_base_time:
            if current_pressure == 'CRITICAL':
                extension = min(20, current_base_time * 0.4)  # Up to 40% extension
                adjustment_reason = f"CRITICAL pressure in {current_lane} - extending by {extension}s"
            else:
                extension = min(15, current_base_time * 0.3)  # Up to 30% extension
                adjustment_reason = f"HIGH pressure in {current_lane} - extending by {extension}s"
            
            new_duration = current_base_time + extension
            should_adjust = True
        
        # Rule 2: Current lane pressure relieved early - consider switching
        elif current_pressure in ['MINIMAL', 'LOW'] and time_in_current_phase >= current_base_time * 0.6:
            # Check if other lanes need attention
            other_lanes = [l for l in ['north', 'east', 'south', 'west'] if l != current_lane]
            high_pressure_lanes = [l for l in other_lanes if pressure_levels[l] in ['CRITICAL', 'HIGH']]
            
            if high_pressure_lanes and time_in_current_phase >= current_base_time * 0.7:
                reduction = min(10, current_base_time * 0.3)
                new_duration = max(15, current_base_time - reduction)  # Minimum 15s
                adjustment_reason = f"LOW pressure in {current_lane}, HIGH pressure in {high_pressure_lanes[0]} - reducing by {reduction}s"
                should_adjust = True
        
        # Rule 3: Waiting time excessive - emergency adjustment
        elif avg_waiting > 60 and time_in_current_phase >= current_base_time * 0.8:
            if current_pressure in ['MINIMAL', 'LOW']:
                reduction = min(15, current_base_time * 0.4)
                new_duration = max(12, current_base_time - reduction)
                adjustment_reason = f"EXCESSIVE waiting ({avg_waiting:.1f}s), {current_lane} has {current_pressure} pressure - emergency reduction"
                should_adjust = True
        
        # Log adjustment decision
        if should_adjust:
            self.adaptation_history.append({
                'time': current_time,
                'phase': current_lane,
                'original_duration': current_base_time,
                'new_duration': new_duration,
                'reason': adjustment_reason,
                'pressure_levels': pressure_levels.copy(),
                'avg_waiting': avg_waiting
            })
            self.total_adaptations += 1
        
        return should_adjust, int(new_duration)
    
    def should_switch_phase(self, current_time: float, lane_vehicles: Dict[str, int], 
                          avg_waiting: float) -> bool:
        """Determine if it's time to switch to next phase."""
        
        time_in_phase = current_time - self.phase_start_time
        current_base_time = self.current_base_times[self.current_phase]
        
        # Check for dynamic adjustment
        should_adjust, new_duration = self.calculate_dynamic_adjustment(
            current_time, lane_vehicles, avg_waiting
        )
        
        if should_adjust:
            # Update the base time for this phase dynamically
            self.current_base_times[self.current_phase] = new_duration
            return time_in_phase >= new_duration
        
        # Normal switching based on base time
        return time_in_phase >= current_base_time
    
    def switch_phase(self, current_time: float):
        """Switch to the next phase."""
        self.current_phase = (self.current_phase + 1) % 4
        self.phase_start_time = current_time
    
    def get_current_phase_info(self) -> Dict:
        """Get information about current phase."""
        phase_names = ['North', 'East', 'South', 'West']
        return {
            'current_phase': phase_names[self.current_phase],
            'base_time': self.current_base_times[self.current_phase],
            'total_adaptations': self.total_adaptations
        }

class ScenarioTestFramework:
    """Framework for testing RL predictive algorithm across different scenarios."""
    
    def __init__(self):
        self.rl_controller = RLPredictiveController()
        self.test_results = {}
        
    def simulate_traffic_scenario(self, scenario_name: str, scenario_config: Dict) -> List[Dict]:
        """Simulate a traffic scenario with the RL predictive algorithm."""
        
        print(f"\\n📊 Simulating scenario: {scenario_name}")
        print(f"   Duration: {scenario_config['duration']} minutes")
        
        # Set RL predicted base times for this scenario
        self.rl_controller.set_scenario_base_times(scenario_config['base_time_scenario'])
        
        simulation_data = []
        current_time = 0
        time_step = 0.5  # 30-second intervals
        
        while current_time < scenario_config['duration']:
            # Generate traffic based on scenario pattern
            lane_vehicles = self.generate_scenario_traffic(scenario_name, current_time, scenario_config)
            
            # Calculate performance metrics
            total_vehicles = sum(lane_vehicles.values())
            avg_waiting = self.calculate_waiting_time(lane_vehicles, current_time)
            avg_speed = self.calculate_avg_speed(total_vehicles, avg_waiting)
            
            # RL Algorithm decision making
            if self.rl_controller.should_switch_phase(current_time, lane_vehicles, avg_waiting):
                self.rl_controller.switch_phase(current_time)
            
            # Record data point
            phase_info = self.rl_controller.get_current_phase_info()
            data_point = {
                'time': current_time,
                'scenario': scenario_name,
                'total_vehicles': total_vehicles,
                'lane_vehicles': lane_vehicles.copy(),
                'avg_waiting_time': avg_waiting,
                'avg_speed': avg_speed,
                'current_phase': phase_info['current_phase'],
                'base_time': phase_info['base_time'],
                'adaptations': phase_info['total_adaptations'],
                'algorithm': 'rl_predictive'
            }
            
            simulation_data.append(data_point)
            current_time += time_step
        
        print(f"✅ Scenario complete: {len(simulation_data)} data points, {self.rl_controller.total_adaptations} adaptations")
        return simulation_data
    
    def generate_scenario_traffic(self, scenario_name: str, current_time: float, config: Dict) -> Dict[str, int]:
        """Generate traffic pattern for specific scenario."""
        
        import random
        import math
        
        # Base traffic patterns for each scenario
        patterns = {
            'initial_low': {
                'north': (2, 6), 'east': (2, 6), 'south': (2, 6), 'west': (2, 6)
            },
            'heavy_north': {
                'north': (15, 25), 'east': (3, 8), 'south': (3, 8), 'west': (3, 8)
            },
            'heavy_east': {
                'north': (3, 8), 'east': (15, 25), 'south': (3, 8), 'west': (3, 8)
            },
            'heavy_north_east': {
                'north': (12, 20), 'east': (12, 20), 'south': (4, 9), 'west': (4, 9)
            },
            'all_heavy': {
                'north': (15, 22), 'east': (15, 22), 'south': (15, 22), 'west': (15, 22)
            },
            'gradual_slowdown': {
                'north': (8, 15), 'east': (8, 15), 'south': (8, 15), 'west': (8, 15)
            },
            'evening_light': {
                'north': (2, 5), 'east': (2, 5), 'south': (2, 5), 'west': (2, 5)
            }
        }
        
        if scenario_name not in patterns:
            scenario_name = 'initial_low'
        
        pattern = patterns[scenario_name]
        lane_vehicles = {}
        
        # Add some randomness and time-based variation
        time_factor = 1 + 0.3 * math.sin(current_time * 0.1)  # Slight variation over time
        
        for lane, (min_val, max_val) in pattern.items():
            base_count = random.randint(min_val, max_val)
            varied_count = int(base_count * time_factor)
            lane_vehicles[lane] = max(0, varied_count)
        
        return lane_vehicles
    
    def calculate_waiting_time(self, lane_vehicles: Dict[str, int], current_time: float) -> float:
        """Calculate average waiting time based on vehicle counts."""
        total_vehicles = sum(lane_vehicles.values())
        if total_vehicles == 0:
            return 0.0
        
        # Estimate waiting time based on queue length and current phase
        current_phase_lane = ['north', 'east', 'south', 'west'][self.rl_controller.current_phase]
        
        base_wait = 0
        for lane, count in lane_vehicles.items():
            if lane == current_phase_lane:
                # Current phase lane has lower wait time
                lane_wait = count * 2.5
            else:
                # Other lanes wait longer
                lane_wait = count * 4.0
            base_wait += lane_wait
        
        avg_wait = base_wait / total_vehicles
        return min(avg_wait, 120)  # Cap at 2 minutes
    
    def calculate_avg_speed(self, total_vehicles: int, avg_waiting: float) -> float:
        """Calculate average speed based on congestion."""
        if total_vehicles == 0:
            return 15.0  # Free flow speed
        
        # Speed decreases with more vehicles and longer waits
        congestion_factor = min(total_vehicles / 50.0, 1.0)
        wait_factor = min(avg_waiting / 60.0, 1.0)
        
        speed = 15.0 * (1 - 0.7 * congestion_factor - 0.3 * wait_factor)
        return max(speed, 2.0)  # Minimum speed
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive test across all scenarios."""
        
        print("🤖 RL PREDICTIVE ALGORITHM COMPREHENSIVE TEST")
        print("=" * 70)
        print("Testing algorithm with RL-predicted base times and dynamic adaptation")
        
        # Define test scenarios
        test_scenarios = {
            'initial_low': {
                'duration': 10,
                'base_time_scenario': 'initial_low',
                'description': 'Initial low traffic with 20s base times'
            },
            'heavy_north': {
                'duration': 10,
                'base_time_scenario': 'heavy_north',
                'description': 'Heavy North traffic with 50s North, 30s others'
            },
            'heavy_east': {
                'duration': 10,
                'base_time_scenario': 'heavy_east',
                'description': 'Heavy East traffic with 50s East, 30s others'
            },
            'heavy_north_east': {
                'duration': 10,
                'base_time_scenario': 'heavy_north_east',
                'description': 'Heavy North-East with 50s both, 30s others'
            },
            'all_heavy': {
                'duration': 10,
                'base_time_scenario': 'all_heavy',
                'description': 'All lanes heavy with 45s equal times'
            },
            'gradual_slowdown': {
                'duration': 10,
                'base_time_scenario': 'gradual_slowdown',
                'description': 'Gradual slowdown with 30s moderate times'
            },
            'evening_light': {
                'duration': 10,
                'base_time_scenario': 'evening_light',
                'description': 'Evening light traffic with 20s base times'
            }
        }
        
        all_scenario_data = {}
        
        for scenario_name, config in test_scenarios.items():
            # Reset controller for each scenario
            self.rl_controller = RLPredictiveController()
            
            # Run scenario simulation
            scenario_data = self.simulate_traffic_scenario(scenario_name, config)
            all_scenario_data[scenario_name] = scenario_data
        
        return all_scenario_data
    
    def analyze_results(self, all_data: Dict) -> Dict:
        """Analyze results across all scenarios."""
        
        print("\\n📊 ANALYZING RL PREDICTIVE ALGORITHM RESULTS")
        print("=" * 60)
        
        scenario_summaries = {}
        
        for scenario_name, data in all_data.items():
            if not data:
                continue
                
            avg_wait = statistics.mean([d['avg_waiting_time'] for d in data])
            avg_speed = statistics.mean([d['avg_speed'] for d in data])
            total_adaptations = data[-1]['adaptations'] if data else 0
            avg_vehicles = statistics.mean([d['total_vehicles'] for d in data])
            
            scenario_summaries[scenario_name] = {
                'avg_wait_time': avg_wait,
                'avg_speed': avg_speed,
                'total_adaptations': total_adaptations,
                'avg_vehicles': avg_vehicles,
                'data_points': len(data)
            }
            
            print(f"\\n📈 {scenario_name.upper()}:")
            print(f"   Average Wait Time: {avg_wait:.1f}s")
            print(f"   Average Speed: {avg_speed:.1f} m/s")
            print(f"   Total Adaptations: {total_adaptations}")
            print(f"   Average Vehicles: {avg_vehicles:.1f}")
        
        # Overall performance
        all_waits = [s['avg_wait_time'] for s in scenario_summaries.values()]
        all_speeds = [s['avg_speed'] for s in scenario_summaries.values()]
        all_adaptations = [s['total_adaptations'] for s in scenario_summaries.values()]
        
        overall_summary = {
            'overall_avg_wait': statistics.mean(all_waits),
            'overall_avg_speed': statistics.mean(all_speeds),
            'total_adaptations': sum(all_adaptations),
            'scenarios_tested': len(scenario_summaries)
        }
        
        print(f"\\n🎯 OVERALL PERFORMANCE:")
        print(f"   Average Wait Time: {overall_summary['overall_avg_wait']:.1f}s")
        print(f"   Average Speed: {overall_summary['overall_avg_speed']:.1f} m/s")
        print(f"   Total Adaptations: {overall_summary['total_adaptations']}")
        print(f"   Scenarios Tested: {overall_summary['scenarios_tested']}")
        
        return {
            'scenario_summaries': scenario_summaries,
            'overall_summary': overall_summary,
            'all_data': all_data
        }

if __name__ == "__main__":
    # Run the comprehensive RL predictive algorithm test
    framework = ScenarioTestFramework()
    
    # Execute all scenarios
    all_results = framework.run_comprehensive_test()
    
    # Analyze results
    analysis = framework.analyze_results(all_results)
    
    # Save results
    import os
    results_dir = os.path.join(os.path.dirname(__file__), "rl_predictive_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    with open(os.path.join(results_dir, "rl_predictive_test_data.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open(os.path.join(results_dir, "rl_predictive_analysis.json"), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\\n📁 Results saved to: {results_dir}")