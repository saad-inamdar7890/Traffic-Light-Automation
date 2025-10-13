"""
Focused Scenario Test - Balanced Algorithm
Tests the new balanced algorithm with a condensed version of scenarios
"""

import os
import sys
import traci
import statistics
import time

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class FocusedScenarioTest:
    """Focused test of balanced algorithm across multiple scenarios"""
    
    def __init__(self):
        self.sumo_cmd = ["sumo", "-c", "demo.sumocfg"]
        self.traffic_controller = AdaptiveTrafficController("J4")
        self.analyzer = TrafficAnalyzer()
        
        # Define test scenarios (shorter duration for faster testing)
        self.scenarios = {
            'heavy_ns': {
                'name': 'Heavy North-South',
                'description': 'Heavy NS traffic, light EW traffic',
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 300, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 300, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 40, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 40, 'vtype': 'car'}
                }
            },
            'balanced': {
                'name': 'Balanced Traffic',
                'description': 'Equal traffic in all directions',
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 120, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 120, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 120, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 120, 'vtype': 'car'}
                }
            },
            'light': {
                'name': 'Light Traffic',
                'description': 'Light traffic in all directions',
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 60, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 60, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 60, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 60, 'vtype': 'car'}
                }
            }
        }
    
    def run_focused_tests(self):
        """Run focused scenario tests - 5 minutes each scenario, each mode"""
        print("🎯 FOCUSED BALANCED ALGORITHM TESTS")
        print("=" * 60)
        print("⏱️  Testing: 5 min Normal + 5 min Adaptive per scenario")
        print("🔬 Scenarios: Heavy NS, Balanced, Light Traffic")
        print("🎯 Goal: Evaluate balanced algorithm performance")
        print("=" * 60)
        
        results = {}
        
        for scenario_key, scenario in self.scenarios.items():
            print(f"\n🚦 TESTING SCENARIO: {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print("-" * 50)
            
            scenario_results = self._run_single_scenario(scenario_key, scenario)
            results[scenario_key] = scenario_results
            
            print(f"✅ {scenario['name']} completed")
        
        # Comprehensive analysis
        self._analyze_all_results(results)
        
        print("\n🎉 FOCUSED TESTS COMPLETED!")
        print("📊 Balanced algorithm evaluation ready!")
    
    def _run_single_scenario(self, scenario_key, scenario):
        """Run a single scenario test"""
        traci.start(self.sumo_cmd)
        
        scenario_data = {
            'normal': {'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 'pressures_ew': []},
            'adaptive': {'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 'pressures_ew': []}
        }
        
        try:
            # Phase 1: Normal mode (0-300 seconds = 5 minutes)
            print("🔴 Normal mode (5 min)")
            self._run_scenario_phase(scenario_data['normal'], 0, 300, scenario['flows'], adaptive=False)
            
            # Clear vehicles
            self._clear_vehicles()
            
            # Phase 2: Adaptive mode (300-600 seconds = 5 minutes)
            print("🟢 Adaptive mode (5 min)")
            self._run_scenario_phase(scenario_data['adaptive'], 300, 600, scenario['flows'], adaptive=True)
            
        finally:
            traci.close()
        
        # Quick scenario analysis
        return self._analyze_scenario_results(scenario_key, scenario_data)
    
    def _run_scenario_phase(self, data_dict, start_time, end_time, flows, adaptive=False):
        """Run a single phase with specified flows"""
        adaptations_count = 0
        last_log = 0
        
        for step in range(start_time, end_time):
            traci.simulationStep()
            
            step_data = self.analyzer.collect_traffic_metrics(step, traci)
            
            if step_data:
                # Calculate pressures
                ns_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E1', '-E1']
                )
                ew_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E0', '-E0']
                )
                
                if adaptive:
                    result = self.traffic_controller.apply_adaptive_control(step_data, step)
                    if result.get('applied'):
                        adaptations_count += 1
                
                # Store data every 20 seconds
                if step % 20 == 0:
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['pressures_ns'].append(ns_pressure)
                    data_dict['pressures_ew'].append(ew_pressure)
                
                # Progress log every minute
                if step % 60 == 0 and step != last_log:
                    last_log = step
                    mode = "Adaptive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    print(f"   {mode} - {minute:3.0f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | NS: {ns_pressure:4.0f} | "
                          f"EW: {ew_pressure:4.0f} | Adaptations: {adaptations_count}")
    
    def _clear_vehicles(self):
        """Clear all vehicles"""
        try:
            for veh_id in traci.vehicle.getIDList():
                traci.vehicle.remove(veh_id)
            print("🧹 Vehicles cleared")
            # Wait a moment for cleanup
            for _ in range(3):
                traci.simulationStep()
        except:
            pass
    
    def _analyze_scenario_results(self, scenario_key, data):
        """Analyze single scenario results"""
        if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
            return {'improvement': 0, 'rating': 'NO DATA'}
        
        # Skip first 2 data points for warmup
        warmup = 2
        
        normal_avg_wait = statistics.mean(data['normal']['waiting_times'][warmup:])
        adaptive_avg_wait = statistics.mean(data['adaptive']['waiting_times'][warmup:])
        
        normal_avg_vehicles = statistics.mean(data['normal']['vehicle_counts'][warmup:])
        adaptive_avg_vehicles = statistics.mean(data['adaptive']['vehicle_counts'][warmup:])
        
        normal_avg_ns = statistics.mean(data['normal']['pressures_ns'][warmup:])
        adaptive_avg_ns = statistics.mean(data['adaptive']['pressures_ns'][warmup:])
        
        normal_avg_ew = statistics.mean(data['normal']['pressures_ew'][warmup:])
        adaptive_avg_ew = statistics.mean(data['adaptive']['pressures_ew'][warmup:])
        
        # Calculate improvements
        wait_improvement = ((normal_avg_wait - adaptive_avg_wait) / normal_avg_wait * 100) if normal_avg_wait > 0 else 0
        
        # Pressure balance improvement
        normal_imbalance = abs(normal_avg_ns - normal_avg_ew)
        adaptive_imbalance = abs(adaptive_avg_ns - adaptive_avg_ew)
        balance_improvement = ((normal_imbalance - adaptive_imbalance) / normal_imbalance * 100) if normal_imbalance > 0 else 0
        
        # Performance rating
        if wait_improvement > 5:
            rating = "🟢 EXCELLENT"
        elif wait_improvement > 2:
            rating = "🟡 GOOD"
        elif wait_improvement > -1:
            rating = "🟠 FAIR"
        else:
            rating = "🔴 POOR"
        
        print(f"   📊 Results: Wait: {wait_improvement:+5.1f}% | Balance: {balance_improvement:+5.1f}% | {rating}")
        
        return {
            'wait_improvement': wait_improvement,
            'balance_improvement': balance_improvement,
            'rating': rating,
            'normal_wait': normal_avg_wait,
            'adaptive_wait': adaptive_avg_wait,
            'normal_vehicles': normal_avg_vehicles,
            'adaptive_vehicles': adaptive_avg_vehicles
        }
    
    def _analyze_all_results(self, results):
        """Comprehensive analysis of all scenarios"""
        print(f"\n📊 COMPREHENSIVE BALANCED ALGORITHM ANALYSIS:")
        print("=" * 70)
        
        print(f"{'Scenario':<20} {'Wait Improve':<12} {'Balance Improve':<15} {'Rating':<15}")
        print("-" * 70)
        
        total_improvement = 0
        scenario_count = 0
        
        for scenario_key, result in results.items():
            scenario_name = self.scenarios[scenario_key]['name']
            wait_imp = result['wait_improvement']
            balance_imp = result['balance_improvement']
            rating = result['rating']
            
            print(f"{scenario_name:<20} {wait_imp:+7.1f}%      {balance_imp:+9.1f}%      {rating}")
            
            total_improvement += wait_imp
            scenario_count += 1
        
        print("-" * 70)
        
        avg_improvement = total_improvement / scenario_count if scenario_count > 0 else 0
        
        print(f"{'AVERAGE':<20} {avg_improvement:+7.1f}%")
        print()
        
        # Overall algorithm rating
        if avg_improvement > 3:
            overall_rating = "🟢 EXCELLENT - Algorithm performing very well!"
        elif avg_improvement > 1:
            overall_rating = "🟡 GOOD - Solid improvements achieved"
        elif avg_improvement > -1:
            overall_rating = "🟠 FAIR - Mixed results, some optimization needed"
        else:
            overall_rating = "🔴 NEEDS WORK - Consider algorithm adjustments"
        
        print(f"🎯 OVERALL PERFORMANCE: {overall_rating}")
        print(f"📈 Average Improvement: {avg_improvement:+.1f}%")
        
        return avg_improvement

def main():
    test = FocusedScenarioTest()
    test.run_focused_tests()

if __name__ == "__main__":
    main()