"""
STRESS TEST PERFORMANCE ANALYZER
===============================
Focused stress testing for extreme scenarios and edge cases
to evaluate algorithm robustness and breaking points.
"""

import os
import sys
import traci
import statistics
import time
import random
from collections import defaultdict

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class StressTestAnalyzer:
    """Stress testing for extreme scenarios and algorithm limits"""
    
    def __init__(self):
        self.sumo_cmd = ["sumo", "-c", "demo.sumocfg"]
        self.traffic_controller = AdaptiveTrafficController("J4")
        self.analyzer = TrafficAnalyzer()
        
        # Stress test scenarios
        self.stress_scenarios = {
            'extreme_volume': {
                'name': 'Extreme Volume Stress Test',
                'description': 'Maximum traffic volume to test saturation handling',
                'duration': 20,  # 20 minutes each phase
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 600, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 600, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 600, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 600, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'rate': 400, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 400, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 400, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 400, 'vtype': 'car'}
                }
            },
            
            'rapid_fluctuation': {
                'name': 'Rapid Traffic Fluctuation',
                'description': 'Rapidly changing traffic patterns to test adaptation speed',
                'duration': 20,
                'flows': 'RAPID_CHANGE',  # Special handling
                'patterns': [
                    # Pattern 1: Heavy NS (5 minutes)
                    {'duration': 300, 'flows': {
                        'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 400, 'vtype': 'car'},
                        'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 400, 'vtype': 'car'},
                        'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 50, 'vtype': 'car'},
                        'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 50, 'vtype': 'car'}
                    }},
                    # Pattern 2: Heavy EW (5 minutes)
                    {'duration': 300, 'flows': {
                        'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 50, 'vtype': 'car'},
                        'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 50, 'vtype': 'car'},
                        'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 400, 'vtype': 'car'},
                        'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 400, 'vtype': 'car'}
                    }},
                    # Pattern 3: Extreme imbalance (5 minutes)
                    {'duration': 300, 'flows': {
                        'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 500, 'vtype': 'car'},
                        'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 10, 'vtype': 'car'},
                        'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 10, 'vtype': 'car'},
                        'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 10, 'vtype': 'car'}
                    }},
                    # Pattern 4: Balanced recovery (5 minutes)
                    {'duration': 300, 'flows': {
                        'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 120, 'vtype': 'car'},
                        'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 120, 'vtype': 'car'},
                        'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 120, 'vtype': 'car'},
                        'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 120, 'vtype': 'car'}
                    }}
                ]
            },
            
            'sustained_pressure': {
                'name': 'Sustained High Pressure',
                'description': 'Long-term high traffic to test algorithm stability',
                'duration': 25,  # 25 minutes
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 350, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 350, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 350, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 350, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'rate': 200, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 200, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 200, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 200, 'vtype': 'car'}
                }
            }
        }
    
    def run_stress_analysis(self):
        """Run comprehensive stress testing"""
        print("🔥 STRESS TEST PERFORMANCE ANALYSIS")
        print("=" * 80)
        print("⚡ Focus: Algorithm limits and robustness")
        print("🎯 Goal: Find breaking points and edge case performance")
        print("⏱️  Extended durations: 20-25 minutes per phase")
        print("=" * 80)
        
        all_results = {}
        start_time = time.time()
        
        for scenario_key, scenario in self.stress_scenarios.items():
            print(f"\n{'='*60}")
            print(f"🔥 STRESS TEST: {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print(f"⏱️  Duration: {scenario['duration']} minutes per phase")
            print(f"{'='*60}")
            
            scenario_results = self._run_stress_scenario(scenario_key, scenario)
            all_results[scenario_key] = scenario_results
            
            print(f"✅ {scenario['name']} completed")
            
            # Progress update
            elapsed = time.time() - start_time
            print(f"⏱️  Elapsed time: {elapsed/60:.1f} minutes")
        
        # Generate stress analysis report
        self._generate_stress_report(all_results)
        
        total_time = time.time() - start_time
        print(f"\n🔥 STRESS ANALYSIS COMPLETED!")
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
        print(f"📊 Stress test report generated!")
    
    def _run_stress_scenario(self, scenario_key, scenario):
        """Run individual stress test scenario"""
        traci.start(self.sumo_cmd)
        
        duration_seconds = scenario['duration'] * 60
        
        scenario_data = {
            'normal': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'max_waiting': [],
                'extreme_waits': [], 'saturation_points': []
            },
            'adaptive': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'max_waiting': [],
                'extreme_waits': [], 'saturation_points': []
            }
        }
        
        try:
            # Phase 1: Normal mode
            print(f"🔴 Normal mode stress test ({scenario['duration']} min)")
            self._run_stress_phase(scenario_data['normal'], 0, duration_seconds, 
                                 scenario, adaptive=False)
            
            # Clear and reset
            self._clear_and_reset()
            
            # Phase 2: Adaptive mode
            print(f"🟢 Adaptive mode stress test ({scenario['duration']} min)")
            self._run_stress_phase(scenario_data['adaptive'], duration_seconds, 
                                 duration_seconds * 2, scenario, adaptive=True)
            
        finally:
            traci.close()
        
        return self._analyze_stress_results(scenario_key, scenario_data)
    
    def _run_stress_phase(self, data_dict, start_time, end_time, scenario, adaptive=False):
        """Run stress test phase with detailed monitoring"""
        adaptations_count = 0
        last_log_time = 0
        extreme_wait_threshold = 180  # 3 minutes
        saturation_threshold = 200  # vehicles
        current_pattern = 0
        pattern_start_time = start_time
        
        for step in range(start_time, end_time):
            traci.simulationStep()
            
            # Handle rapid fluctuation patterns
            if scenario.get('flows') == 'RAPID_CHANGE':
                self._handle_rapid_change_pattern(step, start_time, scenario, current_pattern, pattern_start_time)
            
            step_data = self.analyzer.collect_traffic_metrics(step, traci)
            
            if step_data:
                # Calculate pressures
                ns_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E1', '-E1']
                )
                ew_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E0', '-E0']
                )
                
                # Apply adaptive control
                if adaptive:
                    result = self.traffic_controller.apply_adaptive_control(step_data, step)
                    if result.get('applied'):
                        adaptations_count += 1
                
                # Stress-specific monitoring
                max_wait = max([v.get('waiting_time', 0) for edge_data in step_data.get('edge_data', {}).values() 
                              for v in edge_data.get('vehicles', [])], default=0)
                
                # Count extreme waiting vehicles
                extreme_waits = len([v for edge_data in step_data.get('edge_data', {}).values() 
                                   for v in edge_data.get('vehicles', []) 
                                   if v.get('waiting_time', 0) > extreme_wait_threshold])
                
                # Check saturation
                is_saturated = step_data['total_vehicles'] > saturation_threshold
                
                # Store data every 60 seconds
                if step % 60 == 0:
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['pressures_ns'].append(ns_pressure)
                    data_dict['pressures_ew'].append(ew_pressure)
                    data_dict['adaptations'].append(adaptations_count)
                    data_dict['max_waiting'].append(max_wait)
                    data_dict['extreme_waits'].append(extreme_waits)
                    data_dict['saturation_points'].append(1 if is_saturated else 0)
                
                # Detailed progress logging every 3 minutes for stress tests
                if step % 180 == 0 and step != last_log_time:
                    last_log_time = step
                    mode = "Adaptive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    print(f"   {mode} - {minute:4.0f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | Max: {max_wait:5.1f}s | "
                          f"Extreme: {extreme_waits:2d} | NS: {ns_pressure:4.0f} | EW: {ew_pressure:4.0f} | "
                          f"Adaptations: {adaptations_count} | {'SATURATED' if is_saturated else 'Normal'}")
    
    def _handle_rapid_change_pattern(self, step, start_time, scenario, current_pattern, pattern_start_time):
        """Handle rapid change pattern updates"""
        relative_step = step - start_time
        
        # Determine current pattern
        for i, pattern in enumerate(scenario['patterns']):
            pattern_end = sum(p['duration'] for p in scenario['patterns'][:i+1])
            if relative_step < pattern_end:
                if i != current_pattern:
                    current_pattern = i
                    pattern_start_time = step
                    print(f"   🔄 Pattern change: {i+1}/4")
                break
    
    def _clear_and_reset(self):
        """Comprehensive clear and reset for stress tests"""
        try:
            # Remove all vehicles
            for veh_id in traci.vehicle.getIDList():
                traci.vehicle.remove(veh_id)
            
            # Reset traffic light
            traci.trafficlight.setPhase(self.traffic_controller.junction_id, 0)
            
            # Reset controller state
            self.traffic_controller = AdaptiveTrafficController("J4")
            
            # Extended wait for stress test cleanup
            for _ in range(20):
                traci.simulationStep()
            
            print("🧹 Stress test system reset completed")
            
        except Exception as e:
            print(f"⚠️  Warning during stress test reset: {e}")
    
    def _analyze_stress_results(self, scenario_key, data):
        """Analyze stress test results with focus on extreme conditions"""
        if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
            return {'improvement': 0, 'rating': 'NO DATA', 'stress_metrics': {}}
        
        # Skip first 3 data points for warmup
        warmup = 3
        
        # Stress-specific metrics
        stress_metrics = {}
        
        for mode in ['normal', 'adaptive']:
            mode_data = data[mode]
            
            stress_metrics[mode] = {
                'avg_waiting': statistics.mean(mode_data['waiting_times'][warmup:]),
                'peak_waiting': max(mode_data['waiting_times'][warmup:]),
                'max_vehicle_peak': max(mode_data['vehicle_counts'][warmup:]),
                'avg_vehicles': statistics.mean(mode_data['vehicle_counts'][warmup:]),
                'max_waiting_peak': max(mode_data['max_waiting'][warmup:]),
                'total_extreme_waits': sum(mode_data['extreme_waits'][warmup:]),
                'saturation_frequency': sum(mode_data['saturation_points'][warmup:]) / len(mode_data['saturation_points'][warmup:]) * 100,
                'pressure_volatility': statistics.stdev(mode_data['pressures_ns'][warmup:] + mode_data['pressures_ew'][warmup:]) if len(mode_data['pressures_ns'][warmup:]) > 1 else 0,
                'adaptations_total': mode_data['adaptations'][-1] if mode_data['adaptations'] else 0,
                'stability_score': 100 - min(100, statistics.stdev(mode_data['waiting_times'][warmup:]) if len(mode_data['waiting_times'][warmup:]) > 1 else 0)
            }
        
        # Stress improvements
        improvements = {
            'avg_waiting': ((stress_metrics['normal']['avg_waiting'] - stress_metrics['adaptive']['avg_waiting']) / 
                           stress_metrics['normal']['avg_waiting'] * 100) if stress_metrics['normal']['avg_waiting'] > 0 else 0,
            'peak_waiting': ((stress_metrics['normal']['peak_waiting'] - stress_metrics['adaptive']['peak_waiting']) / 
                            stress_metrics['normal']['peak_waiting'] * 100) if stress_metrics['normal']['peak_waiting'] > 0 else 0,
            'extreme_waits': ((stress_metrics['normal']['total_extreme_waits'] - stress_metrics['adaptive']['total_extreme_waits']) / 
                             max(1, stress_metrics['normal']['total_extreme_waits']) * 100),
            'stability': ((stress_metrics['adaptive']['stability_score'] - stress_metrics['normal']['stability_score']) / 
                         max(1, stress_metrics['normal']['stability_score']) * 100),
            'saturation': ((stress_metrics['normal']['saturation_frequency'] - stress_metrics['adaptive']['saturation_frequency']) / 
                          max(1, stress_metrics['normal']['saturation_frequency']) * 100) if stress_metrics['normal']['saturation_frequency'] > 0 else 0
        }
        
        # Stress performance score (emphasis on extreme condition handling)
        stress_score = (improvements['avg_waiting'] * 0.3 + improvements['peak_waiting'] * 0.3 + 
                       improvements['extreme_waits'] * 0.25 + improvements['stability'] * 0.15)
        
        # Stress rating
        if stress_score > 15:
            rating = "🟢 ROBUST"
        elif stress_score > 8:
            rating = "🟡 STABLE"
        elif stress_score > 2:
            rating = "🟠 FRAGILE"
        elif stress_score > -5:
            rating = "🔴 UNSTABLE"
        else:
            rating = "❌ FAILS"
        
        print(f"   🔥 Stress Test Results:")
        print(f"      Avg Waiting: {improvements['avg_waiting']:+6.1f}% | Peak Waiting: {improvements['peak_waiting']:+6.1f}%")
        print(f"      Extreme Waits: {improvements['extreme_waits']:+6.1f}% | Stability: {improvements['stability']:+6.1f}%")
        print(f"      Saturation: {improvements['saturation']:+6.1f}%")
        print(f"      Stress Score: {stress_score:+6.1f}% | {rating}")
        print(f"      Adaptations: {stress_metrics['adaptive']['adaptations_total']} total")
        print(f"      Peak Vehicles: Normal {stress_metrics['normal']['max_vehicle_peak']}, Adaptive {stress_metrics['adaptive']['max_vehicle_peak']}")
        
        return {
            'scenario_key': scenario_key,
            'improvements': improvements,
            'stress_score': stress_score,
            'rating': rating,
            'stress_metrics': stress_metrics
        }
    
    def _generate_stress_report(self, all_results):
        """Generate comprehensive stress test report"""
        print(f"\n🔥 STRESS TEST ANALYSIS REPORT")
        print("=" * 100)
        
        # Stress summary table
        print(f"{'Scenario':<25} {'AvgWait':<8} {'PeakWait':<9} {'ExtremeW':<9} {'Stability':<9} {'StressScore':<11} {'Rating'}")
        print("-" * 100)
        
        stress_scores = []
        
        for result in all_results.values():
            imp = result['improvements']
            scenario_name = self.stress_scenarios[result['scenario_key']]['name'][:24]
            
            print(f"{scenario_name:<25} {imp['avg_waiting']:+6.1f}% {imp['peak_waiting']:+7.1f}% "
                  f"{imp['extreme_waits']:+7.1f}% {imp['stability']:+7.1f}% "
                  f"{result['stress_score']:+9.1f}% {result['rating']}")
            
            stress_scores.append(result['stress_score'])
        
        print("-" * 100)
        
        # Overall stress assessment
        avg_stress_score = statistics.mean(stress_scores) if stress_scores else 0
        
        print(f"{'AVERAGE STRESS SCORE':<25} {avg_stress_score:+6.1f}%")
        
        # Algorithm robustness assessment
        if avg_stress_score > 12:
            robustness = "🟢 HIGHLY ROBUST - Excellent under extreme conditions"
        elif avg_stress_score > 6:
            robustness = "🟡 ROBUST - Good stress handling with consistent performance"
        elif avg_stress_score > 0:
            robustness = "🟠 MODERATE - Some benefits but struggles under extreme stress"
        elif avg_stress_score > -8:
            robustness = "🔴 FRAGILE - Poor performance under stress conditions"
        else:
            robustness = "❌ FAILS - Algorithm breaks down under stress"
        
        print(f"\n🎯 ALGORITHM ROBUSTNESS ASSESSMENT: {robustness}")
        print(f"📈 Average Stress Performance: {avg_stress_score:+.1f}%")
        
        # Detailed stress insights
        best_stress = max(all_results.items(), key=lambda x: x[1]['stress_score'])
        worst_stress = min(all_results.items(), key=lambda x: x[1]['stress_score'])
        
        print(f"\n📊 STRESS TEST INSIGHTS:")
        print(f"🏆 Best Stress Performance: {self.stress_scenarios[best_stress[0]]['name']} ({best_stress[1]['stress_score']:+.1f}%)")
        print(f"⚠️  Worst Stress Performance: {self.stress_scenarios[worst_stress[0]]['name']} ({worst_stress[1]['stress_score']:+.1f}%)")
        
        return avg_stress_score

def main():
    """Run stress test analysis"""
    try:
        analyzer = StressTestAnalyzer()
        analyzer.run_stress_analysis()
    except Exception as e:
        print(f"❌ Error in stress test analysis: {e}")

if __name__ == "__main__":
    main()