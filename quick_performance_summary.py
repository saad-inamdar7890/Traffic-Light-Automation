"""
QUICK PERFORMANCE SUMMARY ANALYZER
=================================
Fast performance analysis with multiple shorter scenarios
to quickly evaluate algorithm effectiveness across different conditions.
"""

import os
import sys
import traci
import statistics
import time
from collections import defaultdict

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class QuickPerformanceSummary:
    """Quick performance analysis with shorter durations but multiple scenarios"""
    
    def __init__(self):
        self.sumo_cmd = ["sumo", "-c", "demo.sumocfg"]
        self.traffic_controller = AdaptiveTrafficController("J4")
        self.analyzer = TrafficAnalyzer()
        
        # Quick test scenarios (10 minutes each phase = 20 min total per scenario)
        self.quick_scenarios = {
            'heavy_traffic': {
                'name': 'Heavy Traffic Test',
                'description': 'High volume traffic across all directions',
                'duration': 10,  # 10 minutes each phase
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 300, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 300, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 250, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 250, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'rate': 150, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 150, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 120, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 120, 'vtype': 'car'}
                }
            },
            
            'imbalanced_traffic': {
                'name': 'Imbalanced Traffic Test',
                'description': 'One direction heavily loaded, others light',
                'duration': 10,
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 400, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 400, 'vtype': 'car'},
                    'f_2': {'from': 'E0', 'to': '-E1.238', 'rate': 250, 'vtype': 'car'},
                    'f_3': {'from': '-E1', 'to': '-E0.254', 'rate': 250, 'vtype': 'car'},
                    # Light EW traffic
                    'f_4': {'from': 'E1', 'to': 'E1.200', 'rate': 40, 'vtype': 'car'},
                    'f_5': {'from': '-E0', 'to': '-E0.254', 'rate': 40, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 20, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 20, 'vtype': 'car'}
                }
            },
            
            'moderate_balanced': {
                'name': 'Moderate Balanced Test',
                'description': 'Balanced moderate traffic load',
                'duration': 10,
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 150, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 150, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 150, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 150, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'rate': 80, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 80, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 80, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 80, 'vtype': 'car'}
                }
            },
            
            'light_traffic': {
                'name': 'Light Traffic Test',
                'description': 'Light traffic to test minimal adaptation',
                'duration': 10,
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 60, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 60, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 60, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 60, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'rate': 30, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 30, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 30, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 30, 'vtype': 'car'}
                }
            },
            
            'extreme_peak': {
                'name': 'Extreme Peak Test',
                'description': 'Maximum traffic load test',
                'duration': 10,
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 450, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 450, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 400, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 400, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'rate': 250, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 250, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 200, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 200, 'vtype': 'car'}
                }
            }
        }
    
    def run_quick_analysis(self):
        """Run quick performance analysis across multiple scenarios"""
        print("⚡ QUICK PERFORMANCE SUMMARY ANALYSIS")
        print("=" * 80)
        print("🎯 Focus: Fast comprehensive evaluation across diverse conditions")
        print("⏱️  Duration: 10 minutes per phase (20 min total per scenario)")
        print("📊 Scenarios: 5 different traffic patterns (100 min total)")
        print("🚀 Goal: Quick but thorough algorithm performance assessment")
        print("=" * 80)
        
        all_results = {}
        start_time = time.time()
        total_scenarios = len(self.quick_scenarios)
        
        for i, (scenario_key, scenario) in enumerate(self.quick_scenarios.items(), 1):
            print(f"\n{'='*50}")
            print(f"⚡ QUICK TEST {i}/{total_scenarios}: {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print(f"⏱️  Duration: {scenario['duration']} min per phase ({scenario['duration'] * 2} min total)")
            print(f"{'='*50}")
            
            scenario_results = self._run_quick_scenario(scenario_key, scenario)
            all_results[scenario_key] = scenario_results
            
            print(f"✅ {scenario['name']} completed")
            
            # Progress update
            elapsed = time.time() - start_time
            remaining_scenarios = total_scenarios - i
            estimated_remaining = (elapsed / i) * remaining_scenarios
            print(f"⏱️  Progress: {i}/{total_scenarios} | Elapsed: {elapsed/60:.1f}min | ETA: {estimated_remaining/60:.1f}min")
        
        # Generate quick summary report
        self._generate_quick_summary_report(all_results)
        
        total_time = time.time() - start_time
        print(f"\n⚡ QUICK ANALYSIS COMPLETED!")
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
        print(f"📊 Performance summary generated!")
    
    def _run_quick_scenario(self, scenario_key, scenario):
        """Run individual quick scenario"""
        traci.start(self.sumo_cmd)
        
        duration_seconds = scenario['duration'] * 60
        
        scenario_data = {
            'normal': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'max_waiting': [],
                'throughput': []
            },
            'adaptive': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'max_waiting': [],
                'throughput': []
            }
        }
        
        try:
            # Phase 1: Normal mode
            print(f"🔴 Normal mode ({scenario['duration']} min)")
            self._run_quick_phase(scenario_data['normal'], 0, duration_seconds, 
                                scenario, adaptive=False)
            
            # Quick reset
            self._quick_reset()
            
            # Phase 2: Adaptive mode
            print(f"🟢 Adaptive mode ({scenario['duration']} min)")
            self._run_quick_phase(scenario_data['adaptive'], duration_seconds, 
                                duration_seconds * 2, scenario, adaptive=True)
            
        finally:
            traci.close()
        
        return self._analyze_quick_results(scenario_key, scenario_data)
    
    def _run_quick_phase(self, data_dict, start_time, end_time, scenario, adaptive=False):
        """Run quick phase with essential monitoring"""
        adaptations_count = 0
        vehicles_completed = 0
        last_vehicle_count = 0
        last_log_time = 0
        
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
                
                # Calculate throughput
                current_vehicles = step_data['total_vehicles']
                if current_vehicles < last_vehicle_count:
                    vehicles_completed += (last_vehicle_count - current_vehicles)
                last_vehicle_count = current_vehicles
                
                # Apply adaptive control
                if adaptive:
                    result = self.traffic_controller.apply_adaptive_control(step_data, step)
                    if result.get('applied'):
                        adaptations_count += 1
                
                # Store data every 30 seconds for quick analysis
                if step % 30 == 0:
                    max_wait = max([v.get('waiting_time', 0) for edge_data in step_data.get('edge_data', {}).values() 
                                  for v in edge_data.get('vehicles', [])], default=0)
                    
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['pressures_ns'].append(ns_pressure)
                    data_dict['pressures_ew'].append(ew_pressure)
                    data_dict['adaptations'].append(adaptations_count)
                    data_dict['max_waiting'].append(max_wait)
                    data_dict['throughput'].append(vehicles_completed)
                
                # Progress logging every 2 minutes
                if step % 120 == 0 and step != last_log_time:
                    last_log_time = step
                    mode = "Adaptive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    print(f"   {mode} - {minute:4.0f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | NS: {ns_pressure:4.0f} | "
                          f"EW: {ew_pressure:4.0f} | Adaptations: {adaptations_count}")
    
    def _quick_reset(self):
        """Quick system reset"""
        try:
            # Remove all vehicles
            for veh_id in traci.vehicle.getIDList():
                traci.vehicle.remove(veh_id)
            
            # Reset traffic light
            traci.trafficlight.setPhase(self.traffic_controller.junction_id, 0)
            
            # Reset controller state
            self.traffic_controller = AdaptiveTrafficController("J4")
            
            # Quick cleanup
            for _ in range(10):
                traci.simulationStep()
            
            print("🧹 Quick reset completed")
            
        except Exception as e:
            print(f"⚠️  Warning during reset: {e}")
    
    def _analyze_quick_results(self, scenario_key, data):
        """Quick analysis of scenario results"""
        if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
            return {'improvement': 0, 'rating': 'NO DATA', 'quick_metrics': {}}
        
        # Skip first 2 data points for warmup
        warmup = 2
        
        # Quick metrics
        metrics = {}
        
        for mode in ['normal', 'adaptive']:
            mode_data = data[mode]
            
            metrics[mode] = {
                'avg_waiting': statistics.mean(mode_data['waiting_times'][warmup:]),
                'peak_waiting': max(mode_data['waiting_times'][warmup:]),
                'avg_vehicles': statistics.mean(mode_data['vehicle_counts'][warmup:]),
                'total_throughput': mode_data['throughput'][-1] if mode_data['throughput'] else 0,
                'avg_max_wait': statistics.mean(mode_data['max_waiting'][warmup:]),
                'pressure_ns_avg': statistics.mean(mode_data['pressures_ns'][warmup:]),
                'pressure_ew_avg': statistics.mean(mode_data['pressures_ew'][warmup:]),
                'adaptations_total': mode_data['adaptations'][-1] if mode_data['adaptations'] else 0
            }
        
        # Calculate improvements
        improvements = {
            'waiting_time': ((metrics['normal']['avg_waiting'] - metrics['adaptive']['avg_waiting']) / 
                           metrics['normal']['avg_waiting'] * 100) if metrics['normal']['avg_waiting'] > 0 else 0,
            'peak_waiting': ((metrics['normal']['peak_waiting'] - metrics['adaptive']['peak_waiting']) / 
                           metrics['normal']['peak_waiting'] * 100) if metrics['normal']['peak_waiting'] > 0 else 0,
            'throughput': ((metrics['adaptive']['total_throughput'] - metrics['normal']['total_throughput']) / 
                         max(1, metrics['normal']['total_throughput']) * 100),
            'max_waiting': ((metrics['normal']['avg_max_wait'] - metrics['adaptive']['avg_max_wait']) / 
                          metrics['normal']['avg_max_wait'] * 100) if metrics['normal']['avg_max_wait'] > 0 else 0
        }
        
        # Overall performance score
        overall_score = (improvements['waiting_time'] + improvements['throughput'] + 
                        improvements['peak_waiting'] + improvements['max_waiting']) / 4
        
        # Performance rating
        if overall_score > 8:
            rating = "🟢 EXCELLENT"
        elif overall_score > 3:
            rating = "🟡 GOOD"
        elif overall_score > 0:
            rating = "🟠 FAIR"
        elif overall_score > -5:
            rating = "🔴 POOR"
        else:
            rating = "❌ VERY POOR"
        
        print(f"   📊 Quick Results:")
        print(f"      Wait Time: {improvements['waiting_time']:+6.1f}% | Peak Wait: {improvements['peak_waiting']:+6.1f}%")
        print(f"      Throughput: {improvements['throughput']:+6.1f}% | Max Wait: {improvements['max_waiting']:+6.1f}%")
        print(f"      Score: {overall_score:+6.1f}% | {rating}")
        print(f"      Adaptations: {metrics['adaptive']['adaptations_total']}")
        
        return {
            'scenario_key': scenario_key,
            'improvements': improvements,
            'overall_score': overall_score,
            'rating': rating,
            'metrics': metrics
        }
    
    def _generate_quick_summary_report(self, all_results):
        """Generate quick performance summary report"""
        print(f"\n⚡ QUICK PERFORMANCE SUMMARY REPORT")
        print("=" * 90)
        
        # Summary table
        print(f"{'Scenario':<22} {'Wait':<8} {'Peak':<8} {'Throughput':<11} {'MaxWait':<8} {'Score':<8} {'Rating'}")
        print("-" * 90)
        
        total_scores = []
        
        for result in all_results.values():
            imp = result['improvements']
            scenario_name = self.quick_scenarios[result['scenario_key']]['name'][:21]
            
            print(f"{scenario_name:<22} {imp['waiting_time']:+6.1f}% {imp['peak_waiting']:+6.1f}% "
                  f"{imp['throughput']:+9.1f}% {imp['max_waiting']:+6.1f}% "
                  f"{result['overall_score']:+6.1f}% {result['rating']}")
            
            total_scores.append(result['overall_score'])
        
        print("-" * 90)
        
        # Overall statistics
        avg_score = statistics.mean(total_scores) if total_scores else 0
        best_scenario = max(all_results.items(), key=lambda x: x[1]['overall_score'])
        worst_scenario = min(all_results.items(), key=lambda x: x[1]['overall_score'])
        
        print(f"{'AVERAGE PERFORMANCE':<22} {avg_score:+6.1f}%")
        print(f"{'BEST SCENARIO':<22} {self.quick_scenarios[best_scenario[0]]['name']} ({best_scenario[1]['overall_score']:+.1f}%)")
        print(f"{'WORST SCENARIO':<22} {self.quick_scenarios[worst_scenario[0]]['name']} ({worst_scenario[1]['overall_score']:+.1f}%)")
        
        # Final quick assessment
        if avg_score > 6:
            final_rating = "🟢 EXCELLENT - Algorithm consistently improves traffic flow"
        elif avg_score > 2:
            final_rating = "🟡 GOOD - Solid performance with measurable benefits"
        elif avg_score > -1:
            final_rating = "🟠 FAIR - Mixed results across different conditions"
        elif avg_score > -5:
            final_rating = "🔴 POOR - Limited benefits in most scenarios"
        else:
            final_rating = "❌ VERY POOR - Algorithm needs significant improvement"
        
        print(f"\n🎯 QUICK ALGORITHM ASSESSMENT: {final_rating}")
        print(f"📈 Average Performance Score: {avg_score:+.1f}%")
        
        # Performance insights
        consistent_performers = [r for r in all_results.values() if r['overall_score'] > 2]
        poor_performers = [r for r in all_results.values() if r['overall_score'] < -2]
        
        print(f"\n📊 PERFORMANCE INSIGHTS:")
        print(f"✅ Consistent Performance: {len(consistent_performers)}/{len(all_results)} scenarios")
        print(f"⚠️  Poor Performance: {len(poor_performers)}/{len(all_results)} scenarios")
        
        # Adaptation insights
        total_adaptations = sum(r['metrics']['adaptive']['adaptations_total'] for r in all_results.values())
        avg_adaptations_per_scenario = total_adaptations / len(all_results) if all_results else 0
        
        print(f"🔧 Algorithm Activity: {total_adaptations} total adaptations ({avg_adaptations_per_scenario:.1f} avg per scenario)")
        
        return avg_score

def main():
    """Run quick performance summary analysis"""
    try:
        analyzer = QuickPerformanceSummary()
        analyzer.run_quick_analysis()
    except Exception as e:
        print(f"❌ Error in quick analysis: {e}")

if __name__ == "__main__":
    main()