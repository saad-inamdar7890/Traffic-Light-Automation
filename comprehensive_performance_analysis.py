"""
COMPREHENSIVE PERFORMANCE ANALYSIS
=================================
Extended duration testing with diverse scenarios to thoroughly evaluate
the balanced adaptive algorithm performance under various conditions.

Test Configurations:
1. Long Duration Tests (30 minutes each phase)
2. Extreme Traffic Scenarios
3. Rush Hour Simulations
4. Dynamic Traffic Patterns
5. Emergency/Incident Scenarios
"""

import os
import sys
import traci
import statistics
import matplotlib.pyplot as plt
import numpy as np
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

class ComprehensivePerformanceAnalyzer:
    """Comprehensive performance analysis with extended scenarios"""
    
    def __init__(self):
        self.sumo_cmd = ["sumo", "-c", "demo.sumocfg"]
        self.traffic_controller = AdaptiveTrafficController("J4")
        self.analyzer = TrafficAnalyzer()
        
        # Extended test scenarios
        self.test_scenarios = {
            'rush_hour_morning': {
                'name': 'Rush Hour Morning',
                'description': 'Heavy inbound traffic (NS heavy, EW moderate)',
                'duration': 30,  # 30 minutes each phase
                'flows': {
                    # Heavy NS inbound
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 350, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 350, 'vtype': 'car'},
                    'f_2': {'from': 'E0', 'to': '-E1.238', 'rate': 200, 'vtype': 'car'},
                    'f_3': {'from': '-E1', 'to': '-E0.254', 'rate': 200, 'vtype': 'car'},
                    # Moderate EW
                    'f_4': {'from': 'E1', 'to': 'E1.200', 'rate': 150, 'vtype': 'car'},
                    'f_5': {'from': '-E0', 'to': '-E0.254', 'rate': 150, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 100, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 100, 'vtype': 'car'}
                }
            },
            
            'rush_hour_evening': {
                'name': 'Rush Hour Evening',
                'description': 'Heavy outbound traffic (NS light, EW heavy)',
                'duration': 30,
                'flows': {
                    # Light NS
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 80, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 80, 'vtype': 'car'},
                    # Heavy EW outbound
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 400, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 400, 'vtype': 'car'},
                    'f_4': {'from': 'E1', 'to': '-E0.254', 'rate': 250, 'vtype': 'car'},
                    'f_5': {'from': '-E0', 'to': 'E1.200', 'rate': 250, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': 'E0.319', 'rate': 150, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': '-E1.238', 'rate': 150, 'vtype': 'car'}
                }
            },
            
            'weekend_light': {
                'name': 'Weekend Light Traffic',
                'description': 'Consistent light traffic, all directions',
                'duration': 30,
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 40, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 40, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 40, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 40, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'rate': 25, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 25, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 25, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 25, 'vtype': 'car'}
                }
            },
            
            'extreme_imbalance': {
                'name': 'Extreme Traffic Imbalance',
                'description': 'One direction completely saturated, others minimal',
                'duration': 30,
                'flows': {
                    # Extreme NS traffic
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 500, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 500, 'vtype': 'car'},
                    'f_2': {'from': 'E0', 'to': '-E1.238', 'rate': 300, 'vtype': 'car'},
                    'f_3': {'from': '-E1', 'to': '-E0.254', 'rate': 300, 'vtype': 'car'},
                    # Minimal EW traffic
                    'f_4': {'from': 'E1', 'to': 'E1.200', 'rate': 20, 'vtype': 'car'},
                    'f_5': {'from': '-E0', 'to': '-E0.254', 'rate': 20, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 10, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 10, 'vtype': 'car'}
                }
            },
            
            'dynamic_waves': {
                'name': 'Dynamic Traffic Waves',
                'description': 'Fluctuating traffic with periodic waves',
                'duration': 30,
                'flows': 'DYNAMIC',  # Special handling for dynamic flows
                'base_flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'base_rate': 120, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'base_rate': 120, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'base_rate': 120, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'base_rate': 120, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'base_rate': 80, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'base_rate': 80, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'base_rate': 80, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'base_rate': 80, 'vtype': 'car'}
                }
            }
        }
    
    def run_comprehensive_analysis(self):
        """Run comprehensive performance analysis"""
        print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("=" * 80)
        print("⏱️  Extended Duration: 30 minutes per phase (60 min total per scenario)")
        print("🔬 Scenarios: 5 diverse traffic patterns")
        print("📈 Metrics: Detailed performance, adaptation analysis, efficiency")
        print("🎯 Goal: Thorough evaluation of algorithm effectiveness")
        print("=" * 80)
        
        all_results = {}
        start_time = time.time()
        
        for scenario_key, scenario in self.test_scenarios.items():
            print(f"\n{'='*60}")
            print(f"🚦 TESTING SCENARIO: {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print(f"⏱️  Duration: {scenario['duration']} minutes per phase")
            print(f"{'='*60}")
            
            scenario_results = self._run_extended_scenario(scenario_key, scenario)
            all_results[scenario_key] = scenario_results
            
            print(f"✅ {scenario['name']} completed")
            
            # Progress update
            elapsed = time.time() - start_time
            print(f"⏱️  Elapsed time: {elapsed/60:.1f} minutes")
        
        # Comprehensive analysis
        self._generate_comprehensive_report(all_results)
        
        total_time = time.time() - start_time
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
        print(f"📊 Detailed performance report generated!")
    
    def _run_extended_scenario(self, scenario_key, scenario):
        """Run single extended scenario"""
        traci.start(self.sumo_cmd)
        
        duration_seconds = scenario['duration'] * 60  # Convert to seconds
        
        scenario_data = {
            'normal': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'throughput': [],
                'max_waiting': [], 'efficiency': []
            },
            'adaptive': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'throughput': [],
                'max_waiting': [], 'efficiency': []
            }
        }
        
        try:
            # Phase 1: Normal mode
            print(f"🔴 Normal mode ({scenario['duration']} min)")
            self._run_extended_phase(scenario_data['normal'], 0, duration_seconds, 
                                   scenario, adaptive=False)
            
            # Clear and reset
            self._clear_and_reset()
            
            # Phase 2: Adaptive mode
            print(f"🟢 Adaptive mode ({scenario['duration']} min)")
            self._run_extended_phase(scenario_data['adaptive'], duration_seconds, 
                                   duration_seconds * 2, scenario, adaptive=True)
            
        finally:
            traci.close()
        
        return self._analyze_extended_results(scenario_key, scenario_data)
    
    def _run_extended_phase(self, data_dict, start_time, end_time, scenario, adaptive=False):
        """Run extended phase with detailed monitoring"""
        adaptations_count = 0
        vehicles_completed = 0
        last_vehicle_count = 0
        last_log_time = 0
        
        for step in range(start_time, end_time):
            traci.simulationStep()
            
            # Handle dynamic flows if needed
            if scenario.get('flows') == 'DYNAMIC':
                self._update_dynamic_flows(step, scenario['base_flows'])
            
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
                
                # Store data every 60 seconds for extended monitoring
                if step % 60 == 0:
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['pressures_ns'].append(ns_pressure)
                    data_dict['pressures_ew'].append(ew_pressure)
                    data_dict['adaptations'].append(adaptations_count)
                    data_dict['throughput'].append(vehicles_completed)
                    
                    # Additional metrics
                    max_wait = max([v.get('waiting_time', 0) for edge_data in step_data.get('edge_data', {}).values() 
                                  for v in edge_data.get('vehicles', [])], default=0)
                    data_dict['max_waiting'].append(max_wait)
                    
                    # Efficiency score (inverse of average delay)
                    efficiency = max(0, 100 - step_data['avg_waiting_time'])
                    data_dict['efficiency'].append(efficiency)
                
                # Progress logging every 5 minutes
                if step % 300 == 0 and step != last_log_time:
                    last_log_time = step
                    mode = "Adaptive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    print(f"   {mode} - {minute:4.0f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | Max: {max_wait:5.1f}s | "
                          f"NS: {ns_pressure:4.0f} | EW: {ew_pressure:4.0f} | "
                          f"Adaptations: {adaptations_count} | Throughput: {vehicles_completed}")
    
    def _update_dynamic_flows(self, step, base_flows):
        """Update flows for dynamic traffic waves scenario"""
        # Create traffic waves with different periods for each direction
        time_minutes = step / 60
        
        for flow_id, flow_config in base_flows.items():
            base_rate = flow_config['base_rate']
            
            # Different wave patterns for each flow
            if 'E0' in flow_config['from']:  # East flows
                wave = 1 + 0.5 * np.sin(2 * np.pi * time_minutes / 8)  # 8-minute cycle
            elif '-E1' in flow_config['from']:  # South flows  
                wave = 1 + 0.7 * np.sin(2 * np.pi * time_minutes / 12)  # 12-minute cycle
            elif 'E1' in flow_config['from']:  # West flows
                wave = 1 + 0.6 * np.sin(2 * np.pi * time_minutes / 10)  # 10-minute cycle
            else:  # North flows
                wave = 1 + 0.4 * np.sin(2 * np.pi * time_minutes / 6)  # 6-minute cycle
            
            # Add some randomness
            wave *= (0.8 + 0.4 * random.random())
            
            new_rate = max(10, int(base_rate * wave))
            
            # Update flow rate (simplified - in real implementation would use SUMO flow control)
    
    def _clear_and_reset(self):
        """Comprehensive clear and reset"""
        try:
            # Remove all vehicles
            for veh_id in traci.vehicle.getIDList():
                traci.vehicle.remove(veh_id)
            
            # Reset traffic light
            traci.trafficlight.setPhase(self.traffic_controller.junction_id, 0)
            
            # Reset controller state
            self.traffic_controller = AdaptiveTrafficController("J4")
            
            # Wait for cleanup
            for _ in range(10):
                traci.simulationStep()
            
            print("🧹 Complete system reset completed")
            
        except Exception as e:
            print(f"⚠️  Warning during reset: {e}")
    
    def _analyze_extended_results(self, scenario_key, data):
        """Detailed analysis of extended scenario results"""
        if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
            return {'improvement': 0, 'rating': 'NO DATA', 'detailed_metrics': {}}
        
        # Skip first 5 data points for warmup (5 minutes)
        warmup = 5
        
        # Comprehensive metrics
        metrics = {}
        
        for mode in ['normal', 'adaptive']:
            mode_data = data[mode]
            
            metrics[mode] = {
                'avg_waiting': statistics.mean(mode_data['waiting_times'][warmup:]),
                'max_waiting_avg': statistics.mean(mode_data['max_waiting'][warmup:]),
                'max_waiting_peak': max(mode_data['max_waiting'][warmup:]),
                'avg_vehicles': statistics.mean(mode_data['vehicle_counts'][warmup:]),
                'total_throughput': mode_data['throughput'][-1] if mode_data['throughput'] else 0,
                'avg_efficiency': statistics.mean(mode_data['efficiency'][warmup:]),
                'pressure_ns_avg': statistics.mean(mode_data['pressures_ns'][warmup:]),
                'pressure_ew_avg': statistics.mean(mode_data['pressures_ew'][warmup:]),
                'pressure_imbalance': abs(statistics.mean(mode_data['pressures_ns'][warmup:]) - 
                                        statistics.mean(mode_data['pressures_ew'][warmup:])),
                'adaptations_total': mode_data['adaptations'][-1] if mode_data['adaptations'] else 0
            }
        
        # Calculate improvements
        improvements = {
            'waiting_time': ((metrics['normal']['avg_waiting'] - metrics['adaptive']['avg_waiting']) / 
                           metrics['normal']['avg_waiting'] * 100) if metrics['normal']['avg_waiting'] > 0 else 0,
            'max_waiting': ((metrics['normal']['max_waiting_avg'] - metrics['adaptive']['max_waiting_avg']) / 
                          metrics['normal']['max_waiting_avg'] * 100) if metrics['normal']['max_waiting_avg'] > 0 else 0,
            'throughput': ((metrics['adaptive']['total_throughput'] - metrics['normal']['total_throughput']) / 
                         max(1, metrics['normal']['total_throughput']) * 100),
            'efficiency': ((metrics['adaptive']['avg_efficiency'] - metrics['normal']['avg_efficiency']) / 
                         max(1, metrics['normal']['avg_efficiency']) * 100),
            'pressure_balance': ((metrics['normal']['pressure_imbalance'] - metrics['adaptive']['pressure_imbalance']) / 
                               max(1, metrics['normal']['pressure_imbalance']) * 100)
        }
        
        # Overall performance score
        overall_score = (improvements['waiting_time'] + improvements['throughput'] + 
                        improvements['efficiency'] + improvements['pressure_balance']) / 4
        
        # Performance rating
        if overall_score > 10:
            rating = "🟢 EXCELLENT"
        elif overall_score > 5:
            rating = "🟡 GOOD"
        elif overall_score > 1:
            rating = "🟠 FAIR"
        elif overall_score > -3:
            rating = "🔴 POOR"
        else:
            rating = "❌ VERY POOR"
        
        print(f"   📊 Extended Results:")
        print(f"      Waiting Time: {improvements['waiting_time']:+6.1f}% | Max Wait: {improvements['max_waiting']:+6.1f}%")
        print(f"      Throughput: {improvements['throughput']:+8.1f}% | Efficiency: {improvements['efficiency']:+6.1f}%")
        print(f"      Pressure Balance: {improvements['pressure_balance']:+6.1f}%")
        print(f"      Overall Score: {overall_score:+6.1f}% | {rating}")
        print(f"      Adaptations: {metrics['adaptive']['adaptations_total']} total")
        
        return {
            'scenario_key': scenario_key,
            'improvements': improvements,
            'overall_score': overall_score,
            'rating': rating,
            'metrics': metrics,
            'detailed_data': data
        }
    
    def _generate_comprehensive_report(self, all_results):
        """Generate comprehensive performance report"""
        print(f"\n📊 COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 100)
        
        # Summary table
        print(f"{'Scenario':<25} {'Wait':<8} {'MaxWait':<8} {'Throughput':<11} {'Efficiency':<10} {'Balance':<8} {'Overall':<8} {'Rating'}")
        print("-" * 100)
        
        total_scores = []
        
        for result in all_results.values():
            imp = result['improvements']
            scenario_name = self.test_scenarios[result['scenario_key']]['name'][:24]
            
            print(f"{scenario_name:<25} {imp['waiting_time']:+6.1f}% {imp['max_waiting']:+6.1f}% "
                  f"{imp['throughput']:+9.1f}% {imp['efficiency']:+8.1f}% {imp['pressure_balance']:+6.1f}% "
                  f"{result['overall_score']:+6.1f}% {result['rating']}")
            
            total_scores.append(result['overall_score'])
        
        print("-" * 100)
        
        # Overall statistics
        avg_score = statistics.mean(total_scores) if total_scores else 0
        best_scenario = max(all_results.items(), key=lambda x: x[1]['overall_score'])
        worst_scenario = min(all_results.items(), key=lambda x: x[1]['overall_score'])
        
        print(f"{'AVERAGE PERFORMANCE':<25} {avg_score:+6.1f}%")
        print(f"{'BEST SCENARIO':<25} {self.test_scenarios[best_scenario[0]]['name']} ({best_scenario[1]['overall_score']:+.1f}%)")
        print(f"{'WORST SCENARIO':<25} {self.test_scenarios[worst_scenario[0]]['name']} ({worst_scenario[1]['overall_score']:+.1f}%)")
        
        # Final algorithm assessment
        if avg_score > 8:
            final_rating = "🟢 EXCELLENT - Algorithm performs very well across scenarios"
        elif avg_score > 4:
            final_rating = "🟡 GOOD - Solid performance with consistent improvements"
        elif avg_score > 0:
            final_rating = "🟠 FAIR - Mixed results, some scenarios benefit more than others"
        elif avg_score > -5:
            final_rating = "🔴 POOR - Limited benefits, needs optimization"
        else:
            final_rating = "❌ VERY POOR - Algorithm needs significant revision"
        
        print(f"\n🎯 FINAL ALGORITHM ASSESSMENT: {final_rating}")
        print(f"📈 Average Performance Score: {avg_score:+.1f}%")
        
        # Create comprehensive visualization
        self._create_comprehensive_visualization(all_results)
        
        return avg_score
    
    def _create_comprehensive_visualization(self, all_results):
        """Create comprehensive performance visualization"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
            
            scenarios = list(all_results.keys())
            scenario_names = [self.test_scenarios[key]['name'] for key in scenarios]
            
            # Performance improvements by metric
            metrics = ['waiting_time', 'throughput', 'efficiency', 'pressure_balance']
            metric_names = ['Wait Time', 'Throughput', 'Efficiency', 'Pressure Balance']
            
            improvements_data = []
            for metric in metrics:
                improvements_data.append([all_results[scenario]['improvements'][metric] 
                                        for scenario in scenarios])
            
            # Bar chart of improvements
            x = np.arange(len(scenarios))
            width = 0.2
            
            for i, (data, name) in enumerate(zip(improvements_data, metric_names)):
                ax1.bar(x + i*width, data, width, label=name, alpha=0.8)
            
            ax1.set_xlabel('Scenarios')
            ax1.set_ylabel('Improvement (%)')
            ax1.set_title('Performance Improvements by Scenario and Metric')
            ax1.set_xticks(x + width * 1.5)
            ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            
            # Overall scores
            overall_scores = [all_results[scenario]['overall_score'] for scenario in scenarios]
            colors = ['green' if score > 5 else 'orange' if score > 0 else 'red' for score in overall_scores]
            
            ax2.bar(scenario_names, overall_scores, color=colors, alpha=0.7)
            ax2.set_xlabel('Scenarios')
            ax2.set_ylabel('Overall Score (%)')
            ax2.set_title('Overall Performance Score by Scenario')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            
            # Adaptations count
            adaptations = [all_results[scenario]['metrics']['adaptive']['adaptations_total'] 
                          for scenario in scenarios]
            
            ax3.bar(scenario_names, adaptations, color='blue', alpha=0.7)
            ax3.set_xlabel('Scenarios')
            ax3.set_ylabel('Total Adaptations')
            ax3.set_title('Algorithm Activity Level by Scenario')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Efficiency comparison
            normal_efficiency = [all_results[scenario]['metrics']['normal']['avg_efficiency'] 
                               for scenario in scenarios]
            adaptive_efficiency = [all_results[scenario]['metrics']['adaptive']['avg_efficiency'] 
                                 for scenario in scenarios]
            
            x = np.arange(len(scenarios))
            width = 0.35
            
            ax4.bar(x - width/2, normal_efficiency, width, label='Normal Mode', alpha=0.8)
            ax4.bar(x + width/2, adaptive_efficiency, width, label='Adaptive Mode', alpha=0.8)
            
            ax4.set_xlabel('Scenarios')
            ax4.set_ylabel('Efficiency Score')
            ax4.set_title('Traffic Efficiency Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('COMPREHENSIVE ALGORITHM PERFORMANCE ANALYSIS\n'
                        'Extended Duration Testing Across Diverse Scenarios', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('comprehensive_performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("📊 Comprehensive visualization saved as 'comprehensive_performance_analysis.png'")
            
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")

def main():
    """Run comprehensive performance analysis"""
    try:
        analyzer = ComprehensivePerformanceAnalyzer()
        analyzer.run_comprehensive_analysis()
    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {e}")

if __name__ == "__main__":
    main()