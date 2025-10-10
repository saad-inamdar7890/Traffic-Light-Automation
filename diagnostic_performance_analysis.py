"""
DIAGNOSTIC PERFORMANCE ANALYZER
==============================
Investigates and fixes issues with traffic flow configuration
and provides detailed diagnostic information about algorithm performance.
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

class DiagnosticAnalyzer:
    """Diagnostic analyzer to identify and fix performance testing issues"""
    
    def __init__(self):
        self.sumo_cmd = ["sumo", "-c", "demo.sumocfg"]
        self.traffic_controller = AdaptiveTrafficController("J4")
        self.analyzer = TrafficAnalyzer()
        
        # Simplified diagnostic scenarios
        self.diagnostic_scenarios = {
            'light_traffic': {
                'name': 'Light Traffic Diagnostic',
                'description': 'Basic light traffic for baseline testing',
                'duration': 8,  # 8 minutes each phase
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 60, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 60, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 60, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 60, 'vtype': 'car'}
                }
            },
            
            'imbalanced_diagnostic': {
                'name': 'Imbalanced Traffic Diagnostic',
                'description': 'Heavy NS, light EW for differential testing',
                'duration': 8,
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 250, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 250, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 40, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 40, 'vtype': 'car'}
                }
            },
            
            'heavy_diagnostic': {
                'name': 'Heavy Traffic Diagnostic',
                'description': 'High volume traffic for stress testing',
                'duration': 8,
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 200, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 200, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 200, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 200, 'vtype': 'car'}
                }
            }
        }
    
    def run_diagnostic_analysis(self):
        """Run diagnostic performance analysis with detailed monitoring"""
        print("🔍 DIAGNOSTIC PERFORMANCE ANALYSIS")
        print("=" * 80)
        print("🎯 Focus: Identify and fix testing issues")
        print("🔧 Debugging: Traffic flows, resets, data collection")
        print("⏱️  Duration: 8 minutes per phase (16 min total per scenario)")
        print("📊 Scenarios: 3 diagnostic scenarios (48 min total)")
        print("=" * 80)
        
        all_results = {}
        start_time = time.time()
        
        for i, (scenario_key, scenario) in enumerate(self.diagnostic_scenarios.items(), 1):
            print(f"\n{'='*60}")
            print(f"🔍 DIAGNOSTIC TEST {i}/3: {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print(f"⏱️  Duration: {scenario['duration']} min per phase")
            print(f"{'='*60}")
            
            scenario_results = self._run_diagnostic_scenario(scenario_key, scenario)
            all_results[scenario_key] = scenario_results
            
            print(f"✅ {scenario['name']} completed")
            
            # Progress update
            elapsed = time.time() - start_time
            remaining = 3 - i
            eta = (elapsed / i) * remaining
            print(f"⏱️  Progress: {i}/3 | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
        
        # Generate diagnostic report
        self._generate_diagnostic_report(all_results)
        
        total_time = time.time() - start_time
        print(f"\n🔍 DIAGNOSTIC ANALYSIS COMPLETED!")
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
    
    def _run_diagnostic_scenario(self, scenario_key, scenario):
        """Run diagnostic scenario with detailed monitoring"""
        print(f"\n🔧 Starting SUMO simulation...")
        traci.start(self.sumo_cmd)
        
        duration_seconds = scenario['duration'] * 60
        
        scenario_data = {
            'normal': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'max_waiting': [],
                'flow_diagnostics': []
            },
            'adaptive': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'max_waiting': [],
                'flow_diagnostics': []
            }
        }
        
        try:
            # Phase 1: Normal mode with diagnostics
            print(f"🔴 Normal mode diagnostic ({scenario['duration']} min)")
            self._run_diagnostic_phase(scenario_data['normal'], 0, duration_seconds, 
                                     scenario, adaptive=False)
            
            # Diagnostic reset
            self._diagnostic_reset()
            
            # Phase 2: Adaptive mode with diagnostics
            print(f"🟢 Adaptive mode diagnostic ({scenario['duration']} min)")
            self._run_diagnostic_phase(scenario_data['adaptive'], duration_seconds, 
                                     duration_seconds * 2, scenario, adaptive=True)
            
        finally:
            traci.close()
            print(f"🔧 SUMO simulation closed")
        
        return self._analyze_diagnostic_results(scenario_key, scenario_data)
    
    def _run_diagnostic_phase(self, data_dict, start_time, end_time, scenario, adaptive=False):
        """Run diagnostic phase with detailed flow monitoring"""
        adaptations_count = 0
        last_log_time = 0
        
        # Check initial traffic flows
        print(f"🔧 Checking traffic flows configuration...")
        self._check_traffic_flows(scenario)
        
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
                
                # Apply adaptive control with logging
                if adaptive:
                    result = self.traffic_controller.apply_adaptive_control(step_data, step)
                    if result.get('applied'):
                        adaptations_count += 1
                        print(f"🚦 Adaptation #{adaptations_count}: {result.get('action', 'Unknown')}")
                
                # Detailed flow diagnostics every minute
                if step % 60 == 0:
                    flow_diagnostic = {
                        'step': step,
                        'minute': (step - start_time) / 60,
                        'total_vehicles': step_data['total_vehicles'],
                        'avg_waiting': step_data['avg_waiting_time'],
                        'ns_pressure': ns_pressure,
                        'ew_pressure': ew_pressure,
                        'edge_data': {}
                    }
                    
                    # Collect edge-specific data
                    for edge_id in ['E0', '-E1', 'E1', '-E0']:
                        try:
                            edge_vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
                            edge_waiting = traci.edge.getWaitingTime(edge_id)
                            flow_diagnostic['edge_data'][edge_id] = {
                                'vehicles': edge_vehicles,
                                'waiting_time': edge_waiting
                            }
                        except:
                            flow_diagnostic['edge_data'][edge_id] = {'vehicles': 0, 'waiting_time': 0}
                    
                    data_dict['flow_diagnostics'].append(flow_diagnostic)
                
                # Store data every 30 seconds
                if step % 30 == 0:
                    max_wait = max([v.get('waiting_time', 0) for edge_data in step_data.get('edge_data', {}).values() 
                                  for v in edge_data.get('vehicles', [])], default=0)
                    
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['pressures_ns'].append(ns_pressure)
                    data_dict['pressures_ew'].append(ew_pressure)
                    data_dict['adaptations'].append(adaptations_count)
                    data_dict['max_waiting'].append(max_wait)
                
                # Progress logging every 2 minutes
                if step % 120 == 0 and step != last_log_time:
                    last_log_time = step
                    mode = "Adaptive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    
                    # Edge details for diagnostics
                    edge_details = []
                    for edge_id in ['E0', '-E1', 'E1', '-E0']:
                        try:
                            vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
                            edge_details.append(f"{edge_id}:{vehicles}")
                        except:
                            edge_details.append(f"{edge_id}:0")
                    
                    print(f"   {mode} - {minute:4.0f}min | Total: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | NS: {ns_pressure:4.0f} | "
                          f"EW: {ew_pressure:4.0f} | Adaptations: {adaptations_count}")
                    print(f"      Edge Details: {' '.join(edge_details)}")
    
    def _check_traffic_flows(self, scenario):
        """Check and validate traffic flow configuration"""
        print(f"🔧 Traffic flow diagnostics:")
        
        try:
            # Check available edges
            all_edges = traci.edge.getIDList()
            print(f"   Available edges: {len(all_edges)} - {all_edges[:8]}...")
            
            # Check routes
            all_routes = traci.route.getIDList()
            print(f"   Available routes: {len(all_routes)} - {all_routes[:8]}...")
            
            # Check vehicle types
            all_vtypes = traci.vehicletype.getIDList()
            print(f"   Available vehicle types: {all_vtypes}")
            
            # Check specific edges from scenario
            for flow_id, flow_config in scenario['flows'].items():
                from_edge = flow_config['from']
                to_edge = flow_config['to']
                
                from_exists = from_edge in all_edges
                to_exists = to_edge in all_edges
                
                print(f"   Flow {flow_id}: {from_edge}→{to_edge} | From: {'✓' if from_exists else '✗'} | To: {'✓' if to_exists else '✗'}")
                
                if not from_exists:
                    print(f"      ⚠️  Missing source edge: {from_edge}")
                if not to_exists:
                    print(f"      ⚠️  Missing destination edge: {to_edge}")
            
        except Exception as e:
            print(f"   ⚠️  Error during flow diagnostics: {e}")
    
    def _diagnostic_reset(self):
        """Diagnostic reset with detailed logging"""
        try:
            print(f"🧹 Starting diagnostic reset...")
            
            # Count vehicles before reset
            vehicles_before = len(traci.vehicle.getIDList())
            print(f"   Vehicles before reset: {vehicles_before}")
            
            # Remove all vehicles
            for veh_id in traci.vehicle.getIDList():
                traci.vehicle.remove(veh_id)
            
            # Reset traffic light
            traci.trafficlight.setPhase(self.traffic_controller.junction_id, 0)
            print(f"   Traffic light reset to phase 0")
            
            # Reset controller state
            self.traffic_controller = AdaptiveTrafficController("J4")
            print(f"   Controller state reset")
            
            # Extended cleanup
            for i in range(15):
                traci.simulationStep()
                if i % 5 == 0:
                    remaining_vehicles = len(traci.vehicle.getIDList())
                    print(f"   Cleanup step {i}: {remaining_vehicles} vehicles remaining")
            
            vehicles_after = len(traci.vehicle.getIDList())
            print(f"   Vehicles after reset: {vehicles_after}")
            print(f"🧹 Diagnostic reset completed successfully")
            
        except Exception as e:
            print(f"⚠️  Error during diagnostic reset: {e}")
    
    def _analyze_diagnostic_results(self, scenario_key, data):
        """Analyze diagnostic results with detailed comparisons"""
        if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
            print(f"⚠️  Missing data for {scenario_key}")
            return {'improvement': 0, 'rating': 'NO DATA', 'diagnostic_details': {}}
        
        # Skip first data point for warmup
        warmup = 1
        
        # Diagnostic metrics
        diagnostic_metrics = {}
        
        for mode in ['normal', 'adaptive']:
            mode_data = data[mode]
            
            if len(mode_data['waiting_times']) > warmup:
                diagnostic_metrics[mode] = {
                    'avg_waiting': statistics.mean(mode_data['waiting_times'][warmup:]),
                    'peak_waiting': max(mode_data['waiting_times'][warmup:]),
                    'min_waiting': min(mode_data['waiting_times'][warmup:]),
                    'avg_vehicles': statistics.mean(mode_data['vehicle_counts'][warmup:]),
                    'peak_vehicles': max(mode_data['vehicle_counts'][warmup:]),
                    'avg_pressure_ns': statistics.mean(mode_data['pressures_ns'][warmup:]),
                    'avg_pressure_ew': statistics.mean(mode_data['pressures_ew'][warmup:]),
                    'total_adaptations': mode_data['adaptations'][-1] if mode_data['adaptations'] else 0,
                    'data_points': len(mode_data['waiting_times']),
                    'flow_diagnostics': mode_data['flow_diagnostics']
                }
            else:
                diagnostic_metrics[mode] = {
                    'avg_waiting': 0, 'peak_waiting': 0, 'min_waiting': 0,
                    'avg_vehicles': 0, 'peak_vehicles': 0,
                    'avg_pressure_ns': 0, 'avg_pressure_ew': 0,
                    'total_adaptations': 0, 'data_points': 0,
                    'flow_diagnostics': []
                }
        
        # Calculate improvements
        normal_metrics = diagnostic_metrics['normal']
        adaptive_metrics = diagnostic_metrics['adaptive']
        
        if normal_metrics['avg_waiting'] > 0:
            improvements = {
                'avg_waiting': ((normal_metrics['avg_waiting'] - adaptive_metrics['avg_waiting']) / 
                               normal_metrics['avg_waiting'] * 100),
                'peak_waiting': ((normal_metrics['peak_waiting'] - adaptive_metrics['peak_waiting']) / 
                               normal_metrics['peak_waiting'] * 100) if normal_metrics['peak_waiting'] > 0 else 0,
                'pressure_balance': abs(normal_metrics['avg_pressure_ns'] - normal_metrics['avg_pressure_ew']) - 
                                  abs(adaptive_metrics['avg_pressure_ns'] - adaptive_metrics['avg_pressure_ew'])
            }
        else:
            improvements = {'avg_waiting': 0, 'peak_waiting': 0, 'pressure_balance': 0}
        
        overall_score = (improvements['avg_waiting'] + improvements['peak_waiting']) / 2
        
        # Rating
        if overall_score > 5:
            rating = "🟢 GOOD"
        elif overall_score > 1:
            rating = "🟡 FAIR"
        elif overall_score > -2:
            rating = "🟠 MARGINAL"
        else:
            rating = "🔴 POOR"
        
        print(f"   🔍 Diagnostic Results for {scenario_key}:")
        print(f"      Normal: Avg Wait {normal_metrics['avg_waiting']:.1f}s | Peak {normal_metrics['peak_waiting']:.1f}s | Vehicles {normal_metrics['avg_vehicles']:.0f}")
        print(f"      Adaptive: Avg Wait {adaptive_metrics['avg_waiting']:.1f}s | Peak {adaptive_metrics['peak_waiting']:.1f}s | Vehicles {adaptive_metrics['avg_vehicles']:.0f}")
        print(f"      Improvements: Avg Wait {improvements['avg_waiting']:+.1f}% | Peak {improvements['peak_waiting']:+.1f}%")
        print(f"      Score: {overall_score:+.1f}% | {rating}")
        print(f"      Adaptations: {adaptive_metrics['total_adaptations']}")
        print(f"      Data Points: Normal {normal_metrics['data_points']}, Adaptive {adaptive_metrics['data_points']}")
        
        return {
            'scenario_key': scenario_key,
            'improvements': improvements,
            'overall_score': overall_score,
            'rating': rating,
            'diagnostic_metrics': diagnostic_metrics
        }
    
    def _generate_diagnostic_report(self, all_results):
        """Generate diagnostic performance report"""
        print(f"\n🔍 DIAGNOSTIC ANALYSIS REPORT")
        print("=" * 80)
        
        # Diagnostic summary
        print(f"{'Scenario':<25} {'AvgWait':<8} {'PeakWait':<9} {'Score':<8} {'Rating'}")
        print("-" * 80)
        
        valid_scores = []
        
        for result in all_results.values():
            scenario_name = self.diagnostic_scenarios[result['scenario_key']]['name'][:24]
            imp = result['improvements']
            
            print(f"{scenario_name:<25} {imp['avg_waiting']:+6.1f}% {imp['peak_waiting']:+7.1f}% "
                  f"{result['overall_score']:+6.1f}% {result['rating']}")
            
            if result['overall_score'] != 0:
                valid_scores.append(result['overall_score'])
        
        print("-" * 80)
        
        if valid_scores:
            avg_score = statistics.mean(valid_scores)
            print(f"{'AVERAGE DIAGNOSTIC SCORE':<25} {avg_score:+6.1f}%")
            
            if avg_score > 3:
                assessment = "🟢 ALGORITHM WORKING - Good performance detected"
            elif avg_score > 0:
                assessment = "🟡 ALGORITHM MARGINAL - Some improvements detected"
            elif avg_score > -3:
                assessment = "🟠 ALGORITHM ISSUES - Minimal or inconsistent improvements"
            else:
                assessment = "🔴 ALGORITHM PROBLEMS - Performance degradation detected"
        else:
            avg_score = 0
            assessment = "❌ TESTING ISSUES - No valid performance data collected"
        
        print(f"\n🎯 DIAGNOSTIC ASSESSMENT: {assessment}")
        print(f"📈 Average Diagnostic Score: {avg_score:+.1f}%")
        
        # Technical diagnostics
        print(f"\n🔧 TECHNICAL DIAGNOSTICS:")
        for result in all_results.values():
            scenario_name = self.diagnostic_scenarios[result['scenario_key']]['name']
            metrics = result['diagnostic_metrics']
            
            print(f"\n{scenario_name}:")
            print(f"   Data Collection: Normal {metrics['normal']['data_points']} pts, Adaptive {metrics['adaptive']['data_points']} pts")
            print(f"   Vehicle Ranges: Normal {metrics['normal']['avg_vehicles']:.0f} avg, Adaptive {metrics['adaptive']['avg_vehicles']:.0f} avg")
            print(f"   Pressure Balance: Normal NS:{metrics['normal']['avg_pressure_ns']:.0f} EW:{metrics['normal']['avg_pressure_ew']:.0f}")
            print(f"                    Adaptive NS:{metrics['adaptive']['avg_pressure_ns']:.0f} EW:{metrics['adaptive']['avg_pressure_ew']:.0f}")
            print(f"   Algorithm Activity: {metrics['adaptive']['total_adaptations']} adaptations")
        
        return avg_score

def main():
    """Run diagnostic performance analysis"""
    try:
        analyzer = DiagnosticAnalyzer()
        analyzer.run_diagnostic_analysis()
    except Exception as e:
        print(f"❌ Error in diagnostic analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()