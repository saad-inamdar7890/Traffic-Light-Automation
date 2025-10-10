"""
CORRECTED PERFORMANCE ANALYZER
=============================
Properly generates dynamic traffic flows by creating custom route files
for each scenario to ensure different traffic patterns are actually tested.
"""

import os
import sys
import traci
import statistics
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class CorrectedPerformanceAnalyzer:
    """Corrected analyzer that properly generates different traffic flows"""
    
    def __init__(self):
        self.base_sumo_cmd = ["sumo", "-c", "demo.sumocfg"]
        self.traffic_controller = AdaptiveTrafficController("J4")
        self.analyzer = TrafficAnalyzer()
        
        # Corrected test scenarios with proper flow generation
        self.corrected_scenarios = {
            'light_balanced': {
                'name': 'Light Balanced Traffic',
                'description': 'Light traffic, all directions equal',
                'duration': 8,  # 8 minutes each phase
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 100},      # North straight
                    'f_1': {'from': 'E0', 'to': '-E1.238', 'rate': 50},     # North to West
                    'f_2': {'from': 'E0', 'to': 'E1.200', 'rate': 50},      # North to East
                    'f_3': {'from': '-E1', 'to': '-E1.238', 'rate': 100},   # South straight
                    'f_4': {'from': '-E1', 'to': '-E0.254', 'rate': 50},    # South to East
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 50},     # South to North
                    'f_6': {'from': 'E1', 'to': 'E1.200', 'rate': 100},     # East straight
                    'f_7': {'from': 'E1', 'to': '-E0.254', 'rate': 50},     # East to South
                    'f_8': {'from': 'E1', 'to': 'E0.319', 'rate': 50},      # East to North
                    'f_9': {'from': '-E0', 'to': '-E0.254', 'rate': 100},   # West straight
                    'f_10': {'from': '-E0', 'to': '-E1.238', 'rate': 50},   # West to South
                    'f_11': {'from': '-E0', 'to': 'E1.200', 'rate': 50}     # West to East
                }
            },
            
            'heavy_ns_light_ew': {
                'name': 'Heavy NS, Light EW',
                'description': 'Heavy North-South traffic, light East-West',
                'duration': 8,
                'flows': {
                    # Heavy NS flows
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 400},      # North straight
                    'f_1': {'from': 'E0', 'to': '-E1.238', 'rate': 150},     # North to West
                    'f_2': {'from': 'E0', 'to': 'E1.200', 'rate': 150},      # North to East
                    'f_3': {'from': '-E1', 'to': '-E1.238', 'rate': 400},    # South straight
                    'f_4': {'from': '-E1', 'to': '-E0.254', 'rate': 150},    # South to East
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 150},     # South to North
                    # Light EW flows
                    'f_6': {'from': 'E1', 'to': 'E1.200', 'rate': 60},       # East straight
                    'f_7': {'from': 'E1', 'to': '-E0.254', 'rate': 30},      # East to South
                    'f_8': {'from': 'E1', 'to': 'E0.319', 'rate': 30},       # East to North
                    'f_9': {'from': '-E0', 'to': '-E0.254', 'rate': 60},     # West straight
                    'f_10': {'from': '-E0', 'to': '-E1.238', 'rate': 30},    # West to South
                    'f_11': {'from': '-E0', 'to': 'E1.200', 'rate': 30}      # West to East
                }
            },
            
            'heavy_ew_light_ns': {
                'name': 'Heavy EW, Light NS',
                'description': 'Heavy East-West traffic, light North-South',
                'duration': 8,
                'flows': {
                    # Light NS flows
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 60},       # North straight
                    'f_1': {'from': 'E0', 'to': '-E1.238', 'rate': 30},      # North to West
                    'f_2': {'from': 'E0', 'to': 'E1.200', 'rate': 30},       # North to East
                    'f_3': {'from': '-E1', 'to': '-E1.238', 'rate': 60},     # South straight
                    'f_4': {'from': '-E1', 'to': '-E0.254', 'rate': 30},     # South to East
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 30},      # South to North
                    # Heavy EW flows
                    'f_6': {'from': 'E1', 'to': 'E1.200', 'rate': 400},      # East straight
                    'f_7': {'from': 'E1', 'to': '-E0.254', 'rate': 150},     # East to South
                    'f_8': {'from': 'E1', 'to': 'E0.319', 'rate': 150},      # East to North
                    'f_9': {'from': '-E0', 'to': '-E0.254', 'rate': 400},    # West straight
                    'f_10': {'from': '-E0', 'to': '-E1.238', 'rate': 150},   # West to South
                    'f_11': {'from': '-E0', 'to': 'E1.200', 'rate': 150}     # West to East
                }
            },
            
            'heavy_balanced': {
                'name': 'Heavy Balanced Traffic',
                'description': 'Heavy traffic, all directions balanced',
                'duration': 8,
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 300},      # North straight
                    'f_1': {'from': 'E0', 'to': '-E1.238', 'rate': 120},     # North to West
                    'f_2': {'from': 'E0', 'to': 'E1.200', 'rate': 120},      # North to East
                    'f_3': {'from': '-E1', 'to': '-E1.238', 'rate': 300},    # South straight
                    'f_4': {'from': '-E1', 'to': '-E0.254', 'rate': 120},    # South to East
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 120},     # South to North
                    'f_6': {'from': 'E1', 'to': 'E1.200', 'rate': 300},      # East straight
                    'f_7': {'from': 'E1', 'to': '-E0.254', 'rate': 120},     # East to South
                    'f_8': {'from': 'E1', 'to': 'E0.319', 'rate': 120},      # East to North
                    'f_9': {'from': '-E0', 'to': '-E0.254', 'rate': 300},    # West straight
                    'f_10': {'from': '-E0', 'to': '-E1.238', 'rate': 120},   # West to South
                    'f_11': {'from': '-E0', 'to': 'E1.200', 'rate': 120}     # West to East
                }
            }
        }
    
    def run_corrected_analysis(self):
        """Run corrected performance analysis with proper traffic flows"""
        print("🔧 CORRECTED PERFORMANCE ANALYSIS")
        print("=" * 80)
        print("✅ Fixed: Dynamic route file generation for each scenario")
        print("✅ Fixed: Proper traffic flow differentiation")
        print("⏱️  Duration: 8 minutes per phase (16 min total per scenario)")
        print("📊 Scenarios: 4 distinct traffic patterns (64 min total)")
        print("=" * 80)
        
        all_results = {}
        start_time = time.time()
        total_scenarios = len(self.corrected_scenarios)
        
        for i, (scenario_key, scenario) in enumerate(self.corrected_scenarios.items(), 1):
            print(f"\n{'='*60}")
            print(f"🔧 CORRECTED TEST {i}/{total_scenarios}: {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print(f"⏱️  Duration: {scenario['duration']} min per phase")
            print(f"{'='*60}")
            
            scenario_results = self._run_corrected_scenario(scenario_key, scenario)
            all_results[scenario_key] = scenario_results
            
            print(f"✅ {scenario['name']} completed")
            
            # Progress update
            elapsed = time.time() - start_time
            remaining = total_scenarios - i
            eta = (elapsed / i) * remaining
            print(f"⏱️  Progress: {i}/{total_scenarios} | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
        
        # Generate corrected report
        self._generate_corrected_report(all_results)
        
        total_time = time.time() - start_time
        print(f"\n🔧 CORRECTED ANALYSIS COMPLETED!")
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
    
    def _generate_route_file(self, scenario, filename):
        """Generate a custom route file for the scenario"""
        print(f"🔧 Generating route file: {filename}")
        
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
        
        # Add flows
        for flow_id, flow_config in scenario['flows'].items():
            flow_elem = ET.SubElement(root, "flow")
            flow_elem.set("id", flow_id)
            flow_elem.set("begin", "0.00")
            flow_elem.set("end", str(scenario['duration'] * 60 * 2))  # Total duration for both phases
            flow_elem.set("from", flow_config['from'])
            flow_elem.set("to", flow_config['to'])
            flow_elem.set("vehsPerHour", str(flow_config['rate']))
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(filename, encoding="UTF-8", xml_declaration=True)
        
        print(f"   ✅ Route file generated with {len(scenario['flows'])} flows")
        
        # Print flow summary
        total_rate = sum(flow['rate'] for flow in scenario['flows'].values())
        print(f"   📊 Total traffic rate: {total_rate} vehicles/hour")
        
        # Direction breakdown
        ns_rate = sum(flow['rate'] for flow in scenario['flows'].values() 
                     if flow['from'] in ['E0', '-E1'])
        ew_rate = sum(flow['rate'] for flow in scenario['flows'].values() 
                     if flow['from'] in ['E1', '-E0'])
        print(f"   📊 NS traffic: {ns_rate} veh/h | EW traffic: {ew_rate} veh/h")
    
    def _run_corrected_scenario(self, scenario_key, scenario):
        """Run scenario with dynamically generated route file"""
        # Generate custom route file for this scenario
        custom_route_file = f"scenario_{scenario_key}.rou.xml"
        self._generate_route_file(scenario, custom_route_file)
        
        # Create custom SUMO command with the new route file
        sumo_cmd = ["sumo", "-n", "demo.net.xml", "-r", custom_route_file, 
                   "--time-to-teleport", "300", "--no-warnings"]
        
        print(f"🔧 Starting SUMO with custom route file...")
        traci.start(sumo_cmd)
        
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
            self._run_corrected_phase(scenario_data['normal'], 0, duration_seconds, 
                                    adaptive=False)
            
            # Reset for second phase
            self._corrected_reset()
            
            # Phase 2: Adaptive mode
            print(f"🟢 Adaptive mode ({scenario['duration']} min)")
            self._run_corrected_phase(scenario_data['adaptive'], duration_seconds, 
                                    duration_seconds * 2, adaptive=True)
            
        finally:
            traci.close()
            # Clean up custom route file
            try:
                os.remove(custom_route_file)
                print(f"🧹 Cleaned up {custom_route_file}")
            except:
                pass
        
        return self._analyze_corrected_results(scenario_key, scenario_data)
    
    def _run_corrected_phase(self, data_dict, start_time, end_time, adaptive=False):
        """Run phase with proper monitoring"""
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
                    data_dict['throughput'].append(vehicles_completed)
                
                # Progress logging every 2 minutes
                if step % 120 == 0 and step != last_log_time:
                    last_log_time = step
                    mode = "Adaptive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    print(f"   {mode} - {minute:4.0f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | NS: {ns_pressure:4.0f} | "
                          f"EW: {ew_pressure:4.0f} | Adaptations: {adaptations_count} | "
                          f"Throughput: {vehicles_completed}")
    
    def _corrected_reset(self):
        """Corrected reset procedure"""
        try:
            print(f"🧹 Corrected reset starting...")
            
            # Remove all vehicles
            vehicle_ids = traci.vehicle.getIDList()
            for veh_id in vehicle_ids:
                traci.vehicle.remove(veh_id)
            
            # Reset traffic light
            traci.trafficlight.setPhase(self.traffic_controller.junction_id, 0)
            
            # Reset controller state
            self.traffic_controller = AdaptiveTrafficController("J4")
            
            # Cleanup steps
            for _ in range(15):
                traci.simulationStep()
            
            remaining_vehicles = len(traci.vehicle.getIDList())
            print(f"🧹 Reset completed | Remaining vehicles: {remaining_vehicles}")
            
        except Exception as e:
            print(f"⚠️  Warning during reset: {e}")
    
    def _analyze_corrected_results(self, scenario_key, data):
        """Analyze corrected results"""
        if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
            return {'improvement': 0, 'rating': 'NO DATA', 'corrected_metrics': {}}
        
        # Skip first data point for warmup
        warmup = 1
        
        # Corrected metrics
        metrics = {}
        
        for mode in ['normal', 'adaptive']:
            mode_data = data[mode]
            
            if len(mode_data['waiting_times']) > warmup:
                metrics[mode] = {
                    'avg_waiting': statistics.mean(mode_data['waiting_times'][warmup:]),
                    'peak_waiting': max(mode_data['waiting_times'][warmup:]),
                    'avg_vehicles': statistics.mean(mode_data['vehicle_counts'][warmup:]),
                    'total_throughput': mode_data['throughput'][-1] if mode_data['throughput'] else 0,
                    'avg_pressure_ns': statistics.mean(mode_data['pressures_ns'][warmup:]),
                    'avg_pressure_ew': statistics.mean(mode_data['pressures_ew'][warmup:]),
                    'adaptations_total': mode_data['adaptations'][-1] if mode_data['adaptations'] else 0
                }
            else:
                metrics[mode] = {
                    'avg_waiting': 0, 'peak_waiting': 0, 'avg_vehicles': 0,
                    'total_throughput': 0, 'avg_pressure_ns': 0, 'avg_pressure_ew': 0,
                    'adaptations_total': 0
                }
        
        # Calculate improvements
        improvements = {}
        if metrics['normal']['avg_waiting'] > 0:
            improvements = {
                'waiting_time': ((metrics['normal']['avg_waiting'] - metrics['adaptive']['avg_waiting']) / 
                               metrics['normal']['avg_waiting'] * 100),
                'peak_waiting': ((metrics['normal']['peak_waiting'] - metrics['adaptive']['peak_waiting']) / 
                               metrics['normal']['peak_waiting'] * 100) if metrics['normal']['peak_waiting'] > 0 else 0,
                'throughput': ((metrics['adaptive']['total_throughput'] - metrics['normal']['total_throughput']) / 
                             max(1, metrics['normal']['total_throughput']) * 100),
                'pressure_balance': abs(metrics['normal']['avg_pressure_ns'] - metrics['normal']['avg_pressure_ew']) - 
                                  abs(metrics['adaptive']['avg_pressure_ns'] - metrics['adaptive']['avg_pressure_ew'])
            }
        else:
            improvements = {'waiting_time': 0, 'peak_waiting': 0, 'throughput': 0, 'pressure_balance': 0}
        
        overall_score = (improvements['waiting_time'] + improvements['throughput']) / 2
        
        # Rating
        if overall_score > 5:
            rating = "🟢 EXCELLENT"
        elif overall_score > 2:
            rating = "🟡 GOOD"
        elif overall_score > 0:
            rating = "🟠 FAIR"
        elif overall_score > -3:
            rating = "🔴 POOR"
        else:
            rating = "❌ VERY POOR"
        
        print(f"   🔧 Corrected Results for {scenario_key}:")
        print(f"      Wait Time: {improvements['waiting_time']:+6.1f}% | Peak Wait: {improvements['peak_waiting']:+6.1f}%")
        print(f"      Throughput: {improvements['throughput']:+6.1f}% | Pressure Balance: {improvements['pressure_balance']:+6.1f}")
        print(f"      Score: {overall_score:+6.1f}% | {rating}")
        print(f"      Adaptations: {metrics['adaptive']['adaptations_total']}")
        print(f"      Normal: {metrics['normal']['avg_waiting']:.1f}s wait, {metrics['normal']['avg_vehicles']:.0f} vehicles")
        print(f"      Adaptive: {metrics['adaptive']['avg_waiting']:.1f}s wait, {metrics['adaptive']['avg_vehicles']:.0f} vehicles")
        
        return {
            'scenario_key': scenario_key,
            'improvements': improvements,
            'overall_score': overall_score,
            'rating': rating,
            'metrics': metrics
        }
    
    def _generate_corrected_report(self, all_results):
        """Generate corrected performance report"""
        print(f"\n🔧 CORRECTED PERFORMANCE ANALYSIS REPORT")
        print("=" * 90)
        
        # Corrected summary table
        print(f"{'Scenario':<25} {'Wait':<8} {'Peak':<8} {'Throughput':<11} {'Score':<8} {'Rating'}")
        print("-" * 90)
        
        valid_scores = []
        
        for result in all_results.values():
            scenario_name = self.corrected_scenarios[result['scenario_key']]['name'][:24]
            imp = result['improvements']
            
            print(f"{scenario_name:<25} {imp['waiting_time']:+6.1f}% {imp['peak_waiting']:+6.1f}% "
                  f"{imp['throughput']:+9.1f}% {result['overall_score']:+6.1f}% {result['rating']}")
            
            if result['overall_score'] != 0:
                valid_scores.append(result['overall_score'])
        
        print("-" * 90)
        
        if valid_scores:
            avg_score = statistics.mean(valid_scores)
            best_scenario = max(all_results.items(), key=lambda x: x[1]['overall_score'])
            worst_scenario = min(all_results.items(), key=lambda x: x[1]['overall_score'])
            
            print(f"{'AVERAGE PERFORMANCE':<25} {avg_score:+6.1f}%")
            print(f"{'BEST SCENARIO':<25} {self.corrected_scenarios[best_scenario[0]]['name'][:24]} ({best_scenario[1]['overall_score']:+.1f}%)")
            print(f"{'WORST SCENARIO':<25} {self.corrected_scenarios[worst_scenario[0]]['name'][:24]} ({worst_scenario[1]['overall_score']:+.1f}%)")
            
            # Final assessment
            if avg_score > 4:
                assessment = "🟢 EXCELLENT - Algorithm shows consistent improvements"
            elif avg_score > 1:
                assessment = "🟡 GOOD - Algorithm provides measurable benefits"
            elif avg_score > -1:
                assessment = "🟠 FAIR - Mixed results across scenarios"
            elif avg_score > -5:
                assessment = "🔴 POOR - Limited benefits, needs improvement"
            else:
                assessment = "❌ FAILS - Algorithm degrades performance"
        else:
            avg_score = 0
            assessment = "❌ NO VALID DATA - Technical issues"
        
        print(f"\n🎯 CORRECTED ALGORITHM ASSESSMENT: {assessment}")
        print(f"📈 Average Performance Score: {avg_score:+.1f}%")
        
        # Scenario-specific insights
        print(f"\n📊 SCENARIO-SPECIFIC INSIGHTS:")
        for result in all_results.values():
            scenario_name = self.corrected_scenarios[result['scenario_key']]['name']
            metrics = result['metrics']
            
            print(f"\n{scenario_name}:")
            print(f"   Normal Mode: {metrics['normal']['avg_waiting']:.1f}s avg wait, {metrics['normal']['avg_vehicles']:.0f} vehicles")
            print(f"   Adaptive Mode: {metrics['adaptive']['avg_waiting']:.1f}s avg wait, {metrics['adaptive']['avg_vehicles']:.0f} vehicles")
            print(f"   Improvement: {result['improvements']['waiting_time']:+.1f}% wait time, {result['improvements']['throughput']:+.1f}% throughput")
            print(f"   Adaptations: {metrics['adaptive']['adaptations_total']} total")
        
        return avg_score

def main():
    """Run corrected performance analysis"""
    try:
        analyzer = CorrectedPerformanceAnalyzer()
        analyzer.run_corrected_analysis()
    except Exception as e:
        print(f"❌ Error in corrected analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()