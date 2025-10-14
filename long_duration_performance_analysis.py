"""
LONG DURATION PERFORMANCE ANALYZER
=================================
Extended duration testing (45-60 minutes per phase) to evaluate
algorithm performance over extended periods and identify any
degradation or adaptation patterns.
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

class LongDurationAnalyzer:
    """Long duration performance analysis for extended period evaluation"""
    
    def __init__(self):
        self.sumo_cmd = ["sumo", "-c", "demo.sumocfg"]
        self.traffic_controller = AdaptiveTrafficController("J4")
        self.analyzer = TrafficAnalyzer()
        
        # Long duration test scenarios
        self.long_scenarios = {
            'marathon_balanced': {
                'name': 'Marathon Balanced Traffic',
                'description': 'Extended balanced traffic simulation (60 min each)',
                'duration': 60,  # 60 minutes each phase
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 150, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 150, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 150, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 150, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'rate': 100, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 100, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 100, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 100, 'vtype': 'car'}
                }
            },
            
            'extended_rush_hour': {
                'name': 'Extended Rush Hour Simulation',
                'description': 'Prolonged heavy traffic periods (45 min each)',
                'duration': 45,  # 45 minutes each phase
                'flows': {
                    'f_0': {'from': 'E0', 'to': 'E0.319', 'rate': 300, 'vtype': 'car'},
                    'f_1': {'from': '-E1', 'to': '-E1.238', 'rate': 300, 'vtype': 'car'},
                    'f_2': {'from': 'E1', 'to': 'E1.200', 'rate': 250, 'vtype': 'car'},
                    'f_3': {'from': '-E0', 'to': '-E0.254', 'rate': 250, 'vtype': 'car'},
                    'f_4': {'from': 'E0', 'to': '-E1.238', 'rate': 180, 'vtype': 'car'},
                    'f_5': {'from': '-E1', 'to': 'E0.319', 'rate': 180, 'vtype': 'car'},
                    'f_6': {'from': 'E1', 'to': '-E0.254', 'rate': 150, 'vtype': 'car'},
                    'f_7': {'from': '-E0', 'to': 'E1.200', 'rate': 150, 'vtype': 'car'}
                }
            }
        }
    
    def run_long_duration_analysis(self):
        """Run long duration performance analysis"""
        print("⏰ LONG DURATION PERFORMANCE ANALYSIS")
        print("=" * 80)
        print("🎯 Focus: Extended period algorithm stability and learning")
        print("⏱️  Duration: 45-60 minutes per phase (90-120 min total per scenario)")
        print("📊 Analysis: Performance degradation, adaptation patterns, stability")
        print("=" * 80)
        
        all_results = {}
        start_time = time.time()
        
        for scenario_key, scenario in self.long_scenarios.items():
            print(f"\n{'='*60}")
            print(f"⏰ LONG DURATION TEST: {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print(f"⏱️  Duration: {scenario['duration']} minutes per phase ({scenario['duration'] * 2} min total)")
            print(f"{'='*60}")
            
            scenario_results = self._run_long_duration_scenario(scenario_key, scenario)
            all_results[scenario_key] = scenario_results
            
            print(f"✅ {scenario['name']} completed")
            
            # Progress update
            elapsed = time.time() - start_time
            print(f"⏱️  Total elapsed time: {elapsed/60:.1f} minutes")
        
        # Generate long duration analysis report
        self._generate_long_duration_report(all_results)
        
        total_time = time.time() - start_time
        print(f"\n⏰ LONG DURATION ANALYSIS COMPLETED!")
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        print(f"📊 Long duration performance report generated!")
    
    def _run_long_duration_scenario(self, scenario_key, scenario):
        """Run individual long duration scenario"""
        traci.start(self.sumo_cmd)
        
        duration_seconds = scenario['duration'] * 60
        
        scenario_data = {
            'normal': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'throughput': [],
                'efficiency': [], 'phase_durations': [], 'stability_scores': [],
                'time_segments': []  # Track performance over time segments
            },
            'adaptive': {
                'waiting_times': [], 'vehicle_counts': [], 'pressures_ns': [], 
                'pressures_ew': [], 'adaptations': [], 'throughput': [],
                'efficiency': [], 'phase_durations': [], 'stability_scores': [],
                'time_segments': []  # Track performance over time segments
            }
        }
        
        try:
            # Phase 1: Normal mode
            print(f"🔴 Long duration normal mode ({scenario['duration']} min)")
            self._run_long_duration_phase(scenario_data['normal'], 0, duration_seconds, 
                                        scenario, adaptive=False)
            
            # Clear and reset
            self._clear_and_reset()
            
            # Phase 2: Adaptive mode
            print(f"🟢 Long duration adaptive mode ({scenario['duration']} min)")
            self._run_long_duration_phase(scenario_data['adaptive'], duration_seconds, 
                                        duration_seconds * 2, scenario, adaptive=True)
            
        finally:
            traci.close()
        
        return self._analyze_long_duration_results(scenario_key, scenario_data)
    
    def _run_long_duration_phase(self, data_dict, start_time, end_time, scenario, adaptive=False):
        """Run long duration phase with extensive monitoring"""
        adaptations_count = 0
        vehicles_completed = 0
        last_vehicle_count = 0
        last_log_time = 0
        segment_size = 600  # 10-minute segments for time-based analysis
        current_segment = 0
        segment_data = {'waiting_times': [], 'vehicle_counts': [], 'adaptations': 0}
        
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
                adaptation_made = False
                if adaptive:
                    result = self.traffic_controller.apply_adaptive_control(step_data, step)
                    if result.get('applied'):
                        adaptations_count += 1
                        segment_data['adaptations'] += 1
                        adaptation_made = True
                
                # Collect segment data
                segment_data['waiting_times'].append(step_data['avg_waiting_time'])
                segment_data['vehicle_counts'].append(step_data['total_vehicles'])
                
                # Store detailed data every 2 minutes for long duration monitoring
                if step % 120 == 0:
                    # Calculate efficiency and stability
                    efficiency = max(0, 100 - step_data['avg_waiting_time'])
                    
                    # Get current traffic light phase duration
                    try:
                        current_phase = traci.trafficlight.getPhase(self.traffic_controller.junction_id)
                        phase_duration = traci.trafficlight.getPhaseDuration(self.traffic_controller.junction_id)
                    except:
                        phase_duration = 25  # Default
                    
                    # Calculate stability score (based on variance in recent waiting times)
                    recent_waits = data_dict['waiting_times'][-10:] if len(data_dict['waiting_times']) >= 10 else data_dict['waiting_times']
                    stability = 100 - min(100, statistics.stdev(recent_waits) * 10) if len(recent_waits) > 1 else 100
                    
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['pressures_ns'].append(ns_pressure)
                    data_dict['pressures_ew'].append(ew_pressure)
                    data_dict['adaptations'].append(adaptations_count)
                    data_dict['throughput'].append(vehicles_completed)
                    data_dict['efficiency'].append(efficiency)
                    data_dict['phase_durations'].append(phase_duration)
                    data_dict['stability_scores'].append(stability)
                
                # Process 10-minute segments
                if (step - start_time) % segment_size == 0 and step != start_time:
                    if segment_data['waiting_times']:
                        segment_avg_wait = statistics.mean(segment_data['waiting_times'])
                        segment_avg_vehicles = statistics.mean(segment_data['vehicle_counts'])
                        segment_adaptations = segment_data['adaptations']
                        
                        data_dict['time_segments'].append({
                            'segment': current_segment,
                            'avg_waiting': segment_avg_wait,
                            'avg_vehicles': segment_avg_vehicles,
                            'adaptations': segment_adaptations,
                            'minutes': current_segment * 10
                        })
                    
                    current_segment += 1
                    segment_data = {'waiting_times': [], 'vehicle_counts': [], 'adaptations': 0}
                
                # Detailed progress logging every 10 minutes for long duration
                if step % 600 == 0 and step != last_log_time:
                    last_log_time = step
                    mode = "Adaptive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    
                    # Calculate recent trends
                    recent_waits = data_dict['waiting_times'][-5:] if len(data_dict['waiting_times']) >= 5 else data_dict['waiting_times']
                    trend = "↗️" if len(recent_waits) > 1 and recent_waits[-1] > recent_waits[0] else "↘️" if len(recent_waits) > 1 and recent_waits[-1] < recent_waits[0] else "→"
                    
                    print(f"   {mode} - {minute:4.0f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s {trend} | NS: {ns_pressure:4.0f} | EW: {ew_pressure:4.0f} | "
                          f"Adaptations: {adaptations_count} | Throughput: {vehicles_completed} | "
                          f"Phase: {phase_duration:2.0f}s | Stability: {stability:3.0f}")
        
        # Process final segment
        if segment_data['waiting_times']:
            segment_avg_wait = statistics.mean(segment_data['waiting_times'])
            segment_avg_vehicles = statistics.mean(segment_data['vehicle_counts'])
            segment_adaptations = segment_data['adaptations']
            
            data_dict['time_segments'].append({
                'segment': current_segment,
                'avg_waiting': segment_avg_wait,
                'avg_vehicles': segment_avg_vehicles,
                'adaptations': segment_adaptations,
                'minutes': current_segment * 10
            })
    
    def _clear_and_reset(self):
        """Comprehensive clear and reset for long duration tests"""
        try:
            # Remove all vehicles
            for veh_id in traci.vehicle.getIDList():
                traci.vehicle.remove(veh_id)
            
            # Reset traffic light
            traci.trafficlight.setPhase(self.traffic_controller.junction_id, 0)
            
            # Reset controller state
            self.traffic_controller = AdaptiveTrafficController("J4")
            
            # Extended wait for complete cleanup
            for _ in range(30):
                traci.simulationStep()
            
            print("🧹 Long duration system reset completed")
            
        except Exception as e:
            print(f"⚠️  Warning during long duration reset: {e}")
    
    def _analyze_long_duration_results(self, scenario_key, data):
        """Analyze long duration results with focus on time-based patterns"""
        if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
            return {'improvement': 0, 'rating': 'NO DATA', 'long_duration_metrics': {}}
        
        # Skip first 5 data points for warmup (10 minutes)
        warmup = 5
        
        # Long duration metrics
        long_metrics = {}
        
        for mode in ['normal', 'adaptive']:
            mode_data = data[mode]
            
            # Basic metrics
            avg_waiting = statistics.mean(mode_data['waiting_times'][warmup:])
            avg_vehicles = statistics.mean(mode_data['vehicle_counts'][warmup:])
            total_throughput = mode_data['throughput'][-1] if mode_data['throughput'] else 0
            avg_efficiency = statistics.mean(mode_data['efficiency'][warmup:])
            avg_stability = statistics.mean(mode_data['stability_scores'][warmup:])
            
            # Time-based analysis
            early_period = mode_data['waiting_times'][warmup:warmup+15]  # First 30 minutes after warmup
            late_period = mode_data['waiting_times'][-15:]  # Last 30 minutes
            
            early_avg = statistics.mean(early_period) if early_period else avg_waiting
            late_avg = statistics.mean(late_period) if late_period else avg_waiting
            
            # Performance degradation/improvement over time
            time_trend = ((late_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0
            
            # Adaptation effectiveness (for adaptive mode)
            if mode == 'adaptive' and mode_data['adaptations']:
                adaptations_per_hour = mode_data['adaptations'][-1] / (len(mode_data['waiting_times']) * 2 / 60)  # adaptations per hour
                adaptation_effectiveness = -time_trend  # Negative time_trend is good (improvement over time)
            else:
                adaptations_per_hour = 0
                adaptation_effectiveness = 0
            
            # Consistency analysis
            waiting_variance = statistics.variance(mode_data['waiting_times'][warmup:]) if len(mode_data['waiting_times'][warmup:]) > 1 else 0
            consistency_score = max(0, 100 - waiting_variance)
            
            long_metrics[mode] = {
                'avg_waiting': avg_waiting,
                'avg_vehicles': avg_vehicles,
                'total_throughput': total_throughput,
                'avg_efficiency': avg_efficiency,
                'avg_stability': avg_stability,
                'early_period_avg': early_avg,
                'late_period_avg': late_avg,
                'time_trend': time_trend,
                'consistency_score': consistency_score,
                'adaptations_total': mode_data['adaptations'][-1] if mode_data['adaptations'] else 0,
                'adaptations_per_hour': adaptations_per_hour,
                'adaptation_effectiveness': adaptation_effectiveness,
                'time_segments': mode_data['time_segments']
            }
        
        # Long duration improvements
        improvements = {
            'avg_waiting': ((long_metrics['normal']['avg_waiting'] - long_metrics['adaptive']['avg_waiting']) / 
                           long_metrics['normal']['avg_waiting'] * 100) if long_metrics['normal']['avg_waiting'] > 0 else 0,
            'throughput': ((long_metrics['adaptive']['total_throughput'] - long_metrics['normal']['total_throughput']) / 
                         max(1, long_metrics['normal']['total_throughput']) * 100),
            'efficiency': ((long_metrics['adaptive']['avg_efficiency'] - long_metrics['normal']['avg_efficiency']) / 
                         max(1, long_metrics['normal']['avg_efficiency']) * 100),
            'stability': ((long_metrics['adaptive']['avg_stability'] - long_metrics['normal']['avg_stability']) / 
                        max(1, long_metrics['normal']['avg_stability']) * 100),
            'consistency': ((long_metrics['adaptive']['consistency_score'] - long_metrics['normal']['consistency_score']) / 
                          max(1, long_metrics['normal']['consistency_score']) * 100),
            'time_trend': long_metrics['adaptive']['time_trend'] - long_metrics['normal']['time_trend']  # Negative is better
        }
        
        # Long duration performance score
        long_score = (improvements['avg_waiting'] * 0.3 + improvements['throughput'] * 0.2 + 
                     improvements['efficiency'] * 0.2 + improvements['stability'] * 0.15 + 
                     improvements['consistency'] * 0.15)
        
        # Long duration rating
        if long_score > 10:
            rating = "🟢 EXCELLENT LONG-TERM"
        elif long_score > 5:
            rating = "🟡 GOOD LONG-TERM"
        elif long_score > 1:
            rating = "🟠 FAIR LONG-TERM"
        elif long_score > -3:
            rating = "🔴 POOR LONG-TERM"
        else:
            rating = "❌ DEGRADES OVER TIME"
        
        print(f"   ⏰ Long Duration Results:")
        print(f"      Avg Waiting: {improvements['avg_waiting']:+6.1f}% | Throughput: {improvements['throughput']:+6.1f}%")
        print(f"      Efficiency: {improvements['efficiency']:+6.1f}% | Stability: {improvements['stability']:+6.1f}%")
        print(f"      Consistency: {improvements['consistency']:+6.1f}% | Time Trend: {improvements['time_trend']:+6.1f}%")
        print(f"      Long Score: {long_score:+6.1f}% | {rating}")
        print(f"      Adaptations: {long_metrics['adaptive']['adaptations_total']} total ({long_metrics['adaptive']['adaptations_per_hour']:.1f}/hour)")
        print(f"      Normal: Early {long_metrics['normal']['early_period_avg']:.1f}s → Late {long_metrics['normal']['late_period_avg']:.1f}s")
        print(f"      Adaptive: Early {long_metrics['adaptive']['early_period_avg']:.1f}s → Late {long_metrics['adaptive']['late_period_avg']:.1f}s")
        
        return {
            'scenario_key': scenario_key,
            'improvements': improvements,
            'long_score': long_score,
            'rating': rating,
            'long_metrics': long_metrics
        }
    
    def _generate_long_duration_report(self, all_results):
        """Generate comprehensive long duration analysis report"""
        print(f"\n⏰ LONG DURATION ANALYSIS REPORT")
        print("=" * 120)
        
        # Long duration summary table
        print(f"{'Scenario':<30} {'AvgWait':<8} {'Throughput':<10} {'Stability':<9} {'Consistency':<11} {'TimeTrend':<9} {'LongScore':<9} {'Rating'}")
        print("-" * 120)
        
        long_scores = []
        
        for result in all_results.values():
            imp = result['improvements']
            scenario_name = self.long_scenarios[result['scenario_key']]['name'][:29]
            
            print(f"{scenario_name:<30} {imp['avg_waiting']:+6.1f}% {imp['throughput']:+8.1f}% "
                  f"{imp['stability']:+7.1f}% {imp['consistency']:+9.1f}% "
                  f"{imp['time_trend']:+7.1f}% {result['long_score']:+7.1f}% {result['rating']}")
            
            long_scores.append(result['long_score'])
        
        print("-" * 120)
        
        # Overall long duration assessment
        avg_long_score = statistics.mean(long_scores) if long_scores else 0
        
        print(f"{'AVERAGE LONG DURATION SCORE':<30} {avg_long_score:+6.1f}%")
        
        # Algorithm long-term stability assessment
        if avg_long_score > 8:
            stability_assessment = "🟢 EXCELLENT LONG-TERM STABILITY - Algorithm improves over time"
        elif avg_long_score > 4:
            stability_assessment = "🟡 GOOD LONG-TERM STABILITY - Consistent performance over time"
        elif avg_long_score > 0:
            stability_assessment = "🟠 MODERATE STABILITY - Some benefits but inconsistent over time"
        elif avg_long_score > -5:
            stability_assessment = "🔴 POOR STABILITY - Performance degrades over extended periods"
        else:
            stability_assessment = "❌ UNSTABLE - Significant performance degradation over time"
        
        print(f"\n🎯 LONG-TERM ALGORITHM STABILITY: {stability_assessment}")
        print(f"📈 Average Long Duration Performance: {avg_long_score:+.1f}%")
        
        # Time-based analysis
        print(f"\n📊 TIME-BASED PERFORMANCE ANALYSIS:")
        for result in all_results.values():
            scenario_name = self.long_scenarios[result['scenario_key']]['name']
            metrics = result['long_metrics']
            
            print(f"\n{scenario_name}:")
            print(f"   Normal Mode: {metrics['normal']['early_period_avg']:.1f}s → {metrics['normal']['late_period_avg']:.1f}s "
                  f"(Trend: {metrics['normal']['time_trend']:+.1f}%)")
            print(f"   Adaptive Mode: {metrics['adaptive']['early_period_avg']:.1f}s → {metrics['adaptive']['late_period_avg']:.1f}s "
                  f"(Trend: {metrics['adaptive']['time_trend']:+.1f}%)")
            print(f"   Adaptation Rate: {metrics['adaptive']['adaptations_per_hour']:.1f} adaptations/hour")
            print(f"   Effectiveness: {metrics['adaptive']['adaptation_effectiveness']:+.1f}%")
        
        return avg_long_score

def main():
    """Run long duration analysis"""
    try:
        analyzer = LongDurationAnalyzer()
        analyzer.run_long_duration_analysis()
    except Exception as e:
        print(f"❌ Error in long duration analysis: {e}")

if __name__ == "__main__":
    main()