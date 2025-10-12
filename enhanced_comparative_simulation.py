"""
Enhanced Comparative Traffic Light Simulation
Phase 1: 10 min normal mode → Complete stop → Phase 2: 10 min adaptive mode
With reduced vehicle flows for cleaner analysis
"""

import os
import sys
import traci
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
import time

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

# Import our modular components
from dynamic_flow_manager import DynamicFlowManager
from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class EnhancedComparativeSimulation:
    def __init__(self, config_file="demo.sumocfg", junction_id="J4"):
        """Initialize enhanced comparative simulation"""
        self.config_file = config_file
        self.junction_id = junction_id
        self.sumo_cmd = ["sumo-gui", "-c", config_file]
        
        # Simulation phases - Now 10 minutes each with separation
        self.normal_phase_duration = 600     # 10 minutes normal mode
        self.flow_stop_duration = 60         # 1 minute to clear all vehicles
        self.adaptive_phase_duration = 600   # 10 minutes adaptive mode
        self.total_duration = self.normal_phase_duration + self.flow_stop_duration + self.adaptive_phase_duration
        
        # Data collection for comparison
        self.normal_phase_data = {
            'waiting_times': [],
            'pressures_ns': [],
            'pressures_ew': [],
            'vehicle_counts': [],
            'throughput': [],
            'avg_speeds': [],
            'timestamps': [],
            'phase_states': []
        }
        
        self.adaptive_phase_data = {
            'waiting_times': [],
            'pressures_ns': [],
            'pressures_ew': [],
            'vehicle_counts': [],
            'throughput': [],
            'avg_speeds': [],
            'timestamps': [],
            'phase_states': []
        }
        
        # Initialize components
        self.flow_manager = DynamicFlowManager()
        self.traffic_controller = AdaptiveTrafficController(junction_id)
        self.analyzer = TrafficAnalyzer()
        
        # Simulation state tracking
        self.current_phase = "NORMAL"
        self.flow_active = True
        
    def stop_all_flows(self):
        """Stop all vehicle flows to clear the network"""
        print("🛑 STOPPING ALL VEHICLE FLOWS")
        # Set all current rates to 0
        for flow_id, config in self.flow_manager.flow_configs.items():
            config['current_rate'] = 0
        self.flow_active = False
        
    def restart_flows_for_adaptive(self):
        """Restart flows for adaptive phase"""
        print("🟢 RESTARTING FLOWS FOR ADAPTIVE PHASE")
        # Reset flows to base rates
        for flow_id, config in self.flow_manager.flow_configs.items():
            config['current_rate'] = config['base_rate']
        self.flow_active = True
        
    def clear_network_vehicles(self):
        """Remove all vehicles from the network"""
        try:
            vehicle_ids = traci.vehicle.getIDList()
            removed_count = 0
            for veh_id in vehicle_ids:
                try:
                    traci.vehicle.remove(veh_id)
                    removed_count += 1
                except:
                    pass
            print(f"🧹 Cleared {removed_count} vehicles from network")
        except Exception as e:
            print(f"Note: Vehicle clearing: {e}")
    
    def run_enhanced_simulation(self):
        """Run the enhanced comparative simulation"""
        print("🚦 ENHANCED COMPARATIVE TRAFFIC LIGHT SIMULATION")
        print("=" * 90)
        print("📊 Phase 1 (0-10 min): Normal Traffic Light Operation")
        print("🛑 Transition (10-11 min): Complete Flow Stop & Network Clear")
        print("🤖 Phase 2 (11-21 min): Adaptive Traffic Light Control")
        print("🏍️  Reduced Traffic Mix: Cars (~960 veh/h) + Motorcycles (~480 veh/h)")
        print("📈 Clean phase separation for accurate comparison")
        print("=" * 90)
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        
        try:
            # Phase 1: Normal Operation (0-600 seconds)
            print(f"\n🔴 PHASE 1: NORMAL TRAFFIC LIGHT MODE (0-{self.normal_phase_duration}s)")
            print("-" * 70)
            self._run_normal_phase()
            
            # Transition: Stop flows and clear network (600-660 seconds)
            print(f"\n🛑 TRANSITION: FLOW STOP & NETWORK CLEAR ({self.normal_phase_duration}-{self.normal_phase_duration + self.flow_stop_duration}s)")
            print("-" * 70)
            self._run_transition_phase()
            
            # Phase 2: Adaptive Operation (660-1260 seconds)
            print(f"\n🟢 PHASE 2: ADAPTIVE TRAFFIC LIGHT MODE ({self.normal_phase_duration + self.flow_stop_duration}-{self.total_duration}s)")
            print("-" * 70)
            self._run_adaptive_phase()
            
            # Generate comparison analysis
            self._analyze_enhanced_results()
            
        except KeyboardInterrupt:
            print("\n⚠️  Simulation interrupted by user")
        except Exception as e:
            print(f"\n❌ Simulation error: {e}")
        finally:
            traci.close()
            
        # Generate graphs and final report
        self._create_enhanced_graphs()
        self._generate_enhanced_report()
        print("\n🎉 Enhanced Comparative Simulation Complete!")
    
    def _run_normal_phase(self):
        """Run normal traffic light phase for 10 minutes"""
        self.current_phase = "NORMAL"
        
        for step in range(self.normal_phase_duration):
            traci.simulationStep()
            
            # Update flows (reduced rates)
            if self.flow_active:
                self.flow_manager.update_flow_rates(step)
            
            # Collect data without adaptive control
            step_data = self.analyzer.collect_traffic_metrics(step, traci)
            
            if step_data:
                # Calculate pressure manually for comparison
                ns_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E1', '-E1']
                )
                ew_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E0', '-E0']
                )
                
                # Get current traffic light phase
                try:
                    current_tl_phase = traci.trafficlight.getPhase(self.junction_id)
                except:
                    current_tl_phase = -1
                
                # Store data every 15 seconds for detailed analysis
                if step % 15 == 0:
                    self.normal_phase_data['waiting_times'].append(step_data['avg_waiting_time'])
                    self.normal_phase_data['pressures_ns'].append(ns_pressure)
                    self.normal_phase_data['pressures_ew'].append(ew_pressure)
                    self.normal_phase_data['vehicle_counts'].append(step_data['total_vehicles'])
                    self.normal_phase_data['timestamps'].append(step)
                    self.normal_phase_data['phase_states'].append(current_tl_phase)
                    
                    # Calculate average speed
                    total_speed = 0
                    vehicle_count = 0
                    for edge_data in step_data['edge_data'].values():
                        if edge_data['vehicle_count'] > 0:
                            total_speed += edge_data['avg_speed'] * edge_data['vehicle_count']
                            vehicle_count += edge_data['vehicle_count']
                    
                    avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0
                    self.normal_phase_data['avg_speeds'].append(avg_speed)
                
                # Display progress every 60 seconds
                if step % 60 == 0:
                    minute = step / 60
                    print(f"⏱️  Normal Mode - {minute:4.1f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Waiting: {step_data['avg_waiting_time']:5.1f}s | "
                          f"NS: {ns_pressure:5.1f} | EW: {ew_pressure:5.1f} | TL Phase: {current_tl_phase}")
    
    def _run_transition_phase(self):
        """Run transition phase - stop flows and clear network"""
        print("🛑 Stopping all vehicle flows...")
        self.stop_all_flows()
        
        transition_start = self.normal_phase_duration
        transition_end = self.normal_phase_duration + self.flow_stop_duration
        
        for step in range(transition_start, transition_end):
            traci.simulationStep()
            
            # Monitor network clearing
            vehicle_ids = traci.vehicle.getIDList()
            vehicle_count = len(vehicle_ids)
            
            # Display progress every 15 seconds
            if (step - transition_start) % 15 == 0:
                elapsed = (step - transition_start)
                print(f"⏱️  Transition - {elapsed:2d}s | Vehicles remaining: {vehicle_count:3d}")
            
            # If network is almost clear, force clear remaining vehicles
            if step == transition_end - 10 and vehicle_count > 5:
                self.clear_network_vehicles()
        
        # Restart flows for adaptive phase
        self.restart_flows_for_adaptive()
        print("🟢 Network cleared and flows restarted for adaptive phase")
    
    def _run_adaptive_phase(self):
        """Run adaptive traffic light phase for 10 minutes"""
        self.current_phase = "ADAPTIVE"
        
        adaptive_start = self.normal_phase_duration + self.flow_stop_duration
        adaptive_end = self.total_duration
        
        for step in range(adaptive_start, adaptive_end):
            traci.simulationStep()
            
            # Update flows (reduced rates)
            if self.flow_active:
                self.flow_manager.update_flow_rates(step)
            
            # Collect data with adaptive control
            step_data = self.analyzer.collect_traffic_metrics(step, traci)
            
            if step_data:
                # Apply adaptive traffic light control
                tl_result = self.traffic_controller.apply_adaptive_control(step_data, step)
                
                # Extract pressure data
                if tl_result and 'analysis' in tl_result:
                    analysis = tl_result['analysis']
                    ns_pressure = analysis.get('ns_pressure', 0)
                    ew_pressure = analysis.get('ew_pressure', 0)
                else:
                    ns_pressure = ew_pressure = 0
                
                # Get current traffic light phase
                try:
                    current_tl_phase = traci.trafficlight.getPhase(self.junction_id)
                except:
                    current_tl_phase = -1
                
                # Store data every 15 seconds for detailed analysis
                if step % 15 == 0:
                    self.adaptive_phase_data['waiting_times'].append(step_data['avg_waiting_time'])
                    self.adaptive_phase_data['pressures_ns'].append(ns_pressure)
                    self.adaptive_phase_data['pressures_ew'].append(ew_pressure)
                    self.adaptive_phase_data['vehicle_counts'].append(step_data['total_vehicles'])
                    self.adaptive_phase_data['timestamps'].append(step)
                    self.adaptive_phase_data['phase_states'].append(current_tl_phase)
                    
                    # Calculate average speed
                    total_speed = 0
                    vehicle_count = 0
                    for edge_data in step_data['edge_data'].values():
                        if edge_data['vehicle_count'] > 0:
                            total_speed += edge_data['avg_speed'] * edge_data['vehicle_count']
                            vehicle_count += edge_data['vehicle_count']
                    
                    avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0
                    self.adaptive_phase_data['avg_speeds'].append(avg_speed)
                
                # Display progress every 60 seconds
                if step % 60 == 0:
                    minute = (step - adaptive_start) / 60
                    print(f"⏱️  Adaptive Mode - {minute:4.1f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Waiting: {step_data['avg_waiting_time']:5.1f}s | "
                          f"NS: {ns_pressure:5.1f} | EW: {ew_pressure:5.1f} | TL Phase: {current_tl_phase}")
    
    def _analyze_enhanced_results(self):
        """Analyze and compare results between phases"""
        print(f"\n📊 ANALYZING ENHANCED COMPARATIVE RESULTS")
        print("=" * 70)
        
        # Calculate metrics for normal phase (exclude first 2 minutes for warmup)
        warmup_samples = 8  # 2 minutes * 4 samples per minute
        normal_data_clean = {
            'waiting_times': self.normal_phase_data['waiting_times'][warmup_samples:],
            'pressures_ns': self.normal_phase_data['pressures_ns'][warmup_samples:],
            'pressures_ew': self.normal_phase_data['pressures_ew'][warmup_samples:],
            'vehicle_counts': self.normal_phase_data['vehicle_counts'][warmup_samples:],
            'avg_speeds': self.normal_phase_data['avg_speeds'][warmup_samples:]
        }
        
        # Calculate metrics for adaptive phase (exclude first 2 minutes for warmup)
        adaptive_data_clean = {
            'waiting_times': self.adaptive_phase_data['waiting_times'][warmup_samples:],
            'pressures_ns': self.adaptive_phase_data['pressures_ns'][warmup_samples:],
            'pressures_ew': self.adaptive_phase_data['pressures_ew'][warmup_samples:],
            'vehicle_counts': self.adaptive_phase_data['vehicle_counts'][warmup_samples:],
            'avg_speeds': self.adaptive_phase_data['avg_speeds'][warmup_samples:]
        }
        
        if normal_data_clean['waiting_times'] and adaptive_data_clean['waiting_times']:
            normal_metrics = {
                'avg_waiting_time': statistics.mean(normal_data_clean['waiting_times']),
                'max_waiting_time': max(normal_data_clean['waiting_times']),
                'avg_ns_pressure': statistics.mean(normal_data_clean['pressures_ns']),
                'avg_ew_pressure': statistics.mean(normal_data_clean['pressures_ew']),
                'avg_vehicle_count': statistics.mean(normal_data_clean['vehicle_counts']),
                'avg_speed': statistics.mean(normal_data_clean['avg_speeds']),
                'total_samples': len(normal_data_clean['waiting_times'])
            }
            
            adaptive_metrics = {
                'avg_waiting_time': statistics.mean(adaptive_data_clean['waiting_times']),
                'max_waiting_time': max(adaptive_data_clean['waiting_times']),
                'avg_ns_pressure': statistics.mean(adaptive_data_clean['pressures_ns']),
                'avg_ew_pressure': statistics.mean(adaptive_data_clean['pressures_ew']),
                'avg_vehicle_count': statistics.mean(adaptive_data_clean['vehicle_counts']),
                'avg_speed': statistics.mean(adaptive_data_clean['avg_speeds']),
                'total_samples': len(adaptive_data_clean['waiting_times'])
            }
            
            self.comparison_results = {
                'normal': normal_metrics,
                'adaptive': adaptive_metrics
            }
            
            # Display enhanced comparison
            print(f"📈 ENHANCED COMPARISON (Excluding 2-min warmup periods):")
            print(f"   Average Waiting Time:")
            print(f"     Normal: {normal_metrics['avg_waiting_time']:.2f}s")
            print(f"     Adaptive: {adaptive_metrics['avg_waiting_time']:.2f}s")
            
            improvement = ((normal_metrics['avg_waiting_time'] - adaptive_metrics['avg_waiting_time']) / 
                          normal_metrics['avg_waiting_time'] * 100)
            print(f"     Improvement: {improvement:+.1f}%")
            
            print(f"   Average Vehicle Count:")
            print(f"     Normal: {normal_metrics['avg_vehicle_count']:.1f}")
            print(f"     Adaptive: {adaptive_metrics['avg_vehicle_count']:.1f}")
            
            print(f"   Average Speed:")
            print(f"     Normal: {normal_metrics['avg_speed']:.2f} m/s")
            print(f"     Adaptive: {adaptive_metrics['avg_speed']:.2f} m/s")
        else:
            print("❌ Insufficient data for comparison")
            self.comparison_results = {'normal': {}, 'adaptive': {}}
    
    def _create_enhanced_graphs(self):
        """Create enhanced comparison graphs"""
        print(f"\n📊 GENERATING ENHANCED COMPARISON GRAPHS")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Traffic Light Performance Comparison\n(10 min Normal → Stop → 10 min Adaptive, Reduced Flows)', fontsize=14)
        
        # Convert timestamps to minutes for better readability
        normal_time_min = [t/60 for t in self.normal_phase_data['timestamps']]
        adaptive_time_min = [(t-660)/60 for t in self.adaptive_phase_data['timestamps']]  # Reset to 0 for adaptive phase
        
        # Graph 1: Waiting Times
        ax1.plot(normal_time_min, self.normal_phase_data['waiting_times'], 
                'r-o', label='Normal Mode', linewidth=2, markersize=4, alpha=0.8)
        ax1.plot(adaptive_time_min, self.adaptive_phase_data['waiting_times'], 
                'g-s', label='Adaptive Mode', linewidth=2, markersize=4, alpha=0.8)
        ax1.set_title('Average Waiting Time Comparison')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Waiting Time (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graph 2: Traffic Pressure
        ax2.plot(normal_time_min, self.normal_phase_data['pressures_ns'], 
                'r-', label='Normal NS', alpha=0.8, linewidth=2)
        ax2.plot(normal_time_min, self.normal_phase_data['pressures_ew'], 
                'r--', label='Normal EW', alpha=0.8, linewidth=2)
        ax2.plot(adaptive_time_min, self.adaptive_phase_data['pressures_ns'], 
                'g-', label='Adaptive NS', alpha=0.8, linewidth=2)
        ax2.plot(adaptive_time_min, self.adaptive_phase_data['pressures_ew'], 
                'g--', label='Adaptive EW', alpha=0.8, linewidth=2)
        ax2.set_title('Traffic Pressure Comparison')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Traffic Pressure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Graph 3: Vehicle Count
        ax3.plot(normal_time_min, self.normal_phase_data['vehicle_counts'], 
                'r-o', label='Normal Mode', linewidth=2, markersize=4, alpha=0.8)
        ax3.plot(adaptive_time_min, self.adaptive_phase_data['vehicle_counts'], 
                'g-s', label='Adaptive Mode', linewidth=2, markersize=4, alpha=0.8)
        ax3.set_title('Vehicle Count in Network')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Number of Vehicles')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Graph 4: Average Speed
        ax4.plot(normal_time_min, self.normal_phase_data['avg_speeds'], 
                'r-o', label='Normal Mode', linewidth=2, markersize=4, alpha=0.8)
        ax4.plot(adaptive_time_min, self.adaptive_phase_data['avg_speeds'], 
                'g-s', label='Adaptive Mode', linewidth=2, markersize=4, alpha=0.8)
        ax4.set_title('Average Network Speed')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_traffic_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary bar chart
        self._create_enhanced_summary()
    
    def _create_enhanced_summary(self):
        """Create enhanced summary comparison"""
        if not (self.comparison_results['normal'] and self.comparison_results['adaptive']):
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        normal = self.comparison_results['normal']
        adaptive = self.comparison_results['adaptive']
        
        metrics = ['Avg Waiting\nTime (s)', 'Avg Vehicle\nCount', 'Avg Speed\n(m/s)', 'NS Pressure', 'EW Pressure']
        normal_values = [normal['avg_waiting_time'], normal['avg_vehicle_count'], 
                        normal['avg_speed'], normal['avg_ns_pressure'], normal['avg_ew_pressure']]
        adaptive_values = [adaptive['avg_waiting_time'], adaptive['avg_vehicle_count'], 
                          adaptive['avg_speed'], adaptive['avg_ns_pressure'], adaptive['avg_ew_pressure']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_values, width, label='Normal Mode (10 min)', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, adaptive_values, width, label='Adaptive Mode (10 min)', color='green', alpha=0.7)
        
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Enhanced Traffic Management Comparison\n(Reduced Flows: Cars ~960 veh/h + Motorcycles ~480 veh/h)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('enhanced_summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_enhanced_report(self):
        """Generate enhanced comparative report"""
        print(f"\n" + "=" * 100)
        print(f"🎯 ENHANCED COMPARATIVE ANALYSIS REPORT")
        print(f"=" * 100)
        
        if not (self.comparison_results['normal'] and self.comparison_results['adaptive']):
            print("❌ Insufficient data for comparison")
            return
        
        normal = self.comparison_results['normal']
        adaptive = self.comparison_results['adaptive']
        
        print(f"📊 ENHANCED PERFORMANCE COMPARISON:")
        print(f"• Normal Mode: 10 minutes with standard traffic light operation")
        print(f"• Adaptive Mode: 10 minutes with AI-controlled adaptive timing")
        print(f"• Clean Separation: Complete network clearing between phases")
        print(f"• Reduced Flows: Cars ~960 veh/h + Motorcycles ~480 veh/h")
        print(f"")
        
        # Calculate improvements
        waiting_improvement = ((normal['avg_waiting_time'] - adaptive['avg_waiting_time']) / normal['avg_waiting_time'] * 100)
        speed_improvement = ((adaptive['avg_speed'] - normal['avg_speed']) / normal['avg_speed'] * 100)
        vehicle_change = ((adaptive['avg_vehicle_count'] - normal['avg_vehicle_count']) / normal['avg_vehicle_count'] * 100)
        
        # Pressure balance analysis
        normal_pressure_diff = abs(normal['avg_ns_pressure'] - normal['avg_ew_pressure'])
        adaptive_pressure_diff = abs(adaptive['avg_ns_pressure'] - adaptive['avg_ew_pressure'])
        balance_improvement = ((normal_pressure_diff - adaptive_pressure_diff) / normal_pressure_diff * 100) if normal_pressure_diff > 0 else 0
        
        print(f"📈 KEY PERFORMANCE METRICS:")
        print(f"{'Metric':<25} {'Normal Mode':<15} {'Adaptive Mode':<15} {'Improvement':<15}")
        print("-" * 80)
        print(f"{'Avg Waiting Time (s)':<25} {normal['avg_waiting_time']:<15.2f} {adaptive['avg_waiting_time']:<15.2f} {waiting_improvement:<14.1f}%")
        print(f"{'Avg Vehicle Count':<25} {normal['avg_vehicle_count']:<15.1f} {adaptive['avg_vehicle_count']:<15.1f} {vehicle_change:<14.1f}%")
        print(f"{'Avg Speed (m/s)':<25} {normal['avg_speed']:<15.2f} {adaptive['avg_speed']:<15.2f} {speed_improvement:<14.1f}%")
        print(f"{'NS Pressure':<25} {normal['avg_ns_pressure']:<15.1f} {adaptive['avg_ns_pressure']:<15.1f}")
        print(f"{'EW Pressure':<25} {normal['avg_ew_pressure']:<15.1f} {adaptive['avg_ew_pressure']:<15.1f}")
        print(f"{'Pressure Balance':<25} {normal_pressure_diff:<15.1f} {adaptive_pressure_diff:<15.1f} {balance_improvement:<14.1f}%")
        
        print(f"\n🎯 ENHANCED ANALYSIS RESULTS:")
        
        # Overall performance assessment
        if waiting_improvement > 5:
            print(f"   ✅ Adaptive control IMPROVED waiting times by {waiting_improvement:.1f}%")
        elif waiting_improvement > 0:
            print(f"   🟡 Adaptive control slightly improved waiting times by {waiting_improvement:.1f}%")
        else:
            print(f"   ❌ Adaptive control increased waiting times by {abs(waiting_improvement):.1f}%")
        
        if speed_improvement > 3:
            print(f"   ✅ Adaptive control IMPROVED speeds by {speed_improvement:.1f}%")
        elif speed_improvement > 0:
            print(f"   🟡 Adaptive control slightly improved speeds by {speed_improvement:.1f}%")
        else:
            print(f"   ❌ Adaptive control reduced speeds by {abs(speed_improvement):.1f}%")
        
        if balance_improvement > 10:
            print(f"   ✅ Adaptive control IMPROVED traffic balance by {balance_improvement:.1f}%")
        elif balance_improvement > 0:
            print(f"   🟡 Adaptive control improved balance by {balance_improvement:.1f}%")
        else:
            print(f"   ⚠️  Adaptive control worsened balance by {abs(balance_improvement):.1f}%")
        
        print(f"\n💡 ENHANCED RECOMMENDATIONS:")
        
        if waiting_improvement > 10:
            print(f"   🎯 EXCELLENT RESULTS - Deploy adaptive system")
            print(f"      • Significant improvement in waiting times")
            print(f"      • Consider expanding to network-wide deployment")
        elif waiting_improvement > 0:
            print(f"   🎯 GOOD RESULTS - Fine-tune and deploy")
            print(f"      • Positive trend in performance")
            print(f"      • Consider parameter optimization for better results")
        else:
            print(f"   🎯 NEEDS IMPROVEMENT - Algorithm tuning required")
            print(f"      • Review adaptive logic parameters")
            print(f"      • Consider different pressure calculation methods")
        
        print(f"\n📊 SIMULATION QUALITY ENHANCEMENTS:")
        print(f"   ✅ Extended duration: 20 minutes total (10 min each phase)")
        print(f"   ✅ Clean phase separation: Complete network clearing")
        print(f"   ✅ Reduced traffic flows: Better analysis conditions")
        print(f"   ✅ Mixed vehicle types: Realistic traffic composition")
        print(f"   ✅ Warmup exclusion: More accurate performance metrics")
        print(f"   📊 Graphs saved: enhanced_traffic_comparison.png, enhanced_summary_comparison.png")
        
        print(f"\n🔍 DATA SUMMARY:")
        print(f"   📈 Normal phase samples: {normal['total_samples']} (post-warmup)")
        print(f"   📈 Adaptive phase samples: {adaptive['total_samples']} (post-warmup)")
        print(f"   📊 Total simulation time: {self.total_duration/60:.1f} minutes")
        print(f"   🚗 Car flows: ~960 veh/h ({960*100/(960+480):.1f}% of traffic)")
        print(f"   🏍️  Motorcycle flows: ~480 veh/h ({480*100/(960+480):.1f}% of traffic)")

def main():
    """Run the enhanced comparative simulation"""
    try:
        simulation = EnhancedComparativeSimulation()
        simulation.run_enhanced_simulation()
    except Exception as e:
        print(f"❌ Error running simulation: {e}")

if __name__ == "__main__":
    main()