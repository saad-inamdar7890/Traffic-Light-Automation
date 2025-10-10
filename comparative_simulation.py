"""
Comparative Traffic Light Simulation
Compares normal traffic light operation vs adaptive control with diverse vehicle types
"""

import os
import sys
import traci
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque

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

class ComparativeTrafficSimulation:
    def __init__(self, config_file="demo.sumocfg", junction_id="J4"):
        """Initialize comparative simulation"""
        self.config_file = config_file
        self.junction_id = junction_id
        self.sumo_cmd = ["sumo-gui", "-c", config_file]
        
        # Simulation phases
        self.normal_phase_duration = 300    # 5 minutes normal mode
        self.adaptive_phase_duration = 300  # 5 minutes adaptive mode
        self.total_duration = self.normal_phase_duration + self.adaptive_phase_duration
        
        # Data collection for comparison
        self.normal_phase_data = {
            'waiting_times': [],
            'pressures_ns': [],
            'pressures_ew': [],
            'vehicle_counts': [],
            'throughput': [],
            'avg_speeds': [],
            'timestamps': []
        }
        
        self.adaptive_phase_data = {
            'waiting_times': [],
            'pressures_ns': [],
            'pressures_ew': [],
            'vehicle_counts': [],
            'throughput': [],
            'avg_speeds': [],
            'timestamps': []
        }
        
        # Initialize components
        self.flow_manager = DynamicFlowManager()
        self.traffic_controller = AdaptiveTrafficController(junction_id)
        self.analyzer = TrafficAnalyzer()
        
        # Comparison metrics
        self.comparison_results = {}
        
    def run_comparative_simulation(self):
        """Run the full comparative simulation"""
        print("🚦 COMPARATIVE TRAFFIC LIGHT SIMULATION")
        print("=" * 80)
        print("📊 Phase 1 (0-5 min): Normal Traffic Light Operation")
        print("🤖 Phase 2 (5-10 min): Adaptive Traffic Light Control")
        print("🏍️  Traffic Mix: Cars + Motorcycles/Bikes for diverse flow")
        print("📈 Real-time data collection for performance comparison")
        print("=" * 80)
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        
        try:
            # Phase 1: Normal Operation (0-300 seconds)
            print(f"\n🔴 PHASE 1: NORMAL TRAFFIC LIGHT MODE (0-{self.normal_phase_duration}s)")
            print("-" * 60)
            self._run_normal_phase()
            
            # Phase 2: Adaptive Operation (300-600 seconds)
            print(f"\n🟢 PHASE 2: ADAPTIVE TRAFFIC LIGHT MODE ({self.normal_phase_duration}-{self.total_duration}s)")
            print("-" * 60)
            self._run_adaptive_phase()
            
            # Generate comparison analysis
            self._analyze_comparison_results()
            
        except KeyboardInterrupt:
            print("\n⚠️  Simulation interrupted by user")
        except Exception as e:
            print(f"\n❌ Simulation error: {e}")
        finally:
            traci.close()
            
        # Generate graphs and final report
        self._create_comparison_graphs()
        self._generate_comparative_report()
        print("\n🎉 Comparative Simulation Complete!")
    
    def _run_normal_phase(self):
        """Run normal traffic light phase"""
        for step in range(self.normal_phase_duration):
            traci.simulationStep()
            
            # Update flows (with bikes)
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
                
                # Store data every 10 seconds for detailed analysis
                if step % 10 == 0:
                    self.normal_phase_data['waiting_times'].append(step_data['avg_waiting_time'])
                    self.normal_phase_data['pressures_ns'].append(ns_pressure)
                    self.normal_phase_data['pressures_ew'].append(ew_pressure)
                    self.normal_phase_data['vehicle_counts'].append(step_data['total_vehicles'])
                    self.normal_phase_data['timestamps'].append(step)
                    
                    # Calculate average speed
                    total_speed = 0
                    vehicle_count = 0
                    for edge_data in step_data['edge_data'].values():
                        if edge_data['vehicle_count'] > 0:
                            total_speed += edge_data['avg_speed'] * edge_data['vehicle_count']
                            vehicle_count += edge_data['vehicle_count']
                    
                    avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0
                    self.normal_phase_data['avg_speeds'].append(avg_speed)
                
                # Display progress every 30 seconds
                if step % 30 == 0:
                    minute = step / 60
                    print(f"⏱️  Normal Mode - {minute:.1f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Waiting: {step_data['avg_waiting_time']:5.1f}s | "
                          f"NS Pressure: {ns_pressure:5.1f} | EW Pressure: {ew_pressure:5.1f}")
    
    def _run_adaptive_phase(self):
        """Run adaptive traffic light phase"""
        for step in range(self.normal_phase_duration, self.total_duration):
            traci.simulationStep()
            
            # Update flows (with bikes)
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
                
                # Store data every 10 seconds for detailed analysis
                if step % 10 == 0:
                    self.adaptive_phase_data['waiting_times'].append(step_data['avg_waiting_time'])
                    self.adaptive_phase_data['pressures_ns'].append(ns_pressure)
                    self.adaptive_phase_data['pressures_ew'].append(ew_pressure)
                    self.adaptive_phase_data['vehicle_counts'].append(step_data['total_vehicles'])
                    self.adaptive_phase_data['timestamps'].append(step)
                    
                    # Calculate average speed
                    total_speed = 0
                    vehicle_count = 0
                    for edge_data in step_data['edge_data'].values():
                        if edge_data['vehicle_count'] > 0:
                            total_speed += edge_data['avg_speed'] * edge_data['vehicle_count']
                            vehicle_count += edge_data['vehicle_count']
                    
                    avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0
                    self.adaptive_phase_data['avg_speeds'].append(avg_speed)
                
                # Display progress every 30 seconds
                if step % 30 == 0:
                    minute = step / 60
                    print(f"⏱️  Adaptive Mode - {minute:.1f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Waiting: {step_data['avg_waiting_time']:5.1f}s | "
                          f"NS Pressure: {ns_pressure:5.1f} | EW Pressure: {ew_pressure:5.1f}")
    
    def _analyze_comparison_results(self):
        """Analyze and compare results between phases"""
        print(f"\n📊 ANALYZING COMPARATIVE RESULTS")
        print("=" * 60)
        
        # Calculate metrics for normal phase
        if self.normal_phase_data['waiting_times']:
            normal_metrics = {
                'avg_waiting_time': statistics.mean(self.normal_phase_data['waiting_times']),
                'max_waiting_time': max(self.normal_phase_data['waiting_times']),
                'avg_ns_pressure': statistics.mean(self.normal_phase_data['pressures_ns']),
                'avg_ew_pressure': statistics.mean(self.normal_phase_data['pressures_ew']),
                'avg_vehicle_count': statistics.mean(self.normal_phase_data['vehicle_counts']),
                'avg_speed': statistics.mean(self.normal_phase_data['avg_speeds']),
                'total_samples': len(self.normal_phase_data['waiting_times'])
            }
        else:
            normal_metrics = {}
        
        # Calculate metrics for adaptive phase
        if self.adaptive_phase_data['waiting_times']:
            adaptive_metrics = {
                'avg_waiting_time': statistics.mean(self.adaptive_phase_data['waiting_times']),
                'max_waiting_time': max(self.adaptive_phase_data['waiting_times']),
                'avg_ns_pressure': statistics.mean(self.adaptive_phase_data['pressures_ns']),
                'avg_ew_pressure': statistics.mean(self.adaptive_phase_data['pressures_ew']),
                'avg_vehicle_count': statistics.mean(self.adaptive_phase_data['vehicle_counts']),
                'avg_speed': statistics.mean(self.adaptive_phase_data['avg_speeds']),
                'total_samples': len(self.adaptive_phase_data['waiting_times'])
            }
        else:
            adaptive_metrics = {}
        
        self.comparison_results = {
            'normal': normal_metrics,
            'adaptive': adaptive_metrics
        }
        
        # Display quick comparison
        if normal_metrics and adaptive_metrics:
            print(f"📈 QUICK COMPARISON:")
            print(f"   Average Waiting Time:")
            print(f"     Normal: {normal_metrics['avg_waiting_time']:.2f}s")
            print(f"     Adaptive: {adaptive_metrics['avg_waiting_time']:.2f}s")
            
            improvement = ((normal_metrics['avg_waiting_time'] - adaptive_metrics['avg_waiting_time']) / 
                          normal_metrics['avg_waiting_time'] * 100)
            print(f"     Improvement: {improvement:+.1f}%")
            
            print(f"   Average Speed:")
            print(f"     Normal: {normal_metrics['avg_speed']:.2f} m/s")
            print(f"     Adaptive: {adaptive_metrics['avg_speed']:.2f} m/s")
    
    def _create_comparison_graphs(self):
        """Create comparison graphs"""
        print(f"\n📊 GENERATING COMPARISON GRAPHS")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Traffic Light Performance Comparison: Normal vs Adaptive Control', fontsize=16)
        
        # Convert timestamps to minutes for better readability
        normal_time_min = [t/60 for t in self.normal_phase_data['timestamps']]
        adaptive_time_min = [t/60 for t in self.adaptive_phase_data['timestamps']]
        
        # Graph 1: Waiting Times
        ax1.plot(normal_time_min, self.normal_phase_data['waiting_times'], 
                'r-', label='Normal Mode', linewidth=2, alpha=0.8)
        ax1.plot(adaptive_time_min, self.adaptive_phase_data['waiting_times'], 
                'g-', label='Adaptive Mode', linewidth=2, alpha=0.8)
        ax1.set_title('Average Waiting Time Comparison')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Waiting Time (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.7, label='Mode Switch')
        
        # Graph 2: Traffic Pressure
        ax2.plot(normal_time_min, self.normal_phase_data['pressures_ns'], 
                'r-', label='Normal NS', alpha=0.7)
        ax2.plot(normal_time_min, self.normal_phase_data['pressures_ew'], 
                'r--', label='Normal EW', alpha=0.7)
        ax2.plot(adaptive_time_min, self.adaptive_phase_data['pressures_ns'], 
                'g-', label='Adaptive NS', alpha=0.7)
        ax2.plot(adaptive_time_min, self.adaptive_phase_data['pressures_ew'], 
                'g--', label='Adaptive EW', alpha=0.7)
        ax2.set_title('Traffic Pressure Comparison')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Traffic Pressure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.7)
        
        # Graph 3: Vehicle Count
        ax3.plot(normal_time_min, self.normal_phase_data['vehicle_counts'], 
                'r-', label='Normal Mode', linewidth=2, alpha=0.8)
        ax3.plot(adaptive_time_min, self.adaptive_phase_data['vehicle_counts'], 
                'g-', label='Adaptive Mode', linewidth=2, alpha=0.8)
        ax3.set_title('Vehicle Count in Network')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Number of Vehicles')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=5, color='gray', linestyle='--', alpha=0.7)
        
        # Graph 4: Average Speed
        ax4.plot(normal_time_min, self.normal_phase_data['avg_speeds'], 
                'r-', label='Normal Mode', linewidth=2, alpha=0.8)
        ax4.plot(adaptive_time_min, self.adaptive_phase_data['avg_speeds'], 
                'g-', label='Adaptive Mode', linewidth=2, alpha=0.8)
        ax4.set_title('Average Network Speed')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=5, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('traffic_comparison_graphs.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary bar chart
        self._create_summary_bar_chart()
    
    def _create_summary_bar_chart(self):
        """Create summary bar chart comparison"""
        if not (self.comparison_results['normal'] and self.comparison_results['adaptive']):
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        metrics = ['avg_waiting_time', 'avg_speed', 'avg_ns_pressure', 'avg_ew_pressure']
        metric_labels = ['Avg Waiting Time (s)', 'Avg Speed (m/s)', 'Avg NS Pressure', 'Avg EW Pressure']
        
        normal_values = [self.comparison_results['normal'][metric] for metric in metrics]
        adaptive_values = [self.comparison_results['adaptive'][metric] for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_values, width, label='Normal Mode', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, adaptive_values, width, label='Adaptive Mode', color='green', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=15)
        ax.legend()
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        plt.savefig('traffic_summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_comparative_report(self):
        """Generate comprehensive comparative report"""
        print(f"\n" + "=" * 100)
        print(f"🎯 COMPREHENSIVE COMPARATIVE ANALYSIS REPORT")
        print(f"=" * 100)
        
        if not (self.comparison_results['normal'] and self.comparison_results['adaptive']):
            print("❌ Insufficient data for comparison")
            return
        
        normal = self.comparison_results['normal']
        adaptive = self.comparison_results['adaptive']
        
        print(f"📊 PERFORMANCE METRICS COMPARISON:")
        print(f"{'Metric':<25} {'Normal Mode':<15} {'Adaptive Mode':<15} {'Improvement':<15}")
        print("-" * 80)
        
        # Waiting Time
        waiting_improvement = ((normal['avg_waiting_time'] - adaptive['avg_waiting_time']) / 
                              normal['avg_waiting_time'] * 100)
        print(f"{'Avg Waiting Time (s)':<25} {normal['avg_waiting_time']:<15.2f} "
              f"{adaptive['avg_waiting_time']:<15.2f} {waiting_improvement:<15.1f}%")
        
        # Speed
        speed_improvement = ((adaptive['avg_speed'] - normal['avg_speed']) / 
                            normal['avg_speed'] * 100)
        print(f"{'Avg Speed (m/s)':<25} {normal['avg_speed']:<15.2f} "
              f"{adaptive['avg_speed']:<15.2f} {speed_improvement:<15.1f}%")
        
        # Pressure balance
        normal_pressure_diff = abs(normal['avg_ns_pressure'] - normal['avg_ew_pressure'])
        adaptive_pressure_diff = abs(adaptive['avg_ns_pressure'] - adaptive['avg_ew_pressure'])
        pressure_balance_improvement = ((normal_pressure_diff - adaptive_pressure_diff) / 
                                       normal_pressure_diff * 100) if normal_pressure_diff > 0 else 0
        
        print(f"{'Pressure Imbalance':<25} {normal_pressure_diff:<15.2f} "
              f"{adaptive_pressure_diff:<15.2f} {pressure_balance_improvement:<15.1f}%")
        
        print(f"\n🎯 KEY FINDINGS:")
        
        # Determine overall performance
        if waiting_improvement > 5:
            print(f"   ✅ Adaptive control significantly reduced waiting times by {waiting_improvement:.1f}%")
        elif waiting_improvement > 0:
            print(f"   🟡 Adaptive control slightly reduced waiting times by {waiting_improvement:.1f}%")
        else:
            print(f"   ❌ Adaptive control increased waiting times by {abs(waiting_improvement):.1f}%")
        
        if speed_improvement > 3:
            print(f"   ✅ Adaptive control improved average speeds by {speed_improvement:.1f}%")
        elif speed_improvement > 0:
            print(f"   🟡 Adaptive control slightly improved speeds by {speed_improvement:.1f}%")
        else:
            print(f"   ❌ Adaptive control reduced speeds by {abs(speed_improvement):.1f}%")
        
        if pressure_balance_improvement > 10:
            print(f"   ✅ Adaptive control significantly improved traffic balance by {pressure_balance_improvement:.1f}%")
        elif pressure_balance_improvement > 0:
            print(f"   🟡 Adaptive control improved traffic balance by {pressure_balance_improvement:.1f}%")
        else:
            print(f"   ❌ Adaptive control worsened traffic balance")
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        if waiting_improvement > 10:
            print(f"   🎯 Adaptive control shows excellent results - implement permanently")
        elif waiting_improvement > 0:
            print(f"   🎯 Adaptive control shows promise - fine-tune parameters for better results")
        else:
            print(f"   🎯 Review adaptive algorithm - may need parameter adjustments")
        
        print(f"   📈 Graphs saved as: traffic_comparison_graphs.png, traffic_summary_comparison.png")
        print(f"   📊 Total data points analyzed: {normal['total_samples']} (normal) + {adaptive['total_samples']} (adaptive)")

def main():
    """Run the comparative simulation"""
    try:
        simulation = ComparativeTrafficSimulation()
        simulation.run_comparative_simulation()
    except Exception as e:
        print(f"❌ Error running simulation: {e}")

if __name__ == "__main__":
    main()