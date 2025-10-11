"""
Dynamic Simulation with Fixed Edge Traffic Controller
===================================================

Runs comprehensive dynamic traffic scenarios using the FIXED edge traffic controller
and generates detailed performance analysis and visualizations.

Features:
- 6 dynamic traffic phases (Light, Moderate, Heavy, Rush Hour, etc.)
- Comparative analysis: Normal Mode vs Fixed Adaptive Mode
- Real-time performance monitoring
- Comprehensive plotting and analysis
"""

import os
import sys
import json
import time
import subprocess
import traci
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fixed_edge_traffic_controller import EdgeTrafficController

class DynamicSimulationFixed:
    def __init__(self):
        self.base_directory = "fixed_dynamic_results"
        self.simulation_duration = 1800  # 30 minutes simulation
        self.phase_duration = 300  # 5 minutes per phase
        
        # Traffic phases definition
        self.traffic_phases = {
            'light': {'flow_rate': 200, 'description': 'Light traffic'},
            'moderate': {'flow_rate': 400, 'description': 'Moderate traffic'},
            'heavy': {'flow_rate': 600, 'description': 'Heavy traffic'},
            'rush_hour': {'flow_rate': 800, 'description': 'Rush hour peak'},
            'congested': {'flow_rate': 1000, 'description': 'Congested'},
            'recovery': {'flow_rate': 300, 'description': 'Recovery phase'}
        }
        
        # Results storage
        self.results = {
            'normal_mode': [],
            'adaptive_mode': []
        }
        
        # Performance tracking
        self.performance_data = defaultdict(list)
        self.timing_decisions = []
        
        os.makedirs(self.base_directory, exist_ok=True)
    
    def create_dynamic_route_file(self, phase_name, flow_rate):
        """Create route file for specific traffic phase"""
        
        route_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    
    <!-- Vehicle type definitions -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="50.0"/>
    <vType id="motorcycle" accel="3.0" decel="5.0" sigma="0.3" length="2.5" maxSpeed="60.0"/>
    
    <!-- Route definitions for 4-way intersection -->
    <route id="route_E0_to_E1" edges="E0 E1"/>
    <route id="route_E0_to_-E0" edges="E0 -E0"/>
    <route id="route_E0_to_-E1" edges="E0 -E1"/>
    
    <route id="route_-E0_to_E1" edges="-E0 E1"/>
    <route id="route_-E0_to_E0" edges="-E0 E0"/>
    <route id="route_-E0_to_-E1" edges="-E0 -E1"/>
    
    <route id="route_E1_to_E0" edges="E1 E0"/>
    <route id="route_E1_to_-E0" edges="E1 -E0"/>
    <route id="route_E1_to_-E1" edges="E1 -E1"/>
    
    <route id="route_-E1_to_E0" edges="-E1 E0"/>
    <route id="route_-E1_to_-E0" edges="-E1 -E0"/>
    <route id="route_-E1_to_E1" edges="-E1 E1"/>
    
    <!-- Dynamic traffic flows based on phase -->
'''
        
        # Calculate flows for each direction based on phase
        base_flow = flow_rate // 12  # Distribute across 12 routes
        variation = int(base_flow * 0.3)  # ±30% variation
        
        routes = [
            "route_E0_to_E1", "route_E0_to_-E0", "route_E0_to_-E1",
            "route_-E0_to_E1", "route_-E0_to_E0", "route_-E0_to_-E1",
            "route_E1_to_E0", "route_E1_to_-E0", "route_E1_to_-E1",
            "route_-E1_to_E0", "route_-E1_to_-E0", "route_-E1_to_E1"
        ]
        
        for i, route in enumerate(routes):
            # Add some randomness to flows
            route_flow = base_flow + random.randint(-variation, variation)
            route_flow = max(50, route_flow)  # Minimum 50 vehicles/hour
            
            # Mix of cars and motorcycles
            car_probability = 0.7
            motorcycle_probability = 0.3
            
            route_content += f'''
    <!-- {route} flows -->
    <flow id="cars_{route}" type="car" route="{route}" begin="0" end="{self.phase_duration}" 
          vehsPerHour="{int(route_flow * car_probability)}" departLane="best" departSpeed="max"/>
    <flow id="motorcycles_{route}" type="motorcycle" route="{route}" begin="0" end="{self.phase_duration}" 
          vehsPerHour="{int(route_flow * motorcycle_probability)}" departLane="best" departSpeed="max"/>
'''
        
        route_content += '''
</routes>'''
        
        route_filename = f"dynamic_{phase_name}.rou.xml"
        with open(route_filename, 'w') as f:
            f.write(route_content)
        
        return route_filename
    
    def create_dynamic_config(self, route_file, phase_name):
        """Create SUMO configuration for dynamic phase"""
        
        config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    
    <input>
        <net-file value="demo.net.xml"/>
        <route-files value="{route_file}"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="{self.phase_duration}"/>
        <step-length value="1"/>
    </time>
    
    <processing>
        <time-to-teleport value="300"/>
        <max-depart-delay value="300"/>
    </processing>
    
    <output>
        <summary-output value="{self.base_directory}/{phase_name}_summary.xml"/>
        <tripinfo-output value="{self.base_directory}/{phase_name}_tripinfo.xml"/>
    </output>
    
</configuration>'''
        
        config_filename = f"dynamic_{phase_name}.sumocfg"
        with open(config_filename, 'w') as f:
            f.write(config_content)
        
        return config_filename
    
    def run_phase_simulation(self, phase_name, mode='normal'):
        """Run simulation for a specific traffic phase"""
        
        print(f"🚦 Running {phase_name.upper()} phase in {mode.upper()} mode...")
        
        phase_config = self.traffic_phases[phase_name]
        
        # Create route and config files for this phase
        route_file = self.create_dynamic_route_file(phase_name, phase_config['flow_rate'])
        config_file = self.create_dynamic_config(route_file, f"{phase_name}_{mode}")
        
        # Start SUMO simulation
        sumo_cmd = ["sumo", "-c", config_file, "--start", "--quit-on-end", "--no-warnings"]
        traci.start(sumo_cmd)
        
        # Initialize controller for adaptive mode
        controller = None
        if mode == 'adaptive':
            controller = EdgeTrafficController("J4", base_green_time=30)
        
        # Data collection
        phase_results = []
        adaptations = 0
        total_wait_time = 0
        total_vehicles = 0
        
        try:
            for step in range(self.phase_duration):
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                
                # Collect performance data every 10 steps
                if step % 10 == 0:
                    vehicle_ids = traci.vehicle.getIDList()
                    step_wait_time = 0
                    
                    for veh_id in vehicle_ids:
                        wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                        step_wait_time += wait_time
                    
                    avg_wait = step_wait_time / max(len(vehicle_ids), 1)
                    
                    phase_results.append({
                        'time': current_time,
                        'step': step,
                        'phase': phase_name,
                        'mode': mode,
                        'vehicles': len(vehicle_ids),
                        'total_waiting_time': step_wait_time,
                        'avg_waiting_time': avg_wait,
                        'adaptations': adaptations,
                        'flow_rate': phase_config['flow_rate']
                    })
                    
                    total_wait_time += step_wait_time
                    total_vehicles += len(vehicle_ids)
                
                # Apply adaptive algorithm if enabled
                if mode == 'adaptive' and controller:
                    result = controller.apply_edge_algorithm(current_time)
                    if result:
                        adaptations += 1
                        self.timing_decisions.append({
                            'time': current_time,
                            'phase': phase_name,
                            'mode': mode,
                            'decision': result
                        })
            
            # Calculate phase summary
            avg_phase_wait = total_wait_time / max(total_vehicles, 1)
            
            phase_summary = {
                'phase': phase_name,
                'mode': mode,
                'flow_rate': phase_config['flow_rate'],
                'total_vehicles': total_vehicles,
                'avg_waiting_time': avg_phase_wait,
                'total_adaptations': adaptations,
                'duration': self.phase_duration,
                'results': phase_results
            }
            
            print(f"   ✅ {phase_name} {mode}: {total_vehicles} vehicles, {avg_phase_wait:.1f}s avg wait, {adaptations} adaptations")
            
            return phase_summary
            
        except Exception as e:
            print(f"   ❌ Error in {phase_name} {mode}: {e}")
            return None
        
        finally:
            traci.close()
            # Clean up temporary files
            try:
                os.remove(route_file)
                os.remove(config_file)
            except:
                pass
    
    def run_dynamic_simulation(self):
        """Run complete dynamic simulation with all phases"""
        
        print("🚦 DYNAMIC SIMULATION WITH FIXED EDGE TRAFFIC CONTROLLER")
        print("="*70)
        print("Running 6 traffic phases in both Normal and Adaptive modes...")
        
        all_results = {
            'normal_mode': [],
            'adaptive_mode': [],
            'timing_decisions': []
        }
        
        # Run each phase in both modes
        for phase_name in self.traffic_phases.keys():
            # Normal mode
            normal_result = self.run_phase_simulation(phase_name, 'normal')
            if normal_result:
                all_results['normal_mode'].append(normal_result)
            
            time.sleep(1)  # Brief pause between simulations
            
            # Adaptive mode with fixed algorithm
            adaptive_result = self.run_phase_simulation(phase_name, 'adaptive')
            if adaptive_result:
                all_results['adaptive_mode'].append(adaptive_result)
            
            time.sleep(1)  # Brief pause between phases
        
        all_results['timing_decisions'] = self.timing_decisions
        
        # Save comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"{self.base_directory}/dynamic_simulation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Dynamic simulation completed!")
        print(f"📁 Results saved to: {results_file}")
        
        return all_results
    
    def create_comprehensive_plots(self, results):
        """Create comprehensive performance analysis plots"""
        
        print("\n📊 Creating comprehensive performance plots...")
        
        # Extract data for plotting
        normal_phases = results['normal_mode']
        adaptive_phases = results['adaptive_mode']
        timing_decisions = results['timing_decisions']
        
        # Create multiple analysis plots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Phase-by-phase performance comparison
        ax1 = plt.subplot(3, 3, 1)
        phase_names = [p['phase'] for p in normal_phases]
        normal_waits = [p['avg_waiting_time'] for p in normal_phases]
        adaptive_waits = [p['avg_waiting_time'] for p in adaptive_phases]
        
        x = np.arange(len(phase_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, normal_waits, width, label='Normal Mode', color='green', alpha=0.7)
        bars2 = ax1.bar(x + width/2, adaptive_waits, width, label='Fixed Adaptive', color='blue', alpha=0.7)
        
        ax1.set_xlabel('Traffic Phase')
        ax1.set_ylabel('Average Waiting Time (s)')
        ax1.set_title('Phase-by-Phase Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(phase_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Vehicle throughput comparison
        ax2 = plt.subplot(3, 3, 2)
        normal_vehicles = [p['total_vehicles'] for p in normal_phases]
        adaptive_vehicles = [p['total_vehicles'] for p in adaptive_phases]
        
        bars1 = ax2.bar(x - width/2, normal_vehicles, width, label='Normal Mode', color='green', alpha=0.7)
        bars2 = ax2.bar(x + width/2, adaptive_vehicles, width, label='Fixed Adaptive', color='blue', alpha=0.7)
        
        ax2.set_xlabel('Traffic Phase')
        ax2.set_ylabel('Total Vehicles Processed')
        ax2.set_title('Vehicle Throughput Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(phase_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Adaptation frequency
        ax3 = plt.subplot(3, 3, 3)
        adaptations = [p['total_adaptations'] for p in adaptive_phases]
        flow_rates = [p['flow_rate'] for p in adaptive_phases]
        
        ax3.scatter(flow_rates, adaptations, color='red', s=100, alpha=0.7)
        for i, phase in enumerate(phase_names):
            ax3.annotate(phase, (flow_rates[i], adaptations[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Traffic Flow Rate (vehicles/hour)')
        ax3.set_ylabel('Total Adaptations')
        ax3.set_title('Algorithm Adaptations vs Traffic Intensity')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance ratio analysis
        ax4 = plt.subplot(3, 3, 4)
        performance_ratios = []
        for normal, adaptive in zip(normal_phases, adaptive_phases):
            if normal['avg_waiting_time'] > 0:
                ratio = adaptive['avg_waiting_time'] / normal['avg_waiting_time']
            else:
                ratio = 1.0
            performance_ratios.append(ratio)
        
        colors = ['green' if r <= 1.0 else 'orange' if r <= 1.5 else 'red' for r in performance_ratios]
        bars = ax4.bar(phase_names, performance_ratios, color=colors, alpha=0.7)
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Same Performance')
        ax4.set_xlabel('Traffic Phase')
        ax4.set_ylabel('Performance Ratio (Adaptive/Normal)')
        ax4.set_title('Algorithm Performance Ratio')
        ax4.set_xticklabels(phase_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars, performance_ratios):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Traffic flow vs performance
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(flow_rates, normal_waits, 'go-', label='Normal Mode', linewidth=2, markersize=8)
        ax5.plot(flow_rates, adaptive_waits, 'bo-', label='Fixed Adaptive', linewidth=2, markersize=8)
        ax5.set_xlabel('Traffic Flow Rate (vehicles/hour)')
        ax5.set_ylabel('Average Waiting Time (s)')
        ax5.set_title('Performance vs Traffic Intensity')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Timing decisions timeline
        ax6 = plt.subplot(3, 3, 6)
        if timing_decisions:
            decision_times = [d['time'] for d in timing_decisions]
            decision_phases = [d['phase'] for d in timing_decisions]
            
            phase_colors = {'light': 'lightblue', 'moderate': 'yellow', 'heavy': 'orange', 
                           'rush_hour': 'red', 'congested': 'darkred', 'recovery': 'lightgreen'}
            
            for i, (time, phase) in enumerate(zip(decision_times, decision_phases)):
                color = phase_colors.get(phase, 'gray')
                ax6.scatter(time, i, color=color, s=100, alpha=0.7)
            
            ax6.set_xlabel('Simulation Time (s)')
            ax6.set_ylabel('Decision Index')
            ax6.set_title('Algorithm Timing Decisions')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No Timing Decisions Made\n(Conservative Operation)', 
                    transform=ax6.transAxes, ha='center', va='center',
                    fontsize=12, weight='bold')
            ax6.set_title('Algorithm Timing Decisions')
        
        # 7. Summary statistics
        ax7 = plt.subplot(3, 3, 7)
        metrics = ['Avg Wait Time', 'Total Vehicles', 'Total Adaptations']
        normal_stats = [
            np.mean(normal_waits),
            np.sum(normal_vehicles),
            0  # Normal mode has no adaptations
        ]
        adaptive_stats = [
            np.mean(adaptive_waits),
            np.sum(adaptive_vehicles),
            np.sum(adaptations)
        ]
        
        x_metrics = np.arange(len(metrics))
        bars1 = ax7.bar(x_metrics - width/2, normal_stats, width, label='Normal Mode', color='green', alpha=0.7)
        bars2 = ax7.bar(x_metrics + width/2, adaptive_stats, width, label='Fixed Adaptive', color='blue', alpha=0.7)
        
        ax7.set_xlabel('Metrics')
        ax7.set_ylabel('Values')
        ax7.set_title('Overall Performance Summary')
        ax7.set_xticks(x_metrics)
        ax7.set_xticklabels(metrics)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Performance improvement analysis
        ax8 = plt.subplot(3, 3, 8)
        improvements = []
        for normal, adaptive in zip(normal_phases, adaptive_phases):
            if normal['avg_waiting_time'] > 0:
                improvement = ((normal['avg_waiting_time'] - adaptive['avg_waiting_time']) / 
                              normal['avg_waiting_time']) * 100
            else:
                improvement = 0
            improvements.append(improvement)
        
        colors = ['green' if imp >= 0 else 'red' for imp in improvements]
        bars = ax8.bar(phase_names, improvements, color=colors, alpha=0.7)
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        ax8.set_xlabel('Traffic Phase')
        ax8.set_ylabel('Performance Improvement (%)')
        ax8.set_title('Algorithm Performance Improvement')
        ax8.set_xticklabels(phase_names, rotation=45)
        ax8.grid(True, alpha=0.3)
        
        # 9. Overall verdict
        ax9 = plt.subplot(3, 3, 9)
        overall_normal_avg = np.mean(normal_waits)
        overall_adaptive_avg = np.mean(adaptive_waits)
        overall_ratio = overall_adaptive_avg / overall_normal_avg if overall_normal_avg > 0 else 1.0
        total_adaptations = np.sum(adaptations)
        
        # Create verdict visualization
        verdict_data = ['Normal Mode', 'Fixed Adaptive']
        verdict_values = [overall_normal_avg, overall_adaptive_avg]
        verdict_colors = ['green', 'blue']
        
        bars = ax9.bar(verdict_data, verdict_values, color=verdict_colors, alpha=0.7)
        ax9.set_ylabel('Average Waiting Time (s)')
        ax9.set_title(f'Overall Performance\nRatio: {overall_ratio:.2f}x | Adaptations: {total_adaptations}')
        
        # Add value labels
        for bar, value in zip(bars, verdict_values):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Add verdict text
        if overall_ratio <= 1.1:
            verdict = "✅ EXCELLENT"
            verdict_color = 'green'
        elif overall_ratio <= 1.3:
            verdict = "✅ GOOD"
            verdict_color = 'orange'
        elif overall_ratio <= 2.0:
            verdict = "⚠️ ACCEPTABLE"
            verdict_color = 'orange'
        else:
            verdict = "❌ NEEDS WORK"
            verdict_color = 'red'
        
        ax9.text(0.5, 0.8, verdict, transform=ax9.transAxes, ha='center', va='center',
                fontsize=14, weight='bold', color=verdict_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = f"{self.base_directory}/dynamic_simulation_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive plots saved: {plot_file}")
        
        return plot_file
    
    def generate_performance_report(self, results):
        """Generate detailed performance report"""
        
        normal_phases = results['normal_mode']
        adaptive_phases = results['adaptive_mode']
        timing_decisions = results['timing_decisions']
        
        # Calculate overall statistics
        overall_normal_avg = np.mean([p['avg_waiting_time'] for p in normal_phases])
        overall_adaptive_avg = np.mean([p['avg_waiting_time'] for p in adaptive_phases])
        overall_ratio = overall_adaptive_avg / overall_normal_avg if overall_normal_avg > 0 else 1.0
        total_adaptations = sum([p['total_adaptations'] for p in adaptive_phases])
        
        report = f"""
DYNAMIC SIMULATION PERFORMANCE REPORT
====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SIMULATION OVERVIEW:
------------------
Phases Tested: {len(self.traffic_phases)}
Total Duration: {len(self.traffic_phases) * self.phase_duration * 2} seconds
Algorithm: Fixed Edge Traffic Controller

OVERALL PERFORMANCE:
------------------
Normal Mode Average Wait: {overall_normal_avg:.1f} seconds
Fixed Adaptive Average Wait: {overall_adaptive_avg:.1f} seconds
Performance Ratio: {overall_ratio:.2f}x
Total Algorithm Adaptations: {total_adaptations}

PHASE-BY-PHASE ANALYSIS:
-----------------------
"""
        
        for normal, adaptive in zip(normal_phases, adaptive_phases):
            phase_ratio = adaptive['avg_waiting_time'] / normal['avg_waiting_time'] if normal['avg_waiting_time'] > 0 else 1.0
            improvement = ((normal['avg_waiting_time'] - adaptive['avg_waiting_time']) / normal['avg_waiting_time']) * 100 if normal['avg_waiting_time'] > 0 else 0
            
            report += f"""
{normal['phase'].upper()} PHASE (Flow: {normal['flow_rate']} veh/hr):
  Normal Mode: {normal['avg_waiting_time']:.1f}s wait, {normal['total_vehicles']} vehicles
  Fixed Adaptive: {adaptive['avg_waiting_time']:.1f}s wait, {adaptive['total_vehicles']} vehicles, {adaptive['total_adaptations']} adaptations
  Performance Ratio: {phase_ratio:.2f}x
  Improvement: {improvement:+.1f}%
"""
        
        report += f"""

ALGORITHM BEHAVIOR:
-----------------
Total Timing Decisions: {len(timing_decisions)}
Decision Frequency: {len(timing_decisions)/(len(self.traffic_phases) * self.phase_duration) * 60:.1f} per minute
Conservative Operation: {'YES' if total_adaptations < 10 else 'NO'}

VERDICT:
-------
"""
        
        if overall_ratio <= 1.1:
            report += "✅ EXCELLENT: Fixed algorithm performs same or better than normal mode"
        elif overall_ratio <= 1.3:
            report += "✅ GOOD: Fixed algorithm performs well with minor overhead"
        elif overall_ratio <= 2.0:
            report += "⚠️ ACCEPTABLE: Fixed algorithm performance is reasonable"
        else:
            report += "❌ NEEDS WORK: Fixed algorithm performance needs improvement"
        
        report += f"""

RECOMMENDATIONS:
--------------
"""
        
        if overall_ratio <= 1.2:
            report += "- Algorithm is ready for production deployment\n- Monitor performance in real-world conditions\n- Consider gradual parameter tuning for optimization"
        elif overall_ratio <= 2.0:
            report += "- Algorithm shows promise but needs parameter tuning\n- Test with different adaptation intervals\n- Monitor for specific traffic patterns that cause issues"
        else:
            report += "- Algorithm needs significant improvement\n- Review timing calculation logic\n- Consider more conservative parameters"
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"{self.base_directory}/performance_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Performance report saved: {report_file}")
        
        return report_file

def main():
    """Run the complete dynamic simulation with fixed algorithm"""
    
    print("🚦 DYNAMIC SIMULATION WITH FIXED EDGE TRAFFIC CONTROLLER")
    print("="*70)
    print("This simulation will test the FIXED algorithm across 6 dynamic traffic phases:")
    print("🔸 Light Traffic → Moderate → Heavy → Rush Hour → Congested → Recovery")
    print("🔸 Comparing Normal Mode vs Fixed Adaptive Mode")
    print("🔸 Generating comprehensive performance analysis")
    
    # Create simulation instance
    sim = DynamicSimulationFixed()
    
    # Run complete dynamic simulation
    results = sim.run_dynamic_simulation()
    
    if results:
        # Create comprehensive analysis plots
        sim.create_comprehensive_plots(results)
        
        # Generate performance report
        sim.generate_performance_report(results)
        
        print(f"\n🎉 DYNAMIC SIMULATION COMPLETED!")
        print("="*70)
        print(f"📁 Check '{sim.base_directory}' folder for:")
        print("   📊 Comprehensive performance plots")
        print("   📋 Detailed performance report")
        print("   📊 Phase-by-phase analysis")
        print("   📈 Algorithm behavior visualization")
        
        # Print quick summary
        normal_phases = results['normal_mode']
        adaptive_phases = results['adaptive_mode']
        
        overall_normal_avg = np.mean([p['avg_waiting_time'] for p in normal_phases])
        overall_adaptive_avg = np.mean([p['avg_waiting_time'] for p in adaptive_phases])
        overall_ratio = overall_adaptive_avg / overall_normal_avg if overall_normal_avg > 0 else 1.0
        total_adaptations = sum([p['total_adaptations'] for p in adaptive_phases])
        
        print(f"\n📊 QUICK SUMMARY:")
        print(f"   Normal Mode: {overall_normal_avg:.1f}s average wait")
        print(f"   Fixed Adaptive: {overall_adaptive_avg:.1f}s average wait")
        print(f"   Performance Ratio: {overall_ratio:.2f}x")
        print(f"   Total Adaptations: {total_adaptations}")
        
        if overall_ratio <= 1.2:
            print("   🎉 VERDICT: Algorithm performing excellently!")
        elif overall_ratio <= 2.0:
            print("   ✅ VERDICT: Algorithm performance acceptable")
        else:
            print("   ⚠️ VERDICT: Algorithm needs further optimization")
    
    else:
        print("❌ Dynamic simulation failed. Check error messages above.")

if __name__ == "__main__":
    main()