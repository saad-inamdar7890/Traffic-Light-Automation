"""
Comprehensive TRACI-Based Test for Improved Adaptive Algorithm
============================================================

This script tests the improved adaptive algorithm against a baseline using TRACI and SUMO.
Proper SUMO integration with performance comparison and visualization.
"""

import os
import sys
import time
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traci

# Import our controllers
from traci_improved_controller import TraciImprovedController
from baseline_controller import BaselineFixedController

def create_test_routes(scenario_name, traffic_intensity=0.3, duration=600):
    """Create route files for different traffic scenarios."""
    
    route_file = f"routes_{scenario_name}.rou.xml"
    
    # Traffic patterns for different scenarios
    patterns = {
        'uniform_light': {'ns': 0.2, 'ew': 0.2},
        'uniform_medium': {'ns': 0.4, 'ew': 0.4},
        'uniform_heavy': {'ns': 0.6, 'ew': 0.6},
        'north_heavy': {'ns': 0.8, 'ew': 0.2},
        'east_heavy': {'ns': 0.2, 'ew': 0.8},
        'rush_hour': {'ns': 0.7, 'ew': 0.7}
    }
    
    pattern = patterns.get(scenario_name, {'ns': traffic_intensity, 'ew': traffic_intensity})
    
    with open(route_file, 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>
    
    <!-- Valid routes based on demo.rou.xml -->
    <route id="east_straight" edges="E0 E0.319"/>
    <route id="east_left" edges="E0 -E1.238"/>
    <route id="east_right" edges="E0 E1.200"/>
    
    <route id="north_straight" edges="E1 E1.200"/>
    <route id="north_left" edges="E1 E0.319"/>
    <route id="north_right" edges="E1 -E0.254"/>
    
    <route id="south_straight" edges="-E1 -E1.238"/>
    <route id="south_left" edges="-E1 E0.319"/>
    <route id="south_right" edges="-E1 -E0.254"/>
    
    <route id="west_straight" edges="-E0 -E0.254"/>
    <route id="west_left" edges="-E0 E1.200"/>
    <route id="west_right" edges="-E0 -E1.238"/>
    
    <!-- Traffic Flows - North-South Direction -->
    <flow id="north_flow" route="north_straight" begin="0" end="{}" probability="{}" type="car"/>
    <flow id="south_flow" route="south_straight" begin="0" end="{}" probability="{}" type="car"/>
    
    <!-- Traffic Flows - East-West Direction -->
    <flow id="east_flow" route="east_straight" begin="0" end="{}" probability="{}" type="car"/>
    <flow id="west_flow" route="west_straight" begin="0" end="{}" probability="{}" type="car"/>
    
    <!-- Some turning traffic for realism -->
    <flow id="north_left" route="north_left" begin="0" end="{}" probability="{}" type="car"/>
    <flow id="east_left" route="east_left" begin="0" end="{}" probability="{}" type="car"/>
    
</routes>'''.format(
            duration, pattern['ns'],
            duration, pattern['ns'],
            duration, pattern['ew'],
            duration, pattern['ew'],
            duration, pattern['ns'] * 0.3,
            duration, pattern['ew'] * 0.3
        ))
    
    return route_file

def create_sumo_config(scenario_name, controller_type, duration=600):
    """Create SUMO configuration file for the test."""
    
    route_file = f"routes_{scenario_name}.rou.xml"
    config_file = f"config_{scenario_name}_{controller_type}.sumocfg"
    tripinfo_file = f"tripinfo_{scenario_name}_{controller_type}.xml"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="demo.net.xml"/>
        <route-files value="{route_file}"/>
    </input>
    <output>
        <tripinfo-output value="{tripinfo_file}"/>
    </output>
    <time>
        <begin value="0"/>
        <end value="{duration}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <lateral-resolution value="0.8"/>
    </processing>
    <report>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>''')
    
    return config_file, tripinfo_file

def run_simulation_with_controller(scenario_name, controller_type, duration=600):
    """Run SUMO simulation with specified controller."""
    
    print(f"Running {controller_type} simulation for {scenario_name}")
    print("-" * 60)
    
    # Create route and config files
    route_file = create_test_routes(scenario_name, duration=duration)
    config_file, tripinfo_file = create_sumo_config(scenario_name, controller_type, duration)
    
    try:
        # Start SUMO
        sumo_cmd = ["sumo", "-c", config_file, "--start"]
        traci.start(sumo_cmd)
        
        # Initialize controller
        if controller_type == "improved":
            controller = TraciImprovedController()
        else:
            controller = BaselineFixedController()
        
        if not controller.initialize_traffic_lights():
            print("   ERROR: Failed to initialize traffic lights")
            return None
        
        # Run simulation
        step = 0
        print(f"   Running simulation for {duration} seconds...")
        
        while step < duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Collect traffic data
            traffic_data = controller.collect_traffic_data()
            
            # Update performance metrics
            controller.update_performance_metrics(traffic_data)
            
            # Check if phase should change
            if controller.should_change_phase(traffic_data):
                controller.change_phase()
            
            step += 1
            
            # Progress indicator
            if step % 100 == 0:
                progress = (step / duration) * 100
                print(f"   Progress: {progress:.1f}%")
        
        # Get final performance summary
        performance = controller.get_performance_summary()
        
        print(f"   SUCCESS: Simulation completed")
        print(f"   Average waiting time: {performance['avg_waiting_time']:.2f}s")
        print(f"   Total phase changes: {performance['total_phase_changes']}")
        if 'smart_adaptations' in performance:
            print(f"   Smart adaptations: {performance['smart_adaptations']}")
        
        # Close TRACI
        traci.close()
        
        # Parse additional results from tripinfo
        tripinfo_results = parse_tripinfo_file(tripinfo_file)
        if tripinfo_results:
            performance.update(tripinfo_results)
        
        # Cleanup files
        cleanup_files([route_file, config_file])
        
        return performance
        
    except Exception as e:
        print(f"   ERROR: Simulation error: {e}")
        try:
            traci.close()
        except:
            pass
        return None

def parse_tripinfo_file(tripinfo_file):
    """Parse SUMO tripinfo XML file for additional metrics."""
    try:
        if not os.path.exists(tripinfo_file):
            return None
            
        import xml.etree.ElementTree as ET
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        
        waiting_times = []
        travel_times = []
        
        for tripinfo in root.findall('tripinfo'):
            waiting_time = float(tripinfo.get('waitingTime', 0))
            duration = float(tripinfo.get('duration', 0))
            
            waiting_times.append(waiting_time)
            travel_times.append(duration)
        
        if waiting_times:
            return {
                'tripinfo_avg_waiting': np.mean(waiting_times),
                'tripinfo_max_waiting': np.max(waiting_times),
                'tripinfo_avg_travel': np.mean(travel_times),
                'completed_trips': len(waiting_times)
            }
        
        return None
        
    except Exception as e:
        print(f"   WARNING: Error parsing tripinfo: {e}")
        return None

def cleanup_files(files):
    """Clean up temporary files."""
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

def run_comprehensive_test():
    """Run comprehensive test comparing improved vs baseline controllers."""
    
    print("COMPREHENSIVE TRACI-BASED ALGORITHM TEST")
    print("=" * 70)
    print("Testing improved adaptive algorithm vs baseline fixed-time controller")
    print("=" * 70)
    
    # Test scenarios
    scenarios = [
        ('uniform_light', 'Light Uniform Traffic'),
        ('uniform_medium', 'Medium Uniform Traffic'),
        ('uniform_heavy', 'Heavy Uniform Traffic'),
        ('north_heavy', 'North-Heavy Traffic'),
        ('east_heavy', 'East-Heavy Traffic'),
        ('rush_hour', 'Rush Hour Traffic')
    ]
    
    results = {'baseline': [], 'improved': []}
    test_duration = 300  # 5 minutes per test
    
    for scenario_id, scenario_name in scenarios:
        print(f"\nTESTING SCENARIO: {scenario_name}")
        print("=" * 50)
        
        # Test baseline controller
        baseline_result = run_simulation_with_controller(scenario_id, "baseline", test_duration)
        if baseline_result:
            baseline_result['scenario'] = scenario_name
            results['baseline'].append(baseline_result)
        
        # Small pause between tests
        time.sleep(2)
        
        # Test improved controller
        improved_result = run_simulation_with_controller(scenario_id, "improved", test_duration)
        if improved_result:
            improved_result['scenario'] = scenario_name
            results['improved'].append(improved_result)
        
        # Calculate and display improvement
        if baseline_result and improved_result:
            baseline_wait = baseline_result.get('tripinfo_avg_waiting', baseline_result['avg_waiting_time'])
            improved_wait = improved_result.get('tripinfo_avg_waiting', improved_result['avg_waiting_time'])
            
            improvement = ((baseline_wait - improved_wait) / baseline_wait) * 100
            
            print(f"\nSCENARIO RESULTS:")
            print(f"   Baseline:  {baseline_wait:.2f}s avg waiting")
            print(f"   Improved:  {improved_wait:.2f}s avg waiting")
            print(f"   Improvement: {improvement:+.1f}%")
            
            if improvement > 0:
                print(f"   SUCCESS: Improved algorithm is {improvement:.1f}% better!")
            else:
                print(f"   ISSUE: Improved algorithm is {abs(improvement):.1f}% worse")
        
        time.sleep(2)  # Pause between scenarios
    
    # Generate comprehensive report
    generate_comprehensive_report(results)
    create_comprehensive_visualization(results)
    
    return results

def generate_comprehensive_report(results):
    """Generate detailed test report."""
    
    baseline_results = results['baseline']
    improved_results = results['improved']
    
    if not baseline_results or not improved_results:
        print("ERROR: Insufficient results for report generation")
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""COMPREHENSIVE TRACI-BASED ALGORITHM TEST REPORT
======================================================================

Test Date: {timestamp}
Test Framework: SUMO + TRACI
Baseline: Fixed-time controller (45s phases)
Improved: Dynamic adaptive controller with enhanced logic

DETAILED SCENARIO RESULTS:
--------------------------------------------------
"""
    
    improvements = []
    
    for i in range(min(len(baseline_results), len(improved_results))):
        baseline = baseline_results[i]
        improved = improved_results[i]
        
        baseline_wait = baseline.get('tripinfo_avg_waiting', baseline['avg_waiting_time'])
        improved_wait = improved.get('tripinfo_avg_waiting', improved['avg_waiting_time'])
        improvement = ((baseline_wait - improved_wait) / baseline_wait) * 100
        improvements.append(improvement)
        
        report += f"""
{baseline['scenario']}:
  Baseline Controller:
    Avg Waiting Time: {baseline_wait:.2f}s
    Phase Changes: {baseline['total_phase_changes']}
    Completed Trips: {baseline.get('completed_trips', 'N/A')}
    
  Improved Controller:
    Avg Waiting Time: {improved_wait:.2f}s
    Phase Changes: {improved['total_phase_changes']}
    Smart Adaptations: {improved.get('smart_adaptations', 'N/A')}
    Completed Trips: {improved.get('completed_trips', 'N/A')}
    
  Performance: {improvement:+.1f}% {"improvement" if improvement > 0 else "degradation"}
"""
    
    # Overall statistics
    if improvements:
        avg_improvement = np.mean(improvements)
        best_improvement = max(improvements)
        worst_result = min(improvements)
        positive_scenarios = sum(1 for imp in improvements if imp > 0)
        
        report += f"""
OVERALL PERFORMANCE SUMMARY:
--------------------------------------------------
Average Improvement: {avg_improvement:+.1f}%
Best Improvement: {best_improvement:+.1f}%
Worst Result: {worst_result:+.1f}%
Successful Scenarios: {positive_scenarios}/{len(improvements)} ({positive_scenarios/len(improvements)*100:.1f}%)

ALGORITHM ASSESSMENT:
--------------------------------------------------
"""
        
        if avg_improvement > 30:
            assessment = "EXCELLENT: Algorithm shows outstanding performance!"
        elif avg_improvement > 15:
            assessment = "VERY GOOD: Algorithm shows strong improvements"
        elif avg_improvement > 5:
            assessment = "GOOD: Algorithm shows meaningful improvements"
        elif avg_improvement > 0:
            assessment = "MARGINAL: Algorithm shows minor improvements"
        else:
            assessment = "NEEDS WORK: Algorithm requires optimization"
        
        report += assessment
        
        report += f"""

TECHNICAL INSIGHTS:
--------------------------------------------------
TRACI Integration: Successfully tested with real SUMO simulation
Dynamic Adaptation: Algorithm adapts intervals based on traffic urgency
Smart Detection: Enhanced traffic pattern recognition and response
Performance Tracking: Comprehensive metrics collection and analysis

RECOMMENDATIONS:
--------------------------------------------------
"""
        
        if avg_improvement > 15:
            report += "• Deploy improved algorithm - shows significant benefits\n"
            report += "• Consider fine-tuning thresholds for even better performance\n"
        elif avg_improvement > 5:
            report += "• Algorithm shows promise - recommend further optimization\n"
            report += "• Test with longer simulation periods for validation\n"
        else:
            report += "• Algorithm needs refinement before deployment\n"
            report += "• Review traffic detection and adaptation logic\n"
    
    # Save report
    report_file = f"traci_algorithm_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nFINAL TEST SUMMARY")
    print("=" * 50)
    if improvements:
        print(f"Average Performance Improvement: {np.mean(improvements):+.1f}%")
        print(f"Successful Scenarios: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")
    print(f"Detailed report saved: {report_file}")

def create_comprehensive_visualization(results):
    """Create comprehensive visualization of test results."""
    
    baseline_results = results['baseline']
    improved_results = results['improved']
    
    if not baseline_results or not improved_results:
        return
    
    scenarios = [r['scenario'] for r in baseline_results]
    baseline_times = [r.get('tripinfo_avg_waiting', r['avg_waiting_time']) for r in baseline_results]
    improved_times = [r.get('tripinfo_avg_waiting', r['avg_waiting_time']) for r in improved_results]
    improvements = [((b - i) / b) * 100 for b, i in zip(baseline_times, improved_times)]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Waiting time comparison
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_times, width, label='Baseline Fixed-Time', alpha=0.8, color='blue')
    ax1.bar(x + width/2, improved_times, width, label='Improved Adaptive', alpha=0.8, color='green')
    
    ax1.set_xlabel('Traffic Scenarios')
    ax1.set_ylabel('Average Waiting Time (seconds)')
    ax1.set_title('TRACI Test: Waiting Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement percentages
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.bar(scenarios, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Traffic Scenarios')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Improvement by Scenario')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, v in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{v:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Phase changes comparison
    baseline_changes = [r['total_phase_changes'] for r in baseline_results]
    improved_changes = [r['total_phase_changes'] for r in improved_results]
    
    ax3.bar(x - width/2, baseline_changes, width, label='Baseline', alpha=0.8, color='blue')
    ax3.bar(x + width/2, improved_changes, width, label='Improved', alpha=0.8, color='green')
    
    ax3.set_xlabel('Traffic Scenarios')
    ax3.set_ylabel('Total Phase Changes')
    ax3.set_title('Phase Change Frequency')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Smart adaptations (improved only)
    adaptations = [r.get('smart_adaptations', 0) for r in improved_results]
    ax4.bar(scenarios, adaptations, color='orange', alpha=0.7)
    ax4.set_xlabel('Traffic Scenarios')
    ax4.set_ylabel('Smart Adaptations')
    ax4.set_title('Intelligent Adaptations Made')
    ax4.grid(True, alpha=0.3)
    
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    plot_file = f"traci_algorithm_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {plot_file}")

if __name__ == "__main__":
    print("TRACI-Based Improved Adaptive Algorithm Test")
    print("This test uses real SUMO simulation with TRACI integration")
    print("=" * 70)
    
    # Check SUMO installation
    try:
        result = subprocess.run(["sumo", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("SUCCESS: SUMO is properly installed and accessible")
        else:
            print("ERROR: SUMO not found. Please ensure SUMO is installed and in PATH")
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: SUMO not found. Please install SUMO and ensure it's in PATH")
        sys.exit(1)
    
    # Check required files
    required_files = ["demo.net.xml"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        print("Please ensure demo.net.xml is in the current directory")
        sys.exit(1)
    
    print("SUCCESS: All requirements satisfied")
    
    # Run the comprehensive test
    start_time = time.time()
    try:
        results = run_comprehensive_test()
        end_time = time.time()
        
        print(f"\nTEST COMPLETED in {end_time - start_time:.1f} seconds")
        print("TRACI-based algorithm testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTEST INTERRUPTED by user")
        try:
            traci.close()
        except:
            pass
    except Exception as e:
        print(f"\nTEST FAILED with error: {e}")
        try:
            traci.close()
        except:
            pass
    
    print("=" * 70)