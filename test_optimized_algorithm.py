#!/usr/bin/env python3
"""
Test the optimized adaptive algorithm against the normal mode.
Uses the improved adaptive controller with dynamic intervals and enhanced traffic detection.
"""

import os
import sys
import time
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_test_with_improved_algorithm():
    """Run a comprehensive test using the improved adaptive algorithm."""
    
    print("🚀 TESTING IMPROVED ADAPTIVE ALGORITHM")
    print("=" * 70)
    
    # Test configuration
    test_duration = 1800  # 30 minutes
    phases = [
        {"name": "Low Traffic", "duration": 600, "rate": 0.1},
        {"name": "Medium Traffic", "duration": 600, "rate": 0.3}, 
        {"name": "High Traffic", "duration": 600, "rate": 0.6}
    ]
    
    results = {}
    
    for phase in phases:
        print(f"\n📊 Testing Phase: {phase['name']}")
        print(f"   Duration: {phase['duration']}s, Rate: {phase['rate']}")
        
        # Test Normal Mode
        print("   Testing Normal Mode...")
        normal_result = run_simulation_mode("normal", phase['duration'], phase['rate'])
        
        # Test Improved Adaptive Mode
        print("   Testing Improved Adaptive Mode...")
        adaptive_result = run_simulation_mode("improved_adaptive", phase['duration'], phase['rate'])
        
        # Calculate improvement
        if normal_result and adaptive_result:
            improvement = ((normal_result['avg_waiting'] - adaptive_result['avg_waiting']) / normal_result['avg_waiting']) * 100
            
            results[phase['name']] = {
                'normal': normal_result,
                'adaptive': adaptive_result,
                'improvement': improvement
            }
            
            print(f"   Normal: {normal_result['avg_waiting']:.2f}s")
            print(f"   Improved Adaptive: {adaptive_result['avg_waiting']:.2f}s")
            print(f"   Improvement: {improvement:+.1f}%")
        else:
            print("   ❌ Test failed")
    
    # Generate summary report
    generate_summary_report(results)
    create_comparison_plot(results)
    
    return results

def run_simulation_mode(mode, duration, traffic_rate):
    """Run a single simulation with specified mode and parameters."""
    
    try:
        # Create temporary config files
        config_file = create_test_config(mode, duration, traffic_rate)
        
        # Run SUMO simulation
        cmd = ["sumo", "-c", config_file, "--no-warnings", "--no-step-log"]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 60)
        
        if process.returncode != 0:
            print(f"   ⚠️ SUMO error in {mode} mode")
            return None
        
        # Parse results from tripinfo file
        tripinfo_file = f"tripinfo_{mode}.xml"
        if os.path.exists(tripinfo_file):
            result = parse_tripinfo(tripinfo_file)
            # Clean up
            os.remove(tripinfo_file)
            os.remove(config_file)
            return result
        else:
            print(f"   ⚠️ No tripinfo file generated for {mode} mode")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"   ⚠️ Simulation timeout for {mode} mode")
        return None
    except Exception as e:
        print(f"   ⚠️ Error in {mode} mode: {e}")
        return None

def create_test_config(mode, duration, traffic_rate):
    """Create SUMO configuration file for the test."""
    
    # Create simple route file
    route_file = f"test_routes_{mode}.rou.xml"
    with open(route_file, 'w') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>
    
    <route id="north_south" edges="N4 J4 S4"/>
    <route id="south_north" edges="S4 J4 N4"/>
    <route id="east_west" edges="E4 J4 W4"/>
    <route id="west_east" edges="W4 J4 E4"/>
    
    <flow id="ns_flow" route="north_south" begin="0" end="{}" probability="{}"/>
    <flow id="sn_flow" route="south_north" begin="0" end="{}" probability="{}"/>
    <flow id="ew_flow" route="east_west" begin="0" end="{}" probability="{}"/>
    <flow id="we_flow" route="west_east" begin="0" end="{}" probability="{}"/>
</routes>'''.format(duration, traffic_rate, duration, traffic_rate, duration, traffic_rate, duration, traffic_rate))
    
    # Create config file
    config_file = f"test_config_{mode}.sumocfg"
    tripinfo_file = f"tripinfo_{mode}.xml"
    
    with open(config_file, 'w') as f:
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
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>''')
    
    return config_file

def parse_tripinfo(tripinfo_file):
    """Parse SUMO tripinfo XML file to extract waiting times."""
    
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        
        waiting_times = []
        for tripinfo in root.findall('tripinfo'):
            waiting_time = float(tripinfo.get('waitingTime', 0))
            waiting_times.append(waiting_time)
        
        if waiting_times:
            return {
                'avg_waiting': np.mean(waiting_times),
                'max_waiting': np.max(waiting_times),
                'total_vehicles': len(waiting_times)
            }
        else:
            return {'avg_waiting': 0, 'max_waiting': 0, 'total_vehicles': 0}
            
    except Exception as e:
        print(f"   ⚠️ Error parsing tripinfo: {e}")
        return None

def generate_summary_report(results):
    """Generate a summary report of the test results."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""IMPROVED ADAPTIVE ALGORITHM TEST REPORT
======================================================================

Test Date: {timestamp}
Algorithm: Optimized Adaptive Controller with Dynamic Intervals

PHASE-BY-PHASE RESULTS:
--------------------------------------------------
"""
    
    total_improvement = 0
    phase_count = 0
    
    for phase_name, data in results.items():
        normal = data['normal']
        adaptive = data['adaptive']
        improvement = data['improvement']
        
        report += f"""
{phase_name}:
  Normal Mode:     {normal['avg_waiting']:.2f}s average waiting
  Improved Mode:   {adaptive['avg_waiting']:.2f}s average waiting
  Improvement:     {improvement:+.1f}%
  Vehicles:        {adaptive['total_vehicles']} total
"""
        total_improvement += improvement
        phase_count += 1
    
    avg_improvement = total_improvement / phase_count if phase_count > 0 else 0
    
    report += f"""
OVERALL PERFORMANCE SUMMARY:
--------------------------------------------------
Average Improvement: {avg_improvement:+.1f}%
"""
    
    if avg_improvement > 30:
        report += "✅ EXCELLENT: Algorithm shows significant improvement!"
    elif avg_improvement > 10:
        report += "✅ GOOD: Algorithm shows meaningful improvement"
    elif avg_improvement > 0:
        report += "⚠️ MARGINAL: Algorithm shows minor improvement"
    else:
        report += "❌ POOR: Algorithm needs further optimization"
    
    # Save report
    with open("improved_algorithm_test_report.txt", 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 70)
    print("📋 FINAL RESULTS")
    print("=" * 70)
    print(f"Average Improvement: {avg_improvement:+.1f}%")
    print("Report saved: improved_algorithm_test_report.txt")

def create_comparison_plot(results):
    """Create a comparison plot of the results."""
    
    phases = list(results.keys())
    normal_times = [results[phase]['normal']['avg_waiting'] for phase in phases]
    adaptive_times = [results[phase]['adaptive']['avg_waiting'] for phase in phases]
    improvements = [results[phase]['improvement'] for phase in phases]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Waiting times comparison
    x = np.arange(len(phases))
    width = 0.35
    
    ax1.bar(x - width/2, normal_times, width, label='Normal Mode', alpha=0.8, color='blue')
    ax1.bar(x + width/2, adaptive_times, width, label='Improved Adaptive', alpha=0.8, color='green')
    
    ax1.set_xlabel('Traffic Phases')
    ax1.set_ylabel('Average Waiting Time (seconds)')
    ax1.set_title('Waiting Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement percentages
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(phases, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Traffic Phases')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Improvement')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(improvements):
        ax2.text(i, v + (1 if v > 0 else -5), f'{v:+.1f}%', 
                ha='center', va='bottom' if v > 0 else 'top')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('improved_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved: improved_algorithm_comparison.png")

if __name__ == "__main__":
    print("🔬 IMPROVED ADAPTIVE ALGORITHM TESTING")
    print("This test compares our optimized algorithm against normal mode")
    print("=" * 70)
    
    # Check if required files exist
    required_files = ["demo.net.xml"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure you're running this from the project root directory")
        sys.exit(1)
    
    # Run the test
    start_time = time.time()
    results = run_test_with_improved_algorithm()
    end_time = time.time()
    
    print(f"\n⏱️ Test completed in {end_time - start_time:.1f} seconds")
    print("=" * 70)