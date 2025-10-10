"""
Quick Comparison Results Generator
Based on the comparative simulation data collected
"""

import matplotlib.pyplot as plt
import numpy as np

# Data collected from the simulation run
normal_phase_data = {
    'waiting_times': [0.0, 1.1, 7.8, 14.4, 14.9, 14.8, 16.8, 20.2, 26.6, 29.0],
    'pressures_ns': [3.0, 39.3, 77.1, 186.2, 140.7, 257.2, 265.3, 341.5, 464.4, 357.9],
    'pressures_ew': [3.0, 27.9, 93.0, 111.0, 216.0, 157.2, 237.4, 281.6, 346.9, 523.7],
    'vehicle_counts': [4, 44, 78, 101, 118, 134, 149, 168, 184, 191],
    'timestamps': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
}

adaptive_phase_data = {
    'waiting_times': [29.5, 27.1, 28.8, 30.2, 33.6, 28.5, 32.5, 29.8, 31.8, 28.8],
    'pressures_ns': [534.1, 339.2, 534.5, 391.4, 461.7, 415.7, 413.0, 486.6, 353.8, 528.1],
    'pressures_ew': [352.1, 586.6, 444.9, 607.3, 617.1, 612.0, 649.7, 567.8, 691.5, 497.7],
    'vehicle_counts': [200, 207, 222, 223, 220, 222, 222, 222, 220, 221],
    'timestamps': [300, 330, 360, 390, 420, 450, 480, 510, 540, 570]
}

def create_comparison_graphs():
    """Create comparison graphs from collected data"""
    print("📊 CREATING COMPARISON GRAPHS FROM SIMULATION DATA")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Traffic Light Performance Comparison: Normal vs Adaptive Control\n(10-minute simulation with Cars + Motorcycles)', fontsize=14)
    
    # Convert timestamps to minutes
    normal_time_min = [t/60 for t in normal_phase_data['timestamps']]
    adaptive_time_min = [t/60 for t in adaptive_phase_data['timestamps']]
    
    # Graph 1: Waiting Times
    ax1.plot(normal_time_min, normal_phase_data['waiting_times'], 
            'r-o', label='Normal Mode', linewidth=2, markersize=6)
    ax1.plot(adaptive_time_min, adaptive_phase_data['waiting_times'], 
            'g-s', label='Adaptive Mode', linewidth=2, markersize=6)
    ax1.set_title('Average Waiting Time Comparison')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Waiting Time (seconds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.7, label='Mode Switch')
    
    # Graph 2: Traffic Pressure
    ax2.plot(normal_time_min, normal_phase_data['pressures_ns'], 
            'r-', label='Normal NS', alpha=0.8, linewidth=2)
    ax2.plot(normal_time_min, normal_phase_data['pressures_ew'], 
            'r--', label='Normal EW', alpha=0.8, linewidth=2)
    ax2.plot(adaptive_time_min, adaptive_phase_data['pressures_ns'], 
            'g-', label='Adaptive NS', alpha=0.8, linewidth=2)
    ax2.plot(adaptive_time_min, adaptive_phase_data['pressures_ew'], 
            'g--', label='Adaptive EW', alpha=0.8, linewidth=2)
    ax2.set_title('Traffic Pressure Comparison')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Traffic Pressure')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.7)
    
    # Graph 3: Vehicle Count
    ax3.plot(normal_time_min, normal_phase_data['vehicle_counts'], 
            'r-o', label='Normal Mode', linewidth=2, markersize=6)
    ax3.plot(adaptive_time_min, adaptive_phase_data['vehicle_counts'], 
            'g-s', label='Adaptive Mode', linewidth=2, markersize=6)
    ax3.set_title('Vehicle Count in Network')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Number of Vehicles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=5, color='gray', linestyle='--', alpha=0.7)
    
    # Graph 4: Pressure Balance (difference between NS and EW)
    normal_pressure_diff = [abs(ns - ew) for ns, ew in zip(normal_phase_data['pressures_ns'], normal_phase_data['pressures_ew'])]
    adaptive_pressure_diff = [abs(ns - ew) for ns, ew in zip(adaptive_phase_data['pressures_ns'], adaptive_phase_data['pressures_ew'])]
    
    ax4.plot(normal_time_min, normal_pressure_diff, 
            'r-o', label='Normal Mode', linewidth=2, markersize=6)
    ax4.plot(adaptive_time_min, adaptive_pressure_diff, 
            'g-s', label='Adaptive Mode', linewidth=2, markersize=6)
    ax4.set_title('Traffic Pressure Imbalance (|NS - EW|)')
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Pressure Difference')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=5, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('traffic_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary bar chart
    create_summary_bar_chart()

def create_summary_bar_chart():
    """Create summary comparison bar chart"""
    print("📊 CREATING SUMMARY BAR CHART")
    
    # Calculate averages
    normal_avg_waiting = np.mean(normal_phase_data['waiting_times'])
    adaptive_avg_waiting = np.mean(adaptive_phase_data['waiting_times'])
    
    normal_avg_vehicles = np.mean(normal_phase_data['vehicle_counts'])
    adaptive_avg_vehicles = np.mean(adaptive_phase_data['vehicle_counts'])
    
    normal_avg_ns_pressure = np.mean(normal_phase_data['pressures_ns'])
    adaptive_avg_ns_pressure = np.mean(adaptive_phase_data['pressures_ns'])
    
    normal_avg_ew_pressure = np.mean(normal_phase_data['pressures_ew'])
    adaptive_avg_ew_pressure = np.mean(adaptive_phase_data['pressures_ew'])
    
    # Calculate pressure imbalance
    normal_pressure_imbalance = abs(normal_avg_ns_pressure - normal_avg_ew_pressure)
    adaptive_pressure_imbalance = abs(adaptive_avg_ns_pressure - adaptive_avg_ew_pressure)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    metrics = ['Avg Waiting\nTime (s)', 'Avg Vehicle\nCount', 'Avg NS\nPressure', 'Avg EW\nPressure', 'Pressure\nImbalance']
    normal_values = [normal_avg_waiting, normal_avg_vehicles, normal_avg_ns_pressure, normal_avg_ew_pressure, normal_pressure_imbalance]
    adaptive_values = [adaptive_avg_waiting, adaptive_avg_vehicles, adaptive_avg_ns_pressure, adaptive_avg_ew_pressure, adaptive_pressure_imbalance]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, normal_values, width, label='Normal Mode', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, adaptive_values, width, label='Adaptive Mode', color='green', alpha=0.7)
    
    ax.set_xlabel('Performance Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Traffic Management Performance Comparison\n(Cars + Motorcycles, 10-minute simulation)')
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
    plt.savefig('traffic_summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_analysis():
    """Generate detailed analysis report"""
    print("\n" + "=" * 80)
    print("🎯 DETAILED COMPARATIVE ANALYSIS REPORT")
    print("=" * 80)
    
    # Calculate key metrics
    normal_avg_waiting = np.mean(normal_phase_data['waiting_times'])
    adaptive_avg_waiting = np.mean(adaptive_phase_data['waiting_times'])
    
    normal_avg_vehicles = np.mean(normal_phase_data['vehicle_counts'])
    adaptive_avg_vehicles = np.mean(adaptive_phase_data['vehicle_counts'])
    
    normal_avg_ns_pressure = np.mean(normal_phase_data['pressures_ns'])
    adaptive_avg_ns_pressure = np.mean(adaptive_phase_data['pressures_ns'])
    
    normal_avg_ew_pressure = np.mean(normal_phase_data['pressures_ew'])
    adaptive_avg_ew_pressure = np.mean(adaptive_phase_data['pressures_ew'])
    
    # Calculate improvements
    waiting_change = ((adaptive_avg_waiting - normal_avg_waiting) / normal_avg_waiting * 100)
    vehicle_change = ((adaptive_avg_vehicles - normal_avg_vehicles) / normal_avg_vehicles * 100)
    
    # Pressure balance analysis
    normal_pressure_imbalance = abs(normal_avg_ns_pressure - normal_avg_ew_pressure)
    adaptive_pressure_imbalance = abs(adaptive_avg_ns_pressure - adaptive_avg_ew_pressure)
    balance_improvement = ((normal_pressure_imbalance - adaptive_pressure_imbalance) / normal_pressure_imbalance * 100)
    
    print(f"📊 PERFORMANCE METRICS COMPARISON:")
    print(f"{'Metric':<25} {'Normal Mode':<15} {'Adaptive Mode':<15} {'Change':<15}")
    print("-" * 80)
    print(f"{'Avg Waiting Time (s)':<25} {normal_avg_waiting:<15.1f} {adaptive_avg_waiting:<15.1f} {waiting_change:<14.1f}%")
    print(f"{'Avg Vehicle Count':<25} {normal_avg_vehicles:<15.1f} {adaptive_avg_vehicles:<15.1f} {vehicle_change:<14.1f}%")
    print(f"{'NS Traffic Pressure':<25} {normal_avg_ns_pressure:<15.1f} {adaptive_avg_ns_pressure:<15.1f}")
    print(f"{'EW Traffic Pressure':<25} {normal_avg_ew_pressure:<15.1f} {adaptive_avg_ew_pressure:<15.1f}")
    print(f"{'Pressure Imbalance':<25} {normal_pressure_imbalance:<15.1f} {adaptive_pressure_imbalance:<15.1f} {balance_improvement:<14.1f}%")
    
    print(f"\n🔍 KEY OBSERVATIONS:")
    
    # Waiting time analysis
    if waiting_change > 0:
        print(f"   ❌ Adaptive mode increased waiting times by {waiting_change:.1f}%")
        print(f"      This suggests the adaptive algorithm needs optimization")
    else:
        print(f"   ✅ Adaptive mode reduced waiting times by {abs(waiting_change):.1f}%")
    
    # Vehicle count analysis
    if vehicle_change > 0:
        print(f"   📈 Network handled {vehicle_change:.1f}% more vehicles in adaptive mode")
        print(f"      Higher vehicle count with stable waiting times indicates better throughput")
    else:
        print(f"   📉 Vehicle count decreased by {abs(vehicle_change):.1f}% in adaptive mode")
    
    # Pressure analysis
    if balance_improvement > 0:
        print(f"   ⚖️  Traffic balance improved by {balance_improvement:.1f}%")
        print(f"      Adaptive control better distributed traffic between NS and EW directions")
    else:
        print(f"   ⚠️  Traffic balance worsened by {abs(balance_improvement):.1f}%")
    
    print(f"\n🎯 TRAFFIC MIX ANALYSIS:")
    print(f"   🚗 Car flows: Reduced to 60% of original rates")
    print(f"   🏍️  Motorcycle flows: Added as 40% of traffic mix")
    print(f"   📊 Total flow: Cars (~630 veh/h) + Motorcycles (~602 veh/h)")
    print(f"   🌟 Diverse traffic provides more realistic simulation conditions")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    if waiting_change > 10:
        print(f"   🔧 Adaptive algorithm needs tuning - consider:")
        print(f"      • Reducing minimum green time thresholds")
        print(f"      • Adjusting pressure calculation weights")
        print(f"      • Fine-tuning phase transition timing")
    elif abs(waiting_change) < 5:
        print(f"   ✅ Performance is similar - adaptive system ready for deployment")
        print(f"      • Consider longer test periods for more data")
        print(f"      • Test during different time-of-day scenarios")
    
    if balance_improvement > 10:
        print(f"   🎯 Adaptive control excels at traffic balancing")
        print(f"      • Focus on optimizing this strength")
        print(f"      • Consider pressure-based optimization as primary metric")
    
    print(f"\n📈 SIMULATION QUALITY:")
    print(f"   ✅ Successfully completed 10-minute comparative test")
    print(f"   ✅ Mixed vehicle types (cars + motorcycles) for realism")
    print(f"   ✅ Real-time data collection every 30 seconds")
    print(f"   ✅ Clear phase separation (normal vs adaptive)")
    print(f"   📊 Graphs saved: traffic_comparison_results.png, traffic_summary_comparison.png")

if __name__ == "__main__":
    print("🚦 TRAFFIC LIGHT COMPARISON ANALYSIS")
    print("="*50)
    
    create_comparison_graphs()
    generate_detailed_analysis()
    
    print(f"\n🎉 Analysis Complete!")
    print(f"Check the generated PNG files for detailed visual comparisons.")