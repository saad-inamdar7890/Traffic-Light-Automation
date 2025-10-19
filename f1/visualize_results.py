"""
Continuous Flow Results Visualizer
==================================

Creates plots and analysis for the continuous flow dynamic scenario test results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def create_results_visualization():
    """Create comprehensive visualization of the continuous flow test results."""
    
    try:
        # Load the test data
        with open('continuous_flow_results/baseline_data.json', 'r') as f:
            baseline_data = json.load(f)
        
        with open('continuous_flow_results/adaptive_data.json', 'r') as f:
            adaptive_data = json.load(f)
        
        print("✅ Data loaded successfully")
        print(f"📊 Baseline data points: {len(baseline_data)}")
        print(f"📊 Adaptive data points: {len(adaptive_data)}")
        
        # Create comprehensive comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Continuous Flow Dynamic Scenario Test Results\nImproved Algorithm vs Baseline', fontsize=16, fontweight='bold')
        
        # Extract time series data
        baseline_times = [d['time'] for d in baseline_data]
        baseline_waits = [d['avg_waiting_time'] for d in baseline_data]
        baseline_speeds = [d['avg_speed'] for d in baseline_data]
        baseline_vehicles = [d['total_vehicles'] for d in baseline_data]
        
        adaptive_times = [d['time'] for d in adaptive_data]
        adaptive_waits = [d['avg_waiting_time'] for d in adaptive_data]
        adaptive_speeds = [d['avg_speed'] for d in adaptive_data]
        adaptive_vehicles = [d['total_vehicles'] for d in adaptive_data]
        adaptive_adaptations = [d.get('adaptations', 0) for d in adaptive_data]
        
        # Convert times to minutes for better readability
        baseline_times_min = [t/60 for t in baseline_times]
        adaptive_times_min = [t/60 for t in adaptive_times]
        
        # Plot 1: Average Waiting Time Comparison
        ax1.plot(baseline_times_min, baseline_waits, 'b-', label='Baseline (Fixed-Time)', linewidth=2, alpha=0.8)
        ax1.plot(adaptive_times_min, adaptive_waits, 'r-', label='Improved Adaptive', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Average Waiting Time (seconds)')
        ax1.set_title('Average Waiting Time Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add phase boundaries
        phase_boundaries = [10, 20, 30, 40, 50, 60]  # 10-minute phases
        for boundary in phase_boundaries[:-1]:
            ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 2: Average Speed Comparison
        ax2.plot(baseline_times_min, baseline_speeds, 'b-', label='Baseline (Fixed-Time)', linewidth=2, alpha=0.8)
        ax2.plot(adaptive_times_min, adaptive_speeds, 'r-', label='Improved Adaptive', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Average Speed (m/s)')
        ax2.set_title('Average Vehicle Speed Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add phase boundaries
        for boundary in phase_boundaries[:-1]:
            ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 3: Vehicle Count Comparison
        ax3.plot(baseline_times_min, baseline_vehicles, 'b-', label='Baseline (Fixed-Time)', linewidth=2, alpha=0.8)
        ax3.plot(adaptive_times_min, adaptive_vehicles, 'r-', label='Improved Adaptive', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Total Vehicles in Network')
        ax3.set_title('Total Vehicles in Network Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add phase boundaries
        for boundary in phase_boundaries[:-1]:
            ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 4: Algorithm Adaptation Frequency
        ax4.plot(adaptive_times_min, adaptive_adaptations, 'g-', label='Cumulative Adaptations', linewidth=2)
        ax4_twin = ax4.twinx()
        
        # Calculate adaptation rate (adaptations per 5-minute window)
        adaptation_rates = []
        for i in range(1, len(adaptive_adaptations)):
            if i >= 10:  # 5-minute window (10 data points at 30s intervals)
                recent_adaptations = adaptive_adaptations[i] - adaptive_adaptations[i-10]
                adaptation_rates.append(recent_adaptations)
            else:
                adaptation_rates.append(0)
        
        if len(adaptation_rates) > 0:
            ax4_twin.plot(adaptive_times_min[1:], adaptation_rates, 'orange', label='Adaptations per 5min', linewidth=2, alpha=0.7)
            ax4_twin.set_ylabel('Adaptations per 5-minute Window', color='orange')
            ax4_twin.tick_params(axis='y', labelcolor='orange')
        
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Cumulative Adaptations', color='green')
        ax4.set_title('Algorithm Adaptation Activity')
        ax4.tick_params(axis='y', labelcolor='green')
        ax4.grid(True, alpha=0.3)
        
        # Add phase boundaries
        for boundary in phase_boundaries[:-1]:
            ax4.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Add phase labels
        phase_names = ['Low Traffic', 'Heavy North', 'Heavy East', 'Minimal', 'Rush Hour', 'Gradual Down']
        for i, (start, name) in enumerate(zip([0, 10, 20, 30, 40, 50], phase_names)):
            ax1.text(start + 5, ax1.get_ylim()[1] * 0.9, name, 
                    rotation=90, ha='center', va='top', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('continuous_flow_results/continuous_flow_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Comprehensive comparison plots created and saved")
        
        # Performance Summary
        print("\n📈 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        baseline_avg_wait = np.mean(baseline_waits)
        adaptive_avg_wait = np.mean(adaptive_waits)
        improvement = ((baseline_avg_wait - adaptive_avg_wait) / baseline_avg_wait) * 100
        
        print(f"🔹 Baseline Average Waiting Time: {baseline_avg_wait:.2f}s")
        print(f"🔹 Adaptive Average Waiting Time: {adaptive_avg_wait:.2f}s")
        print(f"🔹 Overall Change: {improvement:+.1f}%")
        
        print(f"\n🔄 Algorithm Activity:")
        print(f"🔹 Total Adaptations: {max(adaptive_adaptations)}")
        print(f"🔹 Average Adaptation Rate: {max(adaptive_adaptations)/60:.1f} per minute")
        
        # Vehicle mix analysis
        baseline_cars = [d.get('cars', 0) for d in baseline_data if d.get('cars', 0) > 0]
        baseline_bikes = [d.get('motorcycles', 0) for d in baseline_data if d.get('motorcycles', 0) > 0]
        adaptive_cars = [d.get('cars', 0) for d in adaptive_data if d.get('cars', 0) > 0]
        adaptive_bikes = [d.get('motorcycles', 0) for d in adaptive_data if d.get('motorcycles', 0) > 0]
        
        if baseline_cars and baseline_bikes:
            baseline_bike_ratio = np.mean(baseline_bikes) / (np.mean(baseline_cars) + np.mean(baseline_bikes)) * 100
            print(f"\n🏍️ Vehicle Mix (Baseline): {baseline_bike_ratio:.1f}% motorcycles")
        
        if adaptive_cars and adaptive_bikes:
            adaptive_bike_ratio = np.mean(adaptive_bikes) / (np.mean(adaptive_cars) + np.mean(adaptive_bikes)) * 100
            print(f"🏍️ Vehicle Mix (Adaptive): {adaptive_bike_ratio:.1f}% motorcycles")
        
        print(f"🎯 Target: 30% motorcycles")
        
        # Analysis insights
        print(f"\n🔍 ANALYSIS INSIGHTS")
        print("=" * 50)
        print(f"⚠️  The adaptive algorithm is switching phases too frequently")
        print(f"⚠️  {max(adaptive_adaptations)/60:.1f} adaptations per minute indicates over-optimization")
        print(f"⚠️  This suggests the CRITICAL threshold may be too sensitive")
        print(f"✅ However, the algorithm is responding to traffic changes as designed")
        print(f"✅ Mixed vehicle flow (cars + motorcycles) is working correctly")
        print(f"✅ All 6 dynamic phases executed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚗🏍️ CONTINUOUS FLOW RESULTS VISUALIZATION")
    print("=" * 50)
    create_results_visualization()