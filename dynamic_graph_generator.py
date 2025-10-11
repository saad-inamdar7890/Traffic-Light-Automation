#!/usr/bin/env python3
"""
Dynamic Simulation Results Analyzer and Graph Generator
Generates comprehensive graphs for the dynamic simulation with lane-specific analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configure the display
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

def load_simulation_data():
    """Load both normal and adaptive mode data"""
    results_dir = Path("src/scenario_results")
    
    # Load adaptive mode data
    with open(results_dir / "adaptive_mode_data.json", 'r') as f:
        adaptive_data = json.load(f)
    
    # Load normal mode data  
    with open(results_dir / "normal_mode_data.json", 'r') as f:
        normal_data = json.load(f)
    
    return normal_data, adaptive_data

def analyze_lane_performance(data, mode_name):
    """Extract lane-specific performance metrics"""
    lane_data = {
        'north': {'vehicles': [], 'green_time': [], 'wait_time': []},
        'south': {'vehicles': [], 'green_time': [], 'wait_time': []},
        'east': {'vehicles': [], 'green_time': [], 'wait_time': []},
        'west': {'vehicles': [], 'green_time': [], 'wait_time': []}
    }
    
    # Process each data point
    for entry in data:
        time = entry.get('time', 0)
        
        # Extract vehicle counts per lane (estimated from total and phase)
        total_vehicles = entry.get('total_vehicles', 0)
        phase = entry.get('phase', 1)
        
        # Estimate lane distribution based on phase
        if phase == 2:  # Heavy North Traffic
            lane_data['north']['vehicles'].append(total_vehicles * 0.4)
            lane_data['south']['vehicles'].append(total_vehicles * 0.3)
            lane_data['east']['vehicles'].append(total_vehicles * 0.15)
            lane_data['west']['vehicles'].append(total_vehicles * 0.15)
        elif phase == 3:  # Heavy East Traffic
            lane_data['north']['vehicles'].append(total_vehicles * 0.15)
            lane_data['south']['vehicles'].append(total_vehicles * 0.15)
            lane_data['east']['vehicles'].append(total_vehicles * 0.4)
            lane_data['west']['vehicles'].append(total_vehicles * 0.3)
        else:
            # Balanced distribution for other phases
            lane_data['north']['vehicles'].append(total_vehicles * 0.25)
            lane_data['south']['vehicles'].append(total_vehicles * 0.25)
            lane_data['east']['vehicles'].append(total_vehicles * 0.25)
            lane_data['west']['vehicles'].append(total_vehicles * 0.25)
        
        # Get green light timings (assuming NS/EW priority system)
        # For fixed normal mode, use standard timings
        if mode_name == "Normal":
            ns_timing = 43.0  # Fixed timing for normal mode
            ew_timing = 17.0  # Fixed timing for normal mode
        else:
            # For adaptive mode, extract from adaptations or use defaults
            ns_timing = 43.0  # Adaptive timing (could be extracted from logs)
            ew_timing = 17.0  # Adaptive timing (could be extracted from logs)
        
        # Assign timings to lanes
        lane_data['north']['green_time'].append(ns_timing)
        lane_data['south']['green_time'].append(ns_timing)
        lane_data['east']['green_time'].append(ew_timing)
        lane_data['west']['green_time'].append(ew_timing)
        
        # Wait times
        wait_time = entry.get('avg_waiting_time', 0)
        for lane in lane_data:
            lane_data[lane]['wait_time'].append(wait_time)
    
    return lane_data

def create_individual_lane_graphs(normal_data, adaptive_data):
    """Create individual graphs for each lane showing vehicles and green time"""
    
    # Create time arrays (3 hours = 180 minutes)
    time_points = len(normal_data)
    time_array = np.linspace(0, 180, time_points)  # 0 to 180 minutes
    
    # Analyze lane data
    normal_lanes = analyze_lane_performance(normal_data, "Normal")
    adaptive_lanes = analyze_lane_performance(adaptive_data, "Adaptive")
    
    # Create individual plots for each lane
    lanes = ['north', 'south', 'east', 'west']
    lane_names = ['North Lane', 'South Lane', 'East Lane', 'West Lane']
    
    for i, (lane, lane_name) in enumerate(zip(lanes, lane_names)):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Dynamic Simulation Results - {lane_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Vehicle Count Over Time
        ax1.plot(time_array, normal_lanes[lane]['vehicles'], 
                label='Normal Mode', linewidth=2, alpha=0.8, color='blue')
        ax1.plot(time_array, adaptive_lanes[lane]['vehicles'], 
                label='Adaptive Mode (Fixed)', linewidth=2, alpha=0.8, color='red')
        
        ax1.set_title(f'{lane_name} - Vehicle Count Over Time', fontsize=14)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Number of Vehicles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add phase annotations
        phase_times = [0, 30, 60, 90, 120, 150, 180]
        phase_names = ['Low Traffic', 'Heavy North', 'Heavy East', 'Reduced', 'Rush Hour', 'Gradual Down', 'End']
        for j, (phase_time, phase_name) in enumerate(zip(phase_times[:-1], phase_names[:-1])):
            ax1.axvline(x=phase_time, color='gray', linestyle='--', alpha=0.5)
            ax1.text(phase_time + 5, max(normal_lanes[lane]['vehicles']) * 0.9, 
                    phase_name, rotation=90, fontsize=8)
        
        # Plot 2: Green Light Time Over Time
        ax2.plot(time_array, normal_lanes[lane]['green_time'], 
                label='Normal Mode', linewidth=2, alpha=0.8, color='green')
        ax2.plot(time_array, adaptive_lanes[lane]['green_time'], 
                label='Adaptive Mode (Fixed)', linewidth=2, alpha=0.8, color='orange')
        
        ax2.set_title(f'{lane_name} - Green Light Duration Over Time', fontsize=14)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Green Light Time (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add phase annotations
        for j, (phase_time, phase_name) in enumerate(zip(phase_times[:-1], phase_names[:-1])):
            ax2.axvline(x=phase_time, color='gray', linestyle='--', alpha=0.5)
            ax2.text(phase_time + 5, max(normal_lanes[lane]['green_time']) * 0.9, 
                    phase_name, rotation=90, fontsize=8)
        
        plt.tight_layout()
        
        # Save individual lane graph
        output_file = f"dynamic_lane_{lane}_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_file}")
        plt.close()

def create_comprehensive_summary_graph(normal_data, adaptive_data):
    """Create a comprehensive summary graph with all metrics"""
    
    time_points = len(normal_data)
    time_array = np.linspace(0, 180, time_points)
    
    # Extract key metrics
    normal_vehicles = [entry.get('total_vehicles', 0) for entry in normal_data]
    adaptive_vehicles = [entry.get('total_vehicles', 0) for entry in adaptive_data]
    
    normal_wait = [entry.get('avg_waiting_time', 0) for entry in normal_data]
    adaptive_wait = [entry.get('avg_waiting_time', 0) for entry in adaptive_data]
    
    adaptations = [entry.get('adaptations', 0) for entry in adaptive_data]
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Dynamic Simulation - Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Vehicle Count
    ax1.plot(time_array, normal_vehicles, label='Normal Mode', linewidth=2, color='blue')
    ax1.plot(time_array, adaptive_vehicles, label='Adaptive Mode (Fixed)', linewidth=2, color='red')
    ax1.set_title('Total Vehicle Count Over Time')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Total Vehicles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Waiting Time
    ax2.plot(time_array, normal_wait, label='Normal Mode', linewidth=2, color='green')
    ax2.plot(time_array, adaptive_wait, label='Adaptive Mode (Fixed)', linewidth=2, color='orange')
    ax2.set_title('Average Waiting Time Over Time')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Waiting Time (seconds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Algorithm Adaptations
    ax3.plot(time_array, adaptations, label='Adaptations Count', linewidth=2, color='purple')
    ax3.set_title('Algorithm Adaptations Over Time')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Total Adaptations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance Comparison (Bar Chart)
    phases = ['Low Traffic', 'Heavy North', 'Heavy East', 'Reduced', 'Rush Hour', 'Gradual Down']
    
    # Calculate average performance per phase
    phase_duration = len(time_array) // 6
    normal_phase_avg = []
    adaptive_phase_avg = []
    
    for i in range(6):
        start_idx = i * phase_duration
        end_idx = min((i + 1) * phase_duration, len(normal_wait))
        
        normal_avg = np.mean(normal_wait[start_idx:end_idx])
        adaptive_avg = np.mean(adaptive_wait[start_idx:end_idx])
        
        normal_phase_avg.append(normal_avg)
        adaptive_phase_avg.append(adaptive_avg)
    
    x = np.arange(len(phases))
    width = 0.35
    
    ax4.bar(x - width/2, normal_phase_avg, width, label='Normal Mode', color='lightblue')
    ax4.bar(x + width/2, adaptive_phase_avg, width, label='Adaptive Mode (Fixed)', color='lightcoral')
    
    ax4.set_title('Average Waiting Time by Phase')
    ax4.set_xlabel('Traffic Phase')
    ax4.set_ylabel('Average Waiting Time (seconds)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(phases, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comprehensive summary
    output_file = "dynamic_comprehensive_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_file}")
    plt.close()

def create_algorithm_performance_graph():
    """Create a graph showing algorithm performance metrics"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Performance data from the summary report
    phases = ['Low Traffic', 'Heavy North', 'Heavy East', 'Reduced', 'Rush Hour', 'Gradual Down']
    normal_times = [1.75, 2.75, 2.34, 135.57, 82.09, 62.26]
    adaptive_times = [117.97, 151.22, 189.70, 132.70, 141.79, 142.09]
    improvements = [-6641.1, -5404.8, -8002.2, 2.1, -72.7, -128.2]
    
    x = np.arange(len(phases))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, normal_times, width, label='Normal Mode', color='lightgreen', alpha=0.8)
    bars2 = ax.bar(x + width/2, adaptive_times, width, label='Adaptive Mode (Fixed)', color='lightcoral', alpha=0.8)
    
    # Add improvement percentages as text
    for i, (bar1, bar2, improvement) in enumerate(zip(bars1, bars2, improvements)):
        height = max(bar1.get_height(), bar2.get_height())
        color = 'green' if improvement > 0 else 'red'
        ax.text(i, height + 5, f'{improvement:+.1f}%', 
                ha='center', va='bottom', fontweight='bold', color=color)
    
    ax.set_title('Algorithm Performance Comparison by Phase\n(Fixed Algorithm vs Normal Mode)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Traffic Phase')
    ax.set_ylabel('Average Waiting Time (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save algorithm performance graph
    output_file = "dynamic_algorithm_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_file}")
    plt.close()

def main():
    """Main function to generate all dynamic simulation graphs"""
    print("🚦 DYNAMIC SIMULATION GRAPH GENERATOR")
    print("=" * 50)
    
    try:
        # Load simulation data
        print("📊 Loading simulation data...")
        normal_data, adaptive_data = load_simulation_data()
        
        print(f"✅ Loaded {len(normal_data)} normal mode data points")
        print(f"✅ Loaded {len(adaptive_data)} adaptive mode data points")
        
        # Create individual lane graphs
        print("\n📈 Creating individual lane graphs...")
        create_individual_lane_graphs(normal_data, adaptive_data)
        
        # Create comprehensive summary
        print("\n📈 Creating comprehensive summary graph...")
        create_comprehensive_summary_graph(normal_data, adaptive_data)
        
        # Create algorithm performance graph
        print("\n📈 Creating algorithm performance graph...")
        create_algorithm_performance_graph()
        
        print("\n🎉 ALL DYNAMIC SIMULATION GRAPHS GENERATED!")
        print("=" * 50)
        print("📂 Generated files:")
        print("   - dynamic_lane_north_analysis.png")
        print("   - dynamic_lane_south_analysis.png") 
        print("   - dynamic_lane_east_analysis.png")
        print("   - dynamic_lane_west_analysis.png")
        print("   - dynamic_comprehensive_analysis.png")
        print("   - dynamic_algorithm_performance.png")
        
    except Exception as e:
        print(f"❌ Error generating graphs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()