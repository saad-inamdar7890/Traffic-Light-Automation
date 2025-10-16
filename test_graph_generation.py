"""
Test script to validate graph generation with sample data
"""

import matplotlib.pyplot as plt
import numpy as np
import statistics

# Sample data structure to test graph generation
sample_data = {
    'heavy_one_direction': {
        'normal': {
            'waiting_times': [0, 15, 25, 30, 32, 28, 26, 24, 22, 20],
            'timestamps': [0, 60, 120, 180, 240, 300, 360, 420, 480, 540],
            'pressures_ns': [10, 150, 300, 400, 350, 320, 280, 250, 220, 200],
            'pressures_ew': [10, 80, 120, 140, 130, 125, 120, 115, 110, 105],
            'vehicle_counts': [5, 50, 100, 150, 140, 130, 120, 110, 100, 95],
            'avg_speeds': [15, 12, 8, 6, 7, 8, 9, 10, 11, 12]
        },
        'adaptive': {
            'waiting_times': [0, 12, 20, 24, 26, 22, 18, 16, 14, 12],
            'timestamps': [0, 60, 120, 180, 240, 300, 360, 420, 480, 540],
            'pressures_ns': [10, 120, 250, 300, 280, 260, 240, 220, 200, 180],
            'pressures_ew': [10, 90, 130, 120, 110, 105, 100, 95, 90, 85],
            'vehicle_counts': [5, 45, 90, 130, 125, 120, 115, 110, 105, 100],
            'avg_speeds': [15, 13, 10, 8, 9, 10, 11, 12, 13, 14]
        }
    },
    'light_three_lanes': {
        'normal': {
            'waiting_times': [0, 8, 15, 18, 20, 17, 15, 13, 11, 9],
            'timestamps': [0, 60, 120, 180, 240, 300, 360, 420, 480, 540],
            'pressures_ns': [5, 60, 120, 150, 140, 130, 120, 110, 100, 90],
            'pressures_ew': [5, 30, 50, 60, 55, 50, 45, 40, 35, 30],
            'vehicle_counts': [3, 25, 50, 75, 70, 65, 60, 55, 50, 45],
            'avg_speeds': [18, 15, 12, 10, 11, 12, 13, 14, 15, 16]
        },
        'adaptive': {
            'waiting_times': [0, 6, 12, 14, 15, 12, 10, 8, 6, 5],
            'timestamps': [0, 60, 120, 180, 240, 300, 360, 420, 480, 540],
            'pressures_ns': [5, 50, 100, 120, 110, 100, 90, 80, 70, 60],
            'pressures_ew': [5, 35, 55, 50, 45, 40, 35, 30, 25, 20],
            'vehicle_counts': [3, 22, 45, 65, 60, 55, 50, 45, 40, 35],
            'avg_speeds': [18, 16, 14, 12, 13, 14, 15, 16, 17, 18]
        }
    }
}

def test_graph_generation():
    """Test the graph generation functionality"""
    print("🧪 TESTING GRAPH GENERATION")
    
    # Test 1: Summary comparison chart
    print("📊 Creating summary comparison chart...")
    
    scenarios = list(sample_data.keys())
    scenario_names = [s.replace('_', ' ').title() for s in scenarios]
    
    # Calculate improvements
    improvements = []
    for scenario in scenarios:
        normal_avg = statistics.mean(sample_data[scenario]['normal']['waiting_times'][2:])  # Skip warmup
        adaptive_avg = statistics.mean(sample_data[scenario]['adaptive']['waiting_times'][2:])
        improvement = ((normal_avg - adaptive_avg) / normal_avg * 100) if normal_avg > 0 else 0
        improvements.append(improvement)
    
    # Create summary chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    bars = ax.bar(x, improvements, color=['skyblue', 'lightgreen'], alpha=0.8)
    
    ax.set_xlabel('Test Scenarios')
    ax.set_ylabel('Waiting Time Improvement (%)')
    ax.set_title('Test: Adaptive Control Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3 if height >= 0 else -15),
                  textcoords="offset points",
                  ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('test_summary_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test 2: Individual scenario plots
    print("📈 Creating individual scenario plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, (scenario, short_name) in enumerate(zip(scenarios, ['Heavy 1-Dir', 'Light 3-Lane'])):
        if i < len(axes) and scenario in sample_data:
            ax = axes[i]
            
            # Convert to minutes
            normal_time = [t/60 for t in sample_data[scenario]['normal']['timestamps']]
            adaptive_time = [t/60 for t in sample_data[scenario]['adaptive']['timestamps']]
            
            ax.plot(normal_time, sample_data[scenario]['normal']['waiting_times'], 'r-', 
                   label='Normal Mode', linewidth=2, alpha=0.7)
            ax.plot(adaptive_time, sample_data[scenario]['adaptive']['waiting_times'], 'g-', 
                   label='Adaptive Mode', linewidth=2, alpha=0.7)
            
            ax.set_title(f'{short_name}\nWaiting Time Comparison')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Waiting Time (s)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(scenarios), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Test: Individual Scenario Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_individual_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Graph generation test completed successfully!")
    print("📁 Generated files: test_summary_chart.png, test_individual_charts.png")

if __name__ == "__main__":
    test_graph_generation()