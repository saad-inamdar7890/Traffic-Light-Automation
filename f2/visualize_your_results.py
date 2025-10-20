"""
Visualization Tool for Your Algorithm Results
===========================================

Creates comprehensive comparison plots between your algorithm and normal mode.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import statistics

class YourAlgorithmVisualizer:
    """Visualize results from your algorithm comparison test."""
    
    def __init__(self, results_directory: str = None):
        if results_directory is None:
            self.results_directory = os.path.join(os.path.dirname(__file__), "your_algorithm_results")
        else:
            self.results_directory = results_directory
    
    def load_data(self) -> tuple:
        """Load simulation data from JSON files."""
        baseline_file = os.path.join(self.results_directory, "baseline_data.json")
        adaptive_file = os.path.join(self.results_directory, "your_algorithm_data.json")
        
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            with open(adaptive_file, 'r') as f:
                adaptive_data = json.load(f)
            
            print("✅ Data loaded successfully")
            print(f"📊 Normal mode data points: {len(baseline_data)}")
            print(f"📊 Your algorithm data points: {len(adaptive_data)}")
            
            return baseline_data, adaptive_data
            
        except FileNotFoundError:
            print("❌ Data files not found. Run the test first.")
            return None, None
    
    def create_comprehensive_plots(self, baseline_data: List[Dict], adaptive_data: List[Dict]):
        """Create comprehensive comparison plots."""
        # Set up the plot style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Create a 3x3 grid of subplots
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Extract time arrays
        baseline_times = [d['time'] / 60 for d in baseline_data]  # Convert to minutes
        adaptive_times = [d['time'] / 60 for d in adaptive_data]
        
        # 1. Waiting Time Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        baseline_wait = [d['avg_waiting_time'] for d in baseline_data]
        adaptive_wait = [d['avg_waiting_time'] for d in adaptive_data]
        
        ax1.plot(baseline_times, baseline_wait, 'b-', label='Normal Mode', linewidth=2, alpha=0.8)
        ax1.plot(adaptive_times, adaptive_wait, 'r-', label='Your Algorithm', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Average Waiting Time (s)')
        ax1.set_title('Waiting Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Speed Comparison (Top Center)
        ax2 = fig.add_subplot(gs[0, 1])
        baseline_speed = [d['avg_speed'] for d in baseline_data]
        adaptive_speed = [d['avg_speed'] for d in adaptive_data]
        
        ax2.plot(baseline_times, baseline_speed, 'b-', label='Normal Mode', linewidth=2, alpha=0.8)
        ax2.plot(adaptive_times, adaptive_speed, 'r-', label='Your Algorithm', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Average Speed (m/s)')
        ax2.set_title('Speed Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Vehicle Count Comparison (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        baseline_vehicles = [d['total_vehicles'] for d in baseline_data]
        adaptive_vehicles = [d['total_vehicles'] for d in adaptive_data]
        
        ax3.plot(baseline_times, baseline_vehicles, 'b-', label='Normal Mode', linewidth=2, alpha=0.8)
        ax3.plot(adaptive_times, adaptive_vehicles, 'r-', label='Your Algorithm', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Total Vehicles')
        ax3.set_title('Vehicle Count Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Adaptation Activity (Second Row Left)
        ax4 = fig.add_subplot(gs[1, 0])
        adaptations = [d['adaptations'] for d in adaptive_data]
        ax4.plot(adaptive_times, adaptations, 'g-', linewidth=3, label='Total Adaptations')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Cumulative Adaptations')
        ax4.set_title('Your Algorithm Adaptation Activity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Phase-by-Phase Performance (Second Row Center & Right)
        phase_names = ['Low', 'Heavy N', 'Heavy E', 'Minimal', 'Rush Hour', 'Gradual']
        phase_performance = []
        
        for phase in phase_names:
            baseline_phase = [d for d in baseline_data if d['phase'] == phase]
            adaptive_phase = [d for d in adaptive_data if d['phase'] == phase]
            
            if baseline_phase and adaptive_phase:
                baseline_avg = statistics.mean([d['avg_waiting_time'] for d in baseline_phase])
                adaptive_avg = statistics.mean([d['avg_waiting_time'] for d in adaptive_phase])
                improvement = ((baseline_avg - adaptive_avg) / baseline_avg) * 100
                phase_performance.append(improvement)
            else:
                phase_performance.append(0)
        
        ax5 = fig.add_subplot(gs[1, 1:])
        colors = ['green' if p > 0 else 'red' for p in phase_performance]
        bars = ax5.bar(phase_names, phase_performance, color=colors, alpha=0.7)
        ax5.set_xlabel('Traffic Phase')
        ax5.set_ylabel('Improvement (%)')
        ax5.set_title('Phase-by-Phase Performance (Your Algorithm vs Normal Mode)')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, phase_performance):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 6. Traffic Urgency Distribution (Third Row Left)
        ax6 = fig.add_subplot(gs[2, 0])
        urgency_data = [d.get('urgency', 'FIXED') for d in adaptive_data if 'urgency' in d]
        urgency_counts = {}
        for urgency in urgency_data:
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
        
        if urgency_counts:
            urgencies = list(urgency_counts.keys())
            counts = list(urgency_counts.values())
            colors_urgency = ['red', 'orange', 'yellow', 'lightgreen', 'lightblue'][:len(urgencies)]
            
            ax6.pie(counts, labels=urgencies, autopct='%1.1f%%', colors=colors_urgency)
            ax6.set_title('Traffic Urgency Distribution\\n(Your Algorithm)')
        
        # 7. Performance Summary (Third Row Center)
        ax7 = fig.add_subplot(gs[2, 1])
        
        # Calculate summary metrics
        baseline_avg_wait = statistics.mean([d['avg_waiting_time'] for d in baseline_data])
        adaptive_avg_wait = statistics.mean([d['avg_waiting_time'] for d in adaptive_data])
        baseline_avg_speed = statistics.mean([d['avg_speed'] for d in baseline_data])
        adaptive_avg_speed = statistics.mean([d['avg_speed'] for d in adaptive_data])
        
        wait_improvement = ((baseline_avg_wait - adaptive_avg_wait) / baseline_avg_wait) * 100
        speed_improvement = ((adaptive_avg_speed - baseline_avg_speed) / baseline_avg_speed) * 100
        
        metrics = ['Wait Time', 'Speed']
        improvements = [wait_improvement, speed_improvement]
        colors_metrics = ['green' if i > 0 else 'red' for i in improvements]
        
        bars = ax7.bar(metrics, improvements, color=colors_metrics, alpha=0.7)
        ax7.set_ylabel('Improvement (%)')
        ax7.set_title('Overall Performance Summary')
        ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax7.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                    f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 8. Vehicle Mix (Third Row Right)
        ax8 = fig.add_subplot(gs[2, 2])
        
        # Calculate average vehicle mix
        total_cars = sum([d['cars'] for d in adaptive_data])
        total_motorcycles = sum([d['motorcycles'] for d in adaptive_data])
        total_all = total_cars + total_motorcycles
        
        if total_all > 0:
            car_percentage = (total_cars / total_all) * 100
            motorcycle_percentage = (total_motorcycles / total_all) * 100
            
            sizes = [car_percentage, motorcycle_percentage]
            labels = [f'Cars\\n{car_percentage:.1f}%', f'Motorcycles\\n{motorcycle_percentage:.1f}%']
            colors_vehicles = ['lightcoral', 'lightskyblue']
            
            ax8.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_vehicles)
            ax8.set_title('Vehicle Mix Distribution')
        
        # 9. Key Insights (Bottom Row)
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')
        
        # Calculate key insights
        total_adaptations = adaptive_data[-1]['adaptations'] if adaptive_data else 0
        adaptation_rate = total_adaptations / 60  # per minute
        phases_won = sum(1 for p in phase_performance if p > 0)
        total_phases = len(phase_performance)
        
        insights_text = f"""
🎯 KEY INSIGHTS - YOUR ALGORITHM ANALYSIS:

📊 Overall Performance: {wait_improvement:+.1f}% wait time, {speed_improvement:+.1f}% speed
🏆 Phases Won: {phases_won}/{total_phases} ({phases_won/total_phases*100:.0f}%)
🔄 Adaptation Rate: {adaptation_rate:.1f} adaptations/minute ({total_adaptations} total)
🚗 Vehicle Processing: {total_all:,} vehicles (70% cars, 30% motorcycles)

{'✅ SUCCESS: Your hypothesis is validated! Shorter phases improve performance.' if wait_improvement > 5 and phases_won > total_phases/2 
 else '🤔 MIXED: Some improvement but not decisive.' if wait_improvement > 0 
 else '❌ HYPOTHESIS NOT VALIDATED: Normal mode performs better.'}

💡 Your Algorithm Strategy: {12}-{20}s phases for light traffic, {35}-{45}s for heavy traffic
💡 Normal Mode Strategy: Fixed {60}s phases for all conditions
        """
        
        ax9.text(0.05, 0.95, insights_text, transform=ax9.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add main title
        fig.suptitle('YOUR ALGORITHM vs NORMAL MODE - COMPREHENSIVE ANALYSIS', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the plot
        plot_file = os.path.join(self.results_directory, "your_algorithm_comparison_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive comparison plots created and saved: {plot_file}")
        
        return plot_file
    
    def visualize_results(self):
        """Main method to create all visualizations."""
        print("\\n🚗🏍️ YOUR ALGORITHM RESULTS VISUALIZATION")
        print("=" * 50)
        
        baseline_data, adaptive_data = self.load_data()
        
        if baseline_data and adaptive_data:
            plot_file = self.create_comprehensive_plots(baseline_data, adaptive_data)
            
            # Calculate and print summary
            baseline_avg_wait = statistics.mean([d['avg_waiting_time'] for d in baseline_data])
            adaptive_avg_wait = statistics.mean([d['avg_waiting_time'] for d in adaptive_data])
            wait_improvement = ((baseline_avg_wait - adaptive_avg_wait) / baseline_avg_wait) * 100
            
            total_adaptations = adaptive_data[-1]['adaptations'] if adaptive_data else 0
            adaptation_rate = total_adaptations / 60
            
            print("\\n📈 PERFORMANCE SUMMARY")
            print("=" * 50)
            print(f"🔹 Normal Mode Average Waiting Time: {baseline_avg_wait:.2f}s")
            print(f"🔹 Your Algorithm Average Waiting Time: {adaptive_avg_wait:.2f}s")
            print(f"🔹 Overall Change: {wait_improvement:+.1f}%")
            print()
            print(f"🔄 Algorithm Activity:")
            print(f"🔹 Total Adaptations: {total_adaptations}")
            print(f"🔹 Average Adaptation Rate: {adaptation_rate:.1f} per minute")
            
            # Determine result
            if wait_improvement > 5:
                print("\\n🎉 EXCELLENT! Your algorithm significantly outperforms normal mode!")
            elif wait_improvement > 0:
                print("\\n✅ GOOD! Your algorithm shows improvement over normal mode.")
            else:
                print("\\n🤔 Your algorithm needs optimization - normal mode performs better.")
            
            return plot_file
        else:
            print("❌ Cannot create visualizations - no data available")
            return None

if __name__ == "__main__":
    visualizer = YourAlgorithmVisualizer()
    visualizer.visualize_results()