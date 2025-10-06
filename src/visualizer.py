"""
Traffic Visualization Module
============================
Comprehensive plotting and charting functionality for traffic simulation
performance visualization and analysis.

Features:
- Performance comparison charts
- Time series plots for traffic metrics
- Traffic flow visualization
- Statistical analysis charts
- Export capabilities for reports
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')

# Define color palette manually
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class TrafficVisualizer:
    """
    Comprehensive visualization system for traffic simulation data
    and performance analysis.
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (12, 8)):
        """
        Initialize the traffic visualizer.
        
        Args:
            figure_size: Default figure size for plots
        """
        self.figure_size = figure_size
        self.color_palette = {
            'adaptive': '#2E8B57',     # Sea Green
            'normal': '#DC143C',       # Crimson
            'background': '#F0F0F0',   # Light Gray
            'grid': '#CCCCCC',         # Light Gray
            'text': '#333333'          # Dark Gray
        }
        
    def plot_performance_comparison(self, adaptive_data: Dict, normal_data: Dict, 
                                  title: str = "Traffic Light Performance Comparison",
                                  save_path: Optional[str] = None) -> bool:
        """
        Create a comprehensive performance comparison chart.
        
        Args:
            adaptive_data: Performance data for adaptive algorithm
            normal_data: Performance data for normal algorithm
            title: Chart title
            save_path: Path to save the chart (optional)
            
        Returns:
            True if plot created successfully
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Extract metrics
            metrics = ['avg_waiting_time', 'avg_speed', 'throughput', 'efficiency_score']
            adaptive_values = [adaptive_data.get(m, 0) for m in metrics]
            normal_values = [normal_data.get(m, 0) for m in metrics]
            
            # Plot 1: Waiting Time Comparison
            categories = ['Adaptive', 'Normal']
            waiting_times = [adaptive_data.get('avg_waiting_time', 0), normal_data.get('avg_waiting_time', 0)]
            
            bars1 = ax1.bar(categories, waiting_times, 
                           color=[self.color_palette['adaptive'], self.color_palette['normal']])
            ax1.set_title('Average Waiting Time', fontweight='bold')
            ax1.set_ylabel('Time (seconds)')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, waiting_times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}s', ha='center', va='bottom')
            
            # Plot 2: Speed Comparison
            speeds = [adaptive_data.get('avg_speed', 0), normal_data.get('avg_speed', 0)]
            bars2 = ax2.bar(categories, speeds,
                           color=[self.color_palette['adaptive'], self.color_palette['normal']])
            ax2.set_title('Average Speed', fontweight='bold')
            ax2.set_ylabel('Speed (m/s)')
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars2, speeds):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.1f}', ha='center', va='bottom')
            
            # Plot 3: Throughput Comparison
            throughputs = [adaptive_data.get('throughput', 0), normal_data.get('throughput', 0)]
            bars3 = ax3.bar(categories, throughputs,
                           color=[self.color_palette['adaptive'], self.color_palette['normal']])
            ax3.set_title('Traffic Throughput', fontweight='bold')
            ax3.set_ylabel('Vehicles/Hour')
            ax3.grid(True, alpha=0.3)
            
            for bar, value in zip(bars3, throughputs):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{value:.0f}', ha='center', va='bottom')
            
            # Plot 4: Efficiency Score Comparison
            efficiency_scores = [adaptive_data.get('efficiency_score', 0), normal_data.get('efficiency_score', 0)]
            bars4 = ax4.bar(categories, efficiency_scores,
                           color=[self.color_palette['adaptive'], self.color_palette['normal']])
            ax4.set_title('Efficiency Score', fontweight='bold')
            ax4.set_ylabel('Score (%)')
            ax4.set_ylim(0, 100)
            ax4.grid(True, alpha=0.3)
            
            for bar, value in zip(bars4, efficiency_scores):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Performance comparison chart saved to {save_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"⚠️  Error creating performance comparison chart: {e}")
            return False
    
    def plot_time_series(self, time_series_data: List[Tuple[int, float]], 
                        metric_name: str = "Traffic Metric",
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> bool:
        """
        Create a time series plot for traffic metrics.
        
        Args:
            time_series_data: List of (timestamp, value) tuples
            metric_name: Name of the metric being plotted
            title: Chart title (optional)
            save_path: Path to save the chart (optional)
            
        Returns:
            True if plot created successfully
        """
        try:
            if not time_series_data:
                print("⚠️  No time series data provided")
                return False
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            timestamps = [t[0] for t in time_series_data]
            values = [t[1] for t in time_series_data]
            
            ax.plot(timestamps, values, linewidth=2.5, 
                   color=self.color_palette['adaptive'], marker='o', markersize=4)
            
            # Formatting
            chart_title = title or f"{metric_name} Over Time"
            ax.set_title(chart_title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(timestamps) > 1:
                z = np.polyfit(timestamps, values, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(timestamps), "--", alpha=0.8, color=self.color_palette['normal'])
            
            # Add statistics text
            if values:
                stats_text = f"Avg: {np.mean(values):.2f}\nMin: {min(values):.2f}\nMax: {max(values):.2f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📈 Time series chart saved to {save_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"⚠️  Error creating time series chart: {e}")
            return False
    
    def plot_scenario_comparison(self, scenario_results: Dict[str, Dict],
                               title: str = "Scenario Performance Analysis",
                               save_path: Optional[str] = None) -> bool:
        """
        Create a multi-scenario performance comparison chart.
        
        Args:
            scenario_results: Dictionary of scenario_name -> performance_data
            title: Chart title
            save_path: Path to save the chart (optional)
            
        Returns:
            True if plot created successfully
        """
        try:
            if not scenario_results:
                print("⚠️  No scenario data provided")
                return False
            
            scenarios = list(scenario_results.keys())
            metrics = ['avg_waiting_time', 'avg_speed', 'efficiency_score']
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Prepare data
            waiting_times = [scenario_results[s].get('avg_waiting_time', 0) for s in scenarios]
            speeds = [scenario_results[s].get('avg_speed', 0) for s in scenarios]
            efficiency_scores = [scenario_results[s].get('efficiency_score', 0) for s in scenarios]
            
            # Plot 1: Waiting Times
            bars1 = axes[0].bar(scenarios, waiting_times, color=self.color_palette['adaptive'])
            axes[0].set_title('Average Waiting Time')
            axes[0].set_ylabel('Time (seconds)')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            for bar, value in zip(bars1, waiting_times):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=10)
            
            # Plot 2: Speeds
            bars2 = axes[1].bar(scenarios, speeds, color=self.color_palette['normal'])
            axes[1].set_title('Average Speed')
            axes[1].set_ylabel('Speed (m/s)')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars2, speeds):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=10)
            
            # Plot 3: Efficiency Scores
            bars3 = axes[2].bar(scenarios, efficiency_scores, color='#FF6B6B')
            axes[2].set_title('Efficiency Score')
            axes[2].set_ylabel('Score (%)')
            axes[2].set_ylim(0, 100)
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3)
            
            for bar, value in zip(bars3, efficiency_scores):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Scenario comparison chart saved to {save_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"⚠️  Error creating scenario comparison chart: {e}")
            return False
    
    def plot_traffic_flow_heatmap(self, edge_data: Dict[str, Dict],
                                 title: str = "Traffic Flow Heatmap",
                                 save_path: Optional[str] = None) -> bool:
        """
        Create a heatmap visualization of traffic flow across edges.
        
        Args:
            edge_data: Dictionary of edge_id -> edge_metrics
            title: Chart title
            save_path: Path to save the chart (optional)
            
        Returns:
            True if plot created successfully
        """
        try:
            if not edge_data:
                print("⚠️  No edge data provided")
                return False
            
            # Prepare data for heatmap
            edges = list(edge_data.keys())
            metrics = ['vehicle_count', 'avg_waiting_time', 'mean_speed', 'occupancy']
            
            # Create data matrix
            data_matrix = []
            for metric in metrics:
                row = []
                for edge in edges:
                    value = edge_data[edge].get(metric, 0)
                    row.append(value)
                data_matrix.append(row)
            
            fig, ax = plt.subplots(figsize=(max(8, len(edges)), 6))
            
            # Create heatmap
            im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(edges)))
            ax.set_yticks(np.arange(len(metrics)))
            ax.set_xticklabels(edges)
            ax.set_yticklabels(metrics)
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Metric Value', rotation=-90, va="bottom")
            
            # Add text annotations
            for i in range(len(metrics)):
                for j in range(len(edges)):
                    text = ax.text(j, i, f'{data_matrix[i][j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"🔥 Traffic flow heatmap saved to {save_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"⚠️  Error creating traffic flow heatmap: {e}")
            return False
    
    def plot_performance_matrix(self, results_data: Dict[str, Any],
                              title: str = "Performance Analysis Matrix",
                              save_path: Optional[str] = None) -> bool:
        """
        Create a comprehensive performance matrix visualization.
        
        Args:
            results_data: Comprehensive results data
            title: Chart title
            save_path: Path to save the chart (optional)
            
        Returns:
            True if plot created successfully
        """
        try:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
            
            # Main title
            fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
            
            # Plot 1: Performance Overview (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            if 'adaptive_metrics' in results_data and 'normal_metrics' in results_data and results_data['adaptive_metrics'] and results_data['normal_metrics']:
                categories = ['Wait Time', 'Speed', 'Throughput', 'Efficiency']
                adaptive_vals = [
                    results_data['adaptive_metrics'].get('avg_waiting_time', 0),
                    results_data['adaptive_metrics'].get('avg_speed', 0) * 2,  # Scale for better visualization
                    results_data['adaptive_metrics'].get('throughput', 0) / 1000,  # Scale to thousands
                    results_data['adaptive_metrics'].get('efficiency_score', 0)
                ]
                normal_vals = [
                    results_data['normal_metrics'].get('avg_waiting_time', 0),
                    results_data['normal_metrics'].get('avg_speed', 0) * 2,  # Scale for better visualization
                    results_data['normal_metrics'].get('throughput', 0) / 1000,  # Scale to thousands
                    results_data['normal_metrics'].get('efficiency_score', 0)
                ]
                
                x = np.arange(len(categories))
                width = 0.35
                
                bars1 = ax1.bar(x - width/2, adaptive_vals, width, label='Adaptive', color=self.color_palette['adaptive'], alpha=0.8)
                bars2 = ax1.bar(x + width/2, normal_vals, width, label='Normal', color=self.color_palette['normal'], alpha=0.8)
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                
                ax1.set_title('Algorithm Comparison', fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(categories, rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No comparison data\navailable', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax1.set_title('Algorithm Comparison', fontweight='bold')
                ax1.axis('off')
            
            # Plot 2: Time Series (top-middle and top-right)
            ax2 = fig.add_subplot(gs[0, 1:])
            # Create synthetic time series data from scenario results if available
            if 'scenario_results' in results_data and results_data['scenario_results']:
                scenario_names = list(results_data['scenario_results'].keys())
                adaptive_waiting_times = []
                normal_waiting_times = []
                
                for scenario_name in scenario_names:
                    scenario = results_data['scenario_results'][scenario_name]
                    if 'adaptive' in scenario and 'normal' in scenario:
                        adaptive_waiting_times.append(scenario['adaptive']['performance_metrics'].get('avg_waiting_time', 0))
                        normal_waiting_times.append(scenario['normal']['performance_metrics'].get('avg_waiting_time', 0))
                
                if adaptive_waiting_times and normal_waiting_times:
                    x_pos = range(len(scenario_names))
                    ax2.plot(x_pos, adaptive_waiting_times, 'o-', color=self.color_palette['adaptive'], linewidth=2, markersize=8, label='Adaptive')
                    ax2.plot(x_pos, normal_waiting_times, 's-', color=self.color_palette['normal'], linewidth=2, markersize=8, label='Normal')
                    ax2.set_title('Performance Across Scenarios', fontweight='bold')
                    ax2.set_xlabel('Scenarios')
                    ax2.set_ylabel('Avg Waiting Time (s)')
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No time series data\navailable', ha='center', va='center', 
                            transform=ax2.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    ax2.set_title('Performance Over Time', fontweight='bold')
                    ax2.axis('off')
            else:
                ax2.text(0.5, 0.5, 'No time series data\navailable', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax2.set_title('Performance Over Time', fontweight='bold')
                ax2.axis('off')
            
            # Plot 3: Scenario Analysis (middle row)
            ax3 = fig.add_subplot(gs[1, :])
            if 'scenario_results' in results_data and results_data['scenario_results']:
                scenario_data = results_data['scenario_results']
                scenarios = list(scenario_data.keys())
                improvements = []
                
                for scenario in scenarios:
                    if 'comparison' in scenario_data[scenario]:
                        improvement = scenario_data[scenario]['comparison'].get('waiting_time_improvement', 0)
                        improvements.append(improvement)
                    else:
                        # Calculate improvement manually if not available
                        if 'adaptive' in scenario_data[scenario] and 'normal' in scenario_data[scenario]:
                            adaptive_perf = scenario_data[scenario]['adaptive']['performance_metrics'].get('avg_waiting_time', 0)
                            normal_perf = scenario_data[scenario]['normal']['performance_metrics'].get('avg_waiting_time', 0)
                            if normal_perf > 0:
                                improvement = ((normal_perf - adaptive_perf) / normal_perf) * 100
                            else:
                                improvement = 0
                            improvements.append(improvement)
                        else:
                            improvements.append(0)
                
                if improvements:
                    colors = ['#2E8B57' if imp > 0 else '#DC143C' if imp < 0 else '#888888' for imp in improvements]
                    bars = ax3.bar(scenarios, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                    ax3.set_title('Performance Improvement by Scenario', fontweight='bold')
                    ax3.set_ylabel('Improvement (%)')
                    ax3.set_xlabel('Scenarios')
                    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
                    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                    ax3.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for bar, value in zip(bars, improvements):
                        height = bar.get_height()
                        label_y = height + (1 if height >= 0 else -3)
                        ax3.text(bar.get_x() + bar.get_width()/2., label_y,
                               f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                               fontweight='bold', fontsize=10)
                else:
                    ax3.text(0.5, 0.5, 'No scenario improvement\ndata available', ha='center', va='center', 
                            transform=ax3.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    ax3.set_title('Performance Improvement by Scenario', fontweight='bold')
                    ax3.axis('off')
            else:
                ax3.text(0.5, 0.5, 'No scenario data\navailable', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax3.set_title('Performance Improvement by Scenario', fontweight='bold')
                ax3.axis('off')
            
            # Plot 4: Statistical Summary (bottom-left)
            ax4 = fig.add_subplot(gs[2, 0])
            if 'adaptive_metrics' in results_data and results_data['adaptive_metrics']:
                metrics = results_data['adaptive_metrics']
                summary_text = f"""ADAPTIVE CONTROLLER SUMMARY

Avg Waiting Time: {metrics.get('avg_waiting_time', 0):.1f}s
Avg Speed: {metrics.get('avg_speed', 0):.1f} m/s
Throughput: {metrics.get('throughput', 0):.0f} veh/h
Efficiency: {metrics.get('efficiency_score', 0):.1f}%
Consistency: {metrics.get('consistency_score', 0):.1f}%

Data Points: {metrics.get('data_points', 'N/A')}
Time Span: {metrics.get('time_span', 0):.0f}s
Analysis: {'🟢 Good' if metrics.get('avg_waiting_time', 100) < 15 else '🟡 Fair' if metrics.get('avg_waiting_time', 100) < 30 else '🔴 Poor'}"""
                
                ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy'))
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.set_title('Performance Summary', fontweight='bold')
                ax4.axis('off')
            else:
                ax4.text(0.5, 0.5, 'No performance\nsummary available', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax4.set_title('Performance Summary', fontweight='bold')
                ax4.axis('off')
            
            # Plot 5: Performance Distribution (bottom-middle)
            ax5 = fig.add_subplot(gs[2, 1])
            # Create distribution from scenario data if available
            if 'scenario_results' in results_data and results_data['scenario_results']:
                waiting_times = []
                for scenario_name, scenario_data in results_data['scenario_results'].items():
                    if 'adaptive' in scenario_data and 'performance_metrics' in scenario_data['adaptive']:
                        waiting_time = scenario_data['adaptive']['performance_metrics'].get('avg_waiting_time', 0)
                        if waiting_time > 0:
                            waiting_times.append(waiting_time)
                
                if waiting_times:
                    bins = min(10, len(waiting_times))  # Adjust bins based on data
                    n, bins_edges, patches = ax5.hist(waiting_times, bins=bins, color=self.color_palette['adaptive'], 
                                                     alpha=0.7, edgecolor='black', linewidth=1)
                    
                    # Color bars based on performance (green=good, red=bad)
                    for i, patch in enumerate(patches):
                        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
                        if bin_center < 15:  # Good performance
                            patch.set_facecolor('#2E8B57')
                        elif bin_center < 25:  # Fair performance
                            patch.set_facecolor('#FFA500')
                        else:  # Poor performance
                            patch.set_facecolor('#DC143C')
                    
                    ax5.set_title('Waiting Time Distribution', fontweight='bold')
                    ax5.set_xlabel('Waiting Time (s)')
                    ax5.set_ylabel('Frequency')
                    ax5.grid(True, alpha=0.3, axis='y')
                    
                    # Add mean line
                    mean_time = np.mean(waiting_times)
                    ax5.axvline(mean_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
                    ax5.text(mean_time + 0.5, max(n) * 0.8, f'Mean: {mean_time:.1f}s', 
                            rotation=90, va='top', ha='left', fontweight='bold')
                else:
                    ax5.text(0.5, 0.5, 'No distribution\ndata available', ha='center', va='center', 
                            transform=ax5.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    ax5.set_title('Waiting Time Distribution', fontweight='bold')
                    ax5.axis('off')
            else:
                ax5.text(0.5, 0.5, 'No distribution\ndata available', ha='center', va='center', 
                        transform=ax5.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax5.set_title('Waiting Time Distribution', fontweight='bold')
                ax5.axis('off')
            
            # Plot 6: Improvement Radar (bottom-right)
            ax6 = fig.add_subplot(gs[2, 2], projection='polar')
            if 'improvement_metrics' in results_data and results_data['improvement_metrics']:
                improvement_metrics = results_data['improvement_metrics']
                categories = ['Waiting Time', 'Speed', 'Throughput', 'Efficiency', 'Overall']
                
                # Calculate improvements - use available data or estimate from scenario results
                values = []
                
                # Waiting time improvement
                waiting_improvement = improvement_metrics.get('waiting_time_improvement', 0)
                if waiting_improvement == 0 and 'scenario_results' in results_data:
                    # Calculate from scenario results
                    improvements = []
                    for scenario_data in results_data['scenario_results'].values():
                        if 'comparison' in scenario_data:
                            improvements.append(scenario_data['comparison'].get('waiting_time_improvement', 0))
                    if improvements:
                        waiting_improvement = np.mean(improvements)
                values.append(waiting_improvement)
                
                # Speed improvement
                speed_improvement = improvement_metrics.get('speed_improvement', waiting_improvement * 0.5)  # Estimate
                values.append(speed_improvement)
                
                # Throughput improvement  
                throughput_improvement = improvement_metrics.get('throughput_improvement', -waiting_improvement * 0.3)  # Often inverse
                values.append(throughput_improvement)
                
                # Efficiency improvement
                efficiency_improvement = improvement_metrics.get('efficiency_improvement', waiting_improvement * 0.8)  # Similar to waiting
                values.append(efficiency_improvement)
                
                # Overall improvement
                overall_improvement = np.mean([waiting_improvement, speed_improvement, efficiency_improvement])
                values.append(overall_improvement)
                
                # Normalize values to 0-100 range for radar chart
                normalized_values = []
                for v in values:
                    normalized = max(0, min(100, v + 50))  # Center around 50, clamp to 0-100
                    normalized_values.append(normalized)
                
                # Complete the circle
                normalized_values += normalized_values[:1]
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                
                # Plot the radar chart
                ax6.plot(angles, normalized_values, color=self.color_palette['adaptive'], linewidth=3, alpha=0.8)
                ax6.fill(angles, normalized_values, color=self.color_palette['adaptive'], alpha=0.25)
                
                # Customize the radar chart
                ax6.set_xticks(angles[:-1])
                ax6.set_xticklabels(categories, fontsize=10)
                ax6.set_ylim(0, 100)
                ax6.set_yticks([20, 40, 60, 80, 100])
                ax6.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
                ax6.set_title('Improvement Radar\n(Higher = Better)', pad=20, fontweight='bold')
                ax6.grid(True, alpha=0.3)
                
                # Add reference circles
                ax6.plot(angles, [50] * len(angles), 'k--', alpha=0.5, linewidth=1)  # 50% reference line
                
            else:
                # Show placeholder with explanation
                ax6.text(0.5, 0.5, 'Improvement radar\nnot available\n\n(Requires comparison\nmetrics)', 
                        ha='center', va='center', transform=ax6.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax6.set_title('Improvement Radar', pad=20, fontweight='bold')
                # Remove polar projection elements
                ax6.set_xlim(0, 1)
                ax6.set_ylim(0, 1)
                ax6.set_xticks([])
                ax6.set_yticks([])
                ax6.spines['polar'].set_visible(False)
                ax6.grid(False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Performance matrix saved to {save_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"⚠️  Error creating performance matrix: {e}")
            return False
    
    def create_summary_dashboard(self, comprehensive_data: Dict[str, Any],
                               save_path: Optional[str] = None) -> bool:
        """
        Create a comprehensive summary dashboard with all key visualizations.
        
        Args:
            comprehensive_data: All available data for visualization
            save_path: Path to save the dashboard (optional)
            
        Returns:
            True if dashboard created successfully
        """
        try:
            # Create the performance matrix as the main dashboard
            return self.plot_performance_matrix(comprehensive_data, 
                                              "Traffic Light System - Performance Dashboard",
                                              save_path)
            
        except Exception as e:
            print(f"⚠️  Error creating summary dashboard: {e}")
            return False