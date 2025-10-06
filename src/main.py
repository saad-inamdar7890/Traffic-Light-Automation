"""
Main Adaptive Traffic Light Runner
===================================
Primary application for running the adaptive traffic light system
with default traffic flows for regular operation and performance monitoring.

Features:
- Adaptive traffic light control with real-time optimization
- Continuous performance monitoring and reporting
- Real-time visualization and statistics
- Configurable simulation parameters
- Export capabilities for data analysis
"""

import sys
import os
import time
import argparse
import signal
import traci
from typing import Dict, List, Optional, Any

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from traffic_controller import AdaptiveTrafficController
from analyzer import TrafficAnalyzer
from visualizer import TrafficVisualizer
from utils import RouteGenerator, SUMOConfigManager, FileManager, SimulationUtils


class AdaptiveTrafficSystem:
    """
    Main adaptive traffic light system for real-time traffic control
    and performance monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive traffic system.
        
        Args:
            config: System configuration parameters
        """
        self.config = config or self._get_default_config()
        self.is_running = False
        self.simulation_start_time = 0
        self.total_runtime = 0
        
        # Initialize components
        self.controller = AdaptiveTrafficController()
        self.analyzer = TrafficAnalyzer()
        self.visualizer = TrafficVisualizer()
        self.file_manager = FileManager()
        
        # Setup file paths
        self.base_directory = self.config.get('base_directory', '.')
        self.route_file = self.config.get('route_file', 'adaptive_traffic.rou.xml')
        self.config_file = self.config.get('config_file', 'adaptive_traffic.sumocfg')
        self.results_directory = self.config.get('results_directory', 'adaptive_results')
        
        # Performance tracking
        self.performance_history = []
        self.last_report_time = 0
        self.report_interval = self.config.get('report_interval', 60)  # Report every minute
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print("🚦 Adaptive Traffic Light System initialized")
        self._print_configuration()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration."""
        return {
            'base_directory': '.',
            'simulation_duration': 3600,  # 1 hour default
            'step_length': 1,
            'route_file': 'adaptive_traffic.rou.xml',
            'config_file': 'adaptive_traffic.sumocfg',
            'network_file': '../demo.net.xml',
            'results_directory': 'adaptive_results',
            'report_interval': 60,
            'save_interval': 300,  # Save data every 5 minutes
            'scenario': 'balanced',
            'enable_gui': False,
            'real_time_visualization': False,
            'export_data': True,
            'verbose_logging': True
        }
    
    def _print_configuration(self):
        """Print current system configuration."""
        print(f"\n⚙️  SYSTEM CONFIGURATION:")
        print(f"   📁 Base Directory: {self.config['base_directory']}")
        print(f"   ⏱️  Duration: {self.config['simulation_duration']}s ({self.config['simulation_duration']/60:.1f} min)")
        print(f"   🌐 Network: {self.config['network_file']}")
        print(f"   🚗 Scenario: {self.config['scenario']}")
        print(f"   📊 Reports: Every {self.config['report_interval']}s")
        print(f"   💾 Auto-save: Every {self.config['save_interval']}s")
        print(f"   🖥️  GUI: {'Enabled' if self.config['enable_gui'] else 'Disabled'}")
        print()
    
    def setup_simulation_files(self) -> bool:
        """
        Setup required simulation files (route and config).
        
        Returns:
            True if setup successful
        """
        try:
            print("📂 Setting up simulation files...")
            
            # Ensure results directory exists
            self.file_manager.ensure_directory(self.results_directory)
            
            # Generate route file
            route_generator = RouteGenerator()
            scenario_config = route_generator.create_scenario_flows(self.config['scenario'])
            
            route_path = os.path.join(self.base_directory, self.route_file)
            if not route_generator.generate_route_file(route_path, scenario_config):
                print("⚠️  Failed to generate route file")
                return False
            
            # Create SUMO config file
            config_manager = SUMOConfigManager()
            config_params = {
                'net-file': self.config['network_file'],
                'route-files': self.route_file,
                'begin': '0',
                'end': str(self.config['simulation_duration']),
                'step-length': str(self.config['step_length'])
            }
            
            config_path = os.path.join(self.base_directory, self.config_file)
            if not config_manager.create_config_file(config_path, config_params):
                print("⚠️  Failed to create SUMO config file")
                return False
            
            print("✅ Simulation files setup complete")
            return True
            
        except Exception as e:
            print(f"⚠️  Error setting up simulation files: {e}")
            return False
    
    def start_simulation(self) -> bool:
        """
        Start the SUMO simulation.
        
        Returns:
            True if simulation started successfully
        """
        try:
            print("🚀 Starting SUMO simulation...")
            
            # Build SUMO command
            config_path = os.path.join(self.base_directory, self.config_file)
            
            if self.config['enable_gui']:
                sumo_cmd = ['sumo-gui', '-c', config_path]
            else:
                sumo_cmd = ['sumo', '-c', config_path, '--no-warnings']
                
                if not self.config['verbose_logging']:
                    sumo_cmd.append('--no-step-log')
            
            # Start TraCI
            traci.start(sumo_cmd)
            
            print("✅ SUMO simulation started")
            return True
            
        except Exception as e:
            print(f"⚠️  Error starting simulation: {e}")
            return False
    
    def run_adaptive_control(self) -> bool:
        """
        Run the main adaptive traffic control loop.
        
        Returns:
            True if completed successfully
        """
        try:
            print("🧠 Starting adaptive traffic control system...")
            print("   Press Ctrl+C to stop gracefully")
            
            self.is_running = True
            self.simulation_start_time = time.time()
            step = 0
            last_save_time = 0
            
            while self.is_running and step < self.config['simulation_duration']:
                # Execute simulation step
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                
                # Apply adaptive traffic control
                control_action = self.controller.control_traffic_lights(current_time, traci)
                
                # Collect traffic metrics
                metrics = self.analyzer.collect_traffic_metrics(int(current_time), traci)
                if metrics:
                    metrics.update(control_action)
                
                # Generate periodic reports
                if current_time - self.last_report_time >= self.report_interval:
                    self._generate_periodic_report(current_time)
                    self.last_report_time = current_time
                
                # Auto-save data periodically
                if current_time - last_save_time >= self.config['save_interval']:
                    self._auto_save_data(current_time)
                    last_save_time = current_time
                
                # Real-time visualization (if enabled)
                if self.config['real_time_visualization'] and step % 30 == 0:  # Every 30 seconds
                    self._update_real_time_visualization(current_time)
                
                step += 1
            
            self.total_runtime = time.time() - self.simulation_start_time
            
            print(f"\n✅ Adaptive control completed successfully")
            print(f"   Total simulation time: {current_time:.0f}s")
            print(f"   Total real time: {self.total_runtime:.1f}s")
            print(f"   Simulation speed: {current_time/self.total_runtime:.1f}x real-time")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error in adaptive control: {e}")
            return False
        finally:
            self.is_running = False
    
    def _generate_periodic_report(self, current_time: float):
        """
        Generate and display periodic performance report.
        
        Args:
            current_time: Current simulation time
        """
        try:
            # Calculate performance metrics for recent period
            performance_metrics = self.analyzer.calculate_performance_metrics(
                time_window=self.report_interval
            )
            
            if performance_metrics:
                # Store in history
                performance_metrics['timestamp'] = current_time
                self.performance_history.append(performance_metrics)
                
                # Print report
                print(f"\n📊 PERFORMANCE REPORT - {current_time:.0f}s")
                print(f"   🚗 Vehicles: {performance_metrics.get('avg_vehicle_count', 0):.0f} avg")
                print(f"   ⏰ Wait Time: {performance_metrics.get('avg_waiting_time', 0):.1f}s")
                print(f"   🏃 Speed: {performance_metrics.get('avg_speed', 0):.1f} m/s")
                print(f"   📈 Efficiency: {performance_metrics.get('efficiency_score', 0):.1f}%")
                print(f"   🎯 Throughput: {performance_metrics.get('throughput', 0):.0f} veh/h")
                
                # Controller statistics
                controller_stats = self.controller.get_statistics()
                if controller_stats:
                    print(f"   🔧 Adaptations: {controller_stats.get('total_adaptations', 0)}")
                    print(f"   ⚡ Avg Adjustment: {controller_stats.get('avg_adjustment', 0):.1f}s")
            
        except Exception as e:
            print(f"⚠️  Error generating periodic report: {e}")
    
    def _auto_save_data(self, current_time: float):
        """
        Auto-save current data to files.
        
        Args:
            current_time: Current simulation time
        """
        try:
            timestamp = int(current_time)
            
            # Save analyzer data
            analyzer_filename = f"traffic_data_{timestamp}.json"
            analyzer_path = os.path.join(self.results_directory, analyzer_filename)
            self.analyzer.export_data(analyzer_path)
            
            # Save controller statistics
            controller_stats = self.controller.get_statistics()
            if controller_stats:
                stats_filename = f"controller_stats_{timestamp}.json"
                stats_path = os.path.join(self.results_directory, stats_filename)
                self.file_manager.save_results_json(controller_stats, stats_path)
            
            print(f"💾 Auto-saved data at {current_time:.0f}s")
            
        except Exception as e:
            print(f"⚠️  Error auto-saving data: {e}")
    
    def _update_real_time_visualization(self, current_time: float):
        """
        Update real-time visualization (if enabled).
        
        Args:
            current_time: Current simulation time
        """
        try:
            # Get recent time series data
            waiting_time_series = self.analyzer.get_time_series_data('avg_waiting_time', 300)  # Last 5 minutes
            
            if waiting_time_series:
                # Create time series plot
                plot_path = os.path.join(self.results_directory, "real_time_performance.png")
                self.visualizer.plot_time_series(
                    waiting_time_series,
                    "Average Waiting Time",
                    f"Real-time Performance - {current_time:.0f}s",
                    plot_path
                )
                
        except Exception as e:
            print(f"⚠️  Error updating real-time visualization: {e}")
    
    def generate_final_report(self) -> bool:
        """
        Generate comprehensive final performance report.
        
        Returns:
            True if report generated successfully
        """
        try:
            print("\n📋 Generating final performance report...")
            
            # Calculate overall performance metrics
            overall_metrics = self.analyzer.calculate_performance_metrics()
            
            if not overall_metrics:
                print("⚠️  No performance data available for final report")
                return False
            
            # Generate text report
            text_report = self.analyzer.generate_summary_report()
            
            # Add controller statistics
            controller_stats = self.controller.get_statistics()
            if controller_stats:
                text_report += f"\n\n🧠 ADAPTIVE CONTROLLER STATISTICS:\n"
                text_report += f"   Total Adaptations: {controller_stats.get('total_adaptations', 0)}\n"
                text_report += f"   Average Adjustment: {controller_stats.get('avg_adjustment', 0):.1f}s\n"
                text_report += f"   Phase Switches: {controller_stats.get('phase_switches', 0)}\n"
                text_report += f"   Adaptation Rate: {controller_stats.get('adaptation_rate', 0):.2f}/min\n"
            
            # Add system performance
            text_report += f"\n\n⚡ SYSTEM PERFORMANCE:\n"
            text_report += f"   Simulation Duration: {overall_metrics.get('time_span', 0):.0f}s\n"
            text_report += f"   Real Runtime: {self.total_runtime:.1f}s\n"
            text_report += f"   Simulation Speed: {overall_metrics.get('time_span', 0)/self.total_runtime:.1f}x real-time\n"
            text_report += f"   Data Collection Rate: {overall_metrics.get('data_points', 0)/(overall_metrics.get('time_span', 1)/60):.1f} points/min\n"
            
            # Save text report
            report_path = os.path.join(self.results_directory, "final_performance_report.txt")
            with open(report_path, 'w') as f:
                f.write(text_report)
            
            print(text_report)
            print(f"📄 Final report saved to {report_path}")
            
            # Generate visualizations
            return self._generate_final_visualizations(overall_metrics)
            
        except Exception as e:
            print(f"⚠️  Error generating final report: {e}")
            return False
    
    def _generate_final_visualizations(self, overall_metrics: Dict[str, Any]) -> bool:
        """
        Generate final visualization charts.
        
        Args:
            overall_metrics: Overall performance metrics
            
        Returns:
            True if visualizations generated successfully
        """
        try:
            print("📊 Generating final visualizations...")
            
            # 1. Performance time series
            waiting_time_series = self.analyzer.get_time_series_data('avg_waiting_time')
            if waiting_time_series:
                time_series_path = os.path.join(self.results_directory, "performance_time_series.png")
                self.visualizer.plot_time_series(
                    waiting_time_series,
                    "Average Waiting Time (seconds)",
                    "Traffic Performance Over Time - Adaptive Control",
                    time_series_path
                )
            
            # 2. Performance dashboard
            dashboard_data = {
                'adaptive_metrics': overall_metrics,
                'time_series': waiting_time_series,
                'performance_distribution': [d['avg_waiting_time'] for d in self.analyzer.data_history 
                                           if 'avg_waiting_time' in d],
                'improvement_metrics': {
                    'waiting_time_improvement': 0,  # No comparison baseline
                    'speed_improvement': 0,
                    'throughput_improvement': 0,
                    'efficiency_improvement': overall_metrics.get('efficiency_score', 0) - 50,  # Relative to 50% baseline
                    'consistency_improvement': overall_metrics.get('consistency_score', 0) - 50
                }
            }
            
            dashboard_path = os.path.join(self.results_directory, "adaptive_performance_dashboard.png")
            self.visualizer.plot_performance_matrix(
                dashboard_data,
                "Adaptive Traffic Light System - Performance Dashboard",
                dashboard_path
            )
            
            print(f"✅ Visualizations saved to {self.results_directory}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")
            return False
    
    def export_final_data(self) -> bool:
        """
        Export all collected data for external analysis.
        
        Returns:
            True if export successful
        """
        try:
            if not self.config['export_data']:
                return True
            
            print("💾 Exporting final data...")
            
            # Export analyzer data
            analyzer_export_path = os.path.join(self.results_directory, "complete_traffic_analysis.json")
            if not self.analyzer.export_data(analyzer_export_path):
                return False
            
            # Export controller data
            controller_stats = self.controller.get_statistics()
            if controller_stats:
                controller_export_path = os.path.join(self.results_directory, "controller_statistics.json")
                self.file_manager.save_results_json(controller_stats, controller_export_path)
            
            # Export performance history
            if self.performance_history:
                performance_export_path = os.path.join(self.results_directory, "performance_history.json")
                self.file_manager.save_results_json(self.performance_history, performance_export_path)
            
            # Export system configuration
            config_export_path = os.path.join(self.results_directory, "system_configuration.json")
            export_config = self.config.copy()
            export_config['total_runtime'] = self.total_runtime
            export_config['export_timestamp'] = time.time()
            self.file_manager.save_results_json(export_config, config_export_path)
            
            print(f"✅ Data exported to {self.results_directory}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error exporting final data: {e}")
            return False
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        print(f"\n⚠️  Received interrupt signal. Shutting down gracefully...")
        self.is_running = False
    
    def run(self) -> bool:
        """
        Run the complete adaptive traffic system.
        
        Returns:
            True if completed successfully
        """
        try:
            print("🚀 Starting Adaptive Traffic Light System")
            print("=" * 50)
            
            # Setup simulation files
            if not self.setup_simulation_files():
                return False
            
            # Start SUMO simulation
            if not self.start_simulation():
                return False
            
            # Run adaptive control
            success = self.run_adaptive_control()
            
            # Generate final report and export data
            if success:
                self.generate_final_report()
                self.export_final_data()
            
            return success
            
        except Exception as e:
            print(f"⚠️  System error: {e}")
            return False
        finally:
            # Cleanup
            try:
                traci.close()
            except:
                pass
            
            print(f"\n🏁 Adaptive Traffic System shutdown complete")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Adaptive Traffic Light System")
    
    parser.add_argument("--duration", type=int, default=3600,
                       help="Simulation duration in seconds (default: 3600)")
    parser.add_argument("--scenario", type=str, default="balanced",
                       choices=["balanced", "heavy_ns", "heavy_ew", "rush_hour"],
                       help="Traffic scenario to run (default: balanced)")
    parser.add_argument("--gui", action="store_true",
                       help="Enable SUMO GUI")
    parser.add_argument("--realtime-viz", action="store_true",
                       help="Enable real-time visualization")
    parser.add_argument("--results-dir", type=str, default="adaptive_results",
                       help="Results directory (default: adaptive_results)")
    parser.add_argument("--report-interval", type=int, default=60,
                       help="Report interval in seconds (default: 60)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def main():
    """Main function for the adaptive traffic system."""
    print("🚦 Adaptive Traffic Light System")
    print("=" * 40)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check SUMO installation
    if not SimulationUtils.check_sumo_installation():
        print("⚠️  SUMO not found. Please install SUMO and add it to PATH.")
        return 1
    
    # Create configuration from arguments
    config = {
        'simulation_duration': args.duration,
        'scenario': args.scenario,
        'enable_gui': args.gui,
        'real_time_visualization': args.realtime_viz,
        'results_directory': args.results_dir,
        'report_interval': args.report_interval,
        'verbose_logging': args.verbose,
        'export_data': True
    }
    
    # Initialize and run system
    system = AdaptiveTrafficSystem(config)
    
    try:
        success = system.run()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  System interrupted by user")
        return 1
    except Exception as e:
        print(f"\n⚠️  System failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())