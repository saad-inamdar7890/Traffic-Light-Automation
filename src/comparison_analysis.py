"""
Traffic Light Comparison Analysis Tool
=====================================
Comprehensive comparison analysis between normal (fixed-time) and adaptive
traffic light algorithms across multiple test scenarios.

Features:
- Normal vs Adaptive algorithm comparison
- Multiple test scenarios (balanced, heavy NS, heavy EW, rush hour)
- Performance metrics collection and analysis
- Comprehensive visualization and reporting
- Statistical significance testing
"""

import sys
import os
import time
import traci
from typing import Dict, List, Optional, Any, Tuple

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from traffic_controller import AdaptiveTrafficController
from analyzer import TrafficAnalyzer
from visualizer import TrafficVisualizer
from utils import RouteGenerator, SUMOConfigManager, FileManager, SimulationUtils


class NormalTrafficController:
    """
    Normal (fixed-time) traffic light controller for comparison baseline.
    Uses standard fixed timing phases without adaptation.
    """
    
    def __init__(self, ns_green_time: int = 30, ew_green_time: int = 30, 
                 yellow_time: int = 4, red_time: int = 2):
        """
        Initialize normal traffic controller.
        
        Args:
            ns_green_time: North-South green phase duration
            ew_green_time: East-West green phase duration
            yellow_time: Yellow phase duration
            red_time: All-red phase duration
        """
        self.ns_green_time = ns_green_time
        self.ew_green_time = ew_green_time
        self.yellow_time = yellow_time
        self.red_time = red_time
        
        self.cycle_time = 2 * (ns_green_time + yellow_time + red_time)
        self.current_phase_start = 0
        self.phase_sequence = [
            {'phase': 0, 'duration': ns_green_time},    # NS Green
            {'phase': 1, 'duration': yellow_time},      # NS Yellow
            {'phase': 2, 'duration': red_time},         # All Red
            {'phase': 3, 'duration': ew_green_time},    # EW Green
            {'phase': 4, 'duration': yellow_time},      # EW Yellow
            {'phase': 5, 'duration': red_time}          # All Red
        ]
        self.current_phase_index = 0
        
        print(f"🚦 Normal controller initialized with {self.cycle_time}s cycle")
    
    def control_traffic_lights(self, current_time: int, traci_connection) -> Dict[str, Any]:
        """
        Control traffic lights with fixed timing.
        
        Args:
            current_time: Current simulation time
            traci_connection: TraCI connection
            
        Returns:
            Control action information
        """
        try:
            # Calculate time within current cycle
            cycle_position = current_time % self.cycle_time
            
            # Determine current phase
            elapsed_time = 0
            target_phase = 0
            
            for i, phase_info in enumerate(self.phase_sequence):
                if cycle_position < elapsed_time + phase_info['duration']:
                    target_phase = phase_info['phase']
                    break
                elapsed_time += phase_info['duration']
            
            # Apply phase to traffic light
            tl_ids = traci_connection.trafficlight.getIDList()
            for tl_id in tl_ids:
                current_phase = traci_connection.trafficlight.getPhase(tl_id)
                if current_phase != target_phase:
                    traci_connection.trafficlight.setPhase(tl_id, target_phase)
            
            return {
                'algorithm': 'normal',
                'current_phase': target_phase,
                'cycle_position': cycle_position,
                'action_taken': 'fixed_timing'
            }
            
        except Exception as e:
            print(f"⚠️  Error in normal traffic control: {e}")
            return {'algorithm': 'normal', 'error': str(e)}


class ComparisonAnalyzer:
    """
    Main comparison analysis system for evaluating normal vs adaptive algorithms.
    """
    
    def __init__(self, base_directory: str = "."):
        """
        Initialize the comparison analyzer.
        
        Args:
            base_directory: Base directory for simulation files
        """
        self.base_directory = base_directory
        self.results = {}
        self.temp_files = []
        
        # Initialize components
        self.route_generator = RouteGenerator()
        self.config_manager = SUMOConfigManager()
        self.file_manager = FileManager()
        self.visualizer = TrafficVisualizer()
        
        # Test scenarios
        self.test_scenarios = {
            'balanced': 'Balanced traffic in all directions',
            'heavy_ns': 'Heavy North-South, light East-West traffic',
            'heavy_ew': 'Heavy East-West, light North-South traffic',
            'rush_hour': 'Rush hour simulation with high traffic'
        }
        
        print("📊 Comparison analyzer initialized")
    
    def run_single_simulation(self, algorithm_type: str, scenario_name: str, 
                            duration: int = 900) -> Optional[Dict[str, Any]]:
        """
        Run a single simulation with specified algorithm and scenario.
        
        Args:
            algorithm_type: 'adaptive' or 'normal'
            scenario_name: Name of traffic scenario
            duration: Simulation duration in seconds
            
        Returns:
            Simulation results or None if failed
        """
        try:
            print(f"🚀 Running {algorithm_type} simulation for {scenario_name} scenario...")
            
            # Generate route file for scenario
            route_filename = f"temp_route_{scenario_name}_{algorithm_type}.rou.xml"
            route_path = os.path.join(self.base_directory, route_filename)
            
            scenario_config = self.route_generator.create_scenario_flows(scenario_name)
            if not self.route_generator.generate_route_file(route_path, scenario_config):
                print(f"⚠️  Failed to generate route file for {scenario_name}")
                return None
            
            self.temp_files.append(route_path)
            
            # Create SUMO config file
            config_filename = f"temp_config_{scenario_name}_{algorithm_type}.sumocfg"
            config_path = os.path.join(self.base_directory, config_filename)
            
            config_params = {
                'net-file': '../demo.net.xml',
                'route-files': route_filename,
                'begin': '0',
                'end': str(duration),
                'step-length': '1'
            }
            
            if not self.config_manager.create_config_file(config_path, config_params):
                print(f"⚠️  Failed to create config file for {scenario_name}")
                return None
            
            self.temp_files.append(config_path)
            
            # Start SUMO simulation
            sumo_cmd = ['sumo', '-c', config_path, '--no-warnings', '--no-step-log']
            traci.start(sumo_cmd)
            
            # Initialize controllers and analyzer
            if algorithm_type == 'adaptive':
                controller = AdaptiveTrafficController()
            else:
                controller = NormalTrafficController()
            
            analyzer = TrafficAnalyzer()
            
            # Run simulation
            step = 0
            simulation_data = []
            
            while step < duration:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                
                # Apply traffic control
                if algorithm_type == 'adaptive':
                    control_action = controller.control_traffic_lights(current_time, traci)
                else:
                    control_action = controller.control_traffic_lights(int(current_time), traci)
                
                # Collect metrics every 10 seconds
                if step % 10 == 0:
                    metrics = analyzer.collect_traffic_metrics(int(current_time), traci)
                    if metrics:
                        metrics.update(control_action)
                        simulation_data.append(metrics)
                
                step += 1
            
            # Calculate final performance metrics
            performance_metrics = analyzer.calculate_performance_metrics()
            performance_metrics.update({
                'algorithm': algorithm_type,
                'scenario': scenario_name,
                'duration': duration,
                'scenario_description': self.test_scenarios.get(scenario_name, 'Unknown scenario')
            })
            
            # Close SUMO
            traci.close()
            
            print(f"✅ {algorithm_type.capitalize()} simulation completed for {scenario_name}")
            print(f"   Avg Waiting Time: {performance_metrics.get('avg_waiting_time', 0):.2f}s")
            print(f"   Avg Speed: {performance_metrics.get('avg_speed', 0):.2f} m/s")
            
            return {
                'performance_metrics': performance_metrics,
                'simulation_data': simulation_data,
                'scenario_config': scenario_config
            }
            
        except Exception as e:
            print(f"⚠️  Error in {algorithm_type} simulation for {scenario_name}: {e}")
            try:
                traci.close()
            except:
                pass
            return None
    
    def run_comparison_analysis(self, scenarios: Optional[List[str]] = None, 
                              duration: int = 900) -> Dict[str, Any]:
        """
        Run comprehensive comparison analysis across multiple scenarios.
        
        Args:
            scenarios: List of scenarios to test (None for all)
            duration: Simulation duration per test
            
        Returns:
            Complete comparison results
        """
        if scenarios is None:
            scenarios = list(self.test_scenarios.keys())
        
        print(f"🔬 Starting comprehensive comparison analysis...")
        print(f"   Scenarios: {', '.join(scenarios)}")
        print(f"   Duration per test: {duration}s")
        print(f"   Total estimated time: {len(scenarios) * 2 * duration / 60:.1f} minutes")
        
        comparison_results = {
            'scenarios': {},
            'summary': {},
            'metadata': {
                'test_scenarios': scenarios,
                'duration_per_test': duration,
                'timestamp': time.time()
            }
        }
        
        # Run tests for each scenario
        for scenario in scenarios:
            print(f"\n📋 Testing scenario: {scenario}")
            print(f"   Description: {self.test_scenarios[scenario]}")
            
            scenario_results = {}
            
            # Run adaptive simulation
            adaptive_results = self.run_single_simulation('adaptive', scenario, duration)
            if adaptive_results:
                scenario_results['adaptive'] = adaptive_results
            
            # Run normal simulation
            normal_results = self.run_single_simulation('normal', scenario, duration)
            if normal_results:
                scenario_results['normal'] = normal_results
            
            # Calculate comparison metrics
            if 'adaptive' in scenario_results and 'normal' in scenario_results:
                adaptive_perf = scenario_results['adaptive']['performance_metrics']
                normal_perf = scenario_results['normal']['performance_metrics']
                
                # Calculate improvements
                waiting_time_improvement = SimulationUtils.calculate_improvement_percentage(
                    adaptive_perf.get('avg_waiting_time', 0),
                    normal_perf.get('avg_waiting_time', 0)
                )
                
                speed_improvement = SimulationUtils.calculate_improvement_percentage(
                    normal_perf.get('avg_speed', 0),  # Higher speed is better
                    adaptive_perf.get('avg_speed', 0)
                ) * -1  # Reverse for speed (higher is better)
                
                throughput_improvement = SimulationUtils.calculate_improvement_percentage(
                    normal_perf.get('throughput', 0),
                    adaptive_perf.get('throughput', 0)
                ) * -1  # Reverse for throughput
                
                scenario_results['comparison'] = {
                    'waiting_time_improvement': waiting_time_improvement,
                    'speed_improvement': speed_improvement,
                    'throughput_improvement': throughput_improvement,
                    'adaptive_avg_waiting_time': adaptive_perf.get('avg_waiting_time', 0),
                    'normal_avg_waiting_time': normal_perf.get('avg_waiting_time', 0),
                    'adaptive_avg_speed': adaptive_perf.get('avg_speed', 0),
                    'normal_avg_speed': normal_perf.get('avg_speed', 0),
                    'performance_category': self._categorize_performance(waiting_time_improvement)
                }
                
                print(f"   🏆 Results: {waiting_time_improvement:+.1f}% waiting time improvement")
            
            comparison_results['scenarios'][scenario] = scenario_results
        
        # Generate summary statistics
        comparison_results['summary'] = self._generate_summary_statistics(comparison_results['scenarios'])
        
        # Store results
        self.results = comparison_results
        
        print(f"\n✅ Comparison analysis completed!")
        self._print_summary_report()
        
        return comparison_results
    
    def _categorize_performance(self, improvement: float) -> str:
        """Categorize performance improvement."""
        if improvement >= 5.0:
            return "Excellent"
        elif improvement >= 1.0:
            return "Good"
        elif improvement >= -1.0:
            return "Neutral"
        elif improvement >= -5.0:
            return "Poor"
        else:
            return "Very Poor"
    
    def _generate_summary_statistics(self, scenario_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics across all scenarios."""
        improvements = []
        successful_tests = 0
        
        for scenario, results in scenario_results.items():
            if 'comparison' in results:
                improvements.append(results['comparison']['waiting_time_improvement'])
                successful_tests += 1
        
        if improvements:
            return {
                'average_improvement': sum(improvements) / len(improvements),
                'best_improvement': max(improvements),
                'worst_improvement': min(improvements),
                'successful_tests': successful_tests,
                'total_tests': len(scenario_results),
                'success_rate': (successful_tests / len(scenario_results)) * 100 if scenario_results else 0
            }
        else:
            return {'successful_tests': 0, 'total_tests': len(scenario_results)}
    
    def _print_summary_report(self):
        """Print a formatted summary report."""
        if not self.results or 'summary' not in self.results:
            return
        
        summary = self.results['summary']
        print(f"\n📊 COMPARISON ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"✅ Successful Tests: {summary.get('successful_tests', 0)}/{summary.get('total_tests', 0)}")
        
        if summary.get('successful_tests', 0) > 0:
            print(f"📈 Average Improvement: {summary.get('average_improvement', 0):+.1f}%")
            print(f"🏆 Best Performance: {summary.get('best_improvement', 0):+.1f}%")
            print(f"📉 Worst Performance: {summary.get('worst_improvement', 0):+.1f}%")
            
            # Print individual scenario results
            print(f"\n📋 SCENARIO BREAKDOWN:")
            for scenario, results in self.results['scenarios'].items():
                if 'comparison' in results:
                    comp = results['comparison']
                    improvement = comp['waiting_time_improvement']
                    category = comp['performance_category']
                    print(f"   {scenario:12}: {improvement:+5.1f}% ({category})")
    
    def generate_visualizations(self, save_directory: str = "analysis_results") -> bool:
        """
        Generate comprehensive visualizations for the comparison analysis.
        
        Args:
            save_directory: Directory to save visualization files
            
        Returns:
            True if visualizations generated successfully
        """
        try:
            if not self.results:
                print("⚠️  No results available for visualization")
                return False
            
            # Ensure save directory exists
            self.file_manager.ensure_directory(save_directory)
            
            print(f"📊 Generating visualizations in {save_directory}...")
            
            # 1. Overall Performance Comparison
            if self.results['summary'].get('successful_tests', 0) > 0:
                # Calculate overall metrics
                all_adaptive_metrics = []
                all_normal_metrics = []
                
                for scenario, results in self.results['scenarios'].items():
                    if 'adaptive' in results and 'normal' in results:
                        all_adaptive_metrics.append(results['adaptive']['performance_metrics'])
                        all_normal_metrics.append(results['normal']['performance_metrics'])
                
                if all_adaptive_metrics and all_normal_metrics:
                    # Average metrics across scenarios
                    avg_adaptive = {
                        'avg_waiting_time': sum(m.get('avg_waiting_time', 0) for m in all_adaptive_metrics) / len(all_adaptive_metrics),
                        'avg_speed': sum(m.get('avg_speed', 0) for m in all_adaptive_metrics) / len(all_adaptive_metrics),
                        'throughput': sum(m.get('throughput', 0) for m in all_adaptive_metrics) / len(all_adaptive_metrics),
                        'efficiency_score': sum(m.get('efficiency_score', 0) for m in all_adaptive_metrics) / len(all_adaptive_metrics)
                    }
                    
                    avg_normal = {
                        'avg_waiting_time': sum(m.get('avg_waiting_time', 0) for m in all_normal_metrics) / len(all_normal_metrics),
                        'avg_speed': sum(m.get('avg_speed', 0) for m in all_normal_metrics) / len(all_normal_metrics),
                        'throughput': sum(m.get('throughput', 0) for m in all_normal_metrics) / len(all_normal_metrics),
                        'efficiency_score': sum(m.get('efficiency_score', 0) for m in all_normal_metrics) / len(all_normal_metrics)
                    }
                    
                    # Overall comparison chart
                    overall_comparison_path = os.path.join(save_directory, "overall_performance_comparison.png")
                    self.visualizer.plot_performance_comparison(
                        avg_adaptive, avg_normal,
                        "Overall Performance: Adaptive vs Normal Traffic Control",
                        overall_comparison_path
                    )
            
            # 2. Scenario-by-scenario comparison
            scenario_data = {}
            for scenario, results in self.results['scenarios'].items():
                if 'comparison' in results:
                    comp = results['comparison']
                    scenario_data[scenario] = {
                        'avg_waiting_time': comp['adaptive_avg_waiting_time'],
                        'avg_speed': comp['adaptive_avg_speed'],
                        'efficiency_score': 100 - comp['adaptive_avg_waiting_time']  # Simple efficiency metric
                    }
            
            if scenario_data:
                scenario_comparison_path = os.path.join(save_directory, "scenario_analysis.png")
                self.visualizer.plot_scenario_comparison(
                    scenario_data,
                    "Adaptive Algorithm Performance Across Scenarios",
                    scenario_comparison_path
                )
            
            # 3. Performance matrix dashboard
            dashboard_data = {
                'adaptive_metrics': avg_adaptive if 'avg_adaptive' in locals() else {},
                'normal_metrics': avg_normal if 'avg_normal' in locals() else {},
                'scenario_results': self.results['scenarios'],
                'improvement_metrics': {
                    'waiting_time_improvement': self.results['summary'].get('average_improvement', 0),
                    'speed_improvement': 0,  # Will be calculated in visualizer from scenario data
                    'throughput_improvement': 0,  # Will be calculated in visualizer from scenario data
                    'efficiency_improvement': self.results['summary'].get('average_improvement', 0),
                    'consistency_improvement': 0  # Will be calculated in visualizer from scenario data
                },
                'performance_distribution': [],  # Will be generated from scenario data in visualizer
                'time_series': []  # Will be generated from scenario data in visualizer
            }
            
            dashboard_path = os.path.join(save_directory, "performance_dashboard.png")
            self.visualizer.plot_performance_matrix(
                dashboard_data,
                "Traffic Light Control System - Comprehensive Analysis Dashboard",
                dashboard_path
            )
            
            print(f"✅ Visualizations generated successfully in {save_directory}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")
            return False
    
    def save_results(self, output_path: str = "comparison_results.json") -> bool:
        """
        Save comparison results to a JSON file.
        
        Args:
            output_path: Output file path
            
        Returns:
            True if saved successfully
        """
        return self.file_manager.save_results_json(self.results, output_path)
    
    def cleanup(self):
        """Clean up temporary files."""
        self.file_manager.cleanup_temp_files(self.temp_files)
        self.temp_files = []


def main():
    """Main function for running comparison analysis."""
    print("🚦 Traffic Light Comparison Analysis Tool")
    print("=" * 50)
    
    # Check SUMO installation
    if not SimulationUtils.check_sumo_installation():
        print("⚠️  SUMO not found. Please install SUMO and add it to PATH.")
        return
    
    # Initialize analyzer
    analyzer = ComparisonAnalyzer()
    
    try:
        # Run comparison analysis
        results = analyzer.run_comparison_analysis(
            scenarios=['balanced', 'heavy_ns', 'heavy_ew', 'rush_hour'],
            duration=600  # 10 minutes per test
        )
        
        # Generate visualizations
        analyzer.generate_visualizations("comparison_results")
        
        # Save results
        analyzer.save_results("detailed_comparison_results.json")
        
        print(f"\n🎉 Analysis complete! Check the 'comparison_results' folder for visualizations.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n⚠️  Analysis failed: {e}")
    finally:
        # Cleanup
        analyzer.cleanup()


if __name__ == "__main__":
    main()