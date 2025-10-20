"""
RL Predictive vs F1 Algorithm Comprehensive Comparison
=====================================================

This module compares your RL predictive algorithm with the proven F1 algorithm
across the same set of traffic scenarios to validate the predictive approach.
"""

import os
import sys
import json
import statistics
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime

# Import the RL predictive algorithm
from rl_predictive_algorithm import RLPredictiveController, ScenarioTestFramework

# Add f1 directory to path for F1 algorithm
f1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "f1")
sys.path.append(f1_path)

class F1AlgorithmSimulator:
    """Simulate F1 algorithm performance on the same scenarios."""
    
    def __init__(self):
        self.current_phase = 0
        self.phase_start_time = 0
        self.total_adaptations = 0
        self.adaptation_history = []
        
    def should_switch_phase(self, current_time: float, lane_vehicles: Dict[str, int], 
                          avg_waiting: float) -> bool:
        """F1 algorithm switching logic."""
        
        time_in_phase = current_time - self.phase_start_time
        total_vehicles = sum(lane_vehicles.values())
        
        # F1 Algorithm thresholds (from proven implementation)
        if total_vehicles >= 25:
            urgency = 'CRITICAL'
            min_phase_time = 45
            switch_threshold = 50
        elif total_vehicles >= 18:
            urgency = 'URGENT'
            min_phase_time = 40
            switch_threshold = 45
        elif total_vehicles >= 12:
            urgency = 'NORMAL'
            min_phase_time = 35
            switch_threshold = 40
        elif total_vehicles >= 6:
            urgency = 'LIGHT'
            min_phase_time = 35  # F1 uses longer times for light traffic
            switch_threshold = 35
        else:
            urgency = 'MINIMAL'
            min_phase_time = 35  # F1 is conservative
            switch_threshold = 35
        
        # F1 switching conditions
        if time_in_phase >= min_phase_time:
            if urgency in ['CRITICAL', 'URGENT']:
                if total_vehicles > 15 or avg_waiting > 40:
                    self.total_adaptations += 1
                    return True
            elif urgency in ['NORMAL', 'LIGHT']:
                if total_vehicles > 8 or avg_waiting > 30:
                    self.total_adaptations += 1
                    return True
            else:  # MINIMAL
                if avg_waiting > 25:
                    self.total_adaptations += 1
                    return True
        
        # Maximum phase time limit
        if time_in_phase >= switch_threshold:
            self.total_adaptations += 1
            return True
            
        return False
    
    def switch_phase(self, current_time: float):
        """Switch to next phase."""
        self.current_phase = (self.current_phase + 1) % 4
        self.phase_start_time = current_time

class ComprehensiveComparison:
    """Compare RL Predictive algorithm with F1 algorithm."""
    
    def __init__(self):
        self.results_dir = os.path.join(os.path.dirname(__file__), "comprehensive_comparison")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def simulate_f1_algorithm(self, scenario_name: str, scenario_config: Dict) -> List[Dict]:
        """Simulate F1 algorithm on scenario."""
        
        print(f"🔧 F1 Algorithm simulation: {scenario_name}")
        
        f1_controller = F1AlgorithmSimulator()
        simulation_data = []
        current_time = 0
        time_step = 0.5
        
        # Use same traffic generation as RL test
        framework = ScenarioTestFramework()
        
        while current_time < scenario_config['duration']:
            # Generate same traffic pattern
            lane_vehicles = framework.generate_scenario_traffic(scenario_name, current_time, scenario_config)
            
            # Calculate metrics
            total_vehicles = sum(lane_vehicles.values())
            avg_waiting = framework.calculate_waiting_time(lane_vehicles, current_time)
            avg_speed = framework.calculate_avg_speed(total_vehicles, avg_waiting)
            
            # F1 algorithm decision
            if f1_controller.should_switch_phase(current_time, lane_vehicles, avg_waiting):
                f1_controller.switch_phase(current_time)
            
            # Record data
            data_point = {
                'time': current_time,
                'scenario': scenario_name,
                'total_vehicles': total_vehicles,
                'lane_vehicles': lane_vehicles.copy(),
                'avg_waiting_time': avg_waiting,
                'avg_speed': avg_speed,
                'current_phase': ['North', 'East', 'South', 'West'][f1_controller.current_phase],
                'adaptations': f1_controller.total_adaptations,
                'algorithm': 'f1_proven'
            }
            
            simulation_data.append(data_point)
            current_time += time_step
        
        return simulation_data
    
    def run_comparison_test(self):
        """Run comprehensive comparison between both algorithms."""
        
        print("🏆 RL PREDICTIVE vs F1 ALGORITHM COMPARISON")
        print("=" * 80)
        print("Testing both algorithms on identical traffic scenarios")
        print()
        
        # Define scenarios
        test_scenarios = {
            'initial_low': {
                'duration': 10,
                'base_time_scenario': 'initial_low',
                'description': 'Initial low traffic - YOUR: 20s base, F1: 35s adaptive'
            },
            'heavy_north': {
                'duration': 10,
                'base_time_scenario': 'heavy_north',
                'description': 'Heavy North - YOUR: 50s N/30s others, F1: adaptive 40-45s'
            },
            'heavy_east': {
                'duration': 10,
                'base_time_scenario': 'heavy_east',
                'description': 'Heavy East - YOUR: 50s E/30s others, F1: adaptive 40-45s'
            },
            'heavy_north_east': {
                'duration': 10,
                'base_time_scenario': 'heavy_north_east',
                'description': 'Heavy N-E - YOUR: 50s both/30s others, F1: adaptive'
            },
            'all_heavy': {
                'duration': 10,
                'base_time_scenario': 'all_heavy',
                'description': 'All heavy - YOUR: 45s equal, F1: adaptive 45-50s'
            },
            'gradual_slowdown': {
                'duration': 10,
                'base_time_scenario': 'gradual_slowdown',
                'description': 'Slowdown - YOUR: 30s moderate, F1: adaptive'
            },
            'evening_light': {
                'duration': 10,
                'base_time_scenario': 'evening_light',
                'description': 'Evening light - YOUR: 20s base, F1: 35s adaptive'
            }
        }
        
        # Test RL Predictive Algorithm
        print("🤖 TESTING RL PREDICTIVE ALGORITHM...")
        rl_framework = ScenarioTestFramework()
        rl_results = rl_framework.run_comprehensive_test()
        
        print("\\n🔧 TESTING F1 PROVEN ALGORITHM...")
        f1_results = {}
        
        for scenario_name, config in test_scenarios.items():
            f1_data = self.simulate_f1_algorithm(scenario_name, config)
            f1_results[scenario_name] = f1_data
            print(f"✅ F1 {scenario_name}: {len(f1_data)} data points")
        
        return rl_results, f1_results, test_scenarios
    
    def analyze_comparison(self, rl_results: Dict, f1_results: Dict, scenarios: Dict):
        """Analyze and compare both algorithms' performance."""
        
        print("\\n" + "=" * 80)
        print("📊 COMPREHENSIVE ALGORITHM COMPARISON ANALYSIS")
        print("=" * 80)
        
        comparison_results = {}
        
        for scenario_name in scenarios.keys():
            if scenario_name not in rl_results or scenario_name not in f1_results:
                continue
                
            rl_data = rl_results[scenario_name]
            f1_data = f1_results[scenario_name]
            
            if not rl_data or not f1_data:
                continue
            
            # Calculate metrics for both algorithms
            rl_metrics = {
                'avg_wait': statistics.mean([d['avg_waiting_time'] for d in rl_data]),
                'avg_speed': statistics.mean([d['avg_speed'] for d in rl_data]),
                'total_adaptations': rl_data[-1]['adaptations'],
                'avg_vehicles': statistics.mean([d['total_vehicles'] for d in rl_data])
            }
            
            f1_metrics = {
                'avg_wait': statistics.mean([d['avg_waiting_time'] for d in f1_data]),
                'avg_speed': statistics.mean([d['avg_speed'] for d in f1_data]),
                'total_adaptations': f1_data[-1]['adaptations'],
                'avg_vehicles': statistics.mean([d['total_vehicles'] for d in f1_data])
            }
            
            # Calculate performance differences
            wait_improvement = ((f1_metrics['avg_wait'] - rl_metrics['avg_wait']) / f1_metrics['avg_wait']) * 100
            speed_improvement = ((rl_metrics['avg_speed'] - f1_metrics['avg_speed']) / f1_metrics['avg_speed']) * 100
            adaptation_ratio = rl_metrics['total_adaptations'] / max(f1_metrics['total_adaptations'], 1)
            
            comparison_results[scenario_name] = {
                'rl_metrics': rl_metrics,
                'f1_metrics': f1_metrics,
                'wait_improvement': wait_improvement,
                'speed_improvement': speed_improvement,
                'adaptation_ratio': adaptation_ratio,
                'winner': 'RL_Predictive' if wait_improvement > 0 else 'F1_Proven'
            }
            
            # Print scenario comparison
            print(f"\\n📈 {scenario_name.upper()} COMPARISON:")
            print(f"   RL Predictive: {rl_metrics['avg_wait']:.1f}s wait, {rl_metrics['avg_speed']:.1f} m/s, {rl_metrics['total_adaptations']} adapt")
            print(f"   F1 Proven:     {f1_metrics['avg_wait']:.1f}s wait, {f1_metrics['avg_speed']:.1f} m/s, {f1_metrics['total_adaptations']} adapt")
            
            if wait_improvement > 0:
                print(f"   🏆 RL WINS: {wait_improvement:+.1f}% better wait time")
            else:
                print(f"   🏆 F1 WINS: {abs(wait_improvement):+.1f}% better wait time")
        
        # Overall comparison
        self.print_overall_comparison(comparison_results)
        
        return comparison_results
    
    def print_overall_comparison(self, comparison_results: Dict):
        """Print overall comparison summary."""
        
        print("\\n" + "🏆" * 20)
        print("OVERALL ALGORITHM BATTLE RESULTS")
        print("🏆" * 20)
        
        rl_wins = sum(1 for r in comparison_results.values() if r['winner'] == 'RL_Predictive')
        f1_wins = sum(1 for r in comparison_results.values() if r['winner'] == 'F1_Proven')
        total_scenarios = len(comparison_results)
        
        print(f"\\n📊 SCENARIO BATTLE RESULTS:")
        print(f"   RL Predictive Algorithm:  {rl_wins}/{total_scenarios} scenarios won")
        print(f"   F1 Proven Algorithm:      {f1_wins}/{total_scenarios} scenarios won")
        
        # Calculate overall averages
        all_rl_waits = [r['rl_metrics']['avg_wait'] for r in comparison_results.values()]
        all_f1_waits = [r['f1_metrics']['avg_wait'] for r in comparison_results.values()]
        
        overall_rl_wait = statistics.mean(all_rl_waits)
        overall_f1_wait = statistics.mean(all_f1_waits)
        overall_improvement = ((overall_f1_wait - overall_rl_wait) / overall_f1_wait) * 100
        
        print(f"\\n🎯 OVERALL PERFORMANCE:")
        print(f"   RL Predictive Average:    {overall_rl_wait:.1f}s wait time")
        print(f"   F1 Proven Average:        {overall_f1_wait:.1f}s wait time")
        print(f"   Overall Improvement:      {overall_improvement:+.1f}%")
        
        # Adaptation efficiency
        all_rl_adaptations = [r['rl_metrics']['total_adaptations'] for r in comparison_results.values()]
        all_f1_adaptations = [r['f1_metrics']['total_adaptations'] for r in comparison_results.values()]
        
        total_rl_adaptations = sum(all_rl_adaptations)
        total_f1_adaptations = sum(all_f1_adaptations)
        
        print(f"\\n⚡ ADAPTATION ANALYSIS:")
        print(f"   RL Total Adaptations:     {total_rl_adaptations}")
        print(f"   F1 Total Adaptations:     {total_f1_adaptations}")
        print(f"   Adaptation Ratio:         {total_rl_adaptations/max(total_f1_adaptations,1):.1f}x")
        
        # Final verdict
        print(f"\\n🏁 FINAL VERDICT:")
        
        if rl_wins > f1_wins and overall_improvement > 5:
            print("🎉 BREAKTHROUGH! RL Predictive Algorithm is SUPERIOR!")
            print("   Your predictive base time approach with dynamic adaptation WORKS!")
            print("   ✅ Hypothesis VALIDATED: Predictive base times + dynamic adaptation beats reactive approach")
            
        elif rl_wins > f1_wins:
            print("🎯 SUCCESS! RL Predictive Algorithm wins more scenarios!")
            print("   Your approach shows promise with scenario-specific advantages")
            print("   👍 Predictive base times provide strategic benefits")
            
        elif overall_improvement > -5:
            print("🤝 COMPETITIVE! Both algorithms show comparable performance!")
            print("   RL Predictive demonstrates viability as alternative approach")
            print("   📊 Different strengths in different scenarios")
            
        else:
            print("📚 LEARNING EXPERIENCE! F1 algorithm remains superior overall")
            print("   However, RL Predictive shows benefits in specific scenarios")
            print("   💡 Hybrid approach could combine strengths of both")
        
        # Scenario-specific insights
        light_traffic_scenarios = ['initial_low', 'evening_light']
        heavy_traffic_scenarios = ['heavy_north', 'heavy_east', 'all_heavy']
        
        light_rl_wins = sum(1 for s in light_traffic_scenarios 
                          if s in comparison_results and comparison_results[s]['winner'] == 'RL_Predictive')
        heavy_rl_wins = sum(1 for s in heavy_traffic_scenarios 
                          if s in comparison_results and comparison_results[s]['winner'] == 'RL_Predictive')
        
        print(f"\\n🔍 SCENARIO TYPE ANALYSIS:")
        print(f"   Light Traffic: RL wins {light_rl_wins}/{len(light_traffic_scenarios)} scenarios")
        print(f"   Heavy Traffic: RL wins {heavy_rl_wins}/{len(heavy_traffic_scenarios)} scenarios")
        
        if light_rl_wins > heavy_rl_wins:
            print("   ✅ RL excels in light traffic - predictive short phases work!")
        elif heavy_rl_wins > light_rl_wins:
            print("   ✅ RL excels in heavy traffic - predictive prioritization works!")
        else:
            print("   📊 Balanced performance across traffic types")
    
    def create_comparison_visualizations(self, rl_results: Dict, f1_results: Dict, 
                                       comparison_analysis: Dict):
        """Create comprehensive comparison visualizations."""
        
        scenarios = list(comparison_analysis.keys())
        
        # Extract metrics for plotting
        rl_waits = [comparison_analysis[s]['rl_metrics']['avg_wait'] for s in scenarios]
        f1_waits = [comparison_analysis[s]['f1_metrics']['avg_wait'] for s in scenarios]
        rl_speeds = [comparison_analysis[s]['rl_metrics']['avg_speed'] for s in scenarios]
        f1_speeds = [comparison_analysis[s]['f1_metrics']['avg_speed'] for s in scenarios]
        rl_adaptations = [comparison_analysis[s]['rl_metrics']['total_adaptations'] for s in scenarios]
        f1_adaptations = [comparison_analysis[s]['f1_metrics']['total_adaptations'] for s in scenarios]
        
        # Create comprehensive comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Wait time comparison
        x_pos = range(len(scenarios))
        width = 0.35
        
        ax1.bar([x - width/2 for x in x_pos], rl_waits, width, label='RL Predictive', color='blue', alpha=0.7)
        ax1.bar([x + width/2 for x in x_pos], f1_waits, width, label='F1 Proven', color='green', alpha=0.7)
        ax1.set_title('Average Wait Time by Scenario')
        ax1.set_xlabel('Scenarios')
        ax1.set_ylabel('Wait Time (seconds)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([s.replace('_', '\\n') for s in scenarios], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Speed comparison
        ax2.bar([x - width/2 for x in x_pos], rl_speeds, width, label='RL Predictive', color='blue', alpha=0.7)
        ax2.bar([x + width/2 for x in x_pos], f1_speeds, width, label='F1 Proven', color='green', alpha=0.7)
        ax2.set_title('Average Speed by Scenario')
        ax2.set_xlabel('Scenarios')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([s.replace('_', '\\n') for s in scenarios], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Adaptations comparison
        ax3.bar([x - width/2 for x in x_pos], rl_adaptations, width, label='RL Predictive', color='blue', alpha=0.7)
        ax3.bar([x + width/2 for x in x_pos], f1_adaptations, width, label='F1 Proven', color='green', alpha=0.7)
        ax3.set_title('Total Adaptations by Scenario')
        ax3.set_xlabel('Scenarios')
        ax3.set_ylabel('Number of Adaptations')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([s.replace('_', '\\n') for s in scenarios], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance improvement
        improvements = [comparison_analysis[s]['wait_improvement'] for s in scenarios]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        ax4.bar(x_pos, improvements, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('RL vs F1 Performance Improvement (%)')
        ax4.set_xlabel('Scenarios')
        ax4.set_ylabel('Improvement (%)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([s.replace('_', '\\n') for s in scenarios], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_file = os.path.join(self.results_dir, "rl_vs_f1_comprehensive_comparison.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"\\n📊 Comprehensive comparison chart saved: {chart_file}")
        
        return chart_file
    
    def run_complete_comparison(self):
        """Run the complete comparison analysis."""
        
        # Run comparison test
        rl_results, f1_results, scenarios = self.run_comparison_test()
        
        # Analyze results
        comparison_analysis = self.analyze_comparison(rl_results, f1_results, scenarios)
        
        # Create visualizations
        self.create_comparison_visualizations(rl_results, f1_results, comparison_analysis)
        
        # Save all results
        complete_results = {
            'rl_results': rl_results,
            'f1_results': f1_results,
            'comparison_analysis': comparison_analysis,
            'test_scenarios': scenarios,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.results_dir, "complete_comparison_results.json"), 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"\\n📁 Complete comparison results saved to: {self.results_dir}")
        
        return complete_results

if __name__ == "__main__":
    comparison = ComprehensiveComparison()
    results = comparison.run_complete_comparison()