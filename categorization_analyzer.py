#!/usr/bin/env python3
"""
Categorization Logic Analysis Tool
Analyzes the relationship between vehicle counts and traffic categories
without needing a full SUMO simulation.
"""

import os
import sys
import numpy as np

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from edge_traffic_controller import EdgeTrafficController

class CategorizationAnalyzer:
    def __init__(self):
        self.controller = EdgeTrafficController()
        
    def analyze_categorization_logic(self):
        """Analyze how vehicle counts map to categories and weights"""
        
        print("🔍 CATEGORIZATION LOGIC ANALYSIS")
        print("="*60)
        print("Analyzing how vehicle counts translate to traffic categories")
        print()
        
        # Test various vehicle count scenarios
        test_scenarios = [
            {'name': 'Empty Lanes', 'vehicle_counts': [0, 0, 0, 0]},
            {'name': 'Very Light (Real 30-60 vph)', 'vehicle_counts': [1, 0, 1, 0]},
            {'name': 'Light (Real 100-150 vph)', 'vehicle_counts': [2, 2, 3, 2]},
            {'name': 'Moderate (Real 300-400 vph)', 'vehicle_counts': [4, 5, 4, 3]},
            {'name': 'Heavy (Real 800-1200 vph)', 'vehicle_counts': [8, 9, 10, 8]},
            {'name': 'Critical (Real 1800+ vph)', 'vehicle_counts': [15, 12, 14, 13]},
            {'name': 'Mixed: Heavy North vs Light Others', 'vehicle_counts': [15, 1, 1, 1]},
            {'name': 'Mixed: Heavy East vs Empty Others', 'vehicle_counts': [0, 0, 20, 0]},
        ]
        
        for scenario in test_scenarios:
            self._analyze_scenario(scenario)
            
        print("\n" + "="*60)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*60)
        self._provide_recommendations()
    
    def _analyze_scenario(self, scenario):
        """Analyze a single scenario"""
        print(f"\n📊 {scenario['name']}")
        print("-" * 50)
        
        vehicle_counts = scenario['vehicle_counts']
        directions = ['north', 'south', 'east', 'west']
        
        # Analyze each direction
        categories = {}
        weights = {}
        
        print(f"{'Direction':<8} {'Vehicles':<8} {'Category':<10} {'Base Wt':<8} {'Modifier':<8} {'Final Wt':<8}")
        print("-" * 60)
        
        for i, direction in enumerate(directions):
            vehicles = vehicle_counts[i]
            
            # Test with no waiting time first
            category_info = self.controller.categorize_traffic_level(vehicles, 0, 10)
            categories[direction] = category_info
            weights[direction] = category_info['final_weight']
            
            print(f"{direction:<8} {vehicles:<8} {category_info['category']:<10} "
                  f"{category_info['base_weight']:<8.1f} {category_info['modifier']:<8.1f} "
                  f"{category_info['final_weight']:<8.1f}")
        
        # Calculate timing implications
        print(f"\nTiming Analysis:")
        
        # Create traffic data in the format expected by the function
        traffic_data = {}
        for i, direction in enumerate(directions):
            traffic_data[direction] = {
                'vehicles': vehicle_counts[i],
                'waiting_time': 0,  # No waiting time for basic analysis
                'speed': 10  # Normal speed
            }
        
        timing_result = self.controller.calculate_weighted_average_timing(traffic_data, 100)  # Use dummy time
        print(f"  North-South: {timing_result['north_south_green']:.1f}s")
        print(f"  East-West:   {timing_result['east_west_green']:.1f}s") 
        print(f"  Priority:    {timing_result['priority_direction']}")
        print(f"  Total Cycle: {timing_result['cycle_time']:.1f}s")
        
        # Analyze if this makes sense
        self._evaluate_scenario_logic(scenario, categories, timing_result)
    
    def _evaluate_scenario_logic(self, scenario, categories, timing_result):
        """Evaluate if the scenario results make logical sense"""
        print(f"\n💡 Logic Evaluation:")
        
        # Get max weight direction
        direction_weights = {}
        for direction in ['north', 'south', 'east', 'west']:
            direction_weights[direction] = categories[direction]['final_weight']
        
        max_weight_dir = max(direction_weights, key=direction_weights.get)
        min_weight_dir = min(direction_weights, key=direction_weights.get)
        
        max_weight = direction_weights[max_weight_dir]
        min_weight = direction_weights[min_weight_dir]
        
        print(f"  Heaviest: {max_weight_dir} (weight={max_weight:.1f}, category={categories[max_weight_dir]['category']})")
        print(f"  Lightest: {min_weight_dir} (weight={min_weight:.1f}, category={categories[min_weight_dir]['category']})")
        print(f"  Weight Ratio: {max_weight/max(min_weight, 0.1):.1f}:1")
        
        # Check if timing priority makes sense
        priority_direction = timing_result['priority_direction']
        if 'north' in priority_direction:
            priority_weight = (direction_weights['north'] + direction_weights['south']) / 2
            other_weight = (direction_weights['east'] + direction_weights['west']) / 2
        else:
            priority_weight = (direction_weights['east'] + direction_weights['west']) / 2
            other_weight = (direction_weights['north'] + direction_weights['south']) / 2
            
        print(f"  Priority makes sense: {priority_weight > other_weight}")
    
    def _provide_recommendations(self):
        """Provide recommendations for improving categorization"""
        print("Current Category Thresholds:")
        for category, config in sorted(self.controller.traffic_categories.items(), 
                                     key=lambda x: x[1]['threshold']):
            print(f"  {category:<10}: {config['threshold']}+ vehicles → weight {config['weight']}")
        
        print(f"\n🎯 ANALYSIS INSIGHTS:")
        print(f"1. Very light traffic (30-60 vph) typically results in 0-2 vehicles")
        print(f"   → Should be EMPTY (1.0) or LIGHT (2.0) category")
        print(f"   → Current thresholds: EMPTY=0+, LIGHT=2+")
        
        print(f"\n2. Light traffic (100-150 vph) typically results in 2-5 vehicles")
        print(f"   → Should be LIGHT (2.0) or MODERATE (3.0) category")
        print(f"   → Current thresholds: LIGHT=2+, MODERATE=4+")
        
        print(f"\n3. Heavy traffic (1000+ vph) typically results in 8+ vehicles")
        print(f"   → Should be HEAVY (5.0) or CRITICAL (6.0) category")
        print(f"   → Current thresholds: HEAVY=8+, CRITICAL=12+")
        
        print(f"\n🔧 POTENTIAL ISSUES:")
        print(f"1. Waiting time modifiers can significantly increase weights")
        print(f"   → Even EMPTY lanes get +0.5 weight if vehicles are waiting")
        print(f"   → This might cause over-prioritization of stuck vehicles")
        
        print(f"\n2. Speed modifiers add complexity")
        print(f"   → Stopped traffic gets +0.4 weight modifier")
        print(f"   → This can push LIGHT traffic into MODERATE+ territory")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"1. Consider adjusting waiting time thresholds")
        print(f"2. Review speed modifier impact")
        print(f"3. Test with actual flow scenarios to validate")
        
        # Test specific problematic cases
        print(f"\n🚨 PROBLEMATIC CASE ANALYSIS:")
        print(f"Case: Light traffic with some waiting (realistic scenario)")
        
        # Simulate light traffic with waiting
        test_cases = [
            {'vehicles': 2, 'waiting': 25, 'speed': 3, 'description': 'Light traffic, moderate waiting'},
            {'vehicles': 1, 'waiting': 40, 'speed': 0, 'description': 'Very light traffic, stopped'},
            {'vehicles': 3, 'waiting': 15, 'speed': 8, 'description': 'Light traffic, normal flow'},
        ]
        
        for case in test_cases:
            result = self.controller.categorize_traffic_level(case['vehicles'], case['waiting'], case['speed'])
            print(f"  {case['description']:<35}: {result['category']:<8} (weight={result['final_weight']:.1f})")
            if result['final_weight'] > 4.0:
                print(f"    ⚠️  Warning: Light traffic categorized as heavy due to modifiers!")

if __name__ == "__main__":
    analyzer = CategorizationAnalyzer()
    analyzer.analyze_categorization_logic()