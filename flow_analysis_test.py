#!/usr/bin/env python3
"""
Quick test to analyze how different flow rates translate to actual vehicle counts
and verify our categorization thresholds make sense.
"""

import os
import sys
import subprocess
import time
import traci
import numpy as np
from collections import defaultdict, deque

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from edge_traffic_controller import EdgeTrafficController

class FlowAnalysisTest:
    def __init__(self):
        self.sumo_cmd = ["sumo", "-c", "demo.sumocfg", "--no-warnings", "--no-step-log"]
        self.edge_controller = None
        
    def test_flow_patterns(self):
        """Test different flow patterns and see actual vehicle accumulation"""
        
        # Test scenarios with different flow rates (vehicles per hour)
        test_scenarios = [
            {
                'name': 'Very Light Traffic', 
                'description': 'Should result in EMPTY/LIGHT categories',
                'flows': {'north': 50, 'south': 40, 'east': 60, 'west': 30}
            },
            {
                'name': 'Light Traffic', 
                'description': 'Should result in LIGHT/MODERATE categories', 
                'flows': {'north': 150, 'south': 120, 'east': 140, 'west': 100}
            },
            {
                'name': 'Heavy North Only', 
                'description': 'North should be HEAVY/CRITICAL, others EMPTY/LIGHT',
                'flows': {'north': 1800, 'south': 50, 'east': 40, 'west': 30}
            },
            {
                'name': 'Balanced Heavy', 
                'description': 'All directions should be HEAVY/CRITICAL',
                'flows': {'north': 1200, 'south': 1100, 'east': 1300, 'west': 1000}
            },
        ]
        
        for scenario in test_scenarios:
            print(f"\n{'='*70}")
            print(f"Testing Scenario: {scenario['name']}")
            print(f"Expected: {scenario['description']}")
            print(f"Flow Rates (vph): {scenario['flows']}")
            print(f"{'='*70}")
            
            self._run_scenario_test(scenario)
            
    def _run_scenario_test(self, scenario):
        """Run a single scenario test for 5 minutes"""
        try:
            # Start SUMO
            traci.start(self.sumo_cmd)
            
            # Initialize edge controller
            self.edge_controller = EdgeTrafficController()
            
            # Create flows based on scenario
            self._create_flows(scenario['flows'])
            
            # Run for 5 minutes (300 seconds) to see steady state
            simulation_duration = 300
            data_points = []
            
            for step in range(simulation_duration):
                current_time = traci.simulation.getTime()
                
                # Collect traffic data every 30 seconds
                if step % 30 == 0 and step > 60:  # Skip first minute for warm-up
                    traffic_data = self._collect_traffic_data()
                    data_points.append({
                        'time': current_time,
                        'data': traffic_data
                    })
                
                # Step simulation
                traci.simulationStep()
                
            # Analyze results
            self._analyze_scenario_data(scenario, data_points)
            
            # Close SUMO
            traci.close()
            
        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
            if traci.isLoaded():
                traci.close()
    
    def _create_flows(self, flow_rates):
        """Create SUMO flows based on the flow rates dictionary"""
        
        # Convert vehicles per hour to vehicles per second for SUMO
        for direction, vph in flow_rates.items():
            vps = vph / 3600.0  # vehicles per second
            
            # Define routes for each direction
            if direction == 'north':
                route_id = f"route_{direction}"
                traci.route.add(route_id, ['E1', 'E1.200'])
            elif direction == 'south':
                route_id = f"route_{direction}"
                traci.route.add(route_id, ['-E1', '-E1.238'])
            elif direction == 'east':
                route_id = f"route_{direction}"
                traci.route.add(route_id, ['E0', 'E0.319'])
            elif direction == 'west':
                route_id = f"route_{direction}"
                traci.route.add(route_id, ['-E0', '-E0.254'])
            
            # Create flow with appropriate probability
            if vps > 0:
                # Calculate probability to achieve desired flow rate
                probability = min(1.0, vps)
                
                flow_id = f"flow_{direction}"
                traci.flow.add(
                    flowID=flow_id,
                    routeID=route_id,
                    vehsPerHour=vph,
                    departLane="best",
                    departSpeed="max"
                )
    
    def _collect_traffic_data(self):
        """Collect current traffic data from all directions"""
        traffic_data = {}
        
        # Define the edges for each direction
        direction_edges = {
            'north': 'E1',
            'south': '-E1', 
            'east': 'E0',
            'west': '-E0'
        }
        
        for direction, edge_id in direction_edges.items():
            try:
                # Count vehicles on edge
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                
                # Get waiting time and speed
                lanes = traci.edge.getLaneNumber(edge_id)
                total_waiting = 0
                total_speed = 0
                
                for lane_idx in range(lanes):
                    lane_id = f"{edge_id}_{lane_idx}"
                    total_waiting += traci.lane.getWaitingTime(lane_id)
                    total_speed += traci.lane.getLastStepMeanSpeed(lane_id)
                
                avg_waiting = total_waiting / max(lanes, 1)
                avg_speed = total_speed / max(lanes, 1)
                
                # Get categorization
                category_info = self.edge_controller.categorize_traffic_level(
                    vehicle_count, avg_waiting, avg_speed
                )
                
                traffic_data[direction] = {
                    'vehicles': vehicle_count,
                    'waiting_time': avg_waiting,
                    'speed': avg_speed,
                    'category': category_info['category'],
                    'base_weight': category_info['base_weight'],
                    'modifier': category_info['modifier'],
                    'final_weight': category_info['final_weight']
                }
                
            except Exception as e:
                traffic_data[direction] = {
                    'vehicles': 0,
                    'waiting_time': 0,
                    'speed': 0,
                    'category': 'ERROR',
                    'base_weight': 0,
                    'modifier': 1.0,
                    'final_weight': 0
                }
        
        return traffic_data
    
    def _analyze_scenario_data(self, scenario, data_points):
        """Analyze and display the collected data"""
        print(f"\nAnalysis Results:")
        print("=" * 50)
        
        if not data_points:
            print("No data collected!")
            return
        
        # Calculate averages over all data points
        direction_stats = defaultdict(lambda: {
            'vehicles': [], 'weights': [], 'categories': [], 
            'waiting': [], 'speeds': [], 'modifiers': []
        })
        
        for point in data_points:
            for direction, data in point['data'].items():
                direction_stats[direction]['vehicles'].append(data['vehicles'])
                direction_stats[direction]['weights'].append(data['final_weight'])
                direction_stats[direction]['categories'].append(data['category'])
                direction_stats[direction]['waiting'].append(data['waiting_time'])
                direction_stats[direction]['speeds'].append(data['speed'])
                direction_stats[direction]['modifiers'].append(data['modifier'])
        
        # Display results
        print(f"{'Direction':<8} {'Flow(vph)':<10} {'AvgVeh':<8} {'MaxVeh':<8} {'Category':<10} {'Weight':<8} {'Waiting':<8} {'Speed':<8}")
        print("-" * 80)
        
        for direction in ['north', 'south', 'east', 'west']:
            stats = direction_stats[direction]
            if stats['vehicles']:
                flow_rate = scenario['flows'][direction]
                avg_vehicles = np.mean(stats['vehicles'])
                max_vehicles = np.max(stats['vehicles'])
                avg_weight = np.mean(stats['weights'])
                avg_waiting = np.mean(stats['waiting'])
                avg_speed = np.mean(stats['speeds'])
                avg_modifier = np.mean(stats['modifiers'])
                most_common_category = max(set(stats['categories']), key=stats['categories'].count)
                
                print(f"{direction.upper():<8} {flow_rate:<10} {avg_vehicles:<8.1f} {max_vehicles:<8} "
                      f"{most_common_category:<10} {avg_weight:<8.1f} {avg_waiting:<8.1f} {avg_speed:<8.1f}")
        
        # Show timing implications
        print(f"\nTiming Analysis:")
        print("-" * 30)
        
        # Get sample weights for timing calculation
        sample_data = data_points[-1]['data'] if data_points else {}
        if sample_data:
            weights = {
                'north': sample_data['north']['final_weight'],
                'south': sample_data['south']['final_weight'],
                'east': sample_data['east']['final_weight'],
                'west': sample_data['west']['final_weight']
            }
            
            timing_result = self.edge_controller.calculate_weighted_average_timing(weights)
            
            print(f"North-South: {timing_result['north_south_time']:.1f}s")
            print(f"East-West:   {timing_result['east_west_time']:.1f}s")
            print(f"Priority:    {timing_result['priority_direction']}")
            
            # Show if this matches expectations
            expected = scenario['description']
            print(f"Expected:    {expected}")

if __name__ == "__main__":
    print("🔍 TRAFFIC FLOW ANALYSIS TEST")
    print("Analyzing how different flow rates translate to vehicle counts and categories")
    print("This helps verify if our categorization thresholds are appropriate")
    
    test = FlowAnalysisTest()
    test.test_flow_patterns()
    
    print(f"\n✅ Flow analysis complete!")
    print(f"\nInterpretation Guide:")
    print(f"- Very light (30-60 vph) should result in 0-2 vehicles → EMPTY/LIGHT")
    print(f"- Light (100-150 vph) should result in 2-4 vehicles → LIGHT/MODERATE") 
    print(f"- Heavy (1000+ vph) should result in 8+ vehicles → HEAVY/CRITICAL")
    print(f"- If categories don't match expectations, thresholds need adjustment")