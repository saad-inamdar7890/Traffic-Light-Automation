"""
Edge Traffic Controller - Phase 1 of Smart Traffic Management
===========================================================

This module implements the edge-level traffic light controller that:
1. Uses 30s base timing for each phase (configurable from cloud)
2. Adjusts green time between 10s (min) to 50s (max) based on real-time traffic density
3. Makes gradual timing changes using statistical calculations
4. Analyzes traffic density across all lanes at the edge/junction
5. Prepares for Phase 2: RL model integration for base timing optimization

Architecture:
- Edge Controller: Real-time traffic density analysis and adaptive timing
- Cloud Integration: Ready for RL model to provide optimized base timings
"""

import traci
import statistics
import math
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

class EdgeTrafficController:
    def __init__(self, junction_id="J4", base_green_time=30):
        """
        Initialize Edge Traffic Controller for Phase 1
        
        Args:
            junction_id: Junction identifier in SUMO
            base_green_time: Base timing from cloud (Phase 2) or default 30s
        """
        self.junction_id = junction_id
        
        # Edge Algorithm Configuration (Your Project Specs)
        self.base_green_time = base_green_time  # 30s default, will come from cloud in Phase 2
        self.min_green_time = 10  # Minimum green time
        self.max_green_time = 50  # Maximum green time
        
        # Gradual Change Parameters
        self.max_change_percent = 0.25  # Maximum 25% change per adjustment
        self.adaptation_interval = 15   # Check every 15 seconds
        
        # Traffic Level Categorization System
        self.traffic_categories = {
            'EMPTY': {'threshold': 0, 'weight': 1.0, 'description': 'No vehicles'},
            'LIGHT': {'threshold': 2, 'weight': 2.0, 'description': 'Few vehicles'},
            'MODERATE': {'threshold': 4, 'weight': 3.0, 'description': 'Normal traffic'},
            'NORMAL': {'threshold': 6, 'weight': 4.0, 'description': 'Regular flow'},
            'HEAVY': {'threshold': 8, 'weight': 5.0, 'description': 'Dense traffic'},
            'CRITICAL': {'threshold': 12, 'weight': 6.0, 'description': 'Very heavy traffic'}
        }
        
        # Category thresholds for waiting time consideration
        self.waiting_time_threshold = 20  # Seconds
        self.speed_threshold_low = 5.0    # m/s (18 km/h)
        
        # State Tracking
        self.current_phase = 0
        self.phase_start_time = 0
        self.last_adaptation_time = 0
        self.current_green_duration = base_green_time
        
        # Traffic Density History (for statistical calculations)
        self.density_history = defaultdict(lambda: deque(maxlen=10))
        self.waiting_history = defaultdict(lambda: deque(maxlen=10))
        self.speed_history = defaultdict(lambda: deque(maxlen=10))
        
        # Performance Metrics
        self.adaptations_count = 0
        self.timing_adjustments = []
        
        # Phase definitions for 4-way intersection
        self.phase_directions = {
            0: "yellow_transition",
            1: "north_south",      # E1, -E1 (North-South)
            2: "yellow_transition", 
            3: "east_west",        # E0, -E0 (East-West)
            4: "yellow_transition",
            5: "north_south_left", # Left turns N-S
            6: "yellow_transition",
            7: "east_west_left"    # Left turns E-W
        }
        
        # Advanced timing management
        self.max_red_time = 120  # Maximum 2 minutes red time per lane
        self.min_green_time_per_lane = 8  # Minimum green per lane
        self.yellow_time = 3  # Yellow transition time
        self.red_clearance_time = 1  # All-red clearance
        
        # Lane timing tracking
        self.lane_last_green_time = {}
        self.lane_green_durations = {}
        self.lane_pressures = {}
        self.total_cycle_time = 0
        
        print(f"🚦 Edge Traffic Controller initialized")
        print(f"   Base Green Time: {self.base_green_time}s")
        print(f"   Timing Range: {self.min_green_time}s - {self.max_green_time}s")
        print(f"   Max Change: {self.max_change_percent*100}% per adjustment")
    
    def collect_lane_traffic_density(self):
        """
        Collect traffic density data from all lanes at the edge/junction
        Returns comprehensive density analysis for statistical calculations
        """
        try:
            traffic_data = {}
            
            # Define edge-to-direction mapping
            edge_directions = {
                "E0": "east_approach",     # From East
                "-E0": "west_approach",    # From West  
                "E1": "north_approach",    # From North
                "-E1": "south_approach"    # From South
            }
            
            for edge_id, direction in edge_directions.items():
                try:
                    # Get all lanes for this edge
                    lanes = traci.edge.getLaneNumber(edge_id)
                    total_vehicles = 0
                    total_waiting_time = 0
                    total_speed = 0
                    lane_count = 0
                    
                    for lane_idx in range(lanes):
                        lane_id = f"{edge_id}_{lane_idx}"
                        
                        # Traffic density metrics
                        vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
                        waiting_time = traci.lane.getWaitingTime(lane_id)
                        mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                        lane_length = traci.lane.getLength(lane_id)
                        
                        # Calculate density (vehicles per 100m)
                        density = (vehicles / (lane_length / 100)) if lane_length > 0 else 0
                        
                        total_vehicles += vehicles
                        total_waiting_time += waiting_time
                        total_speed += mean_speed
                        lane_count += 1
                    
                    # Calculate average metrics for this direction
                    avg_density = total_vehicles / max(lane_count, 1)
                    avg_waiting = total_waiting_time / max(lane_count, 1) 
                    avg_speed = total_speed / max(lane_count, 1)
                    
                    traffic_data[direction] = {
                        'vehicles': total_vehicles,
                        'density': avg_density,
                        'waiting_time': avg_waiting,
                        'speed': avg_speed,
                        'lanes': lane_count
                    }
                    
                    # Store in history for statistical analysis
                    self.density_history[direction].append(avg_density)
                    self.waiting_history[direction].append(avg_waiting)
                    self.speed_history[direction].append(avg_speed)
                    
                except Exception as lane_error:
                    # Handle missing edges gracefully
                    traffic_data[direction] = {
                        'vehicles': 0, 'density': 0, 'waiting_time': 0, 
                        'speed': 0, 'lanes': 0
                    }
            
            return traffic_data
            
        except Exception as e:
            print(f"⚠️  Error collecting traffic density: {e}")
            return {}
    
    def categorize_traffic_level(self, vehicles, waiting_time=0, speed=0):
        """
        Categorize traffic level based on vehicle count, waiting time, and speed
        
        Args:
            vehicles: Number of vehicles in the lane
            waiting_time: Average waiting time (optional modifier)
            speed: Average speed (optional modifier)
            
        Returns:
            dict: Category information with name, weight, and modifiers
        """
        # Base categorization by vehicle count
        base_category = 'EMPTY'
        for category, config in sorted(self.traffic_categories.items(), 
                                     key=lambda x: x[1]['threshold'], reverse=True):
            if vehicles >= config['threshold']:
                base_category = category
                break
        
        base_weight = self.traffic_categories[base_category]['weight']
        
        # Apply modifiers based on waiting time and speed
        modifier = 1.0
        
        # Waiting time modifier (increases urgency)
        if waiting_time > self.waiting_time_threshold:
            modifier += 0.5  # Increase weight for high waiting time
        elif waiting_time > self.waiting_time_threshold / 2:
            modifier += 0.2  # Slight increase for moderate waiting
        
        # Speed modifier (lower speed indicates congestion)
        if speed > 0:
            if speed < self.speed_threshold_low:
                modifier += 0.3  # Increase weight for slow traffic
            elif speed < self.speed_threshold_low * 2:
                modifier += 0.1  # Slight increase for moderate speed
        else:
            modifier += 0.4  # Stopped traffic gets higher priority
        
        # Calculate final weight
        final_weight = base_weight * modifier
        
        return {
            'category': base_category,
            'base_weight': base_weight,
            'modifier': modifier,
            'final_weight': final_weight,
            'vehicles': vehicles,
            'waiting_time': waiting_time,
            'speed': speed
        }
    
    def calculate_lane_categories_and_weights(self, traffic_data):
        """
        Calculate traffic categories and weights for each lane/direction
        
        Returns:
            dict: Lane categories, weights, and overall scenario analysis
        """
        if not traffic_data:
            return {}, 0.0, 'EMPTY'
        
        lane_analysis = {}
        total_weight = 0.0
        lane_weights = []
        
        # Categorize each lane
        for direction, data in traffic_data.items():
            category_info = self.categorize_traffic_level(
                vehicles=data['vehicles'],
                waiting_time=data['waiting_time'],
                speed=data['speed']
            )
            
            lane_analysis[direction] = category_info
            total_weight += category_info['final_weight']
            lane_weights.append(category_info['final_weight'])
        
        # Calculate average weight across all lanes
        avg_weight = statistics.mean(lane_weights) if lane_weights else 0.0
        
        # Determine overall scenario category
        if avg_weight >= 5.0:
            scenario_category = 'CRITICAL'
        elif avg_weight >= 4.0:
            scenario_category = 'HEAVY'
        elif avg_weight >= 3.0:
            scenario_category = 'NORMAL'
        elif avg_weight >= 2.0:
            scenario_category = 'MODERATE'
        elif avg_weight >= 1.0:
            scenario_category = 'LIGHT'
        else:
            scenario_category = 'EMPTY'
        
        return lane_analysis, avg_weight, scenario_category
    
    def calculate_weighted_average_timing(self, traffic_data, current_time):
        """
        Calculate green timing using weighted average categorization approach
        
        Key Principles:
        1. Categorize each lane's traffic level (EMPTY to CRITICAL)
        2. Calculate average weight across all lanes
        3. Compare each direction's weight to average
        4. Adjust green time based on deviation from average
        5. Ensure safety constraints and gradual changes
        """
        if not traffic_data:
            return self._get_default_timing()
        
        # Get lane categories and weights
        lane_analysis, avg_weight, scenario_category = self.calculate_lane_categories_and_weights(traffic_data)
        
        # Calculate direction weights (NS vs EW)
        ns_weights = []
        ew_weights = []
        
        for direction, analysis in lane_analysis.items():
            if direction in ['north', 'south', 'north_approach', 'south_approach']:
                ns_weights.append(analysis['final_weight'])
            elif direction in ['east', 'west', 'east_approach', 'west_approach']:
                ew_weights.append(analysis['final_weight'])
        
        ns_avg_weight = statistics.mean(ns_weights) if ns_weights else 0.0
        ew_avg_weight = statistics.mean(ew_weights) if ew_weights else 0.0
        total_avg_weight = (ns_avg_weight + ew_avg_weight) / 2
        
        # Calculate timing based on weighted deviations with FAST-CYCLE OPTIMIZATION
        timing_plan = self._calculate_deviation_based_timing(
            ns_avg_weight, ew_avg_weight, total_avg_weight, scenario_category
        )
        
        # Apply FAST-CYCLE OPTIMIZATION for low traffic scenarios
        if scenario_category in ['EMPTY', 'LIGHT']:
            timing_plan = self._apply_fast_cycle_optimization(timing_plan, scenario_category, total_avg_weight)
        
        # Add analysis metadata
        timing_plan.update({
            'lane_analysis': lane_analysis,
            'avg_weight': avg_weight,
            'scenario_category': scenario_category,
            'ns_weight': ns_avg_weight,
            'ew_weight': ew_avg_weight
        })
        
        return timing_plan
    
    def _calculate_deviation_based_timing(self, ns_weight, ew_weight, total_avg, scenario):
        """
        Calculate green time based on deviation from average weight with DYNAMIC MINIMUM TIMING
        
        NEW FEATURES:
        - Calculates actual clearance time based on vehicle count
        - Reduces minimum green time when traffic is low
        - Enables faster cycles for better traffic flow
        - If 4 vehicles can pass in 10s, green time ~12-15s (not fixed 30s)
        
        Logic:
        - If a direction's weight is above average: increase green time
        - If a direction's weight is below average: decrease green time
        - Magnitude of change proportional to deviation
        - DYNAMIC safety constraints based on actual traffic needs
        """
        # Base green time from configuration
        base_green = self.base_green_time
        
        # Calculate deviations from average
        ns_deviation = ns_weight - total_avg
        ew_deviation = ew_weight - total_avg
        
        # Convert deviations to timing adjustments
        # Each weight unit = 5 seconds adjustment
        adjustment_factor = 5.0
        
        ns_adjustment = ns_deviation * adjustment_factor
        ew_adjustment = ew_deviation * adjustment_factor
        
        # Apply adjustments to base timing
        ns_green = base_green + ns_adjustment
        ew_green = base_green + ew_adjustment
        
        # DYNAMIC MINIMUM TIMING - Calculate based on actual traffic needs
        ns_min_time = self._calculate_dynamic_minimum_time(ns_weight, 'north_south')
        ew_min_time = self._calculate_dynamic_minimum_time(ew_weight, 'east_west')
        
        # DYNAMIC MAXIMUM - Reduce max time for low traffic scenarios
        dynamic_max = 45
        if scenario in ['EMPTY', 'LIGHT']:
            dynamic_max = 25  # Reduce max time for low traffic
        elif scenario == 'MODERATE':
            dynamic_max = 35  # Moderate reduction for moderate traffic
        
        # Apply DYNAMIC safety constraints
        ns_green = max(ns_min_time, min(dynamic_max, ns_green))
        ew_green = max(ew_min_time, min(dynamic_max, ew_green))
        
        # Determine priority direction
        if abs(ns_deviation) > abs(ew_deviation):
            if ns_deviation > 0:
                priority = 'north_south'
            else:
                priority = 'east_west'
        else:
            if ew_deviation > 0:
                priority = 'east_west'
            else:
                priority = 'north_south'
        
        # If deviations are small, it's balanced
        if abs(ns_deviation) < 0.5 and abs(ew_deviation) < 0.5:
            priority = 'balanced'
        
        # Calculate cycle time
        cycle_time = ns_green + ew_green + (self.yellow_time * 2) + (self.red_clearance_time * 2)
        
        return {
            'north_south_green': round(ns_green, 1),
            'east_west_green': round(ew_green, 1),
            'cycle_time': round(cycle_time, 1),
            'priority_direction': priority,
            'ns_deviation': round(ns_deviation, 2),
            'ew_deviation': round(ew_deviation, 2),
            'ns_adjustment': round(ns_adjustment, 1),
            'ew_adjustment': round(ew_adjustment, 1),
            'ns_min_time': round(ns_min_time, 1),
            'ew_min_time': round(ew_min_time, 1),
            'dynamic_max': dynamic_max
        }
    
    def _calculate_dynamic_minimum_time(self, direction_weight, direction_name):
        """
        Calculate dynamic minimum green time based on actual traffic conditions
        
        Logic:
        - If 4 vehicles can clear in 10s, set minimum to ~12-15s (not 30s)
        - Very low traffic: 8-10s minimum
        - Light traffic: 10-12s minimum  
        - Normal+ traffic: 15s minimum (safety)
        
        Args:
            direction_weight: Traffic weight for the direction
            direction_name: Direction identifier for debugging
            
        Returns:
            float: Dynamic minimum green time in seconds
        """
        # Vehicle clearance assumptions
        vehicles_per_second = 1.5  # Typical intersection clearance rate
        safety_buffer = 3  # Always add 3s safety buffer
        
        # Estimate vehicles from weight (rough approximation)
        # Weight 1.0 = ~0 vehicles, Weight 2.0 = ~2 vehicles, etc.
        estimated_vehicles = max(0, (direction_weight - 1.0) * 2)
        
        # Calculate actual clearance time needed
        clearance_time = estimated_vehicles / vehicles_per_second
        base_minimum = clearance_time + safety_buffer
        
        # Apply dynamic minimum rules based on traffic category
        if direction_weight <= 1.5:  # EMPTY traffic
            dynamic_min = max(8, base_minimum)  # 8s minimum for very low traffic
        elif direction_weight <= 2.5:  # LIGHT traffic  
            dynamic_min = max(10, base_minimum)  # 10s minimum for light traffic
        elif direction_weight <= 3.5:  # MODERATE traffic
            dynamic_min = max(12, base_minimum)  # 12s minimum for moderate traffic
        else:  # NORMAL, HEAVY, CRITICAL traffic
            dynamic_min = max(15, base_minimum)  # 15s minimum for safety
        
        # Never go below absolute safety minimum of 8 seconds
        dynamic_min = max(8, dynamic_min)
        
        return dynamic_min
    
    def _apply_fast_cycle_optimization(self, timing_plan, scenario_category, avg_weight):
        """
        Apply fast-cycle optimization when ALL lanes have low traffic
        
        Logic:
        - If all lanes are EMPTY/LIGHT, reduce cycle time significantly
        - Faster cycles = better traffic flow when demand is low
        - Maintains safety while avoiding wasted green time
        
        Args:
            timing_plan: Current timing plan
            scenario_category: Overall traffic scenario 
            avg_weight: Average weight across all lanes
            
        Returns:
            dict: Optimized timing plan with faster cycles
        """
        if scenario_category == 'EMPTY' and avg_weight < 1.5:
            # Very low traffic - aggressive cycle reduction
            optimization_factor = 0.6  # 40% reduction
            fast_cycle_type = "ULTRA_FAST"
            
        elif scenario_category == 'EMPTY' and avg_weight < 2.0:
            # Empty but some vehicles - moderate reduction  
            optimization_factor = 0.7  # 30% reduction
            fast_cycle_type = "FAST"
            
        elif scenario_category == 'LIGHT' and avg_weight < 2.5:
            # Light traffic - conservative reduction
            optimization_factor = 0.8  # 20% reduction  
            fast_cycle_type = "MODERATE_FAST"
            
        else:
            # No optimization needed
            return timing_plan
        
        # Apply optimization while respecting minimum safety times
        original_ns = timing_plan['north_south_green']
        original_ew = timing_plan['east_west_green']
        
        optimized_ns = max(8, original_ns * optimization_factor)
        optimized_ew = max(8, original_ew * optimization_factor)
        
        # Recalculate cycle time
        optimized_cycle = optimized_ns + optimized_ew + (self.yellow_time * 2) + (self.red_clearance_time * 2)
        
        # Update timing plan
        timing_plan.update({
            'north_south_green': round(optimized_ns, 1),
            'east_west_green': round(optimized_ew, 1), 
            'cycle_time': round(optimized_cycle, 1),
            'fast_cycle_applied': True,
            'fast_cycle_type': fast_cycle_type,
            'optimization_factor': optimization_factor,
            'original_ns_green': round(original_ns, 1),
            'original_ew_green': round(original_ew, 1),
            'cycle_time_reduction': round(timing_plan['cycle_time'] - optimized_cycle, 1)
        })
        
        return timing_plan
    
    def _get_default_timing(self):
        """Return default timing when no traffic data available"""
        return {
            'north_south_green': self.base_green_time,
            'east_west_green': self.base_green_time,
            'cycle_time': (self.base_green_time * 2) + (self.yellow_time * 2) + (self.red_clearance_time * 2),
            'priority_direction': 'balanced',
            'scenario_category': 'EMPTY'
        }
    
    def _calculate_adaptive_cycle_timing(self, phase_pressures, current_time):
        """
        Calculate adaptive cycle timing with lane prioritization
        """
        ns_pressure = phase_pressures['north_south']
        ew_pressure = phase_pressures['east_west']
        total_pressure = ns_pressure + ew_pressure
        
        if total_pressure == 0:
            # No traffic - use minimum timings
            return {
                'north_south_green': self.min_green_time,
                'east_west_green': self.min_green_time,
                'cycle_time': (self.min_green_time * 2) + (self.yellow_time * 4) + (self.red_clearance_time * 4),
                'priority_direction': 'balanced'
            }
        
        # Determine priority and timing distribution
        pressure_ratio = ns_pressure / total_pressure if total_pressure > 0 else 0.5
        
        # Base cycle time calculation
        base_cycle_time = self.base_green_time * 2 + (self.yellow_time * 4) + (self.red_clearance_time * 4)
        
        # Adaptive timing based on pressure ratios
        if pressure_ratio > 0.8:  # Heavy North-South priority
            ns_green = min(self.max_green_time, self.base_green_time * 1.5)
            ew_green = max(self.min_green_time, self.base_green_time * 0.5)
            priority = 'north_south'
        elif pressure_ratio < 0.2:  # Heavy East-West priority
            ns_green = max(self.min_green_time, self.base_green_time * 0.5)
            ew_green = min(self.max_green_time, self.base_green_time * 1.5)
            priority = 'east_west'
        elif 0.6 < pressure_ratio < 0.8:  # Moderate North-South priority
            ns_green = min(self.max_green_time, self.base_green_time * 1.2)
            ew_green = max(self.min_green_time, self.base_green_time * 0.8)
            priority = 'north_south_moderate'
        elif 0.2 < pressure_ratio < 0.4:  # Moderate East-West priority
            ns_green = max(self.min_green_time, self.base_green_time * 0.8)
            ew_green = min(self.max_green_time, self.base_green_time * 1.2)
            priority = 'east_west_moderate'
        else:  # Balanced traffic (0.4 <= pressure_ratio <= 0.6)
            # Equal timing for equal pressure
            ns_green = self.base_green_time
            ew_green = self.base_green_time
            priority = 'balanced'
        
        # Ensure minimum safety timings
        ns_green = max(self.min_green_time_per_lane, ns_green)
        ew_green = max(self.min_green_time_per_lane, ew_green)
        
        # Calculate total cycle time
        cycle_time = ns_green + ew_green + (self.yellow_time * 4) + (self.red_clearance_time * 4)
        
        # Ensure no lane waits more than 2 minutes (120s)
        max_red_ns = ew_green + (self.yellow_time * 4) + (self.red_clearance_time * 4)
        max_red_ew = ns_green + (self.yellow_time * 4) + (self.red_clearance_time * 4)
        
        if max_red_ns > self.max_red_time:
            # Reduce opposing green time to meet red time constraint
            ew_green = self.max_red_time - (self.yellow_time * 4) - (self.red_clearance_time * 4)
            ew_green = max(self.min_green_time_per_lane, ew_green)
        
        if max_red_ew > self.max_red_time:
            # Reduce opposing green time to meet red time constraint
            ns_green = self.max_red_time - (self.yellow_time * 4) - (self.red_clearance_time * 4)
            ns_green = max(self.min_green_time_per_lane, ns_green)
        
        # Recalculate cycle time after constraints
        final_cycle_time = ns_green + ew_green + (self.yellow_time * 4) + (self.red_clearance_time * 4)
        
        return {
            'north_south_green': int(ns_green),
            'east_west_green': int(ew_green),
            'cycle_time': int(final_cycle_time),
            'priority_direction': priority,
            'ns_pressure': ns_pressure,
            'ew_pressure': ew_pressure,
            'pressure_ratio': pressure_ratio,
            'max_red_ns': int(ew_green + (self.yellow_time * 4) + (self.red_clearance_time * 4)),
            'max_red_ew': int(ns_green + (self.yellow_time * 4) + (self.red_clearance_time * 4))
        }
    
    def calculate_adaptive_green_time(self, current_pressure, opposing_pressure, current_time):
        """
        Calculate adaptive green time using gradual changes and statistical analysis
        Core edge algorithm implementing your project specifications
        """
        # Use base timing as reference (not current duration to prevent runaway)
        base_time = self.base_green_time
        
        # Statistical pressure ratio
        total_pressure = current_pressure + opposing_pressure
        if total_pressure > 0:
            pressure_ratio = current_pressure / total_pressure
        else:
            pressure_ratio = 0.5  # Neutral when no pressure
        
        # Conservative pressure thresholds
        high_pressure_threshold = 0.75
        low_pressure_threshold = 0.25
        
        # More conservative adjustment calculation
        if pressure_ratio > high_pressure_threshold:  # High current pressure
            # Moderate increase based on how much above threshold
            excess_pressure = pressure_ratio - high_pressure_threshold
            increase_factor = min(excess_pressure * 0.8, self.max_change_percent)  # More conservative
            new_duration = base_time * (1 + increase_factor)
        elif pressure_ratio < low_pressure_threshold:  # High opposing pressure
            # Moderate decrease based on how much below threshold
            pressure_deficit = low_pressure_threshold - pressure_ratio
            decrease_factor = min(pressure_deficit * 0.8, self.max_change_percent)  # More conservative
            new_duration = base_time * (1 - decrease_factor)
        else:  # Moderate pressure - stay near base
            # Small adjustments around base timing
            deviation_from_neutral = pressure_ratio - 0.5
            adjustment_factor = deviation_from_neutral * 0.2  # Very small adjustments
            new_duration = base_time * (1 + adjustment_factor)
        
        # Apply project constraints: 10s min, 50s max
        final_duration = max(self.min_green_time, 
                           min(self.max_green_time, new_duration))
        
        # Update current duration for gradual changes
        self.current_green_duration = final_duration
        
        return int(final_duration)
    
    def apply_edge_algorithm(self, current_time):
        """
        NEW APPROACH: Weighted Average Categorization Algorithm
        
        Features:
        - Categorizes traffic into 6 levels (EMPTY to CRITICAL)
        - Uses weighted averages for timing decisions
        - Compares lane weights to average for adjustments
        - Gradual and proportional green time changes
        - Better performance than raw density approach
        """
        # Check if it's time for adaptation
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return None
        
        try:
            # Step 1: Collect traffic density from all lanes
            traffic_data = self.collect_lane_traffic_density()
            if not traffic_data:
                return None
            
            # Step 2: Get current traffic light state
            current_phase = traci.trafficlight.getPhase(self.junction_id)
            
            # Step 3: Calculate weighted average timing plan (NEW APPROACH)
            timing_plan = self.calculate_weighted_average_timing(traffic_data, current_time)
            if not timing_plan:
                return None
            
            # Step 4: Apply adaptive timing based on current phase
            adaptation_made = False
            
            if current_phase in [1, 5]:  # North-South phases
                new_duration = timing_plan['north_south_green']
                phase_name = "North-South"
                weight_info = f"NS Weight: {timing_plan.get('ns_weight', 0):.1f}"
            elif current_phase in [3, 7]:  # East-West phases  
                new_duration = timing_plan['east_west_green']
                phase_name = "East-West"
                weight_info = f"EW Weight: {timing_plan.get('ew_weight', 0):.1f}"
            else:
                # Yellow phases - don't modify
                self.last_adaptation_time = current_time
                return None
            
            # Step 5: Apply new duration if significantly different
            try:
                current_duration = traci.trafficlight.getPhaseDuration(self.junction_id)
                
                if abs(new_duration - current_duration) > 2:  # 2s threshold for changes
                    traci.trafficlight.setPhaseDuration(self.junction_id, new_duration)
                    
                    # Log the comprehensive adaptation with new categorization info
                    adaptation_info = {
                        'time': current_time,
                        'phase': current_phase,
                        'phase_name': phase_name,
                        'old_duration': current_duration,
                        'new_duration': new_duration,
                        'timing_plan': timing_plan,
                        'traffic_data': traffic_data,
                        'priority_direction': timing_plan['priority_direction'],
                        'cycle_time': timing_plan['cycle_time'],
                        'scenario_category': timing_plan.get('scenario_category', 'UNKNOWN'),
                        'avg_weight': timing_plan.get('avg_weight', 0),
                        'ns_deviation': timing_plan.get('ns_deviation', 0),
                        'ew_deviation': timing_plan.get('ew_deviation', 0)
                    }
                    
                    self.timing_adjustments.append(adaptation_info)
                    self.adaptations_count += 1
                    adaptation_made = True
                    
                    # Enhanced logging with categorization and dynamic timing info
                    priority = timing_plan['priority_direction']
                    ns_green = timing_plan['north_south_green']
                    ew_green = timing_plan['east_west_green']
                    scenario = timing_plan.get('scenario_category', 'UNKNOWN')
                    avg_weight = timing_plan.get('avg_weight', 0)
                    ns_deviation = timing_plan.get('ns_deviation', 0)
                    ew_deviation = timing_plan.get('ew_deviation', 0)
                    
                    # Dynamic timing information
                    ns_min = timing_plan.get('ns_min_time', 'N/A')
                    ew_min = timing_plan.get('ew_min_time', 'N/A')
                    fast_cycle = timing_plan.get('fast_cycle_applied', False)
                    
                    print(f"🚦 DYNAMIC EDGE: {phase_name} → {new_duration}s")
                    print(f"   Priority: {priority} | Scenario: {scenario}")
                    print(f"   Timing: NS={ns_green}s, EW={ew_green}s | Cycle: {timing_plan['cycle_time']}s")
                    print(f"   Avg Weight: {avg_weight:.1f} | Deviations: NS={ns_deviation:.1f}, EW={ew_deviation:.1f}")
                    print(f"   Dynamic Min: NS={ns_min}s, EW={ew_min}s | Category Weights: NS={timing_plan.get('ns_weight', 0):.1f}, EW={timing_plan.get('ew_weight', 0):.1f}")
                    
                    if fast_cycle:
                        cycle_reduction = timing_plan.get('cycle_time_reduction', 0)
                        optimization_type = timing_plan.get('fast_cycle_type', 'FAST')
                        print(f"   ⚡ FAST CYCLE: {optimization_type} | Reduced cycle by {cycle_reduction}s")
                    
                    self.last_adaptation_time = current_time
                    return adaptation_info
            
            except Exception as apply_error:
                print(f"⚠️  Error applying weighted timing: {apply_error}")
            
            self.last_adaptation_time = current_time
            return None
            
        except Exception as e:
            print(f"⚠️  Error in categorization algorithm: {e}")
            return None
    
    def update_base_timing_from_cloud(self, new_base_time):
        """
        Phase 2: Update base timing from cloud RL model
        This will be called when the RL model provides optimized base timings
        """
        if self.min_green_time <= new_base_time <= self.max_green_time:
            old_base = self.base_green_time
            self.base_green_time = new_base_time
            print(f"🌐 CLOUD UPDATE: Base timing updated {old_base}s → {new_base_time}s")
            return True
        else:
            print(f"⚠️  Invalid base timing from cloud: {new_base_time}s")
            return False
    
    def get_edge_statistics(self):
        """Get edge controller performance statistics"""
        return {
            'algorithm': 'edge_adaptive',
            'base_green_time': self.base_green_time,
            'current_green_duration': self.current_green_duration,
            'adaptations_made': self.adaptations_count,
            'timing_range': f"{self.min_green_time}s - {self.max_green_time}s",
            'recent_adjustments': self.timing_adjustments[-5:] if self.timing_adjustments else [],
            'density_analysis': {
                direction: {
                    'avg_density': statistics.mean(history) if history else 0,
                    'samples': len(history)
                }
                for direction, history in self.density_history.items()
            }
        }
    
    def reset_controller(self):
        """Reset controller state for new simulation"""
        self.current_phase = 0
        self.phase_start_time = 0
        self.last_adaptation_time = 0
        self.current_green_duration = self.base_green_time
        self.adaptations_count = 0
        self.timing_adjustments.clear()
        
        # Clear history
        self.density_history.clear()
        self.waiting_history.clear() 
        self.speed_history.clear()
        
        print(f"🔄 Edge controller reset - Base timing: {self.base_green_time}s")
    
    # Interface compatibility methods
    def control_traffic_lights(self, current_time: float, traci_connection) -> Dict[str, Any]:
        """Interface method for integration with existing system"""
        result = self.apply_edge_algorithm(current_time)
        
        return {
            'algorithm': 'edge_adaptive',
            'applied': result is not None,
            'action': 'timing_adjustment' if result else 'no_change',
            'details': result if result else {},
            'timestamp': current_time
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Interface method for getting statistics"""
        return self.get_edge_statistics()