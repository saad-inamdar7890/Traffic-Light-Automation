"""
ALGORITHM FIX: Critical Performance Issue Resolution
=================================================

PROBLEM IDENTIFIED:
- Adaptive algorithm performing 7.6x WORSE than normal mode
- 165.6s average wait time vs 21.8s normal
- 717 adaptations suggesting over-adaptation
- Algorithm making traffic significantly worse

ROOT CAUSES:
1. Over-adaptation: Changing timing too frequently (every 15s)
2. Excessive timing changes: 25% max change causing instability
3. Dynamic minimum timing may be too aggressive
4. Fast cycle optimization creating traffic jams
5. Algorithm fighting against natural traffic flow

FIXES IMPLEMENTED:
1. Reduced adaptation frequency from 15s to 45s
2. Reduced max change from 25% to 10% for stability
3. More conservative dynamic minimum timing
4. Disabled fast cycle optimization temporarily
5. Added stability checks before making changes
6. Implemented gradual adjustment approach
"""

import traci
import statistics
import math
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

class FixedEdgeTrafficController:
    def __init__(self, junction_id="J4", base_green_time=30):
        """
        FIXED Edge Traffic Controller - Conservative Approach
        
        Changes from original:
        - Longer adaptation intervals for stability
        - Smaller timing changes to prevent oscillations
        - More conservative minimum timing
        - Better stability checks
        """
        self.junction_id = junction_id
        
        # FIXED: More conservative configuration
        self.base_green_time = base_green_time
        self.min_green_time = 15  # INCREASED from 10s for safety
        self.max_green_time = 45  # REDUCED from 50s to prevent congestion
        
        # FIXED: Reduced change frequency and magnitude
        self.max_change_percent = 0.10  # REDUCED from 0.25 to 0.10 (10%)
        self.adaptation_interval = 45   # INCREASED from 15s to 45s
        
        # FIXED: More conservative traffic categories
        self.traffic_categories = {
            'EMPTY': {'threshold': 0, 'weight': 1.0, 'description': 'No vehicles'},
            'LIGHT': {'threshold': 2, 'weight': 1.5, 'description': 'Few vehicles'},    # REDUCED weight
            'MODERATE': {'threshold': 4, 'weight': 2.0, 'description': 'Normal traffic'}, # REDUCED weight
            'NORMAL': {'threshold': 6, 'weight': 2.5, 'description': 'Regular flow'},     # REDUCED weight
            'HEAVY': {'threshold': 8, 'weight': 3.0, 'description': 'Dense traffic'},     # REDUCED weight
            'CRITICAL': {'threshold': 12, 'weight': 3.5, 'description': 'Very heavy traffic'} # REDUCED weight
        }
        
        # FIXED: More conservative timing parameters
        self.waiting_time_threshold = 30  # INCREASED from 20s
        self.speed_threshold_low = 3.0    # REDUCED from 5.0 m/s
        
        # State Tracking
        self.current_phase = 0
        self.phase_start_time = 0
        self.last_adaptation_time = 0
        self.current_green_duration = base_green_time
        
        # Traffic Density History
        self.density_history = defaultdict(lambda: deque(maxlen=5))  # REDUCED from 10
        self.waiting_history = defaultdict(lambda: deque(maxlen=5))
        self.speed_history = defaultdict(lambda: deque(maxlen=5))
        
        # Performance Metrics
        self.adaptations_count = 0
        self.timing_adjustments = []
        
        # Phase definitions
        self.phase_directions = {
            0: "yellow_transition",
            1: "north_south",
            2: "yellow_transition", 
            3: "east_west",
            4: "yellow_transition",
            5: "north_south_left",
            6: "yellow_transition",
            7: "east_west_left"
        }
        
        # FIXED: More conservative timing management
        self.max_red_time = 90   # REDUCED from 120s
        self.min_green_time_per_lane = 12  # INCREASED from 8s
        self.yellow_time = 3
        self.red_clearance_time = 1
        
        # FIXED: Add stability tracking
        self.recent_changes = deque(maxlen=5)
        self.stability_threshold = 0.15  # Don't change if recent changes > 15%
        
        print(f"🚦 FIXED Edge Traffic Controller initialized")
        print(f"   FIXES: Longer intervals (45s), smaller changes (10%), conservative timing")
        print(f"   Base Green Time: {self.base_green_time}s")
        print(f"   Timing Range: {self.min_green_time}s - {self.max_green_time}s")
        print(f"   Max Change: {self.max_change_percent*100}% per adjustment")
    
    def collect_lane_traffic_density(self):
        """
        FIXED: Same collection method but with conservative processing
        """
        try:
            traffic_data = {}
            
            edge_directions = {
                "E0": "east_approach",
                "-E0": "west_approach",  
                "E1": "north_approach",
                "-E1": "south_approach"
            }
            
            for edge_id, direction in edge_directions.items():
                try:
                    lanes = traci.edge.getLaneNumber(edge_id)
                    total_vehicles = 0
                    total_waiting_time = 0
                    total_speed = 0
                    lane_count = 0
                    
                    for lane_idx in range(lanes):
                        lane_id = f"{edge_id}_{lane_idx}"
                        
                        vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
                        waiting_time = traci.lane.getWaitingTime(lane_id)
                        mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                        lane_length = traci.lane.getLength(lane_id)
                        
                        density = (vehicles / (lane_length / 100)) if lane_length > 0 else 0
                        
                        total_vehicles += vehicles
                        total_waiting_time += waiting_time
                        total_speed += mean_speed
                        lane_count += 1
                    
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
                    
                    self.density_history[direction].append(avg_density)
                    self.waiting_history[direction].append(avg_waiting)
                    self.speed_history[direction].append(avg_speed)
                    
                except Exception:
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
        FIXED: More conservative traffic categorization
        """
        # Base categorization by vehicle count
        base_category = 'EMPTY'
        for category, config in sorted(self.traffic_categories.items(), 
                                     key=lambda x: x[1]['threshold'], reverse=True):
            if vehicles >= config['threshold']:
                base_category = category
                break
        
        base_weight = self.traffic_categories[base_category]['weight']
        
        # FIXED: More conservative modifiers
        modifier = 1.0
        
        # REDUCED waiting time impact
        if waiting_time > self.waiting_time_threshold:
            modifier += 0.2  # REDUCED from higher values
        
        # REDUCED speed impact
        if speed > 0 and speed < self.speed_threshold_low:
            modifier += 0.1  # REDUCED from higher values
        
        final_weight = base_weight * modifier
        
        return {
            'category': base_category,
            'base_weight': base_weight,
            'modifier': modifier,
            'final_weight': final_weight,
            'description': self.traffic_categories[base_category]['description']
        }
    
    def calculate_weighted_average_timing(self, traffic_data, current_time):
        """
        FIXED: Conservative weighted average timing with stability checks
        """
        if not traffic_data:
            return self._get_default_timing()
        
        # Calculate direction weights
        ns_data = [traffic_data.get('north_approach', {}), traffic_data.get('south_approach', {})]
        ew_data = [traffic_data.get('east_approach', {}), traffic_data.get('west_approach', {})]
        
        # Get weights for each direction
        ns_weights = []
        ew_weights = []
        
        for data in ns_data:
            if data:
                category_info = self.categorize_traffic_level(
                    data.get('vehicles', 0),
                    data.get('waiting_time', 0),
                    data.get('speed', 0)
                )
                ns_weights.append(category_info['final_weight'])
        
        for data in ew_data:
            if data:
                category_info = self.categorize_traffic_level(
                    data.get('vehicles', 0),
                    data.get('waiting_time', 0),
                    data.get('speed', 0)
                )
                ew_weights.append(category_info['final_weight'])
        
        # Calculate averages
        ns_avg_weight = statistics.mean(ns_weights) if ns_weights else 1.0
        ew_avg_weight = statistics.mean(ew_weights) if ew_weights else 1.0
        total_avg = (ns_avg_weight + ew_avg_weight) / 2
        
        # Determine scenario category
        max_weight = max(ns_avg_weight, ew_avg_weight)
        scenario_category = 'LIGHT'  # Default conservative
        for cat, config in self.traffic_categories.items():
            if max_weight >= config['weight']:
                scenario_category = cat
        
        # FIXED: Use conservative timing calculation
        timing_plan = self._calculate_conservative_timing(ns_avg_weight, ew_avg_weight, total_avg, scenario_category)
        
        # Add metadata
        timing_plan.update({
            'ns_weight': ns_avg_weight,
            'ew_weight': ew_avg_weight,
            'avg_weight': total_avg,
            'scenario_category': scenario_category,
            'timestamp': current_time
        })
        
        return timing_plan
    
    def _calculate_conservative_timing(self, ns_weight, ew_weight, total_avg, scenario):
        """
        FIXED: Conservative timing calculation to prevent traffic jams
        """
        # FIXED: Use smaller adjustment factor
        base_green = self.base_green_time
        adjustment_factor = 2.0  # REDUCED from 5.0
        
        # Calculate smaller deviations
        ns_deviation = ns_weight - total_avg
        ew_deviation = ew_weight - total_avg
        
        # Apply smaller adjustments
        ns_adjustment = ns_deviation * adjustment_factor
        ew_adjustment = ew_deviation * adjustment_factor
        
        ns_green = base_green + ns_adjustment
        ew_green = base_green + ew_adjustment
        
        # FIXED: More conservative minimum timing
        ns_min_time = self._calculate_conservative_minimum_time(ns_weight)
        ew_min_time = self._calculate_conservative_minimum_time(ew_weight)
        
        # FIXED: Conservative maximum timing
        conservative_max = 40  # REDUCED from 45
        if scenario in ['EMPTY', 'LIGHT']:
            conservative_max = 30  # More conservative for low traffic
        
        # Apply constraints
        ns_green = max(ns_min_time, min(conservative_max, ns_green))
        ew_green = max(ew_min_time, min(conservative_max, ew_green))
        
        # Determine priority conservatively
        priority = 'balanced'
        if abs(ns_deviation) > 0.8:  # INCREASED threshold for changes
            priority = 'north_south' if ns_deviation > 0 else 'east_west'
        elif abs(ew_deviation) > 0.8:
            priority = 'east_west' if ew_deviation > 0 else 'north_south'
        
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
            'ew_min_time': round(ew_min_time, 1)
        }
    
    def _calculate_conservative_minimum_time(self, direction_weight):
        """
        FIXED: Conservative minimum timing calculation
        """
        # FIXED: More conservative minimum times
        if direction_weight <= 1.2:  # EMPTY
            return 15  # INCREASED from 8s
        elif direction_weight <= 1.8:  # LIGHT
            return 18  # INCREASED from 10s
        elif direction_weight <= 2.5:  # MODERATE
            return 20  # INCREASED from 12s
        else:  # NORMAL+
            return 22  # INCREASED from 15s
    
    def _get_default_timing(self):
        """Return conservative default timing"""
        return {
            'north_south_green': self.base_green_time,
            'east_west_green': self.base_green_time,
            'cycle_time': (self.base_green_time * 2) + (self.yellow_time * 2) + (self.red_clearance_time * 2),
            'priority_direction': 'balanced',
            'ns_deviation': 0,
            'ew_deviation': 0
        }
    
    def _check_stability(self, new_duration, current_duration):
        """
        FIXED: Check if making changes would cause instability
        """
        # Don't change if recent changes are too frequent
        if len(self.recent_changes) >= 3:
            recent_variance = statistics.variance(self.recent_changes) if len(self.recent_changes) > 1 else 0
            if recent_variance > self.stability_threshold:
                return False  # Too much recent instability
        
        # Don't make tiny changes
        change_magnitude = abs(new_duration - current_duration) / current_duration
        if change_magnitude < 0.05:  # Less than 5% change
            return False
        
        # Don't make huge changes
        if change_magnitude > self.max_change_percent:
            return False
        
        return True
    
    def apply_edge_algorithm(self, current_time):
        """
        FIXED: Conservative algorithm application with stability checks
        """
        # FIXED: Longer adaptation interval
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return None
        
        try:
            traffic_data = self.collect_lane_traffic_density()
            if not traffic_data:
                return None
            
            current_phase = traci.trafficlight.getPhase(self.junction_id)
            timing_plan = self.calculate_weighted_average_timing(traffic_data, current_time)
            if not timing_plan:
                return None
            
            adaptation_made = False
            
            if current_phase in [1, 5]:  # North-South phases
                new_duration = timing_plan['north_south_green']
                phase_name = "North-South"
            elif current_phase in [3, 7]:  # East-West phases  
                new_duration = timing_plan['east_west_green']
                phase_name = "East-West"
            else:
                self.last_adaptation_time = current_time
                return None
            
            try:
                current_duration = traci.trafficlight.getPhaseDuration(self.junction_id)
                
                # FIXED: Use stability check before making changes
                if self._check_stability(new_duration, current_duration):
                    # FIXED: Limit change magnitude
                    max_change = current_duration * self.max_change_percent
                    if new_duration > current_duration:
                        new_duration = min(new_duration, current_duration + max_change)
                    else:
                        new_duration = max(new_duration, current_duration - max_change)
                    
                    traci.trafficlight.setPhaseDuration(self.junction_id, new_duration)
                    
                    # Track change for stability monitoring
                    change_ratio = new_duration / current_duration
                    self.recent_changes.append(change_ratio)
                    
                    adaptation_info = {
                        'time': current_time,
                        'phase': current_phase,
                        'phase_name': phase_name,
                        'old_duration': current_duration,
                        'new_duration': new_duration,
                        'change_ratio': change_ratio,
                        'timing_plan': timing_plan,
                        'adaptation_reason': 'conservative_adjustment'
                    }
                    
                    self.timing_adjustments.append(adaptation_info)
                    self.adaptations_count += 1
                    adaptation_made = True
                    
                    print(f"🔧 FIXED Algorithm: {phase_name} {current_duration:.1f}s → {new_duration:.1f}s")
                
                self.last_adaptation_time = current_time
                return adaptation_info if adaptation_made else None
                
            except Exception as e:
                print(f"⚠️  Error applying timing: {e}")
                return None
            
        except Exception as e:
            print(f"⚠️  Error in edge algorithm: {e}")
            return None
    
    def control_traffic_lights(self, current_time: float, traci_connection) -> Dict[str, Any]:
        """Interface method for integration"""
        result = self.apply_edge_algorithm(current_time)
        
        return {
            'algorithm': 'fixed_edge_adaptive',
            'applied': result is not None,
            'action': 'conservative_timing_adjustment' if result else 'no_change',
            'details': result if result else {},
            'timestamp': current_time
        }
    
    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        recent_adjustments = self.timing_adjustments[-10:] if self.timing_adjustments else []
        
        return {
            'total_adaptations': self.adaptations_count,
            'recent_adjustments': len(recent_adjustments),
            'avg_change_ratio': statistics.mean([adj.get('change_ratio', 1.0) for adj in recent_adjustments]) if recent_adjustments else 1.0,
            'stability_score': 1.0 - (statistics.variance(self.recent_changes) if len(self.recent_changes) > 1 else 0),
            'algorithm_version': 'fixed_conservative_v1.0'
        }

# Export the fixed controller
EdgeTrafficController = FixedEdgeTrafficController