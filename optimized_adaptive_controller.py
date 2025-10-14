"""
OPTIMIZED ADAPTIVE TRAFFIC CONTROLLER
==================================

Key Improvements for Better Performance:
1. Dynamic adaptation intervals based on traffic conditions
2. Improved traffic detection and categorization
3. Phase-aware timing adjustments
4. Smarter minimum/maximum constraints
5. Traffic flow prediction algorithms
"""

import traci
import statistics
import math
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import time

class OptimizedAdaptiveController:
    def __init__(self, junction_id: str = "J1", base_green_time: int = 43):
        self.junction_id = junction_id
        self.base_green_time = base_green_time
        
        # OPTIMIZATION 1: Dynamic Adaptation Intervals
        self.adaptation_intervals = {
            'RUSH_HOUR': 20,      # Quick response during peak traffic
            'HEAVY': 25,          # Medium response for heavy traffic
            'MODERATE': 35,       # Balanced response for normal traffic
            'LIGHT': 45,          # Conservative for light traffic
            'EMPTY': 60           # Minimal changes when empty
        }
        
        # OPTIMIZATION 2: Improved Change Constraints
        self.max_change_by_traffic = {
            'CRITICAL': 0.30,     # Allow bigger changes for critical traffic
            'HEAVY': 0.25,        # Good responsiveness for heavy traffic
            'NORMAL': 0.20,       # Moderate changes for normal traffic
            'MODERATE': 0.15,     # Conservative for moderate traffic
            'LIGHT': 0.10,        # Minimal changes for light traffic
            'EMPTY': 0.05         # Very small changes when empty
        }
        
        # OPTIMIZATION 3: Context-Aware Traffic Categories
        self.traffic_categories = {
            'EMPTY': {'threshold': 0, 'weight': 0.8, 'min_green': 15},
            'LIGHT': {'threshold': 1, 'weight': 1.2, 'min_green': 18},
            'MODERATE': {'threshold': 3, 'weight': 1.8, 'min_green': 22},
            'NORMAL': {'threshold': 5, 'weight': 2.5, 'min_green': 25},
            'HEAVY': {'threshold': 8, 'weight': 3.5, 'min_green': 30},
            'CRITICAL': {'threshold': 12, 'weight': 4.5, 'min_green': 35}
        }
        
        # OPTIMIZATION 4: Phase-Specific Timing
        self.phase_timing_strategy = {
            'morning_rush': {'ns_factor': 1.3, 'ew_factor': 0.8},
            'evening_rush': {'ns_factor': 0.8, 'ew_factor': 1.3},
            'midday': {'ns_factor': 1.0, 'ew_factor': 1.0},
            'night': {'ns_factor': 0.7, 'ew_factor': 0.7}
        }
        
        # OPTIMIZATION 5: Smart Thresholds
        self.dynamic_thresholds = {
            'waiting_time_critical': 45,   # When to prioritize immediately
            'waiting_time_high': 25,       # When to increase priority
            'waiting_time_normal': 15,     # Normal waiting threshold
            'speed_slow': 2.0,             # Very slow traffic
            'speed_moderate': 5.0,         # Moderate speed traffic
            'queue_length_critical': 10    # Critical queue length
        }
        
        # OPTIMIZATION 6: Traffic Flow Prediction
        self.flow_prediction = {
            'window_size': 8,              # Historical data points
            'prediction_weight': 0.3,      # How much to trust predictions
            'trend_sensitivity': 0.4       # Sensitivity to traffic trends
        }
        
        # Enhanced state tracking
        self.current_phase = 0
        self.phase_start_time = 0
        self.last_adaptation_time = 0
        self.current_green_duration = base_green_time
        
        # OPTIMIZATION 7: Multi-layer History
        self.traffic_history = {
            'short_term': defaultdict(lambda: deque(maxlen=3)),    # Last 3 readings
            'medium_term': defaultdict(lambda: deque(maxlen=8)),   # Last 8 readings
            'long_term': defaultdict(lambda: deque(maxlen=15))     # Last 15 readings
        }
        
        # Performance optimization tracking
        self.performance_metrics = {
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'average_improvement': 0,
            'adaptation_history': deque(maxlen=20)
        }
        
    def get_current_traffic_phase(self):
        """Determine current traffic phase for context-aware decisions"""
        current_hour = time.localtime().tm_hour
        
        if 7 <= current_hour <= 10:
            return 'morning_rush'
        elif 16 <= current_hour <= 19:
            return 'evening_rush'
        elif 11 <= current_hour <= 15:
            return 'midday'
        else:
            return 'night'
    
    def predict_traffic_trend(self, direction: str):
        """Predict if traffic is increasing, decreasing, or stable"""
        if direction not in self.traffic_history['medium_term']:
            return 'stable', 1.0
            
        recent_data = list(self.traffic_history['medium_term'][direction])
        if len(recent_data) < 4:
            return 'stable', 1.0
            
        # Calculate trend using linear regression approach
        x = range(len(recent_data))
        y = [data.get('vehicles', 0) for data in recent_data if isinstance(data, dict)]
        
        if len(y) < 4:
            return 'stable', 1.0
            
        # Simple trend calculation
        mid_point = len(y) // 2
        first_half_avg = sum(y[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_avg = sum(y[mid_point:]) / (len(y) - mid_point)
        
        trend_ratio = second_half_avg / max(first_half_avg, 1)
        
        if trend_ratio > 1.2:
            return 'increasing', trend_ratio
        elif trend_ratio < 0.8:
            return 'decreasing', trend_ratio
        else:
            return 'stable', trend_ratio
    
    def calculate_optimal_timing(self, traffic_data: Dict[str, Any]):
        """OPTIMIZATION 8: Smart timing calculation with multiple factors"""
        
        # Get current traffic phase context
        current_phase = self.get_current_traffic_phase()
        phase_factors = self.phase_timing_strategy[current_phase]
        
        # Collect directional data
        ns_data = {
            'vehicles': traffic_data.get('north', {}).get('vehicles', 0) + traffic_data.get('south', {}).get('vehicles', 0),
            'waiting': (traffic_data.get('north', {}).get('waiting_time', 0) + traffic_data.get('south', {}).get('waiting_time', 0)) / 2,
            'speed': (traffic_data.get('north', {}).get('speed', 0) + traffic_data.get('south', {}).get('speed', 0)) / 2
        }
        
        ew_data = {
            'vehicles': traffic_data.get('east', {}).get('vehicles', 0) + traffic_data.get('west', {}).get('vehicles', 0),
            'waiting': (traffic_data.get('east', {}).get('waiting_time', 0) + traffic_data.get('west', {}).get('waiting_time', 0)) / 2,
            'speed': (traffic_data.get('east', {}).get('speed', 0) + traffic_data.get('west', {}).get('speed', 0)) / 2
        }
        
        # Calculate category weights
        ns_category = self.categorize_traffic_enhanced(ns_data['vehicles'], ns_data['waiting'], ns_data['speed'])
        ew_category = self.categorize_traffic_enhanced(ew_data['vehicles'], ew_data['waiting'], ew_data['speed'])
        
        ns_weight = self.traffic_categories[ns_category]['weight']
        ew_weight = self.traffic_categories[ew_category]['weight']
        
        # Apply phase-specific factors
        ns_weight *= phase_factors['ns_factor']
        ew_weight *= phase_factors['ew_factor']
        
        # OPTIMIZATION 9: Priority-based timing with traffic prediction
        ns_trend, ns_trend_ratio = self.predict_traffic_trend('north_south')
        ew_trend, ew_trend_ratio = self.predict_traffic_trend('east_west')
        
        # Adjust weights based on trends
        if ns_trend == 'increasing':
            ns_weight *= (1 + self.flow_prediction['trend_sensitivity'])
        elif ns_trend == 'decreasing':
            ns_weight *= (1 - self.flow_prediction['trend_sensitivity'] * 0.5)
            
        if ew_trend == 'increasing':
            ew_weight *= (1 + self.flow_prediction['trend_sensitivity'])
        elif ew_trend == 'decreasing':
            ew_weight *= (1 - self.flow_prediction['trend_sensitivity'] * 0.5)
        
        # Calculate timing based on relative weights
        total_weight = ns_weight + ew_weight
        if total_weight == 0:
            return self.base_green_time, self.base_green_time * 0.4
            
        total_cycle_time = 60  # Base cycle time
        
        ns_time = (ns_weight / total_weight) * total_cycle_time
        ew_time = (ew_weight / total_weight) * total_cycle_time
        
        # Apply minimum constraints based on traffic category
        ns_min = self.traffic_categories[ns_category]['min_green']
        ew_min = self.traffic_categories[ew_category]['min_green']
        
        ns_time = max(ns_time, ns_min)
        ew_time = max(ew_time, ew_min)
        
        return ns_time, ew_time
    
    def categorize_traffic_enhanced(self, vehicles, waiting_time=0, speed=0):
        """Enhanced traffic categorization with multiple factors"""
        
        # Base categorization
        base_category = 'EMPTY'
        for category, config in sorted(self.traffic_categories.items(), 
                                     key=lambda x: x[1]['threshold'], reverse=True):
            if vehicles >= config['threshold']:
                base_category = category
                break
        
        # Upgrade category based on waiting time and speed
        category_order = ['EMPTY', 'LIGHT', 'MODERATE', 'NORMAL', 'HEAVY', 'CRITICAL']
        current_index = category_order.index(base_category)
        
        # Critical waiting time upgrades category
        if waiting_time > self.dynamic_thresholds['waiting_time_critical']:
            current_index = min(current_index + 2, len(category_order) - 1)
        elif waiting_time > self.dynamic_thresholds['waiting_time_high']:
            current_index = min(current_index + 1, len(category_order) - 1)
        
        # Very slow speed upgrades category
        if 0 < speed < self.dynamic_thresholds['speed_slow']:
            current_index = min(current_index + 1, len(category_order) - 1)
        
        return category_order[current_index]
    
    def should_adapt_now(self, traffic_data: Dict[str, Any]):
        """OPTIMIZATION 10: Smart adaptation timing"""
        
        current_time = traci.simulation.getTime()
        
        # Determine current traffic intensity
        total_vehicles = sum(data.get('vehicles', 0) for data in traffic_data.values())
        max_waiting = max(data.get('waiting_time', 0) for data in traffic_data.values())
        
        # Choose adaptation interval based on traffic conditions
        if total_vehicles >= 20 or max_waiting > self.dynamic_thresholds['waiting_time_critical']:
            interval = self.adaptation_intervals['RUSH_HOUR']
        elif total_vehicles >= 12 or max_waiting > self.dynamic_thresholds['waiting_time_high']:
            interval = self.adaptation_intervals['HEAVY']
        elif total_vehicles >= 6:
            interval = self.adaptation_intervals['MODERATE']
        elif total_vehicles >= 2:
            interval = self.adaptation_intervals['LIGHT']
        else:
            interval = self.adaptation_intervals['EMPTY']
        
        return (current_time - self.last_adaptation_time) >= interval
    
    def apply_optimization_constraints(self, new_ns_time, new_ew_time, current_ns_time, current_ew_time, traffic_data):
        """OPTIMIZATION 11: Smart change constraints"""
        
        # Determine maximum allowed change based on traffic conditions
        total_vehicles = sum(data.get('vehicles', 0) for data in traffic_data.values())
        max_waiting = max(data.get('waiting_time', 0) for data in traffic_data.values())
        
        # Choose change limit based on urgency
        if max_waiting > self.dynamic_thresholds['waiting_time_critical']:
            max_change = self.max_change_by_traffic['CRITICAL']
        elif total_vehicles >= 12:
            max_change = self.max_change_by_traffic['HEAVY']
        elif total_vehicles >= 6:
            max_change = self.max_change_by_traffic['NORMAL']
        elif total_vehicles >= 3:
            max_change = self.max_change_by_traffic['MODERATE']
        elif total_vehicles >= 1:
            max_change = self.max_change_by_traffic['LIGHT']
        else:
            max_change = self.max_change_by_traffic['EMPTY']
        
        # Apply constraints
        max_ns_change = current_ns_time * max_change
        max_ew_change = current_ew_time * max_change
        
        # Constrain changes
        ns_change = max(-max_ns_change, min(max_ns_change, new_ns_time - current_ns_time))
        ew_change = max(-max_ew_change, min(max_ew_change, new_ew_time - current_ew_time))
        
        final_ns_time = current_ns_time + ns_change
        final_ew_time = current_ew_time + ew_change
        
        return final_ns_time, final_ew_time