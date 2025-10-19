"""
TRACI-Based Improved Adaptive Traffic Light Controller
=====================================================

This controller implements the improved adaptive algorithm using TRACI for SUMO integration.
Features dynamic intervals, enhanced traffic detection, and smart adaptation logic.
"""

import traci
import time
import statistics
import math
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

class TraciImprovedController:
    """Improved adaptive traffic light controller using TRACI."""
    
    def __init__(self, junction_id: str = "J4"):
        self.junction_id = junction_id
        self.current_phase = 0  # Start with phase 0 (North-South green)
        self.phase_start_time = 0
        self.total_adaptations = 0
        self.last_adaptation_reason = ""
        self.last_adaptation_time = 0  # Track when last adaptation occurred
        
        # Traffic data storage
        self.traffic_history = deque(maxlen=60)  # Store 60 seconds of history
        self.performance_metrics = {
            'total_wait_time': 0,
            'total_vehicles': 0,
            'phase_changes': 0,
            'smart_adaptations': 0
        }
        
        # IMPROVEMENT: Add stability tracking (ENHANCED)
        self.phase_stability_counter = 0  # Count stable periods
        self.min_adaptation_interval = 30  # Increased from 20s - minimum seconds between adaptations
        self.consecutive_adaptations = 0   # Track consecutive adaptations to prevent oscillation
        self.max_consecutive_adaptations = 3  # Limit consecutive adaptations
        
        # IMPROVEMENT 1: More conservative adaptation intervals (FINAL FIX)
        self.adaptation_intervals = {
            'CRITICAL': 40,    # Increased significantly - allow critical situations to stabilize
            'URGENT': 45,      # Increased from 30s
            'NORMAL': 50,      # Increased from 35s  
            'LIGHT': 60,       # Increased from 45s
            'MINIMAL': 75      # Increased from 60s - let minimal traffic run much longer
        }
        
        # IMPROVEMENT 2: Much more conservative traffic categorization (FINAL FIX)
        self.traffic_thresholds = {
            'CRITICAL': 30,    # Increased from 20 - only truly critical situations
            'HEAVY': 25,       # Increased from 15
            'MODERATE': 15,    # Increased from 10  
            'LIGHT': 8,        # Increased from 5
            'MINIMAL': 3       # Increased from 2
        }
        
        # IMPROVEMENT 3: Longer minimum phase durations (FINAL FIX)
        self.min_phase_durations = {
            'CRITICAL': 35,    # Increased from 25s - ensure phases get adequate time
            'URGENT': 40,      # Increased from 28s
            'NORMAL': 45,      # Increased from 30s
            'LIGHT': 50,       # Increased from 35s
            'MINIMAL': 60      # Increased from 40s
        }
        
        print("TRACI-Based Improved Adaptive Controller initialized")
        print(f"   Junction ID: {junction_id}")
        print(f"   Dynamic intervals: {list(self.adaptation_intervals.keys())}")
    
    def initialize_traffic_lights(self):
        """Initialize traffic light control."""
        try:
            # Get available traffic light programs
            programs = traci.trafficlight.getAllProgramLogics(self.junction_id)
            if programs:
                print(f"   Available programs: {len(programs)}")
            
            # Set initial phase
            traci.trafficlight.setPhase(self.junction_id, self.current_phase)
            self.phase_start_time = traci.simulation.getTime()
            self.last_adaptation_time = self.phase_start_time  # Initialize adaptation timing
            self.consecutive_adaptations = 0  # Initialize consecutive counter
            
            print(f"   Traffic lights initialized. Starting phase: {self.current_phase}")
            return True
            
        except Exception as e:
            print(f"   WARNING: Error initializing traffic lights: {e}")
            return False
    
    def apply_adaptive_control(self, current_time: int) -> Dict[str, Any]:
        """Apply improved adaptive traffic light control logic."""
        try:
            # Collect traffic data
            traffic_data = self.collect_traffic_data()
            
            # Update performance metrics
            self.update_performance_metrics(traffic_data)
            
            # Assess traffic urgency
            urgency = self.assess_traffic_urgency(traffic_data)
            
            # Check if adaptation is needed
            should_adapt = self.should_adapt_phase(traffic_data, current_time, urgency)
            adaptation_made = False
            
            if should_adapt:
                self.change_phase()
                adaptation_made = True
                self.total_adaptations += 1  # Track total adaptations
            
            return {
                'adaptation_made': adaptation_made,
                'urgency_level': urgency,
                'reason': self.last_adaptation_reason,
                'total_adaptations': self.total_adaptations,
                'traffic_data': traffic_data
            }
                
        except Exception as e:
            print(f"   WARNING: Error in adaptive control: {e}")
            return {
                'adaptation_made': False,
                'urgency_level': 'UNKNOWN',
                'reason': f"Error: {e}",
                'total_adaptations': self.total_adaptations,
                'traffic_data': self.get_empty_traffic_data()
            }
    
    def collect_traffic_data(self) -> Dict[str, Any]:
        """Collect comprehensive traffic data from TRACI."""
        try:
            current_time = traci.simulation.getTime()
            
            # Get lane data for all approaches
            lanes = {
                'north': 'E1_0',     # North approach (coming from north)
                'south': '-E1_0',    # South approach (coming from south)  
                'east': 'E0_0',      # East approach (coming from east)
                'west': '-E0_0'      # West approach (coming from west)
            }
            
            traffic_data = {
                'time': current_time,
                'vehicles': {},
                'waiting_times': {},
                'speeds': {},
                'total_vehicles': 0,
                'total_waiting': 0,
                'total_speed': 0,
                'lanes_with_speed': 0
            }
            
            for direction, lane_id in lanes.items():
                try:
                    # Get vehicle count
                    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                    
                    # Get waiting time
                    waiting_time = traci.lane.getWaitingTime(lane_id)
                    
                    # Get average speed
                    avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                    
                    traffic_data['vehicles'][direction] = vehicle_count
                    traffic_data['waiting_times'][direction] = waiting_time
                    traffic_data['speeds'][direction] = avg_speed
                    
                    traffic_data['total_vehicles'] += vehicle_count
                    traffic_data['total_waiting'] += waiting_time
                    
                    if avg_speed > 0:
                        traffic_data['total_speed'] += avg_speed
                        traffic_data['lanes_with_speed'] += 1
                    
                except Exception as e:
                    # Handle missing lanes gracefully
                    traffic_data['vehicles'][direction] = 0
                    traffic_data['waiting_times'][direction] = 0
                    traffic_data['speeds'][direction] = 0
            
            # Calculate averages for test compatibility
            total_vehicles = max(traffic_data['total_vehicles'], 1)
            avg_waiting_time = traffic_data['total_waiting'] / total_vehicles
            avg_speed = traffic_data['total_speed'] / max(traffic_data['lanes_with_speed'], 1)
            throughput = traffic_data['total_vehicles']
            
            # Add expected fields for test compatibility
            traffic_data.update({
                'avg_waiting_time': avg_waiting_time,
                'avg_speed': avg_speed,
                'throughput': throughput
            })
            
            # Store in history
            self.traffic_history.append(traffic_data)
            
            return traffic_data
            
        except Exception as e:
            print(f"   WARNING: Error collecting traffic data: {e}")
            return self.get_empty_traffic_data()
    
    def get_empty_traffic_data(self) -> Dict[str, Any]:
        """Return empty traffic data structure."""
        return {
            'time': 0,
            'vehicles': {'north': 0, 'south': 0, 'east': 0, 'west': 0},
            'waiting_times': {'north': 0, 'south': 0, 'east': 0, 'west': 0},
            'speeds': {'north': 0, 'south': 0, 'east': 0, 'west': 0},
            'total_vehicles': 0,
            'total_waiting': 0,
            'avg_waiting_time': 0,
            'avg_speed': 0,
            'throughput': 0
        }
    
    def assess_traffic_urgency(self, traffic_data: Dict[str, Any]) -> str:
        """Assess traffic urgency level using much more conservative logic (FINAL FIX)."""
        total_vehicles = traffic_data['total_vehicles']
        total_waiting = traffic_data['total_waiting']
        
        # Much more conservative critical situations
        if total_vehicles >= self.traffic_thresholds['CRITICAL'] or total_waiting > 200:
            return 'CRITICAL'
        
        # More conservative heavy traffic assessment
        if total_vehicles >= self.traffic_thresholds['HEAVY'] or total_waiting > 150:
            return 'URGENT'
        
        # Check directional imbalance (MUCH MORE CONSERVATIVE)
        vehicles = traffic_data['vehicles']
        ns_traffic = vehicles['north'] + vehicles['south']
        ew_traffic = vehicles['east'] + vehicles['west']
        
        if total_vehicles > 8:  # Only check imbalance if there's significant traffic
            max_direction = max(ns_traffic, ew_traffic)
            min_direction = min(ns_traffic, ew_traffic)
            
            # Only flag severe imbalances
            if max_direction > 5 * min_direction and max_direction > 12:
                return 'URGENT'
            elif max_direction > 4 * min_direction and max_direction > 10:
                return 'NORMAL'
        
        # Very conservative regular assessment
        if total_vehicles >= self.traffic_thresholds['MODERATE']:
            return 'NORMAL'
        elif total_vehicles >= self.traffic_thresholds['LIGHT']:
            return 'LIGHT'
        else:
            return 'MINIMAL'
    
    def should_adapt_phase(self, traffic_data: Dict[str, Any], current_time: int, urgency: str) -> bool:
        """Determine if phase should change using much more conservative logic (FINAL FIX)."""
        phase_duration = current_time - self.phase_start_time
        
        # IMPROVEMENT: Much stricter adaptation prevention
        time_since_last_adaptation = current_time - self.last_adaptation_time
        
        # Prevent oscillation - if we've had too many consecutive adaptations, wait longer
        required_interval = self.min_adaptation_interval
        if self.consecutive_adaptations >= 2:
            required_interval = self.min_adaptation_interval * 2  # Double the wait time
        
        if time_since_last_adaptation < required_interval:
            return False
        
        # Much longer minimum phase durations
        min_duration = self.min_phase_durations.get(urgency, 45)
        if phase_duration < min_duration:
            return False
        
        # Get traffic for current and opposite phases
        vehicles = traffic_data['vehicles']
        ns_traffic = vehicles['north'] + vehicles['south']
        ew_traffic = vehicles['east'] + vehicles['west']
        
        if self.current_phase == 0:  # North-South green
            current_phase_traffic = ns_traffic
            opposite_phase_traffic = ew_traffic
            current_waiting = traffic_data['waiting_times']['east'] + traffic_data['waiting_times']['west']
        else:  # East-West green
            current_phase_traffic = ew_traffic
            opposite_phase_traffic = ns_traffic
            current_waiting = traffic_data['waiting_times']['north'] + traffic_data['waiting_times']['south']
        
        # IMPROVEMENT 4: Much more conservative adaptation conditions (FINAL FIX)
        should_adapt = False
        reason = ""
        
        if urgency == 'CRITICAL':
            # Only adapt in truly critical situations with significant imbalance
            if opposite_phase_traffic > current_phase_traffic * 3.0 and phase_duration >= min_duration + 10:
                should_adapt = True
                reason = "Critical imbalance detected"
            elif current_waiting > 150 and phase_duration >= min_duration + 10:
                should_adapt = True
                reason = "Critical waiting time"
        
        elif urgency == 'URGENT':
            # Very conservative urgent conditions  
            if opposite_phase_traffic > current_phase_traffic * 2.5 and phase_duration >= min_duration + 15:
                should_adapt = True
                reason = "Urgent traffic imbalance"
            elif current_waiting > 100 and phase_duration >= min_duration + 15:
                should_adapt = True
                reason = "High waiting time"
        
        elif urgency == 'NORMAL':
            # Only switch for very clear benefit with longer phases
            if opposite_phase_traffic > current_phase_traffic * 2.2 and phase_duration >= min_duration + 20:
                should_adapt = True
                reason = "Traffic imbalance"
            elif current_phase_traffic == 0 and opposite_phase_traffic > 5 and phase_duration >= min_duration + 15:
                should_adapt = True
                reason = "Empty current phase"
        
        elif urgency in ['LIGHT', 'MINIMAL']:
            # Extremely conservative for light traffic - only switch if current phase is completely empty
            if current_phase_traffic == 0 and opposite_phase_traffic > 3 and phase_duration >= min_duration + 25:
                should_adapt = True
                reason = "Empty phase with waiting traffic"
        
        # Much higher maximum phase duration limits
        max_duration = self.adaptation_intervals.get(urgency, 75)
        if phase_duration >= max_duration:
            should_adapt = True
            reason = f"Maximum duration reached ({urgency})"
        
        if should_adapt:
            self.last_adaptation_reason = reason
            self.last_adaptation_time = current_time  # Track when adaptation occurred
            self.consecutive_adaptations += 1  # Track consecutive adaptations
        else:
            # Reset consecutive counter if no adaptation
            if time_since_last_adaptation > self.min_adaptation_interval * 2:
                self.consecutive_adaptations = 0
            
        return should_adapt
    
    def change_phase(self):
        """Change traffic light phase (IMPROVED)."""
        try:
            # Switch phase (0 -> 1 or 1 -> 0)
            self.current_phase = 1 - self.current_phase
            traci.trafficlight.setPhase(self.junction_id, self.current_phase)
            current_time = traci.simulation.getTime()
            self.phase_start_time = current_time
            self.last_adaptation_time = current_time  # Update adaptation timing
            
            self.performance_metrics['phase_changes'] += 1
            if self.last_adaptation_reason and ('imbalance' in self.last_adaptation_reason.lower() or 'empty' in self.last_adaptation_reason.lower()):
                self.performance_metrics['smart_adaptations'] += 1
            
            print(f"   Phase changed to {self.current_phase}: {self.last_adaptation_reason}")
            
        except Exception as e:
            print(f"   WARNING: Error changing phase: {e}")
    
    def update_performance_metrics(self, traffic_data: Dict[str, Any]):
        """Update performance tracking metrics."""
        self.performance_metrics['total_wait_time'] += traffic_data['total_waiting']
        self.performance_metrics['total_vehicles'] += traffic_data['total_vehicles']
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_vehicles = max(self.performance_metrics['total_vehicles'], 1)
        
        return {
            'avg_waiting_time': self.performance_metrics['total_wait_time'] / total_vehicles,
            'total_phase_changes': self.performance_metrics['phase_changes'],
            'smart_adaptations': self.performance_metrics['smart_adaptations'],
            'total_adaptations': self.total_adaptations,
            'total_vehicles': total_vehicles
        }