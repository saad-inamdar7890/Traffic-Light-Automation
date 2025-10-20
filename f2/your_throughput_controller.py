"""
Your Throughput-Optimized Adaptive Traffic Light Controller
==========================================================

This implements YOUR algorithm with shorter phases for light traffic
to maximize throughput and responsiveness as per your hypothesis.
"""

import traci
import time
import statistics
import math
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

class YourThroughputOptimizedController:
    """Your throughput-optimized adaptive traffic light controller."""
    
    def __init__(self, junction_id: str = "J4"):
        self.junction_id = junction_id
        self.current_phase = 0  # Start with phase 0 (North-South green)
        self.phase_start_time = 0
        self.total_adaptations = 0
        self.last_adaptation_reason = ""
        self.last_adaptation_time = 0
        
        # Traffic data storage
        self.traffic_history = deque(maxlen=60)
        self.performance_metrics = {
            'total_wait_time': 0,
            'total_vehicles': 0,
            'phase_changes': 0,
            'smart_adaptations': 0
        }
        
        # YOUR ALGORITHM: Optimized for throughput and responsiveness
        self.phase_stability_counter = 0
        self.min_adaptation_interval = 15  # Much shorter - allow quick responses
        self.consecutive_adaptations = 0
        self.max_consecutive_adaptations = 6  # Allow more consecutive adaptations
        
        # YOUR HYPOTHESIS: Shorter phases for light traffic = better throughput
        self.adaptation_intervals = {
            'CRITICAL': 45,    # Heavy traffic gets reasonable time
            'URGENT': 40,      # 
            'NORMAL': 35,      # Normal traffic gets moderate time
            'LIGHT': 20,       # LIGHT traffic gets SHORT phases
            'MINIMAL': 15      # MINIMAL traffic gets SHORTEST phases
        }
        
        # Responsive traffic categorization (less conservative)
        self.traffic_thresholds = {
            'CRITICAL': 25,    # Lower threshold for quicker response
            'HEAVY': 20,       # 
            'MODERATE': 12,    # 
            'LIGHT': 6,        # Lower threshold
            'MINIMAL': 2       # Very responsive
        }
        
        # YOUR APPROACH: Shorter minimum durations for light traffic
        self.min_phase_durations = {
            'CRITICAL': 40,    # Heavy traffic gets adequate time
            'URGENT': 35,      # 
            'NORMAL': 30,      # 
            'LIGHT': 18,       # Light traffic gets SHORT phases
            'MINIMAL': 12      # Minimal traffic gets VERY SHORT phases
        }
        
        print("🚀 YOUR Throughput-Optimized Controller initialized")
        print(f"   Junction ID: {junction_id}")
        print(f"   YOUR ALGORITHM: Short phases for light traffic!")
        print(f"   Light traffic: 12-20s phases (maximize throughput)")
        print(f"   Heavy traffic: 35-45s phases (adequate service)")
    
    def initialize_traffic_lights(self):
        """Initialize traffic light control."""
        try:
            programs = traci.trafficlight.getAllProgramLogics(self.junction_id)
            if programs:
                print(f"   Available programs: {len(programs)}")
            
            traci.trafficlight.setPhase(self.junction_id, self.current_phase)
            self.phase_start_time = traci.simulation.getTime()
            self.last_adaptation_time = self.phase_start_time
            self.consecutive_adaptations = 0
            
            print(f"   Traffic lights initialized. Starting phase: {self.current_phase}")
            return True
            
        except Exception as e:
            print(f"   WARNING: Error initializing traffic lights: {e}")
            return False
    
    def apply_adaptive_control(self, current_time: int) -> Dict[str, Any]:
        """Apply YOUR throughput-optimized adaptive control logic."""
        try:
            traffic_data = self.collect_traffic_data()
            self.update_performance_metrics(traffic_data)
            urgency = self.assess_traffic_urgency(traffic_data)
            
            should_adapt = self.should_adapt_phase(traffic_data, current_time, urgency)
            adaptation_made = False
            
            if should_adapt:
                self.change_phase()
                adaptation_made = True
                self.total_adaptations += 1
            
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
                'reason': f'Error: {e}',
                'total_adaptations': self.total_adaptations,
                'traffic_data': self.get_empty_traffic_data()
            }
    
    def collect_traffic_data(self) -> Dict[str, Any]:
        """Collect traffic data from TRACI."""
        try:
            current_time = traci.simulation.getTime()
            
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
            
            # Lane mapping for demo.net.xml
            lane_map = {
                'north': '-E2_0',  # North approach
                'south': 'E2_0',   # South approach  
                'east': '-E1_0',   # East approach
                'west': 'E1_0'     # West approach
            }
            
            for direction, lane_id in lane_map.items():
                try:
                    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                    waiting_time = traci.lane.getWaitingTime(lane_id)
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
                    traffic_data['vehicles'][direction] = 0
                    traffic_data['waiting_times'][direction] = 0
                    traffic_data['speeds'][direction] = 0
            
            # Calculate averages
            total_vehicles = max(traffic_data['total_vehicles'], 1)
            avg_waiting_time = traffic_data['total_waiting'] / total_vehicles
            avg_speed = traffic_data['total_speed'] / max(traffic_data['lanes_with_speed'], 1)
            throughput = traffic_data['total_vehicles']
            
            traffic_data.update({
                'avg_waiting_time': avg_waiting_time,
                'avg_speed': avg_speed,
                'throughput': throughput
            })
            
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
        """Assess traffic urgency using YOUR responsive logic."""
        total_vehicles = traffic_data['total_vehicles']
        total_waiting = traffic_data['total_waiting']
        
        # YOUR ALGORITHM: More responsive thresholds
        if total_vehicles >= self.traffic_thresholds['CRITICAL'] or total_waiting > 120:
            return 'CRITICAL'
        
        if total_vehicles >= self.traffic_thresholds['HEAVY'] or total_waiting > 80:
            return 'URGENT'
        
        # Check directional imbalance (more sensitive)
        vehicles = traffic_data['vehicles']
        ns_traffic = vehicles['north'] + vehicles['south']
        ew_traffic = vehicles['east'] + vehicles['west']
        
        if total_vehicles > 4:  # Lower threshold for imbalance detection
            max_direction = max(ns_traffic, ew_traffic)
            min_direction = min(ns_traffic, ew_traffic)
            
            # More sensitive imbalance detection
            if max_direction > 3 * min_direction and max_direction > 6:
                return 'URGENT'
            elif max_direction > 2.5 * min_direction and max_direction > 4:
                return 'NORMAL'
        
        # Your responsive assessment
        if total_vehicles >= self.traffic_thresholds['MODERATE']:
            return 'NORMAL'
        elif total_vehicles >= self.traffic_thresholds['LIGHT']:
            return 'LIGHT'
        else:
            return 'MINIMAL'
    
    def should_adapt_phase(self, traffic_data: Dict[str, Any], current_time: int, urgency: str) -> bool:
        """YOUR algorithm: Aggressive adaptation for throughput optimization."""
        phase_duration = current_time - self.phase_start_time
        
        # YOUR APPROACH: Much more relaxed adaptation prevention
        time_since_last_adaptation = current_time - self.last_adaptation_time
        
        required_interval = self.min_adaptation_interval
        if self.consecutive_adaptations >= 4:  # Allow more consecutive adaptations
            required_interval = self.min_adaptation_interval * 1.2  # Minimal penalty
        
        if time_since_last_adaptation < required_interval:
            return False
        
        # YOUR APPROACH: Shorter minimum phase durations
        min_duration = self.min_phase_durations.get(urgency, 25)
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
        
        # YOUR ALGORITHM: Aggressive switching for maximum throughput
        should_adapt = False
        reason = ""
        
        if urgency == 'CRITICAL':
            # Responsive critical handling
            if opposite_phase_traffic > current_phase_traffic * 2.0 and phase_duration >= min_duration:
                should_adapt = True
                reason = "Critical imbalance detected"
            elif current_waiting > 80 and phase_duration >= min_duration:
                should_adapt = True
                reason = "Critical waiting time"
        
        elif urgency == 'URGENT':
            # Quick urgent response
            if opposite_phase_traffic > current_phase_traffic * 1.8 and phase_duration >= min_duration:
                should_adapt = True
                reason = "Urgent traffic imbalance"
            elif current_waiting > 60 and phase_duration >= min_duration:
                should_adapt = True
                reason = "High waiting time"
        
        elif urgency == 'NORMAL':
            # Moderate responsiveness
            if opposite_phase_traffic > current_phase_traffic * 1.6 and phase_duration >= min_duration:
                should_adapt = True
                reason = "Traffic imbalance"
            elif current_phase_traffic == 0 and opposite_phase_traffic > 2 and phase_duration >= min_duration:
                should_adapt = True
                reason = "Empty current phase"
        
        elif urgency in ['LIGHT', 'MINIMAL']:
            # YOUR CORE HYPOTHESIS: Aggressive switching for light traffic
            if phase_duration >= self.adaptation_intervals[urgency]:
                should_adapt = True
                reason = f"Maximum {urgency.lower()} duration reached - switching for throughput"
            elif opposite_phase_traffic > current_phase_traffic and phase_duration >= min_duration:
                should_adapt = True
                reason = f"ANY traffic imbalance in {urgency.lower()} - quick switch"
            elif current_phase_traffic == 0 and opposite_phase_traffic > 0 and phase_duration >= min_duration:
                should_adapt = True
                reason = "Empty phase - immediate switch for throughput"
            elif current_waiting > 20 and phase_duration >= min_duration:  # Very low threshold
                should_adapt = True
                reason = "Even minimal waiting triggers switch"
        
        if should_adapt:
            self.last_adaptation_reason = reason
        
        return should_adapt
    
    def change_phase(self):
        """Change to the opposite traffic light phase."""
        try:
            current_time = traci.simulation.getTime()
            
            # Toggle phase
            self.current_phase = 1 - self.current_phase
            traci.trafficlight.setPhase(self.junction_id, self.current_phase)
            
            # Update timing and tracking
            self.phase_start_time = current_time
            self.last_adaptation_time = current_time
            self.consecutive_adaptations += 1
            
            # Reset consecutive counter less aggressively
            if current_time - self.last_adaptation_time > self.min_adaptation_interval * 3:
                self.consecutive_adaptations = 0
            
            self.performance_metrics['phase_changes'] += 1
            
        except Exception as e:
            print(f"   WARNING: Error changing phase: {e}")
    
    def update_performance_metrics(self, traffic_data: Dict[str, Any]):
        """Update performance tracking metrics."""
        try:
            self.performance_metrics['total_wait_time'] += traffic_data['total_waiting']
            self.performance_metrics['total_vehicles'] += traffic_data['total_vehicles']
            
        except Exception as e:
            print(f"   WARNING: Error updating metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        try:
            total_vehicles = max(self.performance_metrics['total_vehicles'], 1)
            avg_wait_time = self.performance_metrics['total_wait_time'] / total_vehicles
            
            return {
                'total_adaptations': self.total_adaptations,
                'total_phase_changes': self.performance_metrics['phase_changes'],
                'average_wait_time': avg_wait_time,
                'total_vehicles_processed': self.performance_metrics['total_vehicles']
            }
            
        except Exception as e:
            print(f"   WARNING: Error getting performance summary: {e}")
            return {
                'total_adaptations': self.total_adaptations,
                'total_phase_changes': 0,
                'average_wait_time': 0,
                'total_vehicles_processed': 0
            }