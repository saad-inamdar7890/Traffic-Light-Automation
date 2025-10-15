"""
Simple adaptive traffic controller for validation testing.
This version doesn't depend on TRACI and implements the improved algorithm logic.
"""

import statistics
import math
from typing import Dict, Any, List
from collections import deque

class SimpleImprovedController:
    """Simplified version of improved adaptive controller for testing."""
    
    def __init__(self):
        self.current_phase = 'north_south'
        self.phase_start_time = 0
        self.current_time = 0
        self.last_adaptation_reason = None
        self.traffic_history = deque(maxlen=30)
        self.adaptations_made = 0
        
        # Dynamic adaptation intervals based on traffic urgency
        self.adaptation_intervals = {
            'CRITICAL': 15,    # Very urgent situations
            'URGENT': 20,      # Heavy traffic situations  
            'NORMAL': 30,      # Regular traffic
            'LIGHT': 40,       # Light traffic
            'MINIMAL': 50      # Very light traffic
        }
        
        # Dynamic change limits based on situation urgency
        self.change_limits = {
            'CRITICAL': 0.35,  # Allow large changes for critical situations
            'URGENT': 0.25,    # Good responsiveness for urgent situations
            'NORMAL': 0.18,    # Balanced changes for normal traffic
            'LIGHT': 0.12,     # Conservative for light traffic
            'MINIMAL': 0.06    # Very small changes for minimal traffic
        }
        
        # Enhanced traffic categories
        self.traffic_categories = {
            'EMPTY': {'threshold': 0, 'weight': 0.5, 'min_time': 12},
            'MINIMAL': {'threshold': 1, 'weight': 1.0, 'min_time': 15},
            'LIGHT': {'threshold': 2, 'weight': 1.5, 'min_time': 18},
            'MODERATE': {'threshold': 4, 'weight': 2.2, 'min_time': 22},
            'NORMAL': {'threshold': 6, 'weight': 2.8, 'min_time': 25},
            'HEAVY': {'threshold': 9, 'weight': 3.5, 'min_time': 30},
            'CRITICAL': {'threshold': 12, 'weight': 4.5, 'min_time': 35}
        }
    
    def update_traffic_data(self, traffic_data):
        """Update traffic data and time."""
        self.current_time += 1
        self.traffic_history.append(traffic_data)
    
    def should_change_phase(self):
        """Determine if phase should change using improved logic."""
        if len(self.traffic_history) < 3:
            return False
            
        phase_duration = self.current_time - self.phase_start_time
        
        # Minimum time check
        if phase_duration < 15:
            return False
        
        current_traffic = self.traffic_history[-1]
        
        # Calculate traffic metrics
        ns_traffic = current_traffic['north'] + current_traffic['south']
        ew_traffic = current_traffic['east'] + current_traffic['west']
        total_traffic = ns_traffic + ew_traffic
        
        # Assess traffic urgency
        urgency = self.assess_traffic_urgency(total_traffic, ns_traffic, ew_traffic)
        
        # Check if we should adapt based on traffic conditions
        should_adapt = self.should_adapt_based_on_traffic(ns_traffic, ew_traffic, phase_duration, urgency)
        
        if should_adapt:
            self.last_adaptation_reason = f"Smart adaptation ({urgency})"
            self.adaptations_made += 1
            return True
        
        # Check maximum duration based on urgency
        max_duration = self.adaptation_intervals.get(urgency, 45)
        return phase_duration >= max_duration
    
    def assess_traffic_urgency(self, total_traffic, ns_traffic, ew_traffic):
        """Assess the urgency level of current traffic situation."""
        
        # High total traffic
        if total_traffic > 20:
            return 'CRITICAL'
        elif total_traffic > 15:
            return 'URGENT'
        
        # Check for severe imbalance
        if total_traffic > 5:
            max_direction = max(ns_traffic, ew_traffic)
            min_direction = min(ns_traffic, ew_traffic)
            
            if min_direction == 0 and max_direction > 8:
                return 'CRITICAL'
            elif max_direction > 3 * min_direction and max_direction > 6:
                return 'URGENT'
        
        # Regular traffic assessment
        if total_traffic > 8:
            return 'NORMAL'
        elif total_traffic > 3:
            return 'LIGHT'
        else:
            return 'MINIMAL'
    
    def should_adapt_based_on_traffic(self, ns_traffic, ew_traffic, phase_duration, urgency):
        """Check if adaptation is needed based on traffic conditions."""
        
        # Current phase traffic vs opposite phase traffic
        if self.current_phase == 'north_south':
            current_phase_traffic = ns_traffic
            opposite_phase_traffic = ew_traffic
        else:
            current_phase_traffic = ew_traffic
            opposite_phase_traffic = ns_traffic
        
        # Critical situations - immediate adaptation needed
        if urgency == 'CRITICAL':
            if opposite_phase_traffic > current_phase_traffic * 2 and phase_duration >= 15:
                return True
            if opposite_phase_traffic > 12 and current_phase_traffic <= 2 and phase_duration >= 15:
                return True
        
        # Urgent situations - quick adaptation
        elif urgency == 'URGENT':
            if opposite_phase_traffic > current_phase_traffic * 1.8 and phase_duration >= 18:
                return True
            if opposite_phase_traffic > 8 and current_phase_traffic <= 1 and phase_duration >= 18:
                return True
        
        # Normal traffic - balanced adaptation
        elif urgency in ['NORMAL', 'LIGHT']:
            if opposite_phase_traffic > current_phase_traffic * 1.5 and phase_duration >= 22:
                return True
            if opposite_phase_traffic > 5 and current_phase_traffic == 0 and phase_duration >= 20:
                return True
        
        # Check if current phase has been running too long with no traffic
        if current_phase_traffic == 0 and opposite_phase_traffic > 0 and phase_duration >= 25:
            return True
        
        return False
    
    def change_phase(self):
        """Change to the opposite phase."""
        self.current_phase = 'east_west' if self.current_phase == 'north_south' else 'north_south'
        self.phase_start_time = self.current_time