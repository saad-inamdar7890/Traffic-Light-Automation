"""Adaptive Traffic Light Controller Module"""

import traci
from typing import Dict, Any

class AdaptiveTrafficController:
    def __init__(self, junction_id: str = "J4"):
        self.junction_id = junction_id
        self.adaptation_interval = 15
        self.last_adaptation_time = 0
        self.adaptation_count = 0
        self.adaptations_log = []
        self.enable_logging = True
        print(f"Adaptive Traffic Controller initialized for junction {junction_id}")
    
    def control_traffic_lights(self, current_time: float, traci_connection) -> Dict[str, Any]:
        try:
            vehicle_ids = traci_connection.vehicle.getIDList()
            if vehicle_ids:
                total_waiting_time = sum(traci_connection.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
                total_speed = sum(traci_connection.vehicle.getSpeed(vid) for vid in vehicle_ids)
                traffic_data = {
                    'vehicle_count': len(vehicle_ids),
                    'total_waiting_time': total_waiting_time,
                    'avg_waiting_time': total_waiting_time / len(vehicle_ids),
                    'avg_speed': total_speed / len(vehicle_ids),
                    'timestamp': current_time
                }
            else:
                traffic_data = {
                    'vehicle_count': 0,
                    'total_waiting_time': 0,
                    'avg_waiting_time': 0,
                    'avg_speed': 0,
                    'timestamp': current_time
                }
            
            should_adapt = (current_time - self.last_adaptation_time) >= self.adaptation_interval
            if should_adapt and vehicle_ids:
                self.last_adaptation_time = current_time
                self.adaptation_count += 1
                return {
                    'algorithm': 'adaptive',
                    'applied': True,
                    'action': f'adapted_at_{current_time}s',
                    'reason': 'Traffic conditions analyzed',
                    'traffic_data': traffic_data
                }
            else:
                return {
                    'algorithm': 'adaptive',
                    'applied': False,
                    'reason': 'No adaptation needed',
                    'traffic_data': traffic_data
                }
        except Exception as e:
            return {
                'algorithm': 'adaptive',
                'applied': False,
                'error': str(e),
                'timestamp': current_time
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_adaptations': self.adaptation_count,
            'actions': {},
            'adaptations_log': self.adaptations_log[-5:]
        }
    
    def reset_controller(self):
        self.last_adaptation_time = 0
        self.adaptation_count = 0
        self.adaptations_log = []
        if self.enable_logging:
            print(f"Traffic controller reset for junction {self.junction_id}")
