"""
Baseline Fixed-Time Traffic Light Controller using TRACI
=======================================================

Simple fixed-time controller for comparison with the improved adaptive controller.
"""

import traci
from typing import Dict, Any

class BaselineFixedController:
    """Fixed-time traffic light controller for baseline comparison."""
    
    def __init__(self, junction_id: str = "J4", cycle_time: int = 90):
        self.junction_id = junction_id
        self.cycle_time = cycle_time
        self.phase_duration = cycle_time // 2  # 45 seconds per phase
        self.current_phase = 0
        self.phase_start_time = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_wait_time': 0,
            'total_vehicles': 0,
            'phase_changes': 0
        }
        
        print(f"Baseline Fixed-Time Controller initialized")
        print(f"   Junction ID: {junction_id}")
        print(f"   Cycle time: {cycle_time}s ({self.phase_duration}s per phase)")
    
    def initialize_traffic_lights(self):
        """Initialize traffic light control."""
        try:
            traci.trafficlight.setPhase(self.junction_id, self.current_phase)
            self.phase_start_time = traci.simulation.getTime()
            print(f"   Traffic lights initialized. Starting phase: {self.current_phase}")
            return True
        except Exception as e:
            print(f"   WARNING: Error initializing traffic lights: {e}")
            return False
    
    def collect_traffic_data(self) -> Dict[str, Any]:
        """Collect basic traffic data for performance tracking."""
        try:
            current_time = traci.simulation.getTime()
            
            # Get lane data
            lanes = {
                'north': 'E1_0',
                'south': '-E1_0',
                'east': 'E0_0',
                'west': '-E0_0'
            }
            
            traffic_data = {
                'time': current_time,
                'total_vehicles': 0,
                'total_waiting': 0,
                'total_speed': 0,
                'lanes_with_speed': 0
            }
            
            for direction, lane_id in lanes.items():
                try:
                    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                    waiting_time = traci.lane.getWaitingTime(lane_id)
                    avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                    
                    traffic_data['total_vehicles'] += vehicle_count
                    traffic_data['total_waiting'] += waiting_time
                    
                    if avg_speed > 0:
                        traffic_data['total_speed'] += avg_speed
                        traffic_data['lanes_with_speed'] += 1
                    
                except Exception:
                    # Handle missing lanes gracefully
                    pass
            
            # Calculate averages
            total_vehicles = max(traffic_data['total_vehicles'], 1)
            avg_waiting_time = traffic_data['total_waiting'] / total_vehicles
            avg_speed = traffic_data['total_speed'] / max(traffic_data['lanes_with_speed'], 1)
            throughput = traffic_data['total_vehicles']
            
            return {
                'time': current_time,
                'total_vehicles': traffic_data['total_vehicles'],
                'total_waiting': traffic_data['total_waiting'],
                'avg_waiting_time': avg_waiting_time,
                'avg_speed': avg_speed,
                'throughput': throughput
            }
            
        except Exception as e:
            print(f"   WARNING: Error collecting traffic data: {e}")
            return {
                'time': 0, 
                'total_vehicles': 0, 
                'total_waiting': 0,
                'avg_waiting_time': 0,
                'avg_speed': 0,
                'throughput': 0
            }
    
    def should_change_phase(self, traffic_data: Dict[str, Any]) -> bool:
        """Check if it's time to change phase (fixed timing)."""
        current_time = traffic_data['time']
        phase_duration = current_time - self.phase_start_time
        return phase_duration >= self.phase_duration
    
    def change_phase(self):
        """Change traffic light phase."""
        try:
            self.current_phase = 1 - self.current_phase
            traci.trafficlight.setPhase(self.junction_id, self.current_phase)
            self.phase_start_time = traci.simulation.getTime()
            self.performance_metrics['phase_changes'] += 1
            
        except Exception as e:
            print(f"   WARNING: Error changing phase: {e}")
    
    def update_performance_metrics(self, traffic_data: Dict[str, Any]):
        """Update performance tracking metrics."""
        self.performance_metrics['total_wait_time'] += traffic_data['total_waiting']
        self.performance_metrics['total_vehicles'] += traffic_data['total_vehicles']
    
    def apply_fixed_time_control(self):
        """Apply fixed-time traffic light control logic."""
        try:
            traffic_data = self.collect_traffic_data()
            
            # Update performance metrics
            self.update_performance_metrics(traffic_data)
            
            # Check if phase should change
            if self.should_change_phase(traffic_data):
                self.change_phase()
                
        except Exception as e:
            print(f"   WARNING: Error in fixed-time control: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_vehicles = max(self.performance_metrics['total_vehicles'], 1)
        
        return {
            'avg_waiting_time': self.performance_metrics['total_wait_time'] / total_vehicles,
            'total_phase_changes': self.performance_metrics['phase_changes'],
            'total_vehicles': total_vehicles
        }