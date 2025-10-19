"""
Baseline Fixed-Time Traffic Light Controller
==========================================

Simple fixed-time controller for comparison with your adaptive algorithm.
Uses standard 60s cycles with equal time for each direction.
"""

import traci
import time
from typing import Dict, Any

class BaselineFixedTimeController:
    """Fixed-time traffic light controller for baseline comparison."""
    
    def __init__(self, junction_id: str = "J4", cycle_time: int = 120):
        self.junction_id = junction_id
        self.cycle_time = cycle_time  # Total cycle time (120s default)
        self.phase_time = cycle_time // 2  # Each direction gets half the cycle (60s)
        self.current_phase = 0
        self.phase_start_time = 0
        self.total_cycles = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_wait_time': 0,
            'total_vehicles': 0,
            'phase_changes': 0
        }
        
        print("🔄 Baseline Fixed-Time Controller initialized")
        print(f"   Junction ID: {junction_id}")
        print(f"   Cycle Time: {cycle_time}s ({self.phase_time}s per direction)")
        print(f"   NO adaptation - pure fixed timing")
    
    def initialize_traffic_lights(self):
        """Initialize traffic light control."""
        try:
            programs = traci.trafficlight.getAllProgramLogics(self.junction_id)
            if programs:
                print(f"   Available programs: {len(programs)}")
            
            traci.trafficlight.setPhase(self.junction_id, self.current_phase)
            self.phase_start_time = traci.simulation.getTime()
            
            print(f"   Traffic lights initialized. Starting phase: {self.current_phase}")
            return True
            
        except Exception as e:
            print(f"   WARNING: Error initializing traffic lights: {e}")
            return False
    
    def apply_fixed_control(self, current_time: int) -> Dict[str, Any]:
        """Apply fixed-time traffic light control."""
        try:
            # Collect traffic data for metrics
            traffic_data = self.collect_traffic_data()
            self.update_performance_metrics(traffic_data)
            
            # Check if it's time to switch phases
            phase_duration = current_time - self.phase_start_time
            phase_changed = False
            
            if phase_duration >= self.phase_time:
                self.change_phase()
                phase_changed = True
                self.total_cycles += 0.5  # Half cycle completed
            
            return {
                'phase_changed': phase_changed,
                'phase_duration': phase_duration,
                'current_phase': self.current_phase,
                'total_cycles': self.total_cycles,
                'traffic_data': traffic_data
            }
            
        except Exception as e:
            print(f"   WARNING: Error in fixed control: {e}")
            return {
                'phase_changed': False,
                'phase_duration': 0,
                'current_phase': self.current_phase,
                'total_cycles': self.total_cycles,
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
    
    def change_phase(self):
        """Change to the opposite traffic light phase."""
        try:
            current_time = traci.simulation.getTime()
            
            # Toggle phase
            self.current_phase = 1 - self.current_phase
            traci.trafficlight.setPhase(self.junction_id, self.current_phase)
            
            # Update timing
            self.phase_start_time = current_time
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
                'total_cycles': self.total_cycles,
                'total_phase_changes': self.performance_metrics['phase_changes'],
                'average_wait_time': avg_wait_time,
                'total_vehicles_processed': self.performance_metrics['total_vehicles']
            }
            
        except Exception as e:
            print(f"   WARNING: Error getting performance summary: {e}")
            return {
                'total_cycles': self.total_cycles,
                'total_phase_changes': 0,
                'average_wait_time': 0,
                'total_vehicles_processed': 0
            }