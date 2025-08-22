"""
Dynamic Traffic Light Management Module
Handles adaptive traffic light control based on real-time traffic conditions

Network Layout (Junction J4):
- From East: E0 → Junction J4 → E0.319 (To East)
- From West: -E0 → Junction J4 → -E0.254 (To West)  
- From North: E1 → Junction J4 → -E1.238 (To North)
- From South: -E1 → Junction J4 → E1.200 (To South)

Traffic Light Phases (9 phases, 120s cycle):
0: All Red (3s)
1: Mixed NS movements (30s) - Priority: North-South traffic
2: Yellow transition (3s)
3: EW Through movements (30s) - Priority: East-West through traffic
4: Yellow transition (3s) 
5: EW Mixed movements (30s) - Priority: East traffic
6: Yellow transition (3s)
7: NS Through movements (30s) - Priority: West traffic  
8: Yellow transition (3s)
"""

import traci
from collections import defaultdict, deque

class AdaptiveTrafficController:
    def __init__(self, junction_id="J4"):
        """Initialize the adaptive traffic light controller"""
        self.junction_id = junction_id
        self.min_green_time = 10  # Minimum green phase duration
        self.max_green_time = 60  # Maximum green phase duration
        self.current_phase = 0
        self.phase_start_time = 0
        self.traffic_history = defaultdict(lambda: deque(maxlen=5))
        
        # Traffic light phases based on actual network's tlLogic from demo.net.xml
        # State pattern: "rrrrrrrrrrrrrrrrr" represents 17 connections
        # Connections: ["-E1→E0.319", "-E1→-E1.238(0)", "-E1→-E1.238(1)", "-E1→-E0.254", 
        #              "E0→E1.200", "E0→E0.319(0)", "E0→E0.319(1)", "E0→-E1.238", 
        #              "E1→-E0.254", "E1→E1.200(0)", "E1→E1.200(1)", "E1→E0.319", 
        #              "-E0→-E1.238", "-E0→-E0.254(0)", "-E0→-E0.254(1)", "-E0→E1.200"]
        self.phases = {
            0: {"name": "All_Red_Start", "base_duration": 3, "directions": [], "state": "rrrrrrrrrrrrrrrrr"},
            1: {"name": "Mixed_Phase_1", "base_duration": 30, "directions": ["-E1", "E1"], "state": "GrrrgrrrrrrrGGGGr"},
            2: {"name": "Yellow_Transition_1", "base_duration": 3, "directions": [], "state": "yrrryrrrrrrryyyyr"},
            3: {"name": "EW_Through_Phase", "base_duration": 30, "directions": ["E0", "-E0"], "state": "grrrgrrrGGGGGrrrr"},
            4: {"name": "Yellow_Transition_2", "base_duration": 3, "directions": [], "state": "yrrryrrryyyyyrrrr"},
            5: {"name": "EW_Mixed_Phase", "base_duration": 30, "directions": ["E0"], "state": "grrrGGGGGrrrgrrrr"},
            6: {"name": "Yellow_Transition_3", "base_duration": 3, "directions": [], "state": "yrrryyyyyrrryrrrr"},
            7: {"name": "NS_Through_Phase", "base_duration": 30, "directions": ["-E0"], "state": "GGGGGrrrgrrrgrrrr"},
            8: {"name": "Yellow_Transition_4", "base_duration": 3, "directions": [], "state": "yyyyyrrryrrryrrrr"}
        }
        
    def calculate_traffic_pressure(self, step_data, directions):
        """Calculate traffic pressure for given directions"""
        if not step_data or 'edge_data' not in step_data:
            return 0
        
        total_vehicles = 0
        total_waiting = 0
        total_speed_factor = 0
        
        for direction in directions:
            edge_data = step_data['edge_data'].get(direction, {})
            vehicles = edge_data.get('vehicle_count', 0)
            waiting = edge_data.get('waiting_time', 0)
            avg_speed = edge_data.get('avg_speed', 0)
            
            total_vehicles += vehicles
            total_waiting += waiting
            
            # Speed factor: slower speeds indicate congestion
            if avg_speed > 0:
                speed_factor = max(0, (13.89 - avg_speed) / 13.89)  # Normalized speed factor
                total_speed_factor += speed_factor * vehicles
        
        # Weighted pressure score
        pressure = total_vehicles + (total_waiting / 10.0) + total_speed_factor
        return pressure
    
    def get_adaptive_duration(self, step_data, phase_id):
        """Calculate adaptive phase duration based on real-time traffic"""
        phase_info = self.phases.get(phase_id, {})
        phase_name = phase_info.get("name", "")
        base_duration = phase_info.get("base_duration", 30)
        
        # Only adapt main traffic phases (not yellow or all-red phases)
        if phase_id in [2, 4, 6, 8] or "Yellow" in phase_name or "All_Red" in phase_name:
            return base_duration
        
        directions = phase_info.get("directions", [])
        if not directions:
            return base_duration
        
        # Calculate traffic pressure for current directions
        current_pressure = self.calculate_traffic_pressure(step_data, directions)
        
        # Get opposing directions based on phase
        if phase_id == 1:  # Mixed_Phase_1 (NS movements)
            opposing_directions = ["E0", "-E0"]
        elif phase_id == 3:  # EW_Through_Phase 
            opposing_directions = ["E1", "-E1"]
        elif phase_id == 5:  # EW_Mixed_Phase
            opposing_directions = ["E1", "-E1"]
        elif phase_id == 7:  # NS_Through_Phase
            opposing_directions = ["E0"]
        else:
            opposing_directions = []
        
        opposing_pressure = self.calculate_traffic_pressure(step_data, opposing_directions)
        
        # Store historical data
        self.traffic_history[f"phase_{phase_id}"].append(current_pressure)
        
        # Adaptive calculation
        total_pressure = current_pressure + opposing_pressure
        if total_pressure == 0:
            return base_duration
        
        # Allocate time based on pressure ratio
        pressure_ratio = current_pressure / total_pressure
        
        # Calculate adaptive duration with more realistic bounds
        if pressure_ratio > 0.7:  # High pressure on current direction
            adaptive_duration = self.max_green_time * 0.9
        elif pressure_ratio > 0.5:  # Moderate pressure
            adaptive_duration = base_duration + (pressure_ratio * 25)
        else:  # Low pressure
            adaptive_duration = max(self.min_green_time, base_duration * 0.6)
        
        # Apply constraints (15-75 seconds for main phases)
        final_duration = max(15, min(75, adaptive_duration))
        
        return int(final_duration)
    
    def get_current_traffic_light_state(self):
        """Get current traffic light state information"""
        try:
            current_phase = traci.trafficlight.getPhase(self.junction_id)
            
            # Try to get remaining time (compatibility with different SUMO versions)
            try:
                next_switch = traci.trafficlight.getNextSwitch(self.junction_id)
                current_time = traci.simulation.getTime()
                remaining = max(0, next_switch - current_time)
            except:
                remaining = 0
            
            phase_info = self.phases.get(current_phase, {})
            
            return {
                'phase': current_phase,
                'phase_name': phase_info.get('name', f'Phase_{current_phase}'),
                'remaining': remaining,
                'base_duration': phase_info.get('base_duration', 15),
                'directions': phase_info.get('directions', [])
            }
        except Exception as e:
            return {
                'phase': None,
                'phase_name': 'Unknown',
                'remaining': 0,
                'base_duration': 15,
                'directions': []
            }
    
    def analyze_traffic_state(self, step_data):
        """Analyze current traffic state and provide recommendations"""
        if not step_data:
            return {}
        
        # Calculate pressures for main directions based on actual network edges
        # North-South: E1 (from north) and -E1 (from south)
        ns_pressure = self.calculate_traffic_pressure(step_data, ['E1', '-E1'])
        # East-West: E0 (from east) and -E0 (from west) 
        ew_pressure = self.calculate_traffic_pressure(step_data, ['E0', '-E0'])
        
        total_pressure = ns_pressure + ew_pressure
        
        # Calculate optimal timing ratios
        if total_pressure > 0:
            ns_ratio = ns_pressure / total_pressure
            ew_ratio = ew_pressure / total_pressure
        else:
            ns_ratio = ew_ratio = 0.5
        
        # Suggest timing (base cycle of 120 seconds - matching network cycle)
        base_cycle = 120
        suggested_ns_time = max(15, min(60, int(ns_ratio * base_cycle)))
        suggested_ew_time = max(15, min(60, int(ew_ratio * base_cycle)))
        
        # Determine priority
        if ns_pressure > ew_pressure * 1.3:
            priority = "NORTH_SOUTH"
            priority_message = "🔴 PRIORITIZE NORTH-SOUTH"
        elif ew_pressure > ns_pressure * 1.3:
            priority = "EAST_WEST"
            priority_message = "🔴 PRIORITIZE EAST-WEST"
        else:
            priority = "BALANCED"
            priority_message = "🟢 BALANCED TRAFFIC"
        
        return {
            'ns_pressure': ns_pressure,
            'ew_pressure': ew_pressure,
            'total_pressure': total_pressure,
            'suggested_ns_time': suggested_ns_time,
            'suggested_ew_time': suggested_ew_time,
            'priority': priority,
            'priority_message': priority_message,
            'ns_ratio': ns_ratio,
            'ew_ratio': ew_ratio
        }
    
    def apply_adaptive_control(self, step_data, current_step):
        """Apply adaptive control logic (placeholder for actual implementation)"""
        try:
            tl_state = self.get_current_traffic_light_state()
            analysis = self.analyze_traffic_state(step_data)
            
            if tl_state['phase'] is not None and "Green" in tl_state['phase_name']:
                # Calculate adaptive duration
                adaptive_duration = self.get_adaptive_duration(step_data, tl_state['phase'])
                
                # Log recommendation (actual implementation would modify timing)
                if adaptive_duration != tl_state['base_duration']:
                    directions = tl_state['directions']
                    pressure = self.calculate_traffic_pressure(step_data, directions)
                    print(f"🚦 Recommend {tl_state['phase_name']}: {adaptive_duration}s "
                          f"(Base: {tl_state['base_duration']}s, Pressure: {pressure:.1f})")
            
            return {
                'tl_state': tl_state,
                'analysis': analysis,
                'applied': True
            }
            
        except Exception as e:
            print(f"Error in adaptive control: {e}")
            return {
                'tl_state': self.get_current_traffic_light_state(),
                'analysis': {},
                'applied': False
            }
    
    def get_performance_summary(self):
        """Get performance summary of the adaptive controller"""
        total_adaptations = sum(len(hist) for hist in self.traffic_history.values())
        
        summary = {
            'total_adaptations': total_adaptations,
            'phase_history': {}
        }
        
        for phase_key, history in self.traffic_history.items():
            if history:
                import statistics
                summary['phase_history'][phase_key] = {
                    'avg_pressure': statistics.mean(history),
                    'max_pressure': max(history),
                    'min_pressure': min(history),
                    'measurements': len(history)
                }
        
        return summary
