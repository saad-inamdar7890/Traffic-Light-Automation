"""
Dynamic Traffic Light Management Module
Enhanced version with optimized adaptive control algorithms

Network Layout (Junction J4):
- From East: E0 → Junction J4 → E0.319 (To East)
- From West: -E0 → Junction J4 → -E0.254 (To West)  
- From North: E1 → Junction J4 → -E1.238 (To North)
- From South: -E1 → Junction J4 → E1.200 (To South)

Enhanced Features:
- Faster adaptation intervals (15s instead of 30s)
- Predictive traffic modeling
- Queue-aware pressure calculations
- Early phase transition capability
- Optimized timing algorithms
"""

import traci
import statistics
import math
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

class AdaptiveTrafficController:
    def __init__(self, junction_id="J4"):
        """Initialize the enhanced adaptive traffic light controller"""
        self.junction_id = junction_id
        
        # Optimized timing parameters
        self.min_green_time = 8   # Reduced minimum for faster response
        self.max_green_time = 80  # Increased maximum for heavy traffic
        self.default_green_time = 25  # Optimized default duration
        
        # Enhanced monitoring
        self.current_phase = 0
        self.phase_start_time = 0
        self.last_adaptation_time = 0
        self.adaptation_interval = 15  # Less aggressive - adapt every 15 seconds
        
        # Traffic history and prediction
        self.traffic_history = defaultdict(lambda: deque(maxlen=10))  # Increased history
        self.pressure_trend = defaultdict(lambda: deque(maxlen=5))
        self.queue_history = defaultdict(lambda: deque(maxlen=8))
        
        # Performance tracking
        self.adaptations_made = 0
        self.total_pressure_reduced = 0
        self.phase_extensions = defaultdict(int)
        self.phase_reductions = defaultdict(int)
        
        # Enhanced phase definitions with optimized base durations
        self.phases = {
            0: {"name": "All_Red_Start", "base_duration": 2, "directions": [], "state": "rrrrrrrrrrrrrrrrr"},
            1: {"name": "Mixed_Phase_1", "base_duration": 25, "directions": ["-E1", "E1"], "state": "GrrrgrrrrrrrGGGGr"},
            2: {"name": "Yellow_Transition_1", "base_duration": 3, "directions": [], "state": "yrrryrrrrrrryyyyr"},
            3: {"name": "EW_Through_Phase", "base_duration": 25, "directions": ["E0", "-E0"], "state": "grrrgrrrGGGGGrrrr"},
            4: {"name": "Yellow_Transition_2", "base_duration": 3, "directions": [], "state": "yrrryrrryyyyyrrrr"},
            5: {"name": "EW_Mixed_Phase", "base_duration": 25, "directions": ["E0"], "state": "grrrGGGGGrrrgrrrr"},
            6: {"name": "Yellow_Transition_3", "base_duration": 3, "directions": [], "state": "yrrryyyyyrrryrrrr"},
            7: {"name": "NS_Through_Phase", "base_duration": 25, "directions": ["-E0"], "state": "GGGGGrrrgrrrgrrrr"},
            8: {"name": "Yellow_Transition_4", "base_duration": 3, "directions": [], "state": "yyyyyrrryrrryrrrr"}
        }
        
        # Direction mappings for better pressure calculation
        self.direction_groups = {
            'north_south': ['E1', '-E1'],
            'east_west': ['E0', '-E0'],
            'north': ['E1'],
            'south': ['-E1'], 
            'east': ['E0'],
            'west': ['-E0']
        }
        
    def calculate_traffic_pressure(self, step_data, directions):
        """Enhanced pressure calculation with controlled growth to prevent numerical explosion"""
        if not step_data or 'edge_data' not in step_data:
            return 0
        
        total_pressure = 0
        
        for direction in directions:
            edge_data = step_data['edge_data'].get(direction, {})
            vehicles = edge_data.get('vehicle_count', 0)
            waiting_time = edge_data.get('waiting_time', 0)
            avg_speed = edge_data.get('avg_speed', 0)
            
            if vehicles == 0:
                continue
            
            # Controlled pressure components (prevent numerical explosion)
            
            # 1. Vehicle count pressure (logarithmic growth for congestion)
            vehicle_pressure = vehicles * (1 + min(3.0, math.log(max(1, vehicles)) * 0.5))
            
            # 2. Waiting time pressure (controlled exponential penalty)
            if waiting_time > 10:  # Penalty kicks in after 10 seconds
                # Cap the exponential growth to prevent explosion
                exp_factor = min(5.0, (waiting_time - 10) / 20)
                wait_pressure = waiting_time * (1 + exp_factor)
            else:
                wait_pressure = waiting_time * 0.5
            
            # 3. Speed pressure (controlled penalty for slow traffic)
            max_speed = 13.89  # ~50 km/h
            if avg_speed > 0:
                speed_ratio = avg_speed / max_speed
                if speed_ratio < 0.3:  # Very slow traffic
                    speed_pressure = vehicles * (1 - speed_ratio) * 2.0
                else:
                    speed_pressure = vehicles * (1 - speed_ratio) * 1.2
            else:
                speed_pressure = vehicles * 1.5  # Stopped traffic penalty
            
            # 4. Queue buildup pressure (based on recent history)
            historical_vehicles = self.queue_history[direction]
            if len(historical_vehicles) >= 3:
                recent_avg = statistics.mean(list(historical_vehicles)[-3:])
                if vehicles > recent_avg * 1.3:  # Growing queue
                    queue_pressure = min(vehicles * 0.5, (vehicles - recent_avg) * 1.5)  # Cap queue pressure
                else:
                    queue_pressure = 0
            else:
                queue_pressure = 0
            
            # Combine all pressure components with controlled scaling
            edge_pressure = (vehicle_pressure * 0.4 + 
                           wait_pressure * 0.3 + 
                           speed_pressure * 0.2 + 
                           queue_pressure * 0.1)
            
            # Cap individual edge pressure to prevent explosion
            edge_pressure = min(edge_pressure, vehicles * 10)  # Max 10x vehicle count
            
            total_pressure += edge_pressure
            
            # Store for trend analysis
            self.queue_history[direction].append(vehicles)
        
        return total_pressure
    
    def predict_traffic_trend(self, direction_group):
        """Predict traffic trend based on recent pressure history"""
        if direction_group not in self.pressure_trend:
            return 0
        
        history = list(self.pressure_trend[direction_group])
        if len(history) < 3:
            return 0
        
        # Simple linear trend analysis
        recent_values = history[-3:]
        if len(recent_values) >= 2:
            trend = recent_values[-1] - recent_values[0]
            return trend
        
        return 0
    
    def get_adaptive_duration(self, step_data, phase_id, current_phase_time=0):
        """Calculate optimized phase duration with predictive elements"""
        phase_info = self.phases.get(phase_id, {})
        phase_name = phase_info.get("name", "")
        base_duration = phase_info.get("base_duration", 25)
        
        # Don't adapt transition phases
        if phase_id in [0, 2, 4, 6, 8] or "Yellow" in phase_name or "All_Red" in phase_name:
            return base_duration
        
        directions = phase_info.get("directions", [])
        if not directions:
            return base_duration
        
        # Calculate current and opposing pressures
        current_pressure = self.calculate_traffic_pressure(step_data, directions)
        
        # Determine opposing directions
        opposing_directions = self._get_opposing_directions(phase_id)
        opposing_pressure = self.calculate_traffic_pressure(step_data, opposing_directions)
        
        # Store pressure history for trend analysis
        direction_key = "_".join(directions)
        self.pressure_trend[direction_key].append(current_pressure)
        
        # Get traffic trend
        trend = self.predict_traffic_trend(direction_key)
        
        # Enhanced duration calculation
        total_pressure = current_pressure + opposing_pressure
        
        if total_pressure < 5:  # Very light traffic
            return max(self.min_green_time, base_duration * 0.6)
        
        # Pressure ratio with trend adjustment
        if total_pressure > 0:
            pressure_ratio = current_pressure / total_pressure
            
            # Adjust for predicted trend
            if trend > 5:  # Traffic increasing
                pressure_ratio += 0.15
            elif trend < -5:  # Traffic decreasing
                pressure_ratio -= 0.1
            
            pressure_ratio = max(0.1, min(0.9, pressure_ratio))
        else:
            pressure_ratio = 0.5
        
        # Duration calculation with BALANCED logic for stability and performance
        if pressure_ratio > 0.75:  # High pressure - moderate extension
            duration = base_duration + (pressure_ratio - 0.5) * 40  # Reduced multiplier
        elif pressure_ratio > 0.6:  # Moderate-high pressure
            duration = base_duration + (pressure_ratio - 0.5) * 25
        elif pressure_ratio < 0.2:  # Low pressure - moderate cut
            duration = base_duration * 0.7  # Less drastic cut
        elif pressure_ratio < 0.35:  # Moderate-low pressure
            duration = base_duration * 0.8
        else:  # Moderate pressure
            duration = base_duration + (pressure_ratio - 0.5) * 15
        
        # BALANCED time-based adjustments
        if current_phase_time > base_duration * 0.75:  # Later intervention
            # More conservative extension criteria
            if current_pressure > opposing_pressure * 1.8:  # Higher threshold
                duration = min(self.max_green_time, duration * 1.2)  # Smaller extension
            else:
                duration = base_duration * 0.9  # Gentler termination
        
        # BALANCED opposing pressure penalties
        if opposing_pressure > current_pressure * 3:
            duration = min(duration, base_duration * 0.7)  # Moderate penalty
        elif opposing_pressure > current_pressure * 2:
            duration = min(duration, base_duration * 0.85)  # Light penalty
        
        # Balanced constraints
        min_duration = self.min_green_time  # Standard minimum
        max_duration = min(self.max_green_time, base_duration * 2.0)  # More conservative maximum
        
        final_duration = max(min_duration, min(max_duration, duration))
        
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
        """Comprehensive traffic state analysis with enhanced metrics"""
        if not step_data:
            return {}
        
        # Calculate pressures for all direction groups
        pressures = {}
        for group, directions in self.direction_groups.items():
            pressures[group] = self.calculate_traffic_pressure(step_data, directions)
        
        # Overall traffic metrics
        total_pressure = sum(pressures.values())
        total_vehicles = step_data.get('total_vehicles', 0)
        avg_waiting = step_data.get('avg_waiting_time', 0)
        
        # Direction priorities
        ns_pressure = pressures['north_south']
        ew_pressure = pressures['east_west']
        
        # Enhanced priority determination
        pressure_diff = abs(ns_pressure - ew_pressure)
        total_pressure_ne = ns_pressure + ew_pressure
        
        if total_pressure_ne > 0:
            imbalance_ratio = pressure_diff / total_pressure_ne
        else:
            imbalance_ratio = 0
        
        # Priority determination with hysteresis
        if imbalance_ratio > 0.4:  # Significant imbalance
            if ns_pressure > ew_pressure:
                priority = "NORTH_SOUTH"
                priority_message = "🔴 HIGH NS PRIORITY"
                confidence = min(0.95, imbalance_ratio)
            else:
                priority = "EAST_WEST"
                priority_message = "🔴 HIGH EW PRIORITY"
                confidence = min(0.95, imbalance_ratio)
        elif imbalance_ratio > 0.2:  # Moderate imbalance
            if ns_pressure > ew_pressure:
                priority = "NORTH_SOUTH_MODERATE"
                priority_message = "� MODERATE NS PRIORITY"
                confidence = imbalance_ratio
            else:
                priority = "EAST_WEST_MODERATE"
                priority_message = "� MODERATE EW PRIORITY"
                confidence = imbalance_ratio
        else:  # Balanced
            priority = "BALANCED"
            priority_message = "🟢 BALANCED TRAFFIC"
            confidence = 1.0 - imbalance_ratio
        
        # Optimal timing calculation
        base_cycle = 100  # Reduced cycle time for better responsiveness
        if total_pressure_ne > 0:
            ns_ratio = ns_pressure / total_pressure_ne
            ew_ratio = ew_pressure / total_pressure_ne
        else:
            ns_ratio = ew_ratio = 0.5
        
        # Calculate suggested timings with minimum guarantees
        suggested_ns_time = max(self.min_green_time, min(self.max_green_time, int(ns_ratio * base_cycle)))
        suggested_ew_time = max(self.min_green_time, min(self.max_green_time, int(ew_ratio * base_cycle)))
        
        # Traffic efficiency metrics
        if total_vehicles > 0:
            efficiency_score = max(0, 100 - (avg_waiting * 2))  # Penalize waiting time
            congestion_level = min(100, (total_pressure / max(1, total_vehicles)) * 10)
        else:
            efficiency_score = 100
            congestion_level = 0
        
        return {
            'pressures': pressures,
            'total_pressure': total_pressure,
            'ns_pressure': ns_pressure,
            'ew_pressure': ew_pressure,
            'imbalance_ratio': imbalance_ratio,
            'priority': priority,
            'priority_message': priority_message,
            'confidence': confidence,
            'suggested_ns_time': suggested_ns_time,
            'suggested_ew_time': suggested_ew_time,
            'ns_ratio': ns_ratio,
            'ew_ratio': ew_ratio,
            'efficiency_score': efficiency_score,
            'congestion_level': congestion_level,
            'total_vehicles': total_vehicles,
            'avg_waiting': avg_waiting
        }
    
    def apply_adaptive_control(self, step_data, current_step):
        """Apply enhanced adaptive control with faster response"""
        try:
            # Check if it's time to adapt
            if current_step - self.last_adaptation_time < self.adaptation_interval:
                return {'applied': False, 'reason': 'Too soon for adaptation'}
            
            # Get current traffic light state
            tl_state = self.get_current_traffic_light_state()
            analysis = self.analyze_traffic_state(step_data)
            
            # Apply adaptive logic for main phases only
            if tl_state['phase'] in [1, 3, 5, 7]:  # Main green phases
                phase_elapsed = current_step - self.phase_start_time
                
                # Check for early transition
                if self.should_trigger_early_transition(step_data, tl_state['phase'], phase_elapsed):
                    print(f"🚦 EARLY TRANSITION: Phase {tl_state['phase']} after {phase_elapsed}s")
                    self.last_adaptation_time = current_step
                    return {
                        'applied': True,
                        'action': 'early_transition',
                        'analysis': analysis,
                        'tl_state': tl_state
                    }
                
                # Calculate optimized duration
                optimized_duration = self.get_adaptive_duration(step_data, tl_state['phase'], phase_elapsed)
                base_duration = self.phases[tl_state['phase']]['base_duration']
                
                # Log adaptation if significant change
                if abs(optimized_duration - base_duration) > 3:
                    change_type = "EXTEND" if optimized_duration > base_duration else "REDUCE"
                    directions = self.phases[tl_state['phase']]['directions']
                    current_pressure = self.calculate_traffic_pressure(step_data, directions)
                    
                    print(f"🚦 {change_type}: Phase {tl_state['phase']} → {optimized_duration}s "
                          f"(Base: {base_duration}s, Pressure: {current_pressure:.1f})")
                    
                    self.adaptations_made += 1
                    if optimized_duration > base_duration:
                        self.phase_extensions[tl_state['phase']] += 1
                    else:
                        self.phase_reductions[tl_state['phase']] += 1
                
                self.last_adaptation_time = current_step
                
                return {
                    'applied': True,
                    'action': 'duration_optimization',
                    'optimized_duration': optimized_duration,
                    'base_duration': base_duration,
                    'analysis': analysis,
                    'tl_state': tl_state
                }
            
            return {
                'applied': False,
                'reason': 'Non-adaptable phase',
                'analysis': analysis,
                'tl_state': tl_state
            }
            
        except Exception as e:
            print(f"❌ Error in enhanced adaptive control: {e}")
            return {
                'applied': False,
                'error': str(e),
                'analysis': {},
                'tl_state': {}
            }
    
    def should_trigger_early_transition(self, step_data, current_phase, phase_elapsed):
        """Balanced early transition logic - less aggressive for stability"""
        if current_phase not in [1, 3, 5, 7]:  # Only main phases
            return False
        
        if phase_elapsed < self.min_green_time:  # Respect minimum time strictly
            return False
        
        phase_info = self.phases.get(current_phase, {})
        directions = phase_info.get("directions", [])
        
        current_pressure = self.calculate_traffic_pressure(step_data, directions)
        opposing_directions = self._get_opposing_directions(current_phase)
        opposing_pressure = self.calculate_traffic_pressure(step_data, opposing_directions)
        
        # BALANCED EARLY TRANSITION CONDITIONS (less aggressive):
        
        # 1. Strong opposition with reasonable current minimum
        if (opposing_pressure > current_pressure * 2.5 and 
            current_pressure < 20 and 
            phase_elapsed >= self.min_green_time + 3):
            return True
        
        # 2. Very low current traffic with significant opposition
        if (current_pressure < 8 and 
            opposing_pressure > 25 and 
            phase_elapsed >= self.min_green_time + 2):
            return True
            
        # 3. Emergency transition for extreme imbalance only
        if (opposing_pressure > current_pressure * 5 and 
            phase_elapsed >= self.min_green_time):
            return True
        
        return False
    
    def _get_opposing_directions(self, phase_id):
        """Get opposing directions for a given phase"""
        if phase_id == 1:  # Mixed_Phase_1 (NS movements)
            return ["E0", "-E0"]
        elif phase_id == 3:  # EW_Through_Phase 
            return ["E1", "-E1"]
        elif phase_id == 5:  # EW_Mixed_Phase
            return ["E1", "-E1", "-E0"]
        elif phase_id == 7:  # NS_Through_Phase
            return ["E0", "E1", "-E1"]
        else:
            return []
    
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
    
    def control_traffic_lights(self, current_time: float, traci_connection) -> Dict[str, Any]:
        """
        Main interface method for controlling traffic lights using TraCI.
        This method adapts the enhanced algorithm to the expected interface.
        
        Args:
            current_time: Current simulation time
            traci_connection: TraCI connection object
            
        Returns:
            Control action information
        """
        try:
            # Collect traffic data using the analyzer's format
            from analyzer import TrafficAnalyzer
            temp_analyzer = TrafficAnalyzer()
            step_data = temp_analyzer.collect_traffic_metrics(int(current_time), traci_connection)
            
            if step_data is None:
                return {
                    'algorithm': 'adaptive_enhanced',
                    'applied': False,
                    'reason': 'No traffic data available',
                    'timestamp': current_time
                }
            
            # Apply the enhanced adaptive control
            control_result = self.apply_adaptive_control(step_data, int(current_time))
            
            # Format response to match expected interface
            result = {
                'algorithm': 'adaptive_enhanced',
                'applied': control_result.get('action_taken', False),
                'action': control_result.get('action_description', 'no_action'),
                'reason': control_result.get('reason', 'Normal operation'),
                'traffic_data': step_data,
                'timestamp': current_time,
                'enhancement_details': {
                    'pressure_analysis': control_result.get('pressure_analysis', {}),
                    'phase_adjustments': control_result.get('phase_adjustments', {}),
                    'efficiency_gain': control_result.get('efficiency_gain', 0)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️  Error in enhanced traffic light control: {e}")
            return {
                'algorithm': 'adaptive_enhanced',
                'applied': False,
                'error': str(e),
                'timestamp': current_time
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive controller statistics.
        Adapts the enhanced performance summary to the expected interface.
        """
        try:
            performance_summary = self.get_performance_summary()
            
            # Calculate additional statistics
            total_adaptations = performance_summary.get('total_adaptations', 0)
            
            # Extract detailed statistics
            adaptations_log = []
            actions = {}
            
            for phase_key, history in performance_summary.get('phase_history', {}).items():
                if history.get('measurements', 0) > 0:
                    actions[f"{phase_key}_optimizations"] = history['measurements']
                    
                    # Create log entries for recent adaptations
                    adaptations_log.append({
                        'phase': phase_key,
                        'avg_pressure': history.get('avg_pressure', 0),
                        'max_pressure': history.get('max_pressure', 0),
                        'optimizations': history.get('measurements', 0)
                    })
            
            statistics_result = {
                'total_adaptations': total_adaptations,
                'actions': actions,
                'adaptations_log': adaptations_log[-5:],  # Last 5 adaptations
                'algorithm_type': 'enhanced_adaptive',
                'performance_metrics': {
                    'total_pressure_reduced': getattr(self, 'total_pressure_reduced', 0),
                    'phase_extensions': dict(getattr(self, 'phase_extensions', {})),
                    'phase_reductions': dict(getattr(self, 'phase_reductions', {})),
                    'adaptations_made': getattr(self, 'adaptations_made', 0)
                }
            }
            
            return statistics_result
            
        except Exception as e:
            print(f"⚠️  Error getting enhanced controller statistics: {e}")
            return {
                'total_adaptations': 0,
                'actions': {},
                'adaptations_log': [],
                'error': str(e)
            }
    
    def reset_controller(self):
        """Reset the enhanced controller state for a new simulation."""
        try:
            # Reset timing state
            self.current_phase = 0
            self.phase_start_time = 0
            self.last_adaptation_time = 0
            
            # Reset performance tracking
            self.adaptations_made = 0
            self.total_pressure_reduced = 0
            self.phase_extensions.clear()
            self.phase_reductions.clear()
            
            # Reset traffic history and prediction data
            self.traffic_history.clear()
            self.pressure_trend.clear()
            self.queue_history.clear()
            
            print(f"🔄 Enhanced traffic controller reset for junction {self.junction_id}")
            
        except Exception as e:
            print(f"⚠️  Error resetting enhanced controller: {e}")
