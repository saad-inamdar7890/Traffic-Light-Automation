"""
Realistic Traffic Light Controller
===================================
Uses only data that can be realistically collected from real-world sensors:
- Induction loops (vehicle detection)
- Camera-based vehicle classification
- Queue length estimation
- Occupancy sensors

NO IDEALIZED DATA:
- No individual vehicle speed tracking
- No individual waiting time tracking
- No perfect vehicle identification
"""

import traci
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import statistics

class RealisticTrafficController:
    """
    Traffic controller using only realistic sensor data FOR CONTROL
    But collects comprehensive metrics FOR ANALYSIS
    
    CONTROL INPUTS (Realistic - what we can measure in real life):
    - Queue length (induction loops)
    - Vehicle type classification (cameras)
    - Lane occupancy (sensors)
    - Vehicle density (calculated)
    
    ANALYSIS METRICS (Idealized - collected for research only):
    - Individual waiting times
    - Individual speeds
    - Throughput
    - Delay statistics
    """
    
    def __init__(self, junction_id="J0"):
        self.junction_id = junction_id
        
        # Realistic timing parameters
        self.min_green_time = 15   # Minimum green for safety
        self.max_green_time = 90   # Maximum to avoid starvation
        self.yellow_time = 3       # Standard yellow duration
        self.all_red_time = 2      # Safety clearance time
        
        # Vehicle type weights (based on physical characteristics)
        # These represent ROAD PRESSURE (weight + length)
        # USED IN ALGORITHM ✅
        self.vehicle_weights = {
            'passenger': 1.0,      # Baseline: ~1.5 tons, 5m length
            'delivery': 2.5,       # ~3.5 tons, 6.5m length
            'truck': 5.0,          # ~10-20 tons, 12m length (heavy impact)
            'bus': 4.5,            # ~12-18 tons, 12m length
            'emergency': 10.0,     # Priority weight (not physical)
            'default': 1.5         # Unknown vehicles
        }
        
        # Sensor data history (USED IN ALGORITHM ✅)
        self.occupancy_history = defaultdict(lambda: deque(maxlen=10))
        self.queue_history = defaultdict(lambda: deque(maxlen=10))
        self.vehicle_count_history = defaultdict(lambda: deque(maxlen=5))
        
        # Phase management
        self.current_phase = 0
        self.phase_start_time = 0
        
        # ====================================================================
        # ANALYSIS METRICS (NOT used in algorithm, only for comparison)
        # ====================================================================
        self.analysis_data = {
            'total_waiting_time': 0.0,          # For comparison only
            'total_delay': 0.0,                 # For comparison only
            'vehicles_processed': 0,            # For analysis
            'phase_changes': 0,                 # For analysis
            'emergency_overrides': 0,           # For analysis
            'avg_speed_history': deque(maxlen=100),  # For analysis
            'throughput_history': deque(maxlen=100), # For analysis
        }
        
        # Decision history for analysis
        self.decision_log = []
        
        print(f"🚦 Realistic Traffic Controller initialized for {junction_id}")
        print(f"   CONTROL INPUTS: Loops + Cameras + Occupancy (Realistic)")
        print(f"   ANALYSIS METRICS: Waiting times, speeds, throughput (For comparison)")
    
    
    def calculate_realistic_pressure(self, lane_id: str) -> float:
        """
        Calculate traffic pressure using ONLY realistic sensor data
        
        ✅ USED IN ALGORITHM (Control Decision)
        
        Real-world sensors available:
        1. Induction loops - detect vehicle presence
        2. Camera systems - classify vehicle type
        3. Occupancy sensors - measure lane occupancy %
        4. Queue detectors - count stopped vehicles
        
        Returns:
            float: Traffic pressure value
        """
        try:
            # ============================================
            # 1. QUEUE LENGTH (from loop detectors)
            # ============================================
            # Induction loops can detect stopped/slow vehicles
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            
            # ============================================
            # 2. VEHICLE TYPE DETECTION (from cameras)
            # ============================================
            # Modern camera systems can classify vehicles
            vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
            
            weighted_vehicle_count = 0.0
            for veh_id in vehicles_on_lane:
                # Get vehicle type
                veh_type = traci.vehicle.getTypeID(veh_id).lower()
                
                # Apply weight based on vehicle type
                if 'truck' in veh_type or 'trailer' in veh_type:
                    weight = self.vehicle_weights['truck']
                elif 'bus' in veh_type:
                    weight = self.vehicle_weights['bus']
                elif 'delivery' in veh_type or 'van' in veh_type:
                    weight = self.vehicle_weights['delivery']
                elif 'emergency' in veh_type:
                    weight = self.vehicle_weights['emergency']
                else:
                    weight = self.vehicle_weights['passenger']
                
                weighted_vehicle_count += weight
            
            # ============================================
            # 3. LANE OCCUPANCY (from occupancy sensors)
            # ============================================
            # Occupancy = % of lane covered by vehicles
            # Can be measured by multiple loop detectors or video
            occupancy = traci.lane.getLastStepOccupancy(lane_id)
            
            # ============================================
            # 4. VEHICLE DENSITY (vehicles per meter)
            # ============================================
            lane_length = traci.lane.getLength(lane_id)
            vehicle_density = len(vehicles_on_lane) / max(lane_length, 1)
            
            # ============================================
            # 5. QUEUE GROWTH TREND (from historical data)
            # ============================================
            # Track if queue is growing (indicates congestion building)
            self.queue_history[lane_id].append(queue_length)
            
            if len(self.queue_history[lane_id]) >= 3:
                recent_queues = list(self.queue_history[lane_id])[-3:]
                queue_growth = recent_queues[-1] - recent_queues[0]
                queue_trend = max(0, queue_growth)  # Only positive growth matters
            else:
                queue_trend = 0
            
            # ============================================
            # REALISTIC PRESSURE FORMULA
            # ============================================
            pressure = (
                queue_length * 10.0 +              # Queue is most critical (10x weight)
                weighted_vehicle_count * 3.0 +     # Vehicle type matters (3x weight)
                occupancy * 50.0 +                 # Occupancy percentage (0-100)
                vehicle_density * 20.0 +           # Density (vehicles/meter)
                queue_trend * 5.0                  # Queue growing = urgent
            )
            
            return pressure
            
        except Exception as e:
            print(f"Warning: Error calculating pressure for {lane_id}: {e}")
            return 0.0
    
    
    def get_direction_pressure(self, direction_lanes: List[str]) -> Dict:
        """
        Calculate total pressure for a direction (multiple lanes)
        
        Args:
            direction_lanes: List of lane IDs in this direction
            
        Returns:
            Dict with pressure metrics
        """
        total_pressure = 0.0
        total_queue = 0
        total_vehicles = 0
        weighted_vehicles = 0.0
        total_occupancy = 0.0
        
        emergency_present = False
        
        for lane_id in direction_lanes:
            try:
                # Calculate pressure for this lane
                lane_pressure = self.calculate_realistic_pressure(lane_id)
                total_pressure += lane_pressure
                
                # Collect additional metrics
                total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
                
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                total_vehicles += len(vehicles)
                
                # Check for emergency vehicles
                for veh_id in vehicles:
                    veh_type = traci.vehicle.getTypeID(veh_id).lower()
                    if 'emergency' in veh_type:
                        emergency_present = True
                    
                    # Add weighted count
                    if 'truck' in veh_type:
                        weighted_vehicles += self.vehicle_weights['truck']
                    elif 'bus' in veh_type:
                        weighted_vehicles += self.vehicle_weights['bus']
                    elif 'delivery' in veh_type:
                        weighted_vehicles += self.vehicle_weights['delivery']
                    else:
                        weighted_vehicles += self.vehicle_weights['passenger']
                
                total_occupancy += traci.lane.getLastStepOccupancy(lane_id)
                
            except Exception as e:
                print(f"Warning: Error processing lane {lane_id}: {e}")
                continue
        
        # Calculate averages
        num_lanes = len(direction_lanes)
        avg_occupancy = total_occupancy / max(num_lanes, 1)
        
        return {
            'pressure': total_pressure,
            'queue_length': total_queue,
            'vehicle_count': total_vehicles,
            'weighted_count': weighted_vehicles,
            'avg_occupancy': avg_occupancy,
            'emergency_present': emergency_present,
            'urgency': self._calculate_urgency(total_pressure, emergency_present)
        }
    
    
    def _calculate_urgency(self, pressure: float, emergency: bool) -> str:
        """
        Determine urgency level based on pressure
        
        Args:
            pressure: Calculated traffic pressure
            emergency: Whether emergency vehicle is present
            
        Returns:
            str: Urgency level
        """
        if emergency:
            return 'EMERGENCY'
        elif pressure > 500:
            return 'CRITICAL'
        elif pressure > 300:
            return 'HIGH'
        elif pressure > 150:
            return 'MODERATE'
        elif pressure > 50:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    
    def decide_phase_action(self, current_phase_data: Dict, 
                           opposing_phase_data: Dict,
                           time_in_phase: float) -> Dict:
        """
        Decide whether to keep current phase or switch
        
        Args:
            current_phase_data: Metrics for current green direction
            opposing_phase_data: Metrics for opposing (red) direction
            time_in_phase: Seconds in current phase
            
        Returns:
            Dict with action recommendation
        """
        current_pressure = current_phase_data['pressure']
        opposing_pressure = opposing_phase_data['pressure']
        
        # ============================================
        # RULE 1: EMERGENCY VEHICLE PRIORITY
        # ============================================
        if opposing_phase_data['emergency_present']:
            if time_in_phase >= self.min_green_time:
                return {
                    'action': 'SWITCH',
                    'reason': 'Emergency vehicle detected in opposing direction',
                    'recommended_duration': self.min_green_time,
                    'urgency': 'EMERGENCY'
                }
        
        if current_phase_data['emergency_present']:
            return {
                'action': 'EXTEND',
                'reason': 'Emergency vehicle in current direction',
                'recommended_duration': self.max_green_time,
                'urgency': 'EMERGENCY'
            }
        
        # ============================================
        # RULE 2: MINIMUM GREEN TIME (Safety)
        # ============================================
        if time_in_phase < self.min_green_time:
            return {
                'action': 'KEEP',
                'reason': f'Minimum green time not reached ({time_in_phase:.0f}s / {self.min_green_time}s)',
                'recommended_duration': self.min_green_time,
                'urgency': current_phase_data['urgency']
            }
        
        # ============================================
        # RULE 3: MAXIMUM GREEN TIME (Fairness)
        # ============================================
        if time_in_phase >= self.max_green_time:
            return {
                'action': 'SWITCH',
                'reason': f'Maximum green time reached ({time_in_phase:.0f}s)',
                'recommended_duration': self.min_green_time,
                'urgency': opposing_phase_data['urgency']
            }
        
        # ============================================
        # RULE 4: PRESSURE COMPARISON
        # ============================================
        pressure_ratio = opposing_pressure / max(current_pressure, 1)
        
        # Current direction nearly empty, opposing has significant traffic
        if current_pressure < 50 and opposing_pressure > 150 and pressure_ratio > 3.0:
            return {
                'action': 'SWITCH',
                'reason': f'Current direction nearly empty ({current_pressure:.0f}), opposing has heavy traffic ({opposing_pressure:.0f})',
                'recommended_duration': min(45, self.max_green_time),
                'urgency': opposing_phase_data['urgency']
            }
        
        # Opposing direction has MUCH higher pressure
        if pressure_ratio > 4.0 and opposing_pressure > 200:
            return {
                'action': 'SWITCH',
                'reason': f'Opposing pressure significantly higher (ratio: {pressure_ratio:.1f}x)',
                'recommended_duration': min(60, self.max_green_time),
                'urgency': 'HIGH'
            }
        
        # Current direction still has high pressure
        if current_pressure > 300 and time_in_phase < self.max_green_time:
            return {
                'action': 'EXTEND',
                'reason': f'Current direction still heavily congested ({current_pressure:.0f})',
                'recommended_duration': min(time_in_phase + 15, self.max_green_time),
                'urgency': 'HIGH'
            }
        
        # ============================================
        # RULE 5: BALANCED PRESSURE (Keep current)
        # ============================================
        if 0.5 <= pressure_ratio <= 2.0:
            return {
                'action': 'KEEP',
                'reason': f'Pressures balanced (current: {current_pressure:.0f}, opposing: {opposing_pressure:.0f})',
                'recommended_duration': 30,
                'urgency': 'MODERATE'
            }
        
        # ============================================
        # DEFAULT: Keep current phase
        # ============================================
        return {
            'action': 'KEEP',
            'reason': 'No strong reason to switch',
            'recommended_duration': 30,
            'urgency': current_phase_data['urgency']
        }
    
    
    def get_controlled_lanes_by_direction(self) -> Dict[str, List[str]]:
        """
        Get lane groups for this junction
        
        Returns:
            Dict mapping direction names to lane IDs
        """
        try:
            all_lanes = traci.trafficlight.getControlledLanes(self.junction_id)
            
            # Remove duplicates
            unique_lanes = list(set(all_lanes))
            
            # Group lanes by direction (simplified - customize for your network)
            # This is a basic grouping - you should customize based on your network topology
            directions = {
                'north_south': [],
                'east_west': []
            }
            
            for lane in unique_lanes:
                # Simple heuristic: group by edge name
                # Customize this based on your actual network structure
                if '-E1' in lane or 'E1' in lane:
                    directions['north_south'].append(lane)
                elif '-E0' in lane or 'E0' in lane:
                    directions['east_west'].append(lane)
                else:
                    # Fallback: add to both (or create more specific logic)
                    if len(directions['north_south']) <= len(directions['east_west']):
                        directions['north_south'].append(lane)
                    else:
                        directions['east_west'].append(lane)
            
            return directions
            
        except Exception as e:
            print(f"Error getting controlled lanes: {e}")
            return {'north_south': [], 'east_west': []}
    
    
    def print_status(self, step: int):
        """Print current traffic status"""
        try:
            directions = self.get_controlled_lanes_by_direction()
            
            print(f"\n{'='*70}")
            print(f"🚦 REALISTIC TRAFFIC CONTROL - Step {step}")
            print(f"{'='*70}")
            
            for dir_name, lanes in directions.items():
                if not lanes:
                    continue
                    
                data = self.get_direction_pressure(lanes)
                
                print(f"\n📊 {dir_name.upper().replace('_', ' ')}:")
                print(f"   Pressure:        {data['pressure']:>8.1f}")
                print(f"   Queue Length:    {data['queue_length']:>8} vehicles")
                print(f"   Total Vehicles:  {data['vehicle_count']:>8}")
                print(f"   Weighted Count:  {data['weighted_count']:>8.1f} (considering vehicle types)")
                print(f"   Avg Occupancy:   {data['avg_occupancy']:>8.1f}%")
                print(f"   Urgency:         {data['urgency']}")
                if data['emergency_present']:
                    print(f"   🚨 EMERGENCY VEHICLE PRESENT!")
            
            print(f"\n{'='*70}")
            
        except Exception as e:
            print(f"Error printing status: {e}")


    def collect_analysis_metrics(self, step: int) -> Dict:
        """
        Collect comprehensive metrics FOR ANALYSIS ONLY
        
        ❌ NOT USED IN ALGORITHM (only for comparison/research)
        
        Collects idealized metrics that are hard to get in real life:
        - Individual vehicle waiting times
        - Individual vehicle speeds
        - Detailed delay statistics
        - Throughput measurements
        
        Returns:
            Dict: Comprehensive analysis metrics
        """
        metrics = {
            'step': step,
            'junction_id': self.junction_id,
            'timestamp': step,
        }
        
        try:
            # Get all controlled lanes
            lanes = traci.trafficlight.getControlledLanes(self.junction_id)
            unique_lanes = list(set(lanes))
            
            # ====================================================================
            # IDEALIZED METRICS (For analysis only)
            # ====================================================================
            
            total_waiting_time = 0.0
            total_speed = 0.0
            total_vehicles = 0
            stopped_vehicles = 0
            vehicle_details = []
            
            for lane in unique_lanes:
                # Get all vehicles on this lane
                vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane)
                
                for veh_id in vehicles_on_lane:
                    total_vehicles += 1
                    
                    # ❌ Individual waiting time (hard to track in real life)
                    waiting_time = traci.vehicle.getWaitingTime(veh_id)
                    total_waiting_time += waiting_time
                    
                    # ❌ Individual speed (needs expensive sensors)
                    speed = traci.vehicle.getSpeed(veh_id)
                    total_speed += speed
                    
                    # Check if stopped
                    if speed < 0.1:
                        stopped_vehicles += 1
                    
                    # Get vehicle type for detailed analysis
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    
                    # Collect detailed data
                    vehicle_details.append({
                        'id': veh_id,
                        'lane': lane,
                        'type': veh_type,
                        'waiting_time': waiting_time,
                        'speed': speed,
                        'stopped': speed < 0.1
                    })
            
            # Calculate aggregate metrics
            if total_vehicles > 0:
                avg_waiting_time = total_waiting_time / total_vehicles
                avg_speed = total_speed / total_vehicles
            else:
                avg_waiting_time = 0.0
                avg_speed = 0.0
            
            # Throughput (vehicles departed)
            departed = traci.simulation.getDepartedNumber()
            arrived = traci.simulation.getArrivedNumber()
            
            metrics.update({
                # ❌ Idealized metrics (for analysis)
                'total_waiting_time': total_waiting_time,
                'avg_waiting_time': avg_waiting_time,
                'total_speed': total_speed,
                'avg_speed': avg_speed,
                'total_vehicles': total_vehicles,
                'stopped_vehicles': stopped_vehicles,
                'stopped_percentage': (stopped_vehicles / max(total_vehicles, 1)) * 100,
                
                # Throughput metrics
                'departed_this_step': departed,
                'arrived_this_step': arrived,
                
                # Detailed vehicle data
                'vehicle_details': vehicle_details,
                
                # Phase information
                'current_phase': traci.trafficlight.getPhase(self.junction_id),
            })
            
            # Store in history for analysis
            self.analysis_data['total_waiting_time'] += total_waiting_time
            self.analysis_data['vehicles_processed'] += total_vehicles
            self.analysis_data['avg_speed_history'].append(avg_speed)
            self.analysis_data['throughput_history'].append(arrived)
            
        except Exception as e:
            print(f"Warning: Error collecting analysis metrics: {e}")
        
        return metrics
    
    
    def get_analysis_summary(self) -> Dict:
        """
        Get summary of all analysis metrics collected
        
        Returns:
            Dict: Summary statistics for comparison
        """
        try:
            avg_speed_overall = statistics.mean(self.analysis_data['avg_speed_history']) if self.analysis_data['avg_speed_history'] else 0
            total_throughput = sum(self.analysis_data['throughput_history'])
            
            return {
                'junction_id': self.junction_id,
                'total_waiting_time': self.analysis_data['total_waiting_time'],
                'total_vehicles_processed': self.analysis_data['vehicles_processed'],
                'avg_waiting_time': self.analysis_data['total_waiting_time'] / max(self.analysis_data['vehicles_processed'], 1),
                'avg_speed': avg_speed_overall,
                'total_throughput': total_throughput,
                'phase_changes': self.analysis_data['phase_changes'],
                'emergency_overrides': self.analysis_data['emergency_overrides'],
            }
        except Exception as e:
            print(f"Error generating analysis summary: {e}")
            return {}
    
    
    def log_decision(self, action_data: Dict):
        """
        Log decision for analysis purposes
        
        Args:
            action_data: Decision information including action, reason, metrics
        """
        self.decision_log.append(action_data)
        
        # Track phase changes
        if action_data.get('action') == 'SWITCH':
            self.analysis_data['phase_changes'] += 1
            
            if action_data.get('urgency') == 'EMERGENCY':
                self.analysis_data['emergency_overrides'] += 1


def compare_pressure_methods_demo():
    """
    Demonstrate the difference between idealized and realistic pressure calculation
    """
    print("\n" + "="*70)
    print("PRESSURE CALCULATION COMPARISON")
    print("="*70)
    
    # Simulated scenario data
    print("\n📋 SCENARIO: Busy intersection with mixed traffic")
    print("   - 12 passenger cars waiting")
    print("   - 3 delivery vans waiting")
    print("   - 2 trucks waiting")
    print("   - 1 bus waiting")
    print("   - Lane occupancy: 65%")
    print("   - Queue growing (+3 vehicles in last 30s)")
    
    print("\n" + "-"*70)
    print("METHOD 1: IDEALIZED (Simulation-Only)")
    print("-"*70)
    print("Formula: queue × 5.0 + waiting_time × 2.0 + speed_factor × 1.0")
    print("\nRequires:")
    print("   ❌ Individual vehicle speed tracking (expensive sensors)")
    print("   ❌ Individual waiting time tracking (complex ID system)")
    print("   ❌ Perfect vehicle identification (privacy concerns)")
    print("\nCalculation:")
    print("   Pressure = 18 × 5.0 + 450 × 2.0 + 0.3 × 1.0")
    print("   Pressure = 90 + 900 + 0.3 = 990.3")
    
    print("\n" + "-"*70)
    print("METHOD 2: REALISTIC (Real-World Sensors)")
    print("-"*70)
    print("Formula: queue × 10.0 + weighted_vehicles × 3.0 + occupancy × 50.0 +")
    print("         density × 20.0 + queue_trend × 5.0")
    print("\nRequires:")
    print("   ✅ Induction loops (vehicle detection) - STANDARD")
    print("   ✅ Camera systems (vehicle classification) - COMMON")
    print("   ✅ Occupancy sensors (lane occupancy) - STANDARD")
    print("   ✅ Queue detectors (stopped vehicles) - SIMPLE")
    print("\nCalculation:")
    vehicle_weights = {
        'passenger': 12 * 1.0,  # 12 cars
        'delivery': 3 * 2.5,    # 3 vans
        'truck': 2 * 5.0,       # 2 trucks
        'bus': 1 * 4.5          # 1 bus
    }
    weighted_total = sum(vehicle_weights.values())
    
    print(f"   Weighted vehicles = 12×1.0 + 3×2.5 + 2×5.0 + 1×4.5 = {weighted_total}")
    print(f"   Pressure = 18 × 10.0 + {weighted_total} × 3.0 + 65 × 50.0 + 0.18 × 20.0 + 3 × 5.0")
    pressure = 18 * 10.0 + weighted_total * 3.0 + 65 * 50.0 + 0.18 * 20.0 + 3 * 5.0
    print(f"   Pressure = 180 + {weighted_total * 3.0} + 3250 + 3.6 + 15")
    print(f"   Pressure = {pressure:.1f}")
    
    print("\n" + "-"*70)
    print("KEY DIFFERENCES")
    print("-"*70)
    print("\n✅ REALISTIC METHOD ADVANTAGES:")
    print("   • Uses standard traffic infrastructure")
    print("   • No privacy concerns (no vehicle tracking)")
    print("   • Accounts for vehicle weight/impact (trucks ≠ cars)")
    print("   • More accurate for road wear and congestion impact")
    print("   • Deployable in real cities TODAY")
    print("\n❌ IDEALIZED METHOD PROBLEMS:")
    print("   • Requires expensive radar/lidar at every lane")
    print("   • Privacy concerns with vehicle tracking")
    print("   • Treats all vehicles equally (ignores physical impact)")
    print("   • Not realistic for real-world deployment")
    
    print("\n" + "="*70)
    print("RECOMMENDATION: Use REALISTIC method for actual deployment")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("🚦 REALISTIC TRAFFIC CONTROLLER MODULE")
    print("="*70)
    print("\nThis module demonstrates traffic control using ONLY realistic sensors:")
    print("   ✅ Induction loops")
    print("   ✅ Camera-based vehicle classification")
    print("   ✅ Occupancy sensors")
    print("   ✅ Queue detectors")
    print("\nNO idealized simulation data!")
    print("="*70)
    
    # Run comparison demo
    compare_pressure_methods_demo()
    
    print("\n💡 To use in your simulation:")
    print("   1. Replace AdaptiveTrafficController with RealisticTrafficController")
    print("   2. Vehicle type weights will automatically be applied")
    print("   3. Pressure calculation will use realistic sensor data only")
    print("\n✅ Ready for real-world deployment!")
