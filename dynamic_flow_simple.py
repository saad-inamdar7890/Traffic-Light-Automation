import os
import sys
import random
import traci
import time
from collections import defaultdict, deque
import statistics

# ✅ Set SUMO_HOME if not already set in Environment Variables
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

# Path to your SUMO config file
sumoCmd = ["sumo-gui", "-c", "demo.sumocfg"]

# Traffic analysis data storage
traffic_data = {
    'waiting_times': defaultdict(list),
    'queue_lengths': defaultdict(list),
    'avg_speeds': defaultdict(list),
    'flow_rates': defaultdict(list),
    'pressure_history': defaultdict(list)
}

# Define the edges and junction
edges = ["-E0", "-E0.254", "-E1", "-E1.238", "E0", "E0.319", "E1", "E1.200"]
junction_id = "J4"

class DynamicTrafficAnalyzer:
    def __init__(self):
        self.update_interval = 60  # Update analysis every minute
        self.last_update = 0
        self.time_of_day_factors = {
            'night': 0.3,      # 22:00 - 06:00
            'morning': 1.8,    # 07:00 - 09:00  
            'day': 1.0,        # 10:00 - 16:00
            'evening': 1.6,    # 17:00 - 19:00
            'late': 0.7        # 20:00 - 21:00
        }
        
    def get_time_factor(self, step):
        """Get traffic factor based on time of day"""
        hour = (step / 3600) % 24
        
        if 22 <= hour or hour <= 6:
            return self.time_of_day_factors['night']
        elif 7 <= hour <= 9:
            return self.time_of_day_factors['morning']
        elif 10 <= hour <= 16:
            return self.time_of_day_factors['day']
        elif 17 <= hour <= 19:
            return self.time_of_day_factors['evening']
        else:
            return self.time_of_day_factors['late']
    
    def calculate_dynamic_pressure(self, step_data, directions):
        """Calculate traffic pressure for given directions"""
        if not step_data or 'edge_data' not in step_data:
            return 0
        
        pressure = 0
        for direction in directions:
            edge_data = step_data['edge_data'].get(direction, {})
            vehicles = edge_data.get('vehicle_count', 0)
            waiting = edge_data.get('waiting_time', 0)
            speed = edge_data.get('avg_speed', 13.89)
            
            # Pressure calculation considering vehicles, waiting time, and speed reduction
            speed_factor = max(0, (13.89 - speed) / 13.89)
            pressure += vehicles + (waiting / 10.0) + (speed_factor * vehicles * 0.5)
        
        return pressure
    
    def analyze_traffic_patterns(self, step_data, step):
        """Analyze traffic patterns and suggest optimizations"""
        if not step_data:
            return {}
        
        # Calculate pressures for each direction
        ns_pressure = self.calculate_dynamic_pressure(step_data, ['-E1', 'E1'])
        ew_pressure = self.calculate_dynamic_pressure(step_data, ['E0', '-E0'])
        
        # Store historical data
        traffic_data['pressure_history']['ns'].append(ns_pressure)
        traffic_data['pressure_history']['ew'].append(ew_pressure)
        
        # Calculate optimal timing based on pressure
        total_pressure = ns_pressure + ew_pressure
        if total_pressure > 0:
            ns_ratio = ns_pressure / total_pressure
            ew_ratio = ew_pressure / total_pressure
        else:
            ns_ratio = ew_ratio = 0.5
        
        # Suggested timing (in seconds)
        base_cycle = 60  # Base cycle time
        ns_time = max(10, min(40, int(ns_ratio * base_cycle)))
        ew_time = max(10, min(40, int(ew_ratio * base_cycle)))
        
        return {
            'ns_pressure': ns_pressure,
            'ew_pressure': ew_pressure,
            'suggested_ns_time': ns_time,
            'suggested_ew_time': ew_time,
            'total_pressure': total_pressure,
            'hour': (step / 3600) % 24
        }

def collect_comprehensive_metrics(step):
    """Collect comprehensive traffic metrics"""
    vehicle_ids = traci.vehicle.getIDList()
    
    if not vehicle_ids:
        return None
    
    step_data = {
        'step': step,
        'total_vehicles': len(vehicle_ids),
        'waiting_vehicles': 0,
        'edge_data': {},
        'individual_vehicles': []
    }
    
    # Collect edge data
    for edge_id in edges:
        try:
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
            avg_speed = traci.edge.getLastStepMeanSpeed(edge_id)
            waiting_time = traci.edge.getWaitingTime(edge_id)
            
            step_data['edge_data'][edge_id] = {
                'vehicle_count': vehicle_count,
                'avg_speed': avg_speed,
                'waiting_time': waiting_time
            }
            
            # Update global tracking
            traffic_data['queue_lengths'][edge_id].append(vehicle_count)
            traffic_data['avg_speeds'][edge_id].append(avg_speed)
            
        except Exception as e:
            print(f"Error collecting data for edge {edge_id}: {e}")
    
    # Collect individual vehicle data
    total_waiting = 0
    for veh_id in vehicle_ids:
        try:
            waiting = traci.vehicle.getWaitingTime(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            edge = traci.vehicle.getRoadID(veh_id)
            
            step_data['individual_vehicles'].append({
                'id': veh_id,
                'waiting': waiting,
                'speed': speed,
                'edge': edge
            })
            
            if waiting > 0:
                step_data['waiting_vehicles'] += 1
                total_waiting += waiting
                traffic_data['waiting_times'][edge].append(waiting)
                
        except Exception as e:
            continue
    
    # Calculate average waiting time
    step_data['avg_waiting_time'] = total_waiting / len(vehicle_ids) if vehicle_ids else 0
    
    return step_data

def get_current_traffic_light_info():
    """Get current traffic light information"""
    try:
        current_phase = traci.trafficlight.getPhase(junction_id)
        # Try to get remaining time (may not work in all SUMO versions)
        try:
            next_switch = traci.trafficlight.getNextSwitch(junction_id)
            current_time = traci.simulation.getTime()
            remaining = max(0, next_switch - current_time)
        except:
            remaining = 0
        
        return {
            'phase': current_phase,
            'remaining': remaining
        }
    except Exception as e:
        return {'phase': None, 'remaining': 0}

def display_dynamic_analysis(step, step_data, analysis):
    """Display comprehensive dynamic traffic analysis"""
    print(f"\n{'='*90}")
    print(f"🚦 DYNAMIC TRAFFIC LIGHT ANALYSIS - Step {step} (Hour: {analysis['hour']:.1f})")
    print(f"{'='*90}")
    
    # Overall metrics
    print(f"📊 Network Status:")
    print(f"   Total Vehicles: {step_data['total_vehicles']}")
    print(f"   Vehicles Waiting: {step_data['waiting_vehicles']}")
    print(f"   Average Waiting Time: {step_data['avg_waiting_time']:.2f}s")
    
    # Traffic light info
    tl_info = get_current_traffic_light_info()
    print(f"   Current Traffic Light Phase: {tl_info['phase']} (Remaining: {tl_info['remaining']:.1f}s)")
    
    # Dynamic pressure analysis
    print(f"\n🎯 Dynamic Traffic Pressure Analysis:")
    print(f"   North-South Pressure: {analysis['ns_pressure']:.1f}")
    print(f"   East-West Pressure:   {analysis['ew_pressure']:.1f}")
    print(f"   Total Pressure:       {analysis['total_pressure']:.1f}")
    
    # Optimization suggestions
    print(f"\n💡 Adaptive Timing Suggestions:")
    print(f"   Suggested NS Green Time: {analysis['suggested_ns_time']}s")
    print(f"   Suggested EW Green Time: {analysis['suggested_ew_time']}s")
    
    # Determine which direction needs priority
    if analysis['ns_pressure'] > analysis['ew_pressure'] * 1.2:
        priority = "🔴 PRIORITIZE NORTH-SOUTH"
    elif analysis['ew_pressure'] > analysis['ns_pressure'] * 1.2:
        priority = "🔴 PRIORITIZE EAST-WEST"
    else:
        priority = "🟢 BALANCED TRAFFIC"
    
    print(f"   Priority Recommendation: {priority}")
    
    # Edge details
    print(f"\n📈 Edge Performance Details:")
    print(f"{'Edge':<12} {'Vehicles':<10} {'Speed (m/s)':<12} {'Waiting (s)':<12}")
    print("-" * 90)
    
    for edge_id, data in step_data['edge_data'].items():
        if data['vehicle_count'] > 0:
            print(f"{edge_id:<12} {data['vehicle_count']:<10} "
                  f"{data['avg_speed']:<12.2f} {data['waiting_time']:<12.2f}")

def generate_dynamic_summary():
    """Generate comprehensive dynamic traffic summary"""
    print(f"\n{'='*100}")
    print(f"🎯 DYNAMIC TRAFFIC SIMULATION COMPREHENSIVE REPORT")
    print(f"{'='*100}")
    
    # Waiting time analysis
    all_waiting_times = []
    for edge_times in traffic_data['waiting_times'].values():
        all_waiting_times.extend(edge_times)
    
    if all_waiting_times:
        avg_waiting = statistics.mean(all_waiting_times)
        median_waiting = statistics.median(all_waiting_times)
        max_waiting = max(all_waiting_times)
        
        print(f"⏱️  WAITING TIME PERFORMANCE:")
        print(f"   Average: {avg_waiting:.2f}s | Median: {median_waiting:.2f}s | Max: {max_waiting:.2f}s")
        
        if avg_waiting < 20:
            performance = "🟢 EXCELLENT"
        elif avg_waiting < 40:
            performance = "🟡 GOOD"
        elif avg_waiting < 80:
            performance = "🟠 FAIR"
        else:
            performance = "🔴 POOR"
        print(f"   Overall Performance: {performance}")
    
    # Traffic pressure trends
    print(f"\n📊 TRAFFIC PRESSURE TRENDS:")
    if traffic_data['pressure_history']['ns'] and traffic_data['pressure_history']['ew']:
        ns_avg = statistics.mean(traffic_data['pressure_history']['ns'])
        ew_avg = statistics.mean(traffic_data['pressure_history']['ew'])
        ns_peak = max(traffic_data['pressure_history']['ns'])
        ew_peak = max(traffic_data['pressure_history']['ew'])
        
        print(f"   North-South: Avg Pressure {ns_avg:.1f} | Peak {ns_peak:.1f}")
        print(f"   East-West:   Avg Pressure {ew_avg:.1f} | Peak {ew_peak:.1f}")
        
        if ns_avg > ew_avg * 1.2:
            recommendation = "🔴 Consider longer NS green phases"
        elif ew_avg > ns_avg * 1.2:
            recommendation = "🔴 Consider longer EW green phases"
        else:
            recommendation = "🟢 Current timing appears balanced"
        print(f"   Recommendation: {recommendation}")
    
    # Edge efficiency analysis
    print(f"\n🚗 EDGE EFFICIENCY ANALYSIS:")
    print(f"{'Edge':<12} {'Avg Vehicles':<15} {'Avg Speed':<12} {'Peak Queue':<12} {'Efficiency':<12}")
    print("-" * 100)
    
    for edge in edges:
        if edge in traffic_data['avg_speeds'] and traffic_data['avg_speeds'][edge]:
            avg_vehicles = statistics.mean(traffic_data['queue_lengths'][edge])
            avg_speed = statistics.mean(traffic_data['avg_speeds'][edge])
            peak_queue = max(traffic_data['queue_lengths'][edge])
            
            # Calculate efficiency (speed / vehicles ratio)
            efficiency = avg_speed / max(1, avg_vehicles)
            
            print(f"{edge:<12} {avg_vehicles:<15.1f} {avg_speed:<12.2f} "
                  f"{peak_queue:<12} {efficiency:<12.2f}")

# Initialize analyzer
analyzer = DynamicTrafficAnalyzer()

# Start simulation
print("🚦 Starting Dynamic Traffic Light Simulation with Adaptive Analysis")
print("📊 Features: Real-time traffic pressure analysis, adaptive timing suggestions")
traci.start(sumoCmd)

try:
    for step in range(1800):  # Run for 30 minutes
        traci.simulationStep()
        
        # Collect metrics
        step_data = collect_comprehensive_metrics(step)
        
        if step_data:
            # Analyze traffic patterns
            analysis = analyzer.analyze_traffic_patterns(step_data, step)
            
            # Display detailed analysis every 300 steps (5 minutes)
            if step % 300 == 0 and step > 0:
                display_dynamic_analysis(step, step_data, analysis)
            
            # Brief status every 60 steps (1 minute)
            if step % 60 == 0:
                tl_info = get_current_traffic_light_info()
                print(f"Step {step:4d} | Vehicles: {step_data['total_vehicles']:3d} | "
                      f"NS Pressure: {analysis['ns_pressure']:5.1f} | "
                      f"EW Pressure: {analysis['ew_pressure']:5.1f} | "
                      f"Phase: {tl_info['phase']}")
        
        # Check for early termination
        if step > 300 and len(traci.vehicle.getIDList()) == 0:
            print(f"No vehicles remaining at step {step}")
            break

finally:
    generate_dynamic_summary()
    traci.close()
    print("\n🎉 Dynamic Traffic Light Simulation Complete!")
    print("💡 Use the analysis above to optimize your traffic light timing.")
