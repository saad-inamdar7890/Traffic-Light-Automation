import os
import sys
import random
import traci
import time
from collections import defaultdict
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
    'throughput': defaultdict(int),
    'total_vehicles': 0
}

# Define the edges in your network
edges = ["-E0", "-E0.254", "-E1", "-E1.238", "E0", "E0.319", "E1", "E1.200"]
junction_id = "J4"

def collect_traffic_metrics(step):
    """Collect comprehensive traffic metrics for analysis"""
    
    # Get all vehicles currently in simulation
    vehicle_ids = traci.vehicle.getIDList()
    
    if not vehicle_ids:
        return None
    
    # Initialize step data
    step_data = {
        'step': step,
        'total_vehicles': len(vehicle_ids),
        'waiting_vehicles': 0,
        'edge_data': {}
    }
    
    # Collect data for each edge
    for edge_id in edges:
        try:
            # Get basic edge metrics
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
            avg_speed = traci.edge.getLastStepMeanSpeed(edge_id)
            avg_waiting_time = traci.edge.getWaitingTime(edge_id)
            
            # Store edge data
            step_data['edge_data'][edge_id] = {
                'vehicle_count': vehicle_count,
                'avg_speed': avg_speed,
                'waiting_time': avg_waiting_time
            }
            
            # Update global metrics
            traffic_data['avg_speeds'][edge_id].append(avg_speed)
            traffic_data['queue_lengths'][edge_id].append(vehicle_count)
            
        except Exception as e:
            print(f"Error collecting data for edge {edge_id}: {e}")
    
    # Collect individual vehicle data
    total_waiting_time = 0
    for veh_id in vehicle_ids:
        try:
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            edge = traci.vehicle.getRoadID(veh_id)
            
            if waiting_time > 0:
                step_data['waiting_vehicles'] += 1
                total_waiting_time += waiting_time
                traffic_data['waiting_times'][edge].append(waiting_time)
            
        except Exception as e:
            continue
    
    # Calculate average waiting time for this step
    if step_data['waiting_vehicles'] > 0:
        step_data['avg_waiting_time'] = total_waiting_time / step_data['waiting_vehicles']
    else:
        step_data['avg_waiting_time'] = 0
    
    return step_data

def get_traffic_light_state():
    """Get current traffic light phase and remaining duration"""
    try:
        current_phase = traci.trafficlight.getPhase(junction_id)
        remaining_duration = traci.trafficlight.getPhaseDuration(junction_id) - traci.trafficlight.getElapsedTime(junction_id)
        phase_name = traci.trafficlight.getProgramID(junction_id)
        return current_phase, remaining_duration, phase_name
    except Exception as e:
        print(f"Warning: Could not get traffic light state: {e}")
        return None, None, None

def optimize_traffic_light(step_data):
    """Simple traffic light optimization based on queue lengths"""
    if not step_data or 'edge_data' not in step_data:
        return
    
    # Calculate traffic density for each direction
    north_south_traffic = (
        step_data['edge_data'].get('-E1', {}).get('vehicle_count', 0) +
        step_data['edge_data'].get('E1', {}).get('vehicle_count', 0)
    )
    
    east_west_traffic = (
        step_data['edge_data'].get('E0', {}).get('vehicle_count', 0) +
        step_data['edge_data'].get('-E0', {}).get('vehicle_count', 0)
    )
    
    # Get current traffic light state
    current_phase, remaining_duration, _ = get_traffic_light_state()
    
    if current_phase is not None and remaining_duration is not None:
        # Simple optimization: extend green phase for busier direction
        if north_south_traffic > east_west_traffic * 1.5 and current_phase in [1, 3]:
            # Extend north-south green phase
            if remaining_duration < 10:
                try:
                    traci.trafficlight.setPhaseDuration(junction_id, remaining_duration + 10)
                    print(f"Extended North-South green phase by 10s (NS: {north_south_traffic}, EW: {east_west_traffic})")
                except Exception as e:
                    print(f"Could not extend phase: {e}")
                    
        elif east_west_traffic > north_south_traffic * 1.5 and current_phase in [5, 7]:
            # Extend east-west green phase
            if remaining_duration < 10:
                try:
                    traci.trafficlight.setPhaseDuration(junction_id, remaining_duration + 10)
                    print(f"Extended East-West green phase by 10s (NS: {north_south_traffic}, EW: {east_west_traffic})")
                except Exception as e:
                    print(f"Could not extend phase: {e}")

def display_traffic_analysis(step, step_data):
    """Display comprehensive traffic analysis"""
    if not step_data:
        return
    
    print(f"\n{'='*60}")
    print(f"TRAFFIC ANALYSIS - Step {step} (Time: {step}s)")
    print(f"{'='*60}")
    
    # Overall metrics
    print(f"Total Vehicles in Network: {step_data['total_vehicles']}")
    print(f"Vehicles Currently Waiting: {step_data['waiting_vehicles']}")
    print(f"Average Waiting Time: {step_data.get('avg_waiting_time', 0):.2f}s")
    
    # Traffic light state
    phase, remaining, _ = get_traffic_light_state()
    if phase is not None and remaining is not None:
        print(f"Current Traffic Light Phase: {phase} (Remaining: {remaining:.1f}s)")
    else:
        print("Traffic Light State: N/A")
    
    print(f"\n{'Edge Analysis':^60}")
    print(f"{'Edge':<12} {'Vehicles':<10} {'Avg Speed':<12} {'Waiting Time':<15}")
    print("-" * 60)
    
    for edge_id, data in step_data['edge_data'].items():
        print(f"{edge_id:<12} {data['vehicle_count']:<10} "
              f"{data['avg_speed']:<12.2f} {data['waiting_time']:<15.2f}")
    
    # Direction-based analysis
    print(f"\n{'Direction Analysis':^60}")
    north_south = step_data['edge_data'].get('-E1', {}).get('vehicle_count', 0) + \
                  step_data['edge_data'].get('E1', {}).get('vehicle_count', 0)
    east_west = step_data['edge_data'].get('E0', {}).get('vehicle_count', 0) + \
                step_data['edge_data'].get('-E0', {}).get('vehicle_count', 0)
    
    print(f"North-South Traffic: {north_south} vehicles")
    print(f"East-West Traffic: {east_west} vehicles")
    
    if north_south > east_west:
        print("🔴 North-South direction is busier")
    elif east_west > north_south:
        print("🔴 East-West direction is busier")
    else:
        print("🟢 Traffic is balanced")

def generate_summary_report():
    """Generate final traffic analysis summary"""
    print(f"\n{'='*80}")
    print(f"{'TRAFFIC SIMULATION SUMMARY REPORT':^80}")
    print(f"{'='*80}")
    
    # Calculate overall statistics
    all_waiting_times = []
    for edge_times in traffic_data['waiting_times'].values():
        all_waiting_times.extend(edge_times)
    
    if all_waiting_times:
        avg_waiting = statistics.mean(all_waiting_times)
        max_waiting = max(all_waiting_times)
        min_waiting = min(all_waiting_times)
        
        print(f"Overall Average Waiting Time: {avg_waiting:.2f}s")
        print(f"Maximum Waiting Time: {max_waiting:.2f}s")
        print(f"Minimum Waiting Time: {min_waiting:.2f}s")
    else:
        print("No waiting time data collected.")
    
    # Edge performance summary
    print(f"\n{'Edge Performance Summary':^80}")
    print(f"{'Edge':<12} {'Avg Vehicles':<15} {'Avg Speed':<12} {'Peak Queue':<12}")
    print("-" * 80)
    
    for edge in edges:
        if edge in traffic_data['avg_speeds'] and traffic_data['avg_speeds'][edge]:
            avg_vehicles = statistics.mean(traffic_data['queue_lengths'][edge]) if traffic_data['queue_lengths'][edge] else 0
            avg_speed = statistics.mean(traffic_data['avg_speeds'][edge])
            peak_queue = max(traffic_data['queue_lengths'][edge]) if traffic_data['queue_lengths'][edge] else 0
            
            print(f"{edge:<12} {avg_vehicles:<15.1f} {avg_speed:<12.2f} {peak_queue:<12}")

# Start SUMO simulation
print("Starting SUMO Traffic Light Optimization Simulation...")
traci.start(sumoCmd)

try:
    # Main simulation loop
    for step in range(1000):  # Run for 1000 simulation steps
        traci.simulationStep()
        
        # Collect traffic metrics every step
        step_data = collect_traffic_metrics(step)
        
        # Display detailed analysis every 50 steps
        if step % 50 == 0 and step > 0:
            display_traffic_analysis(step, step_data)
        
        # Attempt traffic light optimization every 30 steps
        if step % 30 == 0 and step > 0:
            optimize_traffic_light(step_data)
        
        # Brief status every 10 steps
        if step % 10 == 0:
            vehicle_count = len(traci.vehicle.getIDList())
            phase, remaining, _ = get_traffic_light_state()
            
            # Safe formatting for remaining duration
            remaining_str = f"{remaining:.1f}s" if remaining is not None else "N/A"
            phase_str = str(phase) if phase is not None else "N/A"
            
            print(f"Step {step}: {vehicle_count} vehicles, Phase: {phase_str}, Remaining: {remaining_str}")
        
        # Check if simulation should end (no more vehicles)
        if len(traci.vehicle.getIDList()) == 0 and step > 100:
            print(f"No more vehicles in simulation at step {step}")
            break

finally:
    # Generate final report
    generate_summary_report()
    
    # Close SUMO
    traci.close()
    print("\nSimulation completed successfully!")

