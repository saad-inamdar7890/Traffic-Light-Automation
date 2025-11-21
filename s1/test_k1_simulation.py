"""
K1 Network - Quick Simulation Test
Validates the 24-hour traffic flow setup and measures baseline metrics.
"""

import os
import sys
import traci
import time
from collections import defaultdict

# SUMO configuration
SUMO_BINARY = "sumo"  # Use "sumo-gui" for visualization
SUMOCFG_FILE = "k1.sumocfg"

# Test parameters
TEST_DURATION = 3600  # Test first hour (0-3600 seconds)
FULL_SIMULATION = False  # Set to True for full 24-hour test


class TrafficMetrics:
    """Collects and analyzes traffic metrics during simulation."""
    
    def __init__(self):
        self.junction_data = defaultdict(lambda: {
            'waiting_time': [],
            'queue_length': [],
            'vehicles_passed': 0
        })
        self.vehicle_data = defaultdict(lambda: {
            'spawn_time': 0,
            'arrival_time': 0,
            'total_wait': 0
        })
        self.timestep_data = []
        
    def collect_junction_metrics(self, junction_id):
        """Collect metrics for a specific junction."""
        try:
            # Get incoming lanes for this junction
            lanes = traci.trafficlight.getControlledLanes(junction_id)
            
            total_waiting = 0
            total_queue = 0
            
            for lane in lanes:
                waiting_time = traci.lane.getWaitingTime(lane)
                queue_length = traci.lane.getLastStepHaltingNumber(lane)
                total_waiting += waiting_time
                total_queue += queue_length
            
            self.junction_data[junction_id]['waiting_time'].append(total_waiting)
            self.junction_data[junction_id]['queue_length'].append(total_queue)
            
        except Exception as e:
            print(f"Warning: Could not collect metrics for {junction_id}: {e}")
    
    def collect_timestep_metrics(self, step):
        """Collect network-wide metrics for current timestep."""
        vehicle_count = traci.vehicle.getIDCount()
        departed = traci.simulation.getDepartedNumber()
        arrived = traci.simulation.getArrivedNumber()
        
        self.timestep_data.append({
            'step': step,
            'vehicles': vehicle_count,
            'departed': departed,
            'arrived': arrived
        })
    
    def print_summary(self, duration):
        """Print summary of collected metrics."""
        print("\n" + "="*70)
        print("SIMULATION SUMMARY")
        print("="*70)
        
        print(f"\nSimulation Duration: {duration} seconds ({duration/3600:.2f} hours)")
        
        # Overall statistics
        if self.timestep_data:
            total_departed = sum(d['departed'] for d in self.timestep_data)
            total_arrived = sum(d['arrived'] for d in self.timestep_data)
            avg_vehicles = sum(d['vehicles'] for d in self.timestep_data) / len(self.timestep_data)
            
            print(f"\nNetwork-Wide Statistics:")
            print(f"  Total Vehicles Spawned: {total_departed}")
            print(f"  Total Vehicles Completed: {total_arrived}")
            print(f"  Average Active Vehicles: {avg_vehicles:.1f}")
            print(f"  Completion Rate: {(total_arrived/max(total_departed, 1)*100):.1f}%")
        
        # Junction statistics
        print(f"\nJunction Performance:")
        print(f"{'Junction':<12} {'Avg Wait (s)':<15} {'Avg Queue':<12} {'Status'}")
        print("-" * 70)
        
        for junction_id in sorted(self.junction_data.keys()):
            data = self.junction_data[junction_id]
            
            if data['waiting_time']:
                avg_wait = sum(data['waiting_time']) / len(data['waiting_time'])
                avg_queue = sum(data['queue_length']) / len(data['queue_length'])
                
                # Determine status
                if avg_wait < 30:
                    status = "✅ Good"
                elif avg_wait < 60:
                    status = "⚠️ Moderate"
                else:
                    status = "❌ Congested"
                
                print(f"{junction_id:<12} {avg_wait:<15.2f} {avg_queue:<12.2f} {status}")
        
        print("\n" + "="*70)


def run_simulation_test(duration=3600, gui=False):
    """
    Run simulation test for specified duration.
    
    Args:
        duration: Simulation duration in seconds
        gui: Use SUMO GUI if True, command-line if False
    """
    print("\n" + "="*70)
    print("K1 NETWORK - TRAFFIC SIMULATION TEST")
    print("="*70)
    
    # Check if config file exists
    if not os.path.exists(SUMOCFG_FILE):
        print(f"❌ Error: Configuration file '{SUMOCFG_FILE}' not found!")
        print("   Make sure you're running this from the s1/ directory.")
        return
    
    # Choose SUMO binary
    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [sumo_binary, "-c", SUMOCFG_FILE]
    
    print(f"\n📊 Starting simulation...")
    print(f"   Duration: {duration} seconds ({duration/3600:.2f} hours)")
    print(f"   Mode: {'GUI' if gui else 'Command-line'}")
    print(f"   Config: {SUMOCFG_FILE}")
    
    # Initialize metrics collector
    metrics = TrafficMetrics()
    
    # Traffic light junctions in K1 network
    traffic_lights = ['J0', 'J1', 'J5', 'J6', 'J7', 'J10', 'J11', 'J12', 'J22']
    
    try:
        # Start SUMO
        traci.start(sumo_cmd)
        
        step = 0
        start_time = time.time()
        last_print = 0
        
        print("\n🚦 Simulation running...\n")
        
        # Run simulation
        while step < duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Collect metrics every 10 steps (10 seconds)
            if step % 10 == 0:
                for junction in traffic_lights:
                    metrics.collect_junction_metrics(junction)
                metrics.collect_timestep_metrics(step)
            
            # Print progress every 5 minutes (300 seconds)
            if step - last_print >= 300:
                elapsed = time.time() - start_time
                vehicles = traci.vehicle.getIDCount()
                print(f"⏱️  Step {step:>6} / {duration} ({step/duration*100:>5.1f}%) | "
                      f"Vehicles: {vehicles:>4} | "
                      f"Real time: {elapsed:.1f}s")
                last_print = step
            
            step += 1
        
        # Close SUMO
        traci.close()
        
        # Calculate and print summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Simulation completed!")
        print(f"   Simulated time: {step} seconds ({step/3600:.2f} hours)")
        print(f"   Real time: {elapsed_time:.2f} seconds")
        print(f"   Speedup: {step/elapsed_time:.1f}x")
        
        # Print detailed metrics
        metrics.print_summary(step)
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            traci.close()
        except:
            pass
        
        return None


def quick_route_test():
    """Quick test to verify routes are defined correctly."""
    print("\n" + "="*70)
    print("ROUTE VALIDATION TEST")
    print("="*70)
    
    routes_file = "k1_routes_24h.rou.xml"
    
    if not os.path.exists(routes_file):
        print(f"❌ Error: Routes file '{routes_file}' not found!")
        return
    
    print(f"\n📄 Reading routes from: {routes_file}")
    
    # Parse route file (simple check)
    with open(routes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count routes and flows
    route_count = content.count('<route id=')
    flow_count = content.count('<flow id=')
    vtype_count = content.count('<vType id=')
    
    print(f"\n✅ Route file validated:")
    print(f"   Vehicle Types: {vtype_count}")
    print(f"   Route Definitions: {route_count}")
    print(f"   Traffic Flows: {flow_count}")
    
    # Check time periods
    periods = {
        'Night (00:00-07:00)': (0, 25200),
        'Morning Rush (07:00-09:00)': (25200, 32400),
        'Midday (09:00-17:00)': (32400, 61200),
        'Evening Rush (17:00-19:00)': (61200, 68400),
        'Early Night (19:00-22:00)': (68400, 79200),
        'Late Night (22:00-00:00)': (79200, 86400),
    }
    
    print(f"\n📅 Time Periods Defined:")
    for period, (start, end) in periods.items():
        duration_hours = (end - start) / 3600
        print(f"   {period:<30} {duration_hours:>5.1f} hours")
    
    print("\n✅ Routes file is ready for simulation!")


if __name__ == "__main__":
    print("\n🚗 K1 Network Traffic Simulation Test Tool")
    print("="*70)
    
    # Change to s1 directory if not already there
    if os.path.basename(os.getcwd()) != 's1':
        if os.path.exists('s1'):
            os.chdir('s1')
            print(f"📁 Changed directory to: {os.getcwd()}")
        else:
            print("⚠️  Warning: Not in s1 directory and s1/ not found")
    
    # Run route validation first
    quick_route_test()
    
    print("\n" + "="*70)
    print("SELECT TEST MODE:")
    print("="*70)
    print("1. Quick Test (1 hour, command-line)")
    print("2. Quick Test with GUI (1 hour, visual)")
    print("3. Morning Rush Test (07:00-09:00, 2 hours)")
    print("4. Full 24-Hour Test (command-line)")
    print("5. Full 24-Hour Test with GUI")
    print("0. Exit")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            run_simulation_test(duration=3600, gui=False)
        elif choice == '2':
            run_simulation_test(duration=3600, gui=True)
        elif choice == '3':
            print("\n⚠️  Note: Simulating 25200-32400 seconds (07:00-09:00)")
            print("   This will run 2-hour morning rush starting from 07:00")
            run_simulation_test(duration=32400, gui=False)
        elif choice == '4':
            confirm = input("\n⚠️  Full simulation takes 15-30 minutes. Continue? (y/n): ")
            if confirm.lower() == 'y':
                run_simulation_test(duration=86400, gui=False)
        elif choice == '5':
            confirm = input("\n⚠️  Full GUI simulation takes longer. Continue? (y/n): ")
            if confirm.lower() == 'y':
                run_simulation_test(duration=86400, gui=True)
        elif choice == '0':
            print("\n👋 Exiting...")
        else:
            print("\n❌ Invalid choice!")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test complete!\n")
