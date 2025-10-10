#!/usr/bin/env python3
"""
Debug test to understand why no adaptations are occurring.
"""

import os
import sys
import traci

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from edge_traffic_controller import EdgeTrafficController

def debug_algorithm():
    """Debug why algorithm isn't making adaptations"""
    
    sumo_cmd = ["sumo", "-c", "demo.sumocfg", "--no-warnings", "--no-step-log"]
    
    try:
        # Start SUMO
        traci.start(sumo_cmd)
        
        # Initialize edge controller
        controller = EdgeTrafficController()
        
        print("🐛 DEBUGGING EDGE ALGORITHM")
        print("="*50)
        
        # Run for 60 seconds and debug each step
        for step in range(60):
            current_time = traci.simulation.getTime()
            
            if step > 10 and step % 20 == 0:  # Check every 20 seconds after init
                print(f"\n🔍 Debug at {current_time}s:")
                
                # Check adaptation interval
                time_since_last = current_time - controller.last_adaptation_time
                print(f"   Time since last adaptation: {time_since_last}s")
                print(f"   Adaptation interval required: {controller.adaptation_interval}s")
                print(f"   Ready for adaptation: {time_since_last >= controller.adaptation_interval}")
                
                # Check traffic data collection
                try:
                    traffic_data = controller.collect_lane_traffic_density()
                    if traffic_data:
                        print(f"   Traffic data collected: ✅")
                        for direction, data in traffic_data.items():
                            print(f"     {direction}: {data['vehicles']} vehicles, {data['waiting_time']:.1f}s wait")
                    else:
                        print(f"   Traffic data collected: ❌ (None)")
                except Exception as e:
                    print(f"   Traffic data error: {e}")
                
                # Check current phase
                try:
                    current_phase = traci.trafficlight.getPhase(controller.junction_id)
                    current_duration = traci.trafficlight.getPhaseDuration(controller.junction_id)
                    print(f"   Current phase: {current_phase}, duration: {current_duration}s")
                except Exception as e:
                    print(f"   Phase check error: {e}")
                
                # Try calling timing calculation manually
                if traffic_data:
                    try:
                        timing_plan = controller.calculate_weighted_average_timing(traffic_data, current_time)
                        if timing_plan:
                            print(f"   Timing plan calculated: ✅")
                            print(f"     NS: {timing_plan['north_south_green']:.1f}s, EW: {timing_plan['east_west_green']:.1f}s")
                            print(f"     Priority: {timing_plan['priority_direction']}")
                        else:
                            print(f"   Timing plan calculated: ❌ (None)")
                    except Exception as e:
                        print(f"   Timing calculation error: {e}")
                
                # Now try the full algorithm
                try:
                    result = controller.apply_edge_algorithm(current_time)
                    if result:
                        print(f"   Algorithm result: ✅ {result['phase_name']} → {result['new_duration']}s")
                    else:
                        print(f"   Algorithm result: ❌ (None)")
                except Exception as e:
                    print(f"   Algorithm error: {e}")
                
            traci.simulationStep()
        
        traci.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    debug_algorithm()