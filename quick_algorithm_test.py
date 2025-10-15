#!/usr/bin/env python3
"""
Quick verification test for the edge algorithm functionality.
"""

import os
import sys
import traci

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from edge_traffic_controller import EdgeTrafficController

def quick_test():
    """Quick test to verify algorithm is working"""
    
    sumo_cmd = ["sumo", "-c", "demo.sumocfg", "--no-warnings", "--no-step-log"]
    
    try:
        # Start SUMO
        traci.start(sumo_cmd)
        
        # Initialize edge controller
        controller = EdgeTrafficController()
        
        print("🔍 QUICK EDGE ALGORITHM VERIFICATION")
        print("="*50)
        print(f"Base green: {controller.base_green_time}s")
        print(f"Adaptation interval: {controller.adaptation_interval}s")
        print()
        
        adaptations_counted = 0
        
        # Run for 90 seconds to test basic functionality
        for step in range(90):
            current_time = traci.simulation.getTime()
            
            # Apply edge algorithm
            if step > 10:
                result = controller.apply_edge_algorithm(current_time)
                if result:
                    adaptations_counted += 1
                    print(f"✅ Adaptation #{adaptations_counted}: {result['phase_name']} → {result['new_duration']}s")
            
            # Show state every 30 seconds
            if step % 30 == 0 and step > 0:
                vehicles = len(traci.vehicle.getIDList())
                phase = traci.trafficlight.getPhase("J4")
                duration = traci.trafficlight.getPhaseDuration("J4")
                print(f"📊 {current_time}s: {vehicles} vehicles, phase {phase}, duration {duration:.1f}s")
                
            traci.simulationStep()
        
        print(f"\n✅ Test completed:")
        print(f"   Adaptations counted: {adaptations_counted}")
        print(f"   Controller internal: {controller.adaptations_count}")
        print(f"   Algorithm is working: {adaptations_counted > 0}")
        
        traci.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    quick_test()