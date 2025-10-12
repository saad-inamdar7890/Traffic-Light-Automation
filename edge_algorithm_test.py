"""
Edge Algorithm Test - Phase 1 Implementation
===========================================

Test the edge traffic controller with your project specifications:
- 30s base timing (configurable from cloud in Phase 2)
- 10s minimum, 50s maximum green time
- Gradual changes based on statistical traffic density analysis
- Real-time adaptation using traffic density from all lanes
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from edge_traffic_controller import EdgeTrafficController
import traci
import time

def test_edge_algorithm():
    """Test the edge algorithm with simulated traffic scenarios"""
    
    print("🚦 EDGE TRAFFIC CONTROLLER TEST")
    print("=" * 50)
    print("Phase 1: Edge-level adaptive traffic control")
    print("Base timing: 30s | Range: 10s-50s | Gradual changes")
    print()
    
    # Initialize SUMO simulation
    sumo_cmd = [
        "sumo", "-c", "demo.sumocfg",
        "--start", "--quit-on-end",
        "--step-length", "1"
    ]
    
    try:
        traci.start(sumo_cmd)
        
        # Initialize edge controller with your specifications
        edge_controller = EdgeTrafficController(
            junction_id="J4", 
            base_green_time=30  # This will come from cloud RL model in Phase 2
        )
        
        print("📊 Starting edge algorithm simulation...")
        print()
        
        simulation_time = 0
        test_duration = 300  # 5 minutes test
        
        while simulation_time < test_duration and traci.simulation.getMinExpectedNumber() > 0:
            # Step simulation
            traci.simulationStep()
            simulation_time = traci.simulation.getTime()
            
            # Apply edge algorithm
            if simulation_time > 10:  # Start after initialization
                control_result = edge_controller.apply_edge_algorithm(simulation_time)
                
                # Display traffic density analysis every 30 seconds
                if int(simulation_time) % 30 == 0:
                    traffic_data = edge_controller.collect_lane_traffic_density()
                    print(f"⏰ Time: {simulation_time}s")
                    print("📊 Traffic Density Analysis:")
                    
                    for direction, data in traffic_data.items():
                        density = data['density']
                        waiting = data['waiting_time']
                        vehicles = data['vehicles']
                        
                        # Classify density level
                        if density > edge_controller.density_threshold_high:
                            level = "🔴 HIGH"
                        elif density > edge_controller.density_threshold_low:
                            level = "🟡 MEDIUM"
                        else:
                            level = "🟢 LOW"
                        
                        print(f"   {direction:15}: {vehicles:2d} vehicles, "
                              f"Density: {density:4.1f} {level}, Wait: {waiting:4.1f}s")
                    
                    # Show current green duration
                    current_duration = edge_controller.current_green_duration
                    print(f"🚦 Current Green Duration: {current_duration}s "
                          f"(Base: {edge_controller.base_green_time}s)")
                    print()
        
        # Final statistics
        print("📈 EDGE ALGORITHM PERFORMANCE SUMMARY")
        print("=" * 50)
        stats = edge_controller.get_edge_statistics()
        
        print(f"🎯 Algorithm: {stats['algorithm']}")
        print(f"📊 Base Green Time: {stats['base_green_time']}s")
        print(f"🔧 Current Duration: {stats['current_green_duration']}s")
        print(f"⚡ Adaptations Made: {stats['adaptations_made']}")
        print(f"📏 Timing Range: {stats['timing_range']}")
        print()
        
        print("🧮 Traffic Density Analysis:")
        for direction, analysis in stats['density_analysis'].items():
            avg_density = analysis['avg_density']
            samples = analysis['samples']
            print(f"   {direction:15}: Avg Density {avg_density:4.1f} ({samples} samples)")
        
        print()
        print("📝 Recent Timing Adjustments:")
        recent_adjustments = stats['recent_adjustments']
        if recent_adjustments:
            for adj in recent_adjustments[-3:]:  # Show last 3
                time_stamp = adj['time']
                phase = adj['phase']
                old_dur = adj['old_duration']
                new_dur = adj['new_duration']
                pressure = adj['current_pressure']
                print(f"   {time_stamp:6.0f}s: Phase {phase} → {old_dur}s to {new_dur}s "
                      f"(Pressure: {pressure:.1f})")
        else:
            print("   No timing adjustments made during test")
        
        print()
        print("✅ Edge Algorithm Test Completed Successfully!")
        print("🔄 Ready for Phase 2: Cloud RL model integration")
        
    except Exception as e:
        print(f"❌ Error during edge algorithm test: {e}")
        
    finally:
        if traci.isLoaded():
            traci.close()

def demonstrate_cloud_integration():
    """Demonstrate Phase 2 integration point"""
    print("\n🌐 PHASE 2 INTEGRATION DEMONSTRATION")
    print("=" * 50)
    
    # Create edge controller
    edge_controller = EdgeTrafficController(base_green_time=30)
    
    print("Current configuration:")
    print(f"   Base Green Time: {edge_controller.base_green_time}s")
    print(f"   Timing Range: {edge_controller.min_green_time}s - {edge_controller.max_green_time}s")
    print()
    
    print("Simulating Cloud RL Model Updates:")
    # Simulate cloud updates with different base timings
    test_timings = [25, 35, 28, 32, 40, 15]  # From RL model
    
    for new_timing in test_timings:
        success = edge_controller.update_base_timing_from_cloud(new_timing)
        if success:
            print(f"✅ Cloud update successful: {new_timing}s")
        else:
            print(f"❌ Cloud update rejected: {new_timing}s (out of range)")
    
    print()
    print("🔮 Phase 2 Features (Ready for Implementation):")
    print("   • RL model analyzes traffic patterns")
    print("   • Provides optimized base timings to edges")
    print("   • Edge controllers adapt locally with new base")
    print("   • Continuous learning and improvement")

if __name__ == "__main__":
    # Test the edge algorithm
    test_edge_algorithm()
    
    # Demonstrate cloud integration
    demonstrate_cloud_integration()