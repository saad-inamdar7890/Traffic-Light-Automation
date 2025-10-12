"""
Edge Algorithm Demo - Phase 1 Implementation
===========================================

Demonstrates the edge traffic controller logic without SUMO dependency
Shows your project specifications implementation:
- 30s base timing (configurable from cloud in Phase 2)
- 10s minimum, 50s maximum green time
- Gradual changes based on statistical traffic density analysis
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from edge_traffic_controller import EdgeTrafficController

def simulate_traffic_scenarios():
    """Demonstrate edge algorithm with simulated traffic data"""
    
    print("🚦 EDGE TRAFFIC CONTROLLER DEMONSTRATION")
    print("=" * 60)
    print("Phase 1: Edge-level adaptive traffic control")
    print("Base timing: 30s | Range: 10s-50s | Gradual changes")
    print()
    
    # Initialize edge controller with your specifications
    edge_controller = EdgeTrafficController(
        junction_id="J4", 
        base_green_time=30  # This will come from cloud RL model in Phase 2
    )
    
    print("📊 EDGE ALGORITHM CORE FEATURES:")
    print(f"   ✅ Base Green Time: {edge_controller.base_green_time}s")
    print(f"   ✅ Timing Range: {edge_controller.min_green_time}s - {edge_controller.max_green_time}s")
    print(f"   ✅ Max Change Per Cycle: {edge_controller.max_change_percent*100}%")
    print(f"   ✅ Adaptation Interval: {edge_controller.adaptation_interval}s")
    print()
    
    # Simulate different traffic scenarios
    scenarios = [
        {
            'name': 'Balanced Traffic',
            'description': 'Equal traffic in all directions',
            'traffic_data': {
                'north_approach': {'vehicles': 4, 'density': 3.2, 'waiting_time': 12, 'speed': 8.5},
                'south_approach': {'vehicles': 5, 'density': 3.8, 'waiting_time': 10, 'speed': 9.2},
                'east_approach': {'vehicles': 3, 'density': 2.5, 'waiting_time': 8, 'speed': 10.1},
                'west_approach': {'vehicles': 4, 'density': 3.1, 'waiting_time': 15, 'speed': 7.8}
            }
        },
        {
            'name': 'Heavy North-South',
            'description': 'High traffic from North and South',
            'traffic_data': {
                'north_approach': {'vehicles': 12, 'density': 9.5, 'waiting_time': 25, 'speed': 4.2},
                'south_approach': {'vehicles': 10, 'density': 8.1, 'waiting_time': 22, 'speed': 5.1},
                'east_approach': {'vehicles': 2, 'density': 1.5, 'waiting_time': 5, 'speed': 12.0},
                'west_approach': {'vehicles': 1, 'density': 0.8, 'waiting_time': 3, 'speed': 13.0}
            }
        },
        {
            'name': 'Rush Hour Congestion',
            'description': 'Heavy traffic in all directions',
            'traffic_data': {
                'north_approach': {'vehicles': 15, 'density': 12.0, 'waiting_time': 35, 'speed': 3.0},
                'south_approach': {'vehicles': 13, 'density': 10.5, 'waiting_time': 30, 'speed': 3.5},
                'east_approach': {'vehicles': 11, 'density': 8.8, 'waiting_time': 28, 'speed': 4.1},
                'west_approach': {'vehicles': 14, 'density': 11.2, 'waiting_time': 32, 'speed': 2.8}
            }
        },
        {
            'name': 'Light Traffic',
            'description': 'Minimal traffic, should reduce timing',
            'traffic_data': {
                'north_approach': {'vehicles': 1, 'density': 0.5, 'waiting_time': 2, 'speed': 13.5},
                'south_approach': {'vehicles': 0, 'density': 0.0, 'waiting_time': 0, 'speed': 0},
                'east_approach': {'vehicles': 2, 'density': 1.2, 'waiting_time': 4, 'speed': 12.8},
                'west_approach': {'vehicles': 1, 'density': 0.8, 'waiting_time': 1, 'speed': 13.2}
            }
        }
    ]
    
    print("🔬 TESTING EDGE ALGORITHM WITH TRAFFIC SCENARIOS:")
    print("=" * 60)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print()
        
        # Test North-South phase (phase 1)
        current_pressure, opposing_pressure = edge_controller.calculate_traffic_pressure_statistics(
            scenario['traffic_data'], 1  # North-South phase
        )
        
        # Calculate adaptive timing
        new_duration = edge_controller.calculate_adaptive_green_time(
            current_pressure, opposing_pressure, i * 30
        )
        
        print(f"📊 Traffic Analysis:")
        for direction, data in scenario['traffic_data'].items():
            vehicles = data['vehicles']
            density = data['density']
            waiting = data['waiting_time']
            speed = data['speed']
            
            # Classify density
            if density > edge_controller.density_threshold_high:
                level = "🔴 HIGH"
            elif density > edge_controller.density_threshold_low:
                level = "🟡 MEDIUM"
            else:
                level = "🟢 LOW"
            
            print(f"   {direction:15}: {vehicles:2d} vehicles, Density: {density:4.1f} {level}")
            print(f"                    Wait: {waiting:4.1f}s, Speed: {speed:4.1f} m/s")
        
        print(f"\n🧮 Statistical Pressure Calculation:")
        print(f"   Current Direction (N-S): {current_pressure:6.1f}")
        print(f"   Opposing Direction (E-W): {opposing_pressure:6.1f}")
        
        pressure_ratio = current_pressure / (current_pressure + opposing_pressure) if (current_pressure + opposing_pressure) > 0 else 0.5
        print(f"   Pressure Ratio: {pressure_ratio:6.3f}")
        
        print(f"\n🚦 Edge Algorithm Decision:")
        print(f"   Base Green Time: {edge_controller.base_green_time}s")
        print(f"   Calculated Duration: {new_duration}s")
        
        # Show the change
        change = new_duration - edge_controller.base_green_time
        change_percent = (change / edge_controller.base_green_time) * 100
        
        if change > 0:
            print(f"   📈 INCREASE: +{change}s (+{change_percent:.1f}%)")
        elif change < 0:
            print(f"   📉 DECREASE: {change}s ({change_percent:.1f}%)")
        else:
            print(f"   ➡️  MAINTAIN: No change needed")
        
        # Show gradual change enforcement
        max_change = edge_controller.base_green_time * edge_controller.max_change_percent
        if abs(change) > max_change:
            print(f"   ⚠️  Gradual Change: Limited to ±{max_change:.1f}s per cycle")
        
        print(f"   ✅ Final Duration: {new_duration}s (Range: {edge_controller.min_green_time}s-{edge_controller.max_green_time}s)")
        
        if i < len(scenarios):
            print("\n" + "-" * 60)

def demonstrate_phase2_integration():
    """Demonstrate Phase 2 cloud integration"""
    print("\n\n🌐 PHASE 2: CLOUD RL MODEL INTEGRATION")
    print("=" * 60)
    
    edge_controller = EdgeTrafficController(base_green_time=30)
    
    print("📡 Cloud RL Model Features (Ready for Phase 2):")
    print("   🧠 Analyzes historical traffic patterns")
    print("   📊 Learns optimal base timings for different times/days")
    print("   🎯 Provides optimized base timings to edge controllers")
    print("   🔄 Continuous learning and adaptation")
    print()
    
    print("🔧 Edge-Cloud Communication Protocol:")
    print("   1. Cloud sends optimized base timing")
    print("   2. Edge validates timing constraints")
    print("   3. Edge updates base timing if valid")
    print("   4. Edge continues local adaptation with new base")
    print()
    
    print("💡 Example Cloud Updates:")
    
    # Simulate different base timings from RL model
    cloud_timings = [
        {'time': '08:00', 'base': 35, 'reason': 'Morning rush hour pattern'},
        {'time': '12:00', 'base': 25, 'reason': 'Light midday traffic'},
        {'time': '17:00', 'base': 40, 'reason': 'Evening rush hour pattern'},
        {'time': '22:00', 'base': 20, 'reason': 'Low night traffic'},
        {'time': '02:00', 'base': 15, 'reason': 'Minimal overnight traffic'}
    ]
    
    for update in cloud_timings:
        time_str = update['time']
        new_base = update['base']
        reason = update['reason']
        
        print(f"   {time_str}: Base timing → {new_base}s ({reason})")
        
        # Test the update
        success = edge_controller.update_base_timing_from_cloud(new_base)
        if success:
            print(f"        ✅ Edge accepted: New base {new_base}s")
        else:
            print(f"        ❌ Edge rejected: Out of range")
    
    print(f"\n🎯 Current Edge Configuration:")
    print(f"   Base Timing: {edge_controller.base_green_time}s (from cloud)")
    print(f"   Local Range: {edge_controller.min_green_time}s - {edge_controller.max_green_time}s")
    print(f"   Adaptation: Up to ±{edge_controller.max_change_percent*100}% per cycle")

def show_algorithm_advantages():
    """Show advantages of your edge algorithm design"""
    print("\n\n🏆 EDGE ALGORITHM ADVANTAGES")
    print("=" * 60)
    
    print("✅ Project Specification Compliance:")
    print("   • 30s base timing (configurable from cloud)")
    print("   • 10s-50s adaptive range")
    print("   • Gradual statistical changes")
    print("   • Real-time traffic density analysis")
    print()
    
    print("✅ Technical Advantages:")
    print("   • Multi-factor pressure calculation (density, waiting, speed)")
    print("   • Gradual changes prevent oscillation")
    print("   • Statistical history for trend analysis")
    print("   • Configurable parameters for different intersections")
    print()
    
    print("✅ Phase 2 Ready:")
    print("   • Clean interface for cloud RL model integration")
    print("   • Validates cloud updates for safety")
    print("   • Maintains local adaptation with cloud-optimized base")
    print("   • Comprehensive statistics for RL model feedback")
    print()
    
    print("✅ Real-world Benefits:")
    print("   • Reduces waiting times in high-traffic scenarios")
    print("   • Maintains efficiency in low-traffic periods")
    print("   • Adapts to unexpected traffic patterns")
    print("   • Prepares for intelligent traffic management")

if __name__ == "__main__":
    # Test the edge algorithm
    simulate_traffic_scenarios()
    
    # Demonstrate cloud integration
    demonstrate_phase2_integration()
    
    # Show algorithm advantages
    show_algorithm_advantages()
    
    print("\n\n🎉 EDGE ALGORITHM DEMONSTRATION COMPLETE!")
    print("Your Phase 1 edge controller is ready for deployment!")
    print("Next: Implement Phase 2 RL model for optimal base timings.")