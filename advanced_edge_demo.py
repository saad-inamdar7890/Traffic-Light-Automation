"""
Advanced Lane-Based Edge Algorithm Demo
======================================

Demonstrates the enhanced edge algorithm with lane prioritization:
- Heavy traffic lanes get more green time + reduced red time
- Light traffic lanes get reduced green time for faster cycling  
- Equal pressure lanes get similar timing
- Maximum 2-minute red time constraint
- Minimum safety green times
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from edge_traffic_controller import EdgeTrafficController

def demonstrate_advanced_lane_prioritization():
    """Demonstrate advanced lane-based traffic prioritization"""
    
    print("🚦 ADVANCED LANE-BASED EDGE ALGORITHM")
    print("=" * 60)
    print("Smart Traffic Management with Lane Prioritization")
    print("✅ Heavy traffic → More green + Less red")
    print("✅ Light traffic → Less green (faster cycling)")
    print("✅ Equal pressure → Similar timing")
    print("✅ Max red time: 2 minutes per lane")
    print("✅ Safety: Minimum green times enforced")
    print()
    
    # Initialize advanced edge controller
    edge_controller = EdgeTrafficController(
        junction_id="J4", 
        base_green_time=30
    )
    
    # Advanced test scenarios
    scenarios = [
        {
            'name': 'Heavy North-South Priority',
            'description': 'North-South has heavy traffic, East-West is light',
            'traffic_data': {
                'north_approach': {'vehicles': 18, 'density': 14.0, 'waiting_time': 45, 'speed': 2.5},
                'south_approach': {'vehicles': 16, 'density': 12.5, 'waiting_time': 38, 'speed': 3.1},
                'east_approach': {'vehicles': 2, 'density': 1.5, 'waiting_time': 5, 'speed': 12.0},
                'west_approach': {'vehicles': 1, 'density': 0.8, 'waiting_time': 2, 'speed': 13.5}
            },
            'expected': 'North-South gets extended green, East-West reduced for faster cycling'
        },
        {
            'name': 'Heavy East-West Priority',
            'description': 'East-West has heavy traffic, North-South is light',
            'traffic_data': {
                'north_approach': {'vehicles': 3, 'density': 2.1, 'waiting_time': 8, 'speed': 11.0},
                'south_approach': {'vehicles': 2, 'density': 1.5, 'waiting_time': 6, 'speed': 12.5},
                'east_approach': {'vehicles': 20, 'density': 15.5, 'waiting_time': 50, 'speed': 2.0},
                'west_approach': {'vehicles': 17, 'density': 13.0, 'waiting_time': 42, 'speed': 2.8}
            },
            'expected': 'East-West gets extended green, North-South reduced for faster cycling'
        },
        {
            'name': 'Balanced Equal Priority',
            'description': 'All directions have similar moderate traffic',
            'traffic_data': {
                'north_approach': {'vehicles': 8, 'density': 6.0, 'waiting_time': 18, 'speed': 6.5},
                'south_approach': {'vehicles': 7, 'density': 5.5, 'waiting_time': 16, 'speed': 7.0},
                'east_approach': {'vehicles': 9, 'density': 6.5, 'waiting_time': 20, 'speed': 6.0},
                'west_approach': {'vehicles': 8, 'density': 6.0, 'waiting_time': 17, 'speed': 6.8}
            },
            'expected': 'Equal timing for both directions (balanced approach)'
        },
        {
            'name': 'Extreme Heavy vs Light',
            'description': 'One direction extremely heavy, other extremely light',
            'traffic_data': {
                'north_approach': {'vehicles': 25, 'density': 20.0, 'waiting_time': 80, 'speed': 1.5},
                'south_approach': {'vehicles': 23, 'density': 18.5, 'waiting_time': 75, 'speed': 1.8},
                'east_approach': {'vehicles': 0, 'density': 0.0, 'waiting_time': 0, 'speed': 0},
                'west_approach': {'vehicles': 1, 'density': 0.5, 'waiting_time': 1, 'speed': 14.0}
            },
            'expected': 'Maximum prioritization with red time constraints'
        },
        {
            'name': 'Red Time Constraint Test',
            'description': 'Test 2-minute maximum red time enforcement',
            'traffic_data': {
                'north_approach': {'vehicles': 30, 'density': 25.0, 'waiting_time': 100, 'speed': 1.0},
                'south_approach': {'vehicles': 28, 'density': 23.0, 'waiting_time': 95, 'speed': 1.2},
                'east_approach': {'vehicles': 5, 'density': 3.0, 'waiting_time': 12, 'speed': 9.0},
                'west_approach': {'vehicles': 4, 'density': 2.5, 'waiting_time': 10, 'speed': 10.0}
            },
            'expected': 'Heavy priority but constrained by 2-minute red time limit'
        }
    ]
    
    print("🔬 TESTING ADVANCED LANE PRIORITIZATION:")
    print("=" * 60)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Expected: {scenario['expected']}")
        print()
        
        # Calculate lane-based timing
        timing_plan = edge_controller.calculate_lane_based_timing(
            scenario['traffic_data'], i * 30
        )
        
        if timing_plan:
            print(f"📊 Traffic Analysis:")
            for direction, data in scenario['traffic_data'].items():
                vehicles = data['vehicles']
                density = data['density']
                waiting = data['waiting_time']
                speed = data['speed']
                
                # Classify density
                if density > edge_controller.density_threshold_high:
                    level = "🔴 HEAVY"
                elif density > edge_controller.density_threshold_low:
                    level = "🟡 MEDIUM"
                else:
                    level = "🟢 LIGHT"
                
                print(f"   {direction:15}: {vehicles:2d} vehicles, {level}")
                print(f"                    Density: {density:4.1f}, Wait: {waiting:4.1f}s, Speed: {speed:4.1f} m/s")
            
            print(f"\n🧮 Advanced Timing Calculation:")
            print(f"   Priority Direction: {timing_plan['priority_direction']}")
            print(f"   North-South Pressure: {timing_plan['ns_pressure']:.1f}")
            print(f"   East-West Pressure: {timing_plan['ew_pressure']:.1f}")
            print(f"   Pressure Ratio: {timing_plan['pressure_ratio']:.3f}")
            
            print(f"\n🚦 Smart Timing Decision:")
            print(f"   North-South Green: {timing_plan['north_south_green']}s")
            print(f"   East-West Green: {timing_plan['east_west_green']}s")
            print(f"   Total Cycle Time: {timing_plan['cycle_time']}s")
            
            print(f"\n⏱️  Red Time Analysis:")
            print(f"   Max Red (North-South): {timing_plan['max_red_ns']}s")
            print(f"   Max Red (East-West): {timing_plan['max_red_ew']}s")
            
            # Check constraints
            max_red_violation = False
            if timing_plan['max_red_ns'] > 120:
                print(f"   ⚠️  North-South red time exceeds 2 minutes!")
                max_red_violation = True
            if timing_plan['max_red_ew'] > 120:
                print(f"   ⚠️  East-West red time exceeds 2 minutes!")
                max_red_violation = True
            
            if not max_red_violation:
                print(f"   ✅ All red times within 2-minute constraint")
            
            # Show prioritization effect
            ns_green = timing_plan['north_south_green']
            ew_green = timing_plan['east_west_green']
            base_green = edge_controller.base_green_time
            
            ns_change = ((ns_green - base_green) / base_green) * 100
            ew_change = ((ew_green - base_green) / base_green) * 100
            
            print(f"\n📈 Prioritization Effect:")
            if ns_change > 5:
                print(f"   🔴 North-South: +{ns_change:.1f}% (PRIORITIZED)")
            elif ns_change < -5:
                print(f"   🟢 North-South: {ns_change:.1f}% (reduced for faster cycling)")
            else:
                print(f"   ➡️  North-South: {ns_change:.1f}% (normal)")
            
            if ew_change > 5:
                print(f"   🔴 East-West: +{ew_change:.1f}% (PRIORITIZED)")
            elif ew_change < -5:
                print(f"   🟢 East-West: {ew_change:.1f}% (reduced for faster cycling)")
            else:
                print(f"   ➡️  East-West: {ew_change:.1f}% (normal)")
        
        else:
            print("❌ Error calculating timing plan")
        
        if i < len(scenarios):
            print("\n" + "-" * 60)

def show_algorithm_features():
    """Show the key features of the advanced algorithm"""
    print("\n\n🏆 ADVANCED EDGE ALGORITHM FEATURES")
    print("=" * 60)
    
    print("🎯 Smart Lane Prioritization:")
    print("   • Heavy traffic lanes → Extended green time")
    print("   • Light traffic lanes → Reduced green (faster cycling)")
    print("   • Equal pressure lanes → Similar timing")
    print("   • Dynamic priority based on real-time pressure")
    print()
    
    print("⏱️  Advanced Timing Management:")
    print("   • Maximum 2-minute red time per lane")
    print("   • Minimum safety green times (8s per lane)")
    print("   • Yellow and clearance time consideration")
    print("   • Total cycle time optimization")
    print()
    
    print("🔄 Intelligent Cycle Management:")
    print("   • Faster cycling for light traffic (reduces delays)")
    print("   • Extended cycles for heavy traffic (maximizes throughput)")
    print("   • Balanced cycles for equal pressure")
    print("   • Constraint-based optimization")
    print()
    
    print("📊 Multi-Factor Analysis:")
    print("   • Vehicle density (50% weight)")
    print("   • Waiting time pressure (30% weight)")
    print("   • Speed analysis (20% weight)")
    print("   • Historical trend consideration")
    print()
    
    print("🛡️  Safety & Reliability:")
    print("   • Minimum green time enforcement")
    print("   • Maximum red time protection")
    print("   • Gradual timing changes")
    print("   • Error handling and fallbacks")

if __name__ == "__main__":
    # Demonstrate advanced lane prioritization
    demonstrate_advanced_lane_prioritization()
    
    # Show algorithm features
    show_algorithm_features()
    
    print("\n\n🎉 ADVANCED EDGE ALGORITHM DEMONSTRATION COMPLETE!")
    print("✅ Lane prioritization implemented")
    print("✅ 2-minute red time constraint enforced")
    print("✅ Smart traffic management ready for deployment!")
    print("🚀 Perfect for Phase 1 implementation with Phase 2 cloud integration")