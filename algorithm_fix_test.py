"""
Quick Algorithm Fix Test
=======================

Tests the FIXED algorithm against normal mode to verify performance improvements.
Uses simplified scenario with focus on debugging data.
"""

import os
import sys
import json
import time
import subprocess
import traci
import statistics
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fixed_edge_traffic_controller import EdgeTrafficController

def run_test_scenario(mode='normal', duration=300):
    """Run a test scenario with specified mode"""
    
    print(f"🧪 Running {mode.upper()} mode test...")
    
    # SUMO configuration
    sumo_config = "demo.sumocfg"
    
    # Start SUMO
    if mode == 'normal':
        sumo_cmd = ["sumo-gui", "-c", sumo_config, "--start", "--quit-on-end"]
    else:
        sumo_cmd = ["sumo", "-c", sumo_config, "--start", "--quit-on-end"]
    
    traci.start(sumo_cmd)
    
    # Initialize controller for adaptive mode
    controller = None
    if mode == 'adaptive':
        controller = EdgeTrafficController("J4", base_green_time=30)
    
    # Data collection
    results = []
    total_wait_time = 0
    total_vehicles = 0
    adaptations = 0
    
    start_time = time.time()
    
    try:
        for step in range(duration):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            
            # Collect performance data every 10 steps
            if step % 10 == 0:
                # Get all vehicles
                vehicle_ids = traci.vehicle.getIDList()
                step_wait_time = 0
                
                for veh_id in vehicle_ids:
                    wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    step_wait_time += wait_time
                
                avg_wait = step_wait_time / max(len(vehicle_ids), 1)
                
                results.append({
                    'time': current_time,
                    'step': step,
                    'vehicles': len(vehicle_ids),
                    'total_waiting_time': step_wait_time,
                    'avg_waiting_time': avg_wait,
                    'adaptations': adaptations
                })
                
                total_wait_time += step_wait_time
                total_vehicles += len(vehicle_ids)
            
            # Apply adaptive algorithm if enabled
            if mode == 'adaptive' and controller:
                result = controller.apply_edge_algorithm(current_time)
                if result:
                    adaptations += 1
        
        # Calculate final metrics
        avg_total_wait = total_wait_time / max(total_vehicles, 1)
        simulation_time = time.time() - start_time
        
        print(f"✅ {mode.upper()} mode completed:")
        print(f"   Total vehicles processed: {total_vehicles}")
        print(f"   Average waiting time: {avg_total_wait:.1f}s")
        print(f"   Total adaptations: {adaptations}")
        print(f"   Simulation time: {simulation_time:.1f}s")
        
        return {
            'mode': mode,
            'results': results,
            'summary': {
                'total_vehicles': total_vehicles,
                'avg_waiting_time': avg_total_wait,
                'total_adaptations': adaptations,
                'simulation_duration': duration,
                'real_time': simulation_time
            }
        }
        
    except Exception as e:
        print(f"❌ Error in {mode} mode: {e}")
        return None
    
    finally:
        traci.close()

def compare_results(normal_data, adaptive_data):
    """Compare normal vs adaptive mode results"""
    
    print(f"\n📊 ALGORITHM FIX PERFORMANCE COMPARISON")
    print("="*60)
    
    normal_avg = normal_data['summary']['avg_waiting_time']
    adaptive_avg = adaptive_data['summary']['avg_waiting_time']
    
    normal_vehicles = normal_data['summary']['total_vehicles']
    adaptive_vehicles = adaptive_data['summary']['total_vehicles']
    
    adaptations = adaptive_data['summary']['total_adaptations']
    
    # Calculate performance ratio
    if normal_avg > 0:
        performance_ratio = adaptive_avg / normal_avg
    else:
        performance_ratio = 1.0
    
    print(f"NORMAL MODE:")
    print(f"  Average Waiting Time: {normal_avg:.1f}s")
    print(f"  Total Vehicles: {normal_vehicles}")
    
    print(f"\nFIXED ADAPTIVE MODE:")
    print(f"  Average Waiting Time: {adaptive_avg:.1f}s")
    print(f"  Total Vehicles: {adaptive_vehicles}")
    print(f"  Total Adaptations: {adaptations}")
    
    print(f"\nPERFORMANCE ANALYSIS:")
    print(f"  Performance Ratio: {performance_ratio:.2f}x")
    
    if performance_ratio < 0.9:
        print(f"  🎉 EXCELLENT: Fixed algorithm is {((1-performance_ratio)*100):.1f}% BETTER!")
        verdict = "FIXED_SUCCESS"
    elif performance_ratio < 1.1:
        print(f"  ✅ GOOD: Fixed algorithm performs similarly (+/-10%)")
        verdict = "ACCEPTABLE"
    elif performance_ratio < 1.5:
        print(f"  ⚠️  MINOR ISSUE: Fixed algorithm {((performance_ratio-1)*100):.1f}% worse")
        verdict = "NEEDS_TUNING"
    elif performance_ratio < 3.0:
        print(f"  ❌ PROBLEM: Fixed algorithm {((performance_ratio-1)*100):.1f}% worse")
        verdict = "STILL_BROKEN"
    else:
        print(f"  🚨 CRITICAL: Fixed algorithm {performance_ratio:.1f}x worse - MAJOR ISSUE REMAINS")
        verdict = "CRITICAL_FAILURE"
    
    print(f"\nVERDICT: {verdict}")
    
    # Generate debugging insights
    print(f"\n🔍 DEBUGGING INSIGHTS:")
    print("-"*40)
    
    if adaptations > 50:
        print(f"  📈 High adaptation rate: {adaptations} changes in {adaptive_data['summary']['simulation_duration']}s")
        print(f"     → {adaptations/(adaptive_data['summary']['simulation_duration']/60):.1f} adaptations per minute")
    else:
        print(f"  📉 Conservative adaptation rate: {adaptations} changes")
    
    if performance_ratio > 1.2:
        print(f"  🚨 Algorithm still making traffic worse")
        print(f"     → Need more conservative settings or disable algorithm")
    else:
        print(f"  ✅ Algorithm performance acceptable")
    
    # Save detailed comparison
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'normal_mode': normal_data,
        'fixed_adaptive_mode': adaptive_data,
        'performance_ratio': performance_ratio,
        'verdict': verdict,
        'analysis': {
            'adaptations_per_minute': adaptations/(adaptive_data['summary']['simulation_duration']/60),
            'improvement_percentage': (1-performance_ratio)*100,
            'recommendation': get_recommendation(verdict, performance_ratio, adaptations)
        }
    }
    
    os.makedirs('algorithm_fix_results', exist_ok=True)
    with open('algorithm_fix_results/fix_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Results saved to: algorithm_fix_results/fix_comparison.json")
    
    return verdict, performance_ratio

def get_recommendation(verdict, ratio, adaptations):
    """Get recommendation based on test results"""
    
    if verdict == "FIXED_SUCCESS":
        return "✅ Algorithm fix successful! Deploy with monitoring."
    elif verdict == "ACCEPTABLE":
        return "✅ Algorithm acceptable. Continue testing and monitoring."
    elif verdict == "NEEDS_TUNING":
        return "⚠️  Algorithm needs parameter tuning. Reduce adaptation frequency or change magnitude."
    elif verdict == "STILL_BROKEN":
        return "❌ Algorithm still problematic. Consider disabling or major redesign."
    else:
        return "🚨 Algorithm critically broken. DISABLE immediately and investigate."

def main():
    """Run algorithm fix test"""
    
    print("🧪 ALGORITHM FIX VALIDATION TEST")
    print("="*50)
    print("Testing FIXED algorithm against normal mode...")
    print("This will help verify if the performance issues are resolved.")
    
    # Test both modes with shorter duration for quick validation
    test_duration = 200  # 200 simulation steps
    
    try:
        # Run normal mode test
        normal_results = run_test_scenario('normal', test_duration)
        if not normal_results:
            print("❌ Normal mode test failed")
            return
        
        time.sleep(2)  # Brief pause between tests
        
        # Run fixed adaptive mode test
        adaptive_results = run_test_scenario('adaptive', test_duration)
        if not adaptive_results:
            print("❌ Adaptive mode test failed")
            return
        
        # Compare results
        verdict, ratio = compare_results(normal_results, adaptive_results)
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"Algorithm fix test completed with verdict: {verdict}")
        print(f"Performance ratio: {ratio:.2f}x")
        
        if ratio < 1.2:
            print("🎉 SUCCESS: Algorithm fix appears to be working!")
        else:
            print("⚠️  Algorithm still needs work. Check parameters.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()