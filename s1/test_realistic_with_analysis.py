"""
K1 Network - Realistic Control with Comprehensive Analysis
===========================================================

KEY PRINCIPLE:
- Algorithm uses ONLY realistic measurable data (vehicle types, queue, occupancy)
- Analysis collects ALL metrics for comparison (waiting times, speeds, throughput)

This allows:
1. Realistic control that can be deployed in real cities
2. Comprehensive comparison with fixed-timing and other algorithms
"""

import os
import sys
import traci
import time
import json
from collections import defaultdict
from typing import Dict, List

# Import the realistic controller
from realistic_traffic_controller import RealisticTrafficController

# SUMO configuration
SUMOCFG_FILE = "k1.sumocfg"


class ComprehensiveAnalyzer:
    """
    Collects comprehensive metrics for algorithm comparison
    
    ❌ These metrics are NOT used in the control algorithm
    ✅ These metrics are ONLY for analysis and comparison
    """
    
    def __init__(self, junctions: List[str]):
        self.junctions = junctions
        
        # Metrics for comparison
        self.metrics = {
            'junction_metrics': defaultdict(lambda: {
                'total_waiting_time': 0.0,
                'total_vehicles': 0,
                'total_delay': 0.0,
                'phase_changes': 0,
                'samples': 0
            }),
            'network_metrics': {
                'total_departed': 0,
                'total_arrived': 0,
                'total_waiting_time': 0.0,
                'total_travel_time': 0.0,
                'samples': 0
            },
            'timestep_data': [],
            'vehicle_data': {},  # Track individual vehicles
        }
    
    def collect_timestep_metrics(self, step: int):
        """
        Collect network-wide metrics for this timestep
        
        ❌ NOT used in control algorithm
        ✅ Used for comparison analysis
        """
        try:
            # Network-wide statistics
            departed = traci.simulation.getDepartedNumber()
            arrived = traci.simulation.getArrivedNumber()
            vehicle_count = traci.vehicle.getIDCount()
            
            # Calculate network-wide waiting time and speed
            total_waiting = 0.0
            total_speed = 0.0
            vehicles = traci.vehicle.getIDList()
            
            for veh_id in vehicles:
                waiting = traci.vehicle.getWaitingTime(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                total_waiting += waiting
                total_speed += speed
            
            avg_speed = total_speed / max(len(vehicles), 1)
            
            # Store timestep data
            self.metrics['timestep_data'].append({
                'step': step,
                'vehicles': vehicle_count,
                'departed': departed,
                'arrived': arrived,
                'total_waiting': total_waiting,
                'avg_speed': avg_speed
            })
            
            # Update network totals
            self.metrics['network_metrics']['total_departed'] += departed
            self.metrics['network_metrics']['total_arrived'] += arrived
            self.metrics['network_metrics']['total_waiting_time'] += total_waiting
            self.metrics['network_metrics']['samples'] += 1
            
        except Exception as e:
            print(f"Warning: Error collecting timestep metrics: {e}")
    
    def collect_junction_metrics(self, junction_id: str, step: int):
        """
        Collect detailed metrics for a specific junction
        
        ❌ NOT used in control algorithm
        ✅ Used for comparison analysis
        """
        try:
            lanes = traci.trafficlight.getControlledLanes(junction_id)
            unique_lanes = list(set(lanes))
            
            junction_waiting = 0.0
            junction_vehicles = 0
            
            for lane in unique_lanes:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                for veh_id in vehicles:
                    junction_vehicles += 1
                    waiting = traci.vehicle.getWaitingTime(veh_id)
                    junction_waiting += waiting
            
            # Update junction metrics
            self.metrics['junction_metrics'][junction_id]['total_waiting_time'] += junction_waiting
            self.metrics['junction_metrics'][junction_id]['total_vehicles'] += junction_vehicles
            self.metrics['junction_metrics'][junction_id]['samples'] += 1
            
        except Exception as e:
            print(f"Warning: Error collecting junction metrics for {junction_id}: {e}")
    
    def print_summary(self, duration: int, algorithm_name: str = "Realistic Adaptive"):
        """
        Print comprehensive analysis summary
        """
        print("\n" + "="*80)
        print(f"COMPREHENSIVE ANALYSIS SUMMARY - {algorithm_name}")
        print("="*80)
        
        print(f"\nSimulation Duration: {duration} seconds ({duration/3600:.2f} hours)")
        
        # Network-wide statistics
        net = self.metrics['network_metrics']
        print(f"\n📊 NETWORK-WIDE PERFORMANCE:")
        print(f"   Total Vehicles Spawned:  {net['total_departed']:>8}")
        print(f"   Total Vehicles Arrived:  {net['total_arrived']:>8}")
        print(f"   Completion Rate:         {(net['total_arrived']/max(net['total_departed'], 1)*100):>7.1f}%")
        
        if net['samples'] > 0 and self.metrics['timestep_data']:
            avg_vehicles = sum(d['vehicles'] for d in self.metrics['timestep_data']) / len(self.metrics['timestep_data'])
            avg_waiting = net['total_waiting_time'] / net['samples']
            avg_speed = sum(d['avg_speed'] for d in self.metrics['timestep_data']) / len(self.metrics['timestep_data'])
            
            print(f"   Avg Active Vehicles:     {avg_vehicles:>8.1f}")
            print(f"   Avg Waiting Time/Step:   {avg_waiting:>8.2f}s")
            print(f"   Avg Network Speed:       {avg_speed:>8.2f} km/h")
        
        # Junction-specific statistics
        print(f"\n🚦 JUNCTION PERFORMANCE:")
        print(f"{'Junction':<12} {'Avg Waiting':<15} {'Total Vehicles':<15} {'Status'}")
        print("-" * 80)
        
        for junction_id in sorted(self.metrics['junction_metrics'].keys()):
            data = self.metrics['junction_metrics'][junction_id]
            
            if data['samples'] > 0 and data['total_vehicles'] > 0:
                avg_waiting = data['total_waiting_time'] / data['samples']
                total_veh = data['total_vehicles']
                phase_changes = data['phase_changes']
                
                # Determine status
                if avg_waiting < 20:
                    status = "✅ Excellent"
                elif avg_waiting < 35:
                    status = "✅ Good"
                elif avg_waiting < 50:
                    status = "⚠️ Moderate"
                else:
                    status = "❌ Congested"
                
                print(f"{junction_id:<12} {avg_waiting:<15.2f} {total_veh:<15} {status}")
        
        print("\n" + "="*80)
    
    def export_to_json(self, filename: str, algorithm_name: str):
        """Export all metrics to JSON for comparison"""
        try:
            export_data = {
                'algorithm': algorithm_name,
                'network_metrics': dict(self.metrics['network_metrics']),
                'junction_metrics': dict(self.metrics['junction_metrics']),
                'timestep_count': len(self.metrics['timestep_data']),
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✅ Analysis data exported to: {filename}")
        except Exception as e:
            print(f"Warning: Error exporting analysis: {e}")


def run_realistic_simulation_with_analysis(duration=3600, gui=False, output_file=None):
    """
    Run K1 simulation with:
    - Realistic control (uses only measurable data)
    - Comprehensive analysis (collects all metrics for comparison)
    """
    print("\n" + "="*80)
    print("K1 NETWORK - REALISTIC CONTROL WITH COMPREHENSIVE ANALYSIS")
    print("="*80)
    print("\n🎯 CONTROL ALGORITHM USES:")
    print("   ✅ Vehicle type classification (cameras)")
    print("   ✅ Queue length (induction loops)")
    print("   ✅ Lane occupancy (sensors)")
    print("   ✅ Vehicle density (calculated)")
    print("\n📊 ANALYSIS COLLECTS (for comparison only):")
    print("   📈 Individual waiting times")
    print("   📈 Individual speeds")
    print("   📈 Throughput")
    print("   📈 Delay statistics")
    print("\n" + "="*80)
    
    # Check if config file exists
    if not os.path.exists(SUMOCFG_FILE):
        print(f"❌ Error: Configuration file '{SUMOCFG_FILE}' not found!")
        return None
    
    # Choose SUMO binary
    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [sumo_binary, "-c", SUMOCFG_FILE]
    
    # Initialize controllers for each junction
    junctions = ['J0', 'J1', 'J5', 'J6', 'J7', 'J10', 'J11', 'J12', 'J22']
    controllers = {}
    
    for junction in junctions:
        controllers[junction] = RealisticTrafficController(junction_id=junction)
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer(junctions)
    
    try:
        # Start SUMO
        traci.start(sumo_cmd)
        
        step = 0
        start_time = time.time()
        last_print = 0
        last_analysis = 0
        
        print("\n🚦 Simulation running...\n")
        
        # Run simulation
        while step < duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Collect analysis metrics every 10 seconds
            if step % 10 == 0:
                analyzer.collect_timestep_metrics(step)
                
                for junction in junctions:
                    analyzer.collect_junction_metrics(junction, step)
                    
                    # Also collect controller-specific analysis data
                    controller = controllers[junction]
                    controller_metrics = controller.collect_analysis_metrics(step)
            
            # Apply realistic control every 15 seconds
            if step % 15 == 0:
                for junction in junctions:
                    controller = controllers[junction]
                    
                    try:
                        # Get directions for this junction
                        directions = controller.get_controlled_lanes_by_direction()
                        
                        if len(directions['north_south']) > 0 and len(directions['east_west']) > 0:
                            # Calculate pressure using REALISTIC sensors only ✅
                            ns_data = controller.get_direction_pressure(directions['north_south'])
                            ew_data = controller.get_direction_pressure(directions['east_west'])
                            
                            # Get current phase timing
                            current_phase = traci.trafficlight.getPhase(junction)
                            phase_duration = traci.trafficlight.getPhaseDuration(junction)
                            next_switch = traci.trafficlight.getNextSwitch(junction)
                            time_in_phase = step - (next_switch - phase_duration)
                            
                            # Decide action based on REALISTIC pressure ✅
                            action = controller.decide_phase_action(ns_data, ew_data, time_in_phase)
                            
                            # Log decision for analysis
                            controller.log_decision({
                                'step': step,
                                'action': action['action'],
                                'reason': action['reason'],
                                'urgency': action['urgency'],
                                'ns_pressure': ns_data['pressure'],
                                'ew_pressure': ew_data['pressure']
                            })
                            
                            # Apply action if needed
                            if action['action'] == 'SWITCH':
                                # Switch to next phase
                                num_phases = len(traci.trafficlight.getAllProgramLogics(junction)[0].phases)
                                next_phase = (current_phase + 1) % num_phases
                                traci.trafficlight.setPhase(junction, next_phase)
                                
                                # Update analyzer
                                analyzer.metrics['junction_metrics'][junction]['phase_changes'] += 1
                    
                    except Exception as e:
                        # Skip junctions with errors
                        continue
            
            # Print progress every 5 minutes
            if step - last_print >= 300:
                elapsed = time.time() - start_time
                vehicles = traci.vehicle.getIDCount()
                total_changes = sum(analyzer.metrics['junction_metrics'][j]['phase_changes'] for j in junctions)
                
                print(f"⏱️  Step {step:>6} / {duration} ({step/duration*100:>5.1f}%) | "
                      f"Vehicles: {vehicles:>4} | "
                      f"Phase Changes: {total_changes:>4} | "
                      f"Real time: {elapsed:.1f}s")
                last_print = step
            
            # Show detailed analysis every 10 minutes
            if step - last_analysis >= 600 and step > 0:
                print("\n" + "-"*80)
                controllers['J0'].print_status(step)
                print("-"*80 + "\n")
                last_analysis = step
            
            step += 1
        
        # Close SUMO
        traci.close()
        
        # Calculate final summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Simulation completed!")
        print(f"   Simulated time: {step} seconds ({step/3600:.2f} hours)")
        print(f"   Real time: {elapsed_time:.2f} seconds")
        print(f"   Speedup: {step/elapsed_time:.1f}x")
        
        # Print comprehensive analysis
        analyzer.print_summary(step, "Realistic Adaptive Control")
        
        # Export results
        if output_file:
            analyzer.export_to_json(output_file, "Realistic Adaptive Control")
        else:
            analyzer.export_to_json("realistic_control_results.json", "Realistic Adaptive Control")
        
        return analyzer
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            traci.close()
        except:
            pass
        
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run K1 simulation with realistic control and comprehensive analysis'
    )
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--duration', type=int, default=3600, 
                       help='Simulation duration in seconds (default: 3600)')
    parser.add_argument('--output', type=str, default='realistic_control_results.json',
                       help='Output file for analysis results')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SEPARATION OF CONCERNS:")
    print("="*80)
    print("\n✅ ALGORITHM CONTROL (Realistic - Deployable):")
    print("   Uses: Vehicle types, queue length, occupancy")
    print("   Source: Standard sensors (cameras, loops, occupancy)")
    print("   Cost: $15k per intersection")
    print("\n📊 ANALYSIS METRICS (Idealized - Research Only):")
    print("   Collects: Waiting times, speeds, throughput, delay")
    print("   Purpose: Algorithm comparison and performance evaluation")
    print("   NOT used in control decisions")
    print("\n" + "="*80 + "\n")
    
    run_realistic_simulation_with_analysis(
        duration=args.duration,
        gui=args.gui,
        output_file=args.output
    )
