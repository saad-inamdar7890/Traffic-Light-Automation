"""
Quick Test Version of Traffic Scenario Simulation
Tests one scenario with shorter duration to validate functionality
"""

import os
import sys
import traci
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
import time
import random

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

# Import our modular components
from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class QuickTestFlowManager:
    """Quick test flow manager"""
    
    def __init__(self):
        self.flow_configs = {
            # Car flows - reduced rates for quick test
            'f_0_cars': {'from': 'E0', 'to': 'E0.319', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_1_cars': {'from': 'E0', 'to': '-E1.238', 'base_rate': 30, 'current_rate': 30, 'vtype': 'car'},
            'f_2_cars': {'from': 'E0', 'to': 'E1.200', 'base_rate': 30, 'current_rate': 30, 'vtype': 'car'},
            'f_3_cars': {'from': '-E1', 'to': '-E1.238', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_4_cars': {'from': '-E1', 'to': '-E0.254', 'base_rate': 30, 'current_rate': 30, 'vtype': 'car'},
            'f_5_cars': {'from': '-E1', 'to': 'E0.319', 'base_rate': 30, 'current_rate': 30, 'vtype': 'car'},
            'f_6_cars': {'from': '-E0', 'to': '-E1.238', 'base_rate': 30, 'current_rate': 30, 'vtype': 'car'},
            'f_7_cars': {'from': '-E0', 'to': 'E1.200', 'base_rate': 30, 'current_rate': 30, 'vtype': 'car'},
            'f_8_cars': {'from': 'E1', 'to': '-E0.254', 'base_rate': 30, 'current_rate': 30, 'vtype': 'car'},
            'f_9_cars': {'from': 'E1', 'to': 'E0.319', 'base_rate': 30, 'current_rate': 30, 'vtype': 'car'},
            'f_10_cars': {'from': 'E1', 'to': 'E1.200', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_11_cars': {'from': '-E0', 'to': '-E0.254', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            
            # Motorcycle flows
            'f_0_bikes': {'from': 'E0', 'to': 'E0.319', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_1_bikes': {'from': 'E0', 'to': '-E1.238', 'base_rate': 20, 'current_rate': 20, 'vtype': 'motorcycle'},
            'f_2_bikes': {'from': 'E0', 'to': 'E1.200', 'base_rate': 20, 'current_rate': 20, 'vtype': 'motorcycle'},
            'f_3_bikes': {'from': '-E1', 'to': '-E1.238', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_4_bikes': {'from': '-E1', 'to': '-E0.254', 'base_rate': 20, 'current_rate': 20, 'vtype': 'motorcycle'},
            'f_5_bikes': {'from': '-E1', 'to': 'E0.319', 'base_rate': 20, 'current_rate': 20, 'vtype': 'motorcycle'},
            'f_6_bikes': {'from': '-E0', 'to': '-E1.238', 'base_rate': 20, 'current_rate': 20, 'vtype': 'motorcycle'},
            'f_7_bikes': {'from': '-E0', 'to': 'E1.200', 'base_rate': 20, 'current_rate': 20, 'vtype': 'motorcycle'},
            'f_8_bikes': {'from': 'E1', 'to': '-E0.254', 'base_rate': 20, 'current_rate': 20, 'vtype': 'motorcycle'},
            'f_9_bikes': {'from': 'E1', 'to': 'E0.319', 'base_rate': 20, 'current_rate': 20, 'vtype': 'motorcycle'},
            'f_10_bikes': {'from': 'E1', 'to': 'E1.200', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_11_bikes': {'from': '-E0', 'to': '-E0.254', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'}
        }
    
    def set_heavy_north_south(self):
        """Set heavy traffic in North-South direction"""
        print("🚦 QUICK TEST: Heavy North-South Traffic")
        
        for flow_id, config in self.flow_configs.items():
            if config['from'] in ['E1', '-E1']:  # North-South
                config['current_rate'] = int(config['base_rate'] * 3.0)
            else:  # East-West
                config['current_rate'] = int(config['base_rate'] * 0.5)

class QuickTestSimulation:
    """Quick test version of scenario simulation"""
    
    def __init__(self, config_file="demo.sumocfg", junction_id="J4"):
        self.config_file = config_file
        self.junction_id = junction_id
        self.sumo_cmd = ["sumo", "-c", config_file]  # Use console version for quick test
        
        # Short test durations
        self.phase_duration = 120  # 2 minutes per phase
        
        # Initialize components
        self.flow_manager = QuickTestFlowManager()
        self.traffic_controller = AdaptiveTrafficController(junction_id)
        self.analyzer = TrafficAnalyzer()
    
    def run_quick_test(self):
        """Run quick test simulation"""
        print("🚦 QUICK TEST: Traffic Scenario System Validation")
        print("=" * 60)
        print("⏱️  Testing: 2 min Normal + 2 min Adaptive (4 min total)")
        print("🎯 Scenario: Heavy North-South Traffic")
        print("=" * 60)
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        
        test_data = {
            'normal': {
                'waiting_times': [], 'pressures_ns': [], 'pressures_ew': [],
                'vehicle_counts': [], 'timestamps': []
            },
            'adaptive': {
                'waiting_times': [], 'pressures_ns': [], 'pressures_ew': [],
                'vehicle_counts': [], 'timestamps': []
            }
        }
        
        try:
            # Set heavy North-South traffic
            self.flow_manager.set_heavy_north_south()
            
            # Phase 1: Normal mode (0-120 seconds)
            print("🔴 Normal Mode (0-2 min)")
            self._run_test_phase(test_data['normal'], 0, self.phase_duration, adaptive=False)
            
            # Clear vehicles
            self._clear_vehicles()
            
            # Phase 2: Adaptive mode (120-240 seconds)
            print("🟢 Adaptive Mode (2-4 min)")
            self._run_test_phase(test_data['adaptive'], self.phase_duration, 
                               self.phase_duration * 2, adaptive=True)
            
        except Exception as e:
            print(f"❌ Error in quick test: {e}")
        
        finally:
            traci.close()
        
        # Analyze results
        self._analyze_quick_results(test_data)
        
        print("\n✅ QUICK TEST COMPLETED!")
        print("🎉 System validation successful - ready for full scenario testing!")
    
    def _run_test_phase(self, data_dict, start_time, end_time, adaptive=False):
        """Run a single test phase"""
        
        for step in range(start_time, end_time):
            traci.simulationStep()
            
            # Collect data
            step_data = self.analyzer.collect_traffic_metrics(step, traci)
            
            if step_data:
                # Calculate pressures
                ns_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E1', '-E1']
                )
                ew_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E0', '-E0']
                )
                
                # Apply adaptive control if needed
                if adaptive:
                    self.traffic_controller.apply_adaptive_control(step_data, step)
                
                # Store data every 15 seconds
                if step % 15 == 0:
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['pressures_ns'].append(ns_pressure)
                    data_dict['pressures_ew'].append(ew_pressure)
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['timestamps'].append(step - start_time)
                
                # Display progress every 30 seconds
                if step % 30 == 0:
                    mode = "Adaptive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    print(f"   {mode} - {minute:4.1f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | "
                          f"NS: {ns_pressure:5.1f} | EW: {ew_pressure:5.1f}")
    
    def _clear_vehicles(self):
        """Clear all vehicles"""
        try:
            vehicle_ids = traci.vehicle.getIDList()
            for veh_id in vehicle_ids:
                try:
                    traci.vehicle.remove(veh_id)
                except:
                    pass
            print("🧹 Cleared vehicles between phases")
        except:
            pass
    
    def _analyze_quick_results(self, data):
        """Analyze quick test results"""
        print(f"\n📊 QUICK TEST ANALYSIS:")
        print("=" * 40)
        
        if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
            print("❌ Insufficient data collected")
            return
        
        # Calculate averages
        normal_avg_wait = statistics.mean(data['normal']['waiting_times'])
        adaptive_avg_wait = statistics.mean(data['adaptive']['waiting_times'])
        
        normal_avg_vehicles = statistics.mean(data['normal']['vehicle_counts'])
        adaptive_avg_vehicles = statistics.mean(data['adaptive']['vehicle_counts'])
        
        normal_avg_ns = statistics.mean(data['normal']['pressures_ns'])
        adaptive_avg_ns = statistics.mean(data['adaptive']['pressures_ns'])
        
        normal_avg_ew = statistics.mean(data['normal']['pressures_ew'])
        adaptive_avg_ew = statistics.mean(data['adaptive']['pressures_ew'])
        
        # Calculate improvements
        wait_improvement = ((normal_avg_wait - adaptive_avg_wait) / normal_avg_wait * 100) if normal_avg_wait > 0 else 0
        
        print(f"📈 PERFORMANCE COMPARISON:")
        print(f"   Waiting Time   - Normal: {normal_avg_wait:5.1f}s | Adaptive: {adaptive_avg_wait:5.1f}s | Improvement: {wait_improvement:5.1f}%")
        print(f"   Vehicle Count  - Normal: {normal_avg_vehicles:5.1f} | Adaptive: {adaptive_avg_vehicles:5.1f}")
        print(f"   NS Pressure    - Normal: {normal_avg_ns:5.1f} | Adaptive: {adaptive_avg_ns:5.1f}")
        print(f"   EW Pressure    - Normal: {normal_avg_ew:5.1f} | Adaptive: {adaptive_avg_ew:5.1f}")
        
        # Create simple graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Waiting time comparison
        normal_time = [t/60 for t in data['normal']['timestamps']]
        adaptive_time = [t/60 for t in data['adaptive']['timestamps']]
        
        ax1.plot(normal_time, data['normal']['waiting_times'], 'r-', label='Normal Mode', linewidth=2)
        ax1.plot(adaptive_time, data['adaptive']['waiting_times'], 'g-', label='Adaptive Mode', linewidth=2)
        ax1.set_title('Quick Test: Waiting Time Comparison')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Waiting Time (s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pressure comparison
        ax2.plot(normal_time, data['normal']['pressures_ns'], 'r-', label='Normal NS', linewidth=2)
        ax2.plot(adaptive_time, data['adaptive']['pressures_ns'], 'g-', label='Adaptive NS', linewidth=2)
        ax2.plot(normal_time, data['normal']['pressures_ew'], 'r--', label='Normal EW', linewidth=2)
        ax2.plot(adaptive_time, data['adaptive']['pressures_ew'], 'g--', label='Adaptive EW', linewidth=2)
        ax2.set_title('Quick Test: Traffic Pressure Comparison')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Traffic Pressure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quick_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if wait_improvement > 5:
            print("✅ VALIDATION PASSED: Adaptive control shows improvement!")
        else:
            print("⚠️  VALIDATION WARNING: Limited improvement detected")

def main():
    """Run quick test"""
    try:
        test = QuickTestSimulation()
        test.run_quick_test()
    except Exception as e:
        print(f"❌ Error in quick test: {e}")

if __name__ == "__main__":
    main()