"""
Quick Single Scenario Test to Compare Algorithm Versions
Tests just the "Heavy One Direction" scenario for faster evaluation
"""

import os
import sys
import traci
import statistics
import time

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class QuickScenarioTest:
    """Quick test of single scenario for algorithm comparison"""
    
    def __init__(self):
        self.sumo_cmd = ["sumo", "-c", "demo.sumocfg"]
        self.traffic_controller = AdaptiveTrafficController("J4")
        self.analyzer = TrafficAnalyzer()
        
        # Heavy one direction scenario flows
        self.flow_configs = {
            # Heavy NS traffic
            'f_0_cars': {'from': 'E0', 'to': 'E0.319', 'rate': 400, 'vtype': 'car'},
            'f_1_cars': {'from': 'E0', 'to': '-E1.238', 'rate': 200, 'vtype': 'car'},
            'f_2_cars': {'from': 'E0', 'to': 'E1.200', 'rate': 200, 'vtype': 'car'},
            'f_3_cars': {'from': '-E1', 'to': '-E1.238', 'rate': 400, 'vtype': 'car'},
            'f_4_cars': {'from': '-E1', 'to': '-E0.254', 'rate': 200, 'vtype': 'car'},
            'f_5_cars': {'from': '-E1', 'to': 'E0.319', 'rate': 200, 'vtype': 'car'},
            
            # Light EW traffic
            'f_6_cars': {'from': '-E0', 'to': '-E1.238', 'rate': 30, 'vtype': 'car'},
            'f_7_cars': {'from': '-E0', 'to': 'E1.200', 'rate': 30, 'vtype': 'car'},
            'f_8_cars': {'from': 'E1', 'to': '-E0.254', 'rate': 30, 'vtype': 'car'},
            'f_9_cars': {'from': 'E1', 'to': 'E0.319', 'rate': 30, 'vtype': 'car'},
            'f_10_cars': {'from': 'E1', 'to': 'E1.200', 'rate': 30, 'vtype': 'car'},
            'f_11_cars': {'from': '-E0', 'to': '-E0.254', 'rate': 30, 'vtype': 'car'}
        }
    
    def run_quick_test(self):
        """Run quick comparison test - 7 minutes each mode"""
        print("⚡ QUICK SCENARIO TEST - Heavy One Direction")
        print("=" * 55)
        print("⏱️  Testing: 7 min Normal + 7 min Balanced Adaptive")
        print("🚗 Traffic: Heavy NS (4x), Light EW (0.3x)")
        print("=" * 55)
        
        traci.start(self.sumo_cmd)
        
        test_data = {
            'normal': {'waiting_times': [], 'vehicle_counts': [], 'adaptations': []},
            'adaptive': {'waiting_times': [], 'vehicle_counts': [], 'adaptations': []}
        }
        
        try:
            # Phase 1: Normal mode (0-420 seconds = 7 minutes)
            print("🔴 NORMAL MODE (7 minutes)")
            self._run_test_phase(test_data['normal'], 0, 420, adaptive=False)
            
            # Clear and reset
            self._clear_vehicles()
            
            # Phase 2: Adaptive mode (420-840 seconds = 7 minutes)
            print("🟢 BALANCED ADAPTIVE MODE (7 minutes)")  
            self._run_test_phase(test_data['adaptive'], 420, 840, adaptive=True)
            
        finally:
            traci.close()
        
        # Quick analysis
        self._analyze_quick_results(test_data)
        
        print("\n✅ QUICK TEST COMPLETED!")
    
    def _run_test_phase(self, data_dict, start_time, end_time, adaptive=False):
        """Run single test phase"""
        adaptations_count = 0
        
        for step in range(start_time, end_time):
            traci.simulationStep()
            
            step_data = self.analyzer.collect_traffic_metrics(step, traci)
            
            if step_data:
                if adaptive:
                    result = self.traffic_controller.apply_adaptive_control(step_data, step)
                    if result.get('applied'):
                        adaptations_count += 1
                
                # Store data every 30 seconds
                if step % 30 == 0:
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['adaptations'].append(adaptations_count)
                
                # Progress every minute
                if step % 60 == 0:
                    mode = "Adaptive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    print(f"   {mode} - {minute:3.0f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | Adaptations: {adaptations_count}")
    
    def _clear_vehicles(self):
        """Clear vehicles"""
        try:
            for veh_id in traci.vehicle.getIDList():
                traci.vehicle.remove(veh_id)
            print("🧹 Vehicles cleared")
        except:
            pass
    
    def _analyze_quick_results(self, data):
        """Quick analysis"""
        if not data['normal']['waiting_times'] or not data['adaptive']['waiting_times']:
            print("❌ Insufficient data")
            return
        
        # Skip first 2 data points for warmup
        normal_avg = statistics.mean(data['normal']['waiting_times'][2:])
        adaptive_avg = statistics.mean(data['adaptive']['waiting_times'][2:])
        
        normal_vehicles = statistics.mean(data['normal']['vehicle_counts'][2:])
        adaptive_vehicles = statistics.mean(data['adaptive']['vehicle_counts'][2:])
        
        improvement = ((normal_avg - adaptive_avg) / normal_avg * 100) if normal_avg > 0 else 0
        adaptations = data['adaptive']['adaptations'][-1] if data['adaptive']['adaptations'] else 0
        
        print(f"\n📊 QUICK TEST RESULTS:")
        print(f"   Waiting Time - Normal: {normal_avg:5.1f}s | Adaptive: {adaptive_avg:5.1f}s | Improvement: {improvement:5.1f}%")
        print(f"   Vehicle Count - Normal: {normal_vehicles:5.1f} | Adaptive: {adaptive_vehicles:5.1f}")
        print(f"   Adaptations: {adaptations} total ({adaptations/7:.1f} per minute)")
        
        if improvement > 5:
            rating = "🟢 GOOD"
        elif improvement > 1:
            rating = "🟡 FAIR"
        elif improvement > -2:
            rating = "🟠 NEUTRAL"
        else:
            rating = "🔴 POOR"
        
        print(f"   Performance: {rating}")
        
        return improvement

def main():
    test = QuickScenarioTest()
    test.run_quick_test()

if __name__ == "__main__":
    main()