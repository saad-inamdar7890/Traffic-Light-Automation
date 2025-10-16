"""
Stress Test for Optimized Adaptive Algorithm
Tests with heavier traffic and challenging scenarios to better evaluate performance
"""

import os
import sys
import traci
import statistics
import matplotlib.pyplot as plt
import time

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

# Import optimized components
from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class StressTestAlgorithm:
    """Stress test for the optimized adaptive algorithm with challenging traffic"""
    
    def __init__(self, config_file="demo.sumocfg", junction_id="J4"):
        self.config_file = config_file
        self.junction_id = junction_id
        self.sumo_cmd = ["sumo", "-c", config_file]  # Console version for testing
        
        # Extended test duration for better evaluation
        self.test_duration = 900  # 15 minutes per phase
        
        # Initialize components
        self.traffic_controller = AdaptiveTrafficController(junction_id)
        self.analyzer = TrafficAnalyzer()
        
        # High-stress traffic flows (much heavier than normal)
        self.stress_flows = {
            # Heavy NS traffic (rush hour simulation)
            'f_0_cars': {'from': 'E0', 'to': 'E0.319', 'rate': 200, 'vtype': 'car'},
            'f_1_cars': {'from': 'E0', 'to': '-E1.238', 'rate': 150, 'vtype': 'car'},
            'f_2_cars': {'from': 'E0', 'to': 'E1.200', 'rate': 100, 'vtype': 'car'},
            'f_3_cars': {'from': '-E1', 'to': '-E1.238', 'rate': 200, 'vtype': 'car'},
            'f_4_cars': {'from': '-E1', 'to': '-E0.254', 'rate': 100, 'vtype': 'car'},
            'f_5_cars': {'from': '-E1', 'to': 'E0.319', 'rate': 150, 'vtype': 'car'},
            
            # Moderate EW traffic
            'f_6_cars': {'from': '-E0', 'to': '-E1.238', 'rate': 120, 'vtype': 'car'},
            'f_7_cars': {'from': '-E0', 'to': 'E1.200', 'rate': 100, 'vtype': 'car'},
            'f_8_cars': {'from': 'E1', 'to': '-E0.254', 'rate': 100, 'vtype': 'car'},
            'f_9_cars': {'from': 'E1', 'to': 'E0.319', 'rate': 120, 'vtype': 'car'},
            'f_10_cars': {'from': 'E1', 'to': 'E1.200', 'rate': 200, 'vtype': 'car'},
            'f_11_cars': {'from': '-E0', 'to': '-E0.254', 'rate': 200, 'vtype': 'car'},
            
            # Some motorcycles for mixed traffic
            'f_12_mc': {'from': 'E0', 'to': '-E1.238', 'rate': 50, 'vtype': 'motorcycle'},
            'f_13_mc': {'from': '-E1', 'to': 'E0.319', 'rate': 50, 'vtype': 'motorcycle'},
            'f_14_mc': {'from': 'E1', 'to': '-E0.254', 'rate': 50, 'vtype': 'motorcycle'},
            'f_15_mc': {'from': '-E0', 'to': 'E1.200', 'rate': 50, 'vtype': 'motorcycle'}
        }
    
    def run_stress_test(self):
        """Run comprehensive stress test comparison"""
        print("🔥 AGGRESSIVE ALGORITHM STRESS TEST")
        print("=" * 70)
        print("⏱️  Testing: 15 min Normal + 15 min Aggressive Adaptive")
        print("🚗 Traffic: HEAVY RUSH HOUR simulation (~1800 vehicles/hour)")
        print("🎯 Goal: Test algorithm under realistic challenging conditions")
        print("=" * 70)
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        
        test_data = {
            'normal': {
                'waiting_times': [], 'pressures_ns': [], 'pressures_ew': [],
                'vehicle_counts': [], 'timestamps': [], 'adaptations': [],
                'total_delays': [], 'throughput': []
            },
            'aggressive': {
                'waiting_times': [], 'pressures_ns': [], 'pressures_ew': [],
                'vehicle_counts': [], 'timestamps': [], 'adaptations': [],
                'total_delays': [], 'throughput': []
            }
        }
        
        try:
            # Phase 1: Normal mode (0-900 seconds = 15 minutes)
            print("🔴 NORMAL MODE - 15 minutes of heavy traffic")
            self._run_stress_phase(test_data['normal'], 0, 900, adaptive=False)
            
            # Clear vehicles and reset
            self._clear_all_traffic()
            
            # Phase 2: Aggressive Adaptive mode (900-1800 seconds = 15 minutes)
            print("🟢 AGGRESSIVE ADAPTIVE MODE - 15 minutes of heavy traffic")
            self._run_stress_phase(test_data['aggressive'], 900, 1800, adaptive=True)
            
        except Exception as e:
            print(f"❌ Error in stress test: {e}")
        
        finally:
            traci.close()
        
        # Comprehensive analysis
        self._analyze_stress_results(test_data)
        
        print("\n✅ STRESS TEST COMPLETED!")
        print("🎉 Comprehensive performance analysis ready!")
    
    def _run_stress_phase(self, data_dict, start_time, end_time, adaptive=False):
        """Run a stress test phase with heavy traffic"""
        
        adaptations_count = 0
        last_adaptation_log = 0
        vehicles_completed = 0
        last_vehicle_count = 0
        
        for step in range(start_time, end_time):
            traci.simulationStep()
            
            # Collect comprehensive data
            step_data = self.analyzer.collect_traffic_metrics(step, traci)
            
            if step_data:
                # Calculate pressures
                ns_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E1', '-E1']
                )
                ew_pressure = self.traffic_controller.calculate_traffic_pressure(
                    step_data, ['E0', '-E0']
                )
                
                # Calculate throughput (vehicles completing their journey)
                current_vehicles = step_data['total_vehicles']
                if current_vehicles < last_vehicle_count:
                    vehicles_completed += (last_vehicle_count - current_vehicles)
                last_vehicle_count = current_vehicles
                
                # Apply adaptive control if needed
                if adaptive:
                    result = self.traffic_controller.apply_adaptive_control(step_data, step)
                    if result.get('applied') and step - last_adaptation_log > 60:
                        adaptations_count += 1
                        last_adaptation_log = step
                        action = result.get('action', 'unknown')
                        print(f"   🔧 Adaptation #{adaptations_count}: {action} at step {step}")
                
                # Store detailed data every 30 seconds for better resolution
                if step % 30 == 0:
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['pressures_ns'].append(ns_pressure)
                    data_dict['pressures_ew'].append(ew_pressure)
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['timestamps'].append(step - start_time)
                    data_dict['adaptations'].append(adaptations_count)
                    data_dict['total_delays'].append(step_data.get('total_delay', 0))
                    data_dict['throughput'].append(vehicles_completed)
                
                # Progress reports every 2 minutes
                if step % 120 == 0:
                    mode = "Aggressive" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    print(f"   {mode} - {minute:4.1f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | "
                          f"NS: {ns_pressure:5.0f} | EW: {ew_pressure:5.0f} | "
                          f"Adaptations: {adaptations_count} | Completed: {vehicles_completed}")
    
    def _clear_all_traffic(self):
        """Comprehensive traffic clearing"""
        try:
            # Remove all vehicles
            vehicle_ids = traci.vehicle.getIDList()
            for veh_id in vehicle_ids:
                try:
                    traci.vehicle.remove(veh_id)
                except:
                    pass
            
            # Reset traffic lights to default phase
            traci.trafficlight.setPhase(self.junction_id, 0)
            
            # Wait a few steps for cleanup
            for _ in range(5):
                traci.simulationStep()
            
            print("🧹 Comprehensive traffic reset completed")
            
        except Exception as e:
            print(f"⚠️  Warning during traffic clearing: {e}")
    
    def _analyze_stress_results(self, data):
        """Comprehensive analysis of stress test results"""
        print(f"\n📊 STRESS TEST ANALYSIS (15-minute phases):")
        print("=" * 60)
        
        if (not data['normal']['waiting_times'] or 
            not data['aggressive']['waiting_times']):
            print("❌ Insufficient data collected")
            return
        
        # Skip initial warmup period (first 3 data points = 1.5 minutes)
        warmup = 3
        
        # Comprehensive metrics calculation
        normal_avg_wait = statistics.mean(data['normal']['waiting_times'][warmup:])
        aggressive_avg_wait = statistics.mean(data['aggressive']['waiting_times'][warmup:])
        
        normal_max_wait = max(data['normal']['waiting_times'][warmup:])
        aggressive_max_wait = max(data['aggressive']['waiting_times'][warmup:])
        
        normal_avg_vehicles = statistics.mean(data['normal']['vehicle_counts'][warmup:])
        aggressive_avg_vehicles = statistics.mean(data['aggressive']['vehicle_counts'][warmup:])
        
        normal_final_throughput = data['normal']['throughput'][-1] if data['normal']['throughput'] else 0
        aggressive_final_throughput = data['aggressive']['throughput'][-1] if data['aggressive']['throughput'] else 0
        
        # Pressure analysis
        normal_avg_ns = statistics.mean(data['normal']['pressures_ns'][warmup:])
        aggressive_avg_ns = statistics.mean(data['aggressive']['pressures_ns'][warmup:])
        
        normal_avg_ew = statistics.mean(data['normal']['pressures_ew'][warmup:])
        aggressive_avg_ew = statistics.mean(data['aggressive']['pressures_ew'][warmup:])
        
        # Performance improvements
        wait_improvement = ((normal_avg_wait - aggressive_avg_wait) / normal_avg_wait * 100) if normal_avg_wait > 0 else 0
        max_wait_improvement = ((normal_max_wait - aggressive_max_wait) / normal_max_wait * 100) if normal_max_wait > 0 else 0
        throughput_improvement = ((aggressive_final_throughput - normal_final_throughput) / max(1, normal_final_throughput) * 100)
        
        # Traffic balance analysis
        normal_pressure_imbalance = abs(normal_avg_ns - normal_avg_ew)
        aggressive_pressure_imbalance = abs(aggressive_avg_ns - aggressive_avg_ew)
        balance_improvement = ((normal_pressure_imbalance - aggressive_pressure_imbalance) / normal_pressure_imbalance * 100) if normal_pressure_imbalance > 0 else 0
        
        # Adaptation statistics
        total_adaptations = data['aggressive']['adaptations'][-1] if data['aggressive']['adaptations'] else 0
        adaptation_rate = total_adaptations / 15  # per minute over 15 minutes
        
        print(f"📈 COMPREHENSIVE PERFORMANCE COMPARISON:")
        print(f"   Average Wait Time  - Normal: {normal_avg_wait:5.1f}s | Aggressive: {aggressive_avg_wait:5.1f}s | Improvement: {wait_improvement:5.1f}%")
        print(f"   Maximum Wait Time  - Normal: {normal_max_wait:5.1f}s | Aggressive: {aggressive_max_wait:5.1f}s | Improvement: {max_wait_improvement:5.1f}%")
        print(f"   Average Vehicles   - Normal: {normal_avg_vehicles:5.1f} | Aggressive: {aggressive_avg_vehicles:5.1f}")
        print(f"   Vehicle Throughput - Normal: {normal_final_throughput:4d} | Aggressive: {aggressive_final_throughput:4d} | Improvement: {throughput_improvement:5.1f}%")
        print(f"   NS Pressure        - Normal: {normal_avg_ns:5.0f} | Aggressive: {aggressive_avg_ns:5.0f}")
        print(f"   EW Pressure        - Normal: {normal_avg_ew:5.0f} | Aggressive: {aggressive_avg_ew:5.0f}")
        print(f"   Pressure Balance   - Improvement: {balance_improvement:5.1f}% (lower imbalance is better)")
        print(f"   Adaptations        - {total_adaptations} total ({adaptation_rate:.1f} per minute)")
        
        # Overall performance rating
        overall_score = (wait_improvement + max_wait_improvement + throughput_improvement + balance_improvement) / 4
        
        if overall_score > 15:
            rating = "🟢 EXCELLENT - Significant improvement achieved!"
        elif overall_score > 8:
            rating = "🟡 GOOD - Noticeable improvement"
        elif overall_score > 3:
            rating = "🟠 FAIR - Modest improvement"
        elif overall_score > 0:
            rating = "🔴 POOR - Minimal improvement"
        else:
            rating = "❌ FAILED - Performance degraded"
        
        print(f"   Overall Score      - {overall_score:5.1f}% improvement")
        print(f"   Performance Rating - {rating}")
        
        # Create comprehensive comparison graph
        self._create_stress_test_graph(data)
        
        # Algorithm performance metrics
        performance = self.traffic_controller.get_performance_summary()
        print(f"\n🔧 ALGORITHM DETAILED METRICS:")
        print(f"   Total Adaptations: {performance.get('total_adaptations', 0)}")
        print(f"   Adaptation Frequency: Every {15*60/max(1, total_adaptations):.1f} seconds on average")
        
        return {
            'wait_improvement': wait_improvement,
            'max_wait_improvement': max_wait_improvement,
            'throughput_improvement': throughput_improvement,
            'balance_improvement': balance_improvement,
            'overall_score': overall_score,
            'rating': rating
        }
    
    def _create_stress_test_graph(self, data):
        """Create comprehensive stress test visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # Convert to minutes
        normal_time = [t/60 for t in data['normal']['timestamps']]
        aggressive_time = [t/60 for t in data['aggressive']['timestamps']]
        
        # Waiting time comparison
        ax1.plot(normal_time, data['normal']['waiting_times'], 'r-', 
               label='Normal Mode', linewidth=2.5, alpha=0.8)
        ax1.plot(aggressive_time, data['aggressive']['waiting_times'], 'g-', 
               label='Aggressive Adaptive', linewidth=2.5, alpha=0.8)
        ax1.set_title('Waiting Time Performance - Stress Test', fontweight='bold')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Average Waiting Time (s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Vehicle throughput comparison
        ax2.plot(normal_time, data['normal']['throughput'], 'r-', 
               label='Normal Mode', linewidth=2.5, alpha=0.8)
        ax2.plot(aggressive_time, data['aggressive']['throughput'], 'g-', 
               label='Aggressive Adaptive', linewidth=2.5, alpha=0.8)
        ax2.set_title('Vehicle Throughput - Cumulative', fontweight='bold')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Vehicles Completed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Pressure comparison (combined)
        ax3.plot(normal_time, data['normal']['pressures_ns'], 'r--', 
               label='Normal NS', linewidth=2, alpha=0.7)
        ax3.plot(normal_time, data['normal']['pressures_ew'], 'r:', 
               label='Normal EW', linewidth=2, alpha=0.7)
        ax3.plot(aggressive_time, data['aggressive']['pressures_ns'], 'g--', 
               label='Aggressive NS', linewidth=2, alpha=0.7)
        ax3.plot(aggressive_time, data['aggressive']['pressures_ew'], 'g:', 
               label='Aggressive EW', linewidth=2, alpha=0.7)
        ax3.set_title('Traffic Pressure Comparison', fontweight='bold')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Traffic Pressure')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Vehicle count comparison
        ax4.plot(normal_time, data['normal']['vehicle_counts'], 'r-', 
               label='Normal Mode', linewidth=2.5, alpha=0.8)
        ax4.plot(aggressive_time, data['aggressive']['vehicle_counts'], 'g-', 
               label='Aggressive Adaptive', linewidth=2.5, alpha=0.8)
        ax4.set_title('Traffic Volume - Active Vehicles', fontweight='bold')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Active Vehicle Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('AGGRESSIVE ADAPTIVE ALGORITHM - STRESS TEST RESULTS\n'
                    '15-minute phases with heavy rush hour traffic simulation', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('stress_test_algorithm_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run stress test for algorithm evaluation"""
    try:
        test = StressTestAlgorithm()
        test.run_stress_test()
    except Exception as e:
        print(f"❌ Error in stress test: {e}")

if __name__ == "__main__":
    main()