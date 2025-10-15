"""
Quick Test of Optimized Adaptive Algorithm
Tests the enhanced adaptive controller against the normal mode
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

class OptimizedAlgorithmTest:
    """Test the optimized adaptive algorithm"""
    
    def __init__(self, config_file="demo.sumocfg", junction_id="J4"):
        self.config_file = config_file
        self.junction_id = junction_id
        self.sumo_cmd = ["sumo", "-c", config_file]  # Console version for testing
        
        # Test duration
        self.test_duration = 600  # 10 minutes
        
        # Initialize components
        self.traffic_controller = AdaptiveTrafficController(junction_id)
        self.analyzer = TrafficAnalyzer()
        
        # Flow configs for testing (moderate traffic)
        self.flow_configs = {
            'f_0_cars': {'from': 'E0', 'to': 'E0.319', 'rate': 80, 'vtype': 'car'},
            'f_1_cars': {'from': 'E0', 'to': '-E1.238', 'rate': 40, 'vtype': 'car'},
            'f_2_cars': {'from': 'E0', 'to': 'E1.200', 'rate': 40, 'vtype': 'car'},
            'f_3_cars': {'from': '-E1', 'to': '-E1.238', 'rate': 80, 'vtype': 'car'},
            'f_4_cars': {'from': '-E1', 'to': '-E0.254', 'rate': 40, 'vtype': 'car'},
            'f_5_cars': {'from': '-E1', 'to': 'E0.319', 'rate': 40, 'vtype': 'car'},
            'f_6_cars': {'from': '-E0', 'to': '-E1.238', 'rate': 40, 'vtype': 'car'},
            'f_7_cars': {'from': '-E0', 'to': 'E1.200', 'rate': 40, 'vtype': 'car'},
            'f_8_cars': {'from': 'E1', 'to': '-E0.254', 'rate': 40, 'vtype': 'car'},
            'f_9_cars': {'from': 'E1', 'to': 'E0.319', 'rate': 40, 'vtype': 'car'},
            'f_10_cars': {'from': 'E1', 'to': 'E1.200', 'rate': 80, 'vtype': 'car'},
            'f_11_cars': {'from': '-E0', 'to': '-E0.254', 'rate': 80, 'vtype': 'car'}
        }
    
    def run_comparison_test(self):
        """Run comparison between normal and optimized adaptive modes"""
        print("🚦 OPTIMIZED ADAPTIVE ALGORITHM TEST")
        print("=" * 60)
        print("⏱️  Testing: 5 min Normal + 5 min Optimized Adaptive")
        print("🎯 Focus: Enhanced pressure calculation & faster adaptation")
        print("=" * 60)
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        
        test_data = {
            'normal': {
                'waiting_times': [], 'pressures_ns': [], 'pressures_ew': [],
                'vehicle_counts': [], 'timestamps': [], 'adaptations': []
            },
            'optimized': {
                'waiting_times': [], 'pressures_ns': [], 'pressures_ew': [],
                'vehicle_counts': [], 'timestamps': [], 'adaptations': []
            }
        }
        
        try:
            # Phase 1: Normal mode (0-300 seconds)
            print("🔴 Normal Mode (0-5 min)")
            self._run_test_phase(test_data['normal'], 0, 300, adaptive=False)
            
            # Clear vehicles
            self._clear_vehicles()
            
            # Phase 2: Optimized Adaptive mode (300-600 seconds)
            print("🟢 Optimized Adaptive Mode (5-10 min)")
            self._run_test_phase(test_data['optimized'], 300, 600, adaptive=True)
            
        except Exception as e:
            print(f"❌ Error in optimization test: {e}")
        
        finally:
            traci.close()
        
        # Analyze results
        self._analyze_optimization_results(test_data)
        
        print("\n✅ OPTIMIZATION TEST COMPLETED!")
        print("🎉 Enhanced algorithm performance analysis ready!")
    
    def _run_test_phase(self, data_dict, start_time, end_time, adaptive=False):
        """Run a single test phase"""
        
        adaptations_count = 0
        last_adaptation_log = 0
        
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
                    result = self.traffic_controller.apply_adaptive_control(step_data, step)
                    if result.get('applied') and step - last_adaptation_log > 30:
                        adaptations_count += 1
                        last_adaptation_log = step
                        action = result.get('action', 'unknown')
                        print(f"   🔧 Adaptation #{adaptations_count}: {action} at step {step}")
                
                # Store data every 15 seconds
                if step % 15 == 0:
                    data_dict['waiting_times'].append(step_data['avg_waiting_time'])
                    data_dict['pressures_ns'].append(ns_pressure)
                    data_dict['pressures_ew'].append(ew_pressure)
                    data_dict['vehicle_counts'].append(step_data['total_vehicles'])
                    data_dict['timestamps'].append(step - start_time)
                    data_dict['adaptations'].append(adaptations_count)
                
                # Display progress every minute
                if step % 60 == 0:
                    mode = "Optimized" if adaptive else "Normal"
                    minute = (step - start_time) / 60
                    print(f"   {mode} - {minute:4.1f}min | Vehicles: {step_data['total_vehicles']:3d} | "
                          f"Wait: {step_data['avg_waiting_time']:5.1f}s | "
                          f"NS: {ns_pressure:5.1f} | EW: {ew_pressure:5.1f} | "
                          f"Adaptations: {adaptations_count}")
    
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
    
    def _analyze_optimization_results(self, data):
        """Analyze optimization test results"""
        print(f"\n📊 OPTIMIZATION ANALYSIS:")
        print("=" * 50)
        
        if not data['normal']['waiting_times'] or not data['optimized']['waiting_times']:
            print("❌ Insufficient data collected")
            return
        
        # Calculate averages (skip first 2 data points for warmup)
        warmup = 2
        
        normal_avg_wait = statistics.mean(data['normal']['waiting_times'][warmup:])
        optimized_avg_wait = statistics.mean(data['optimized']['waiting_times'][warmup:])
        
        normal_avg_vehicles = statistics.mean(data['normal']['vehicle_counts'][warmup:])
        optimized_avg_vehicles = statistics.mean(data['optimized']['vehicle_counts'][warmup:])
        
        normal_avg_ns = statistics.mean(data['normal']['pressures_ns'][warmup:])
        optimized_avg_ns = statistics.mean(data['optimized']['pressures_ns'][warmup:])
        
        normal_avg_ew = statistics.mean(data['normal']['pressures_ew'][warmup:])
        optimized_avg_ew = statistics.mean(data['optimized']['pressures_ew'][warmup:])
        
        # Calculate improvements
        wait_improvement = ((normal_avg_wait - optimized_avg_wait) / normal_avg_wait * 100) if normal_avg_wait > 0 else 0
        pressure_balance_normal = abs(normal_avg_ns - normal_avg_ew)
        pressure_balance_optimized = abs(optimized_avg_ns - optimized_avg_ew)
        balance_improvement = ((pressure_balance_normal - pressure_balance_optimized) / pressure_balance_normal * 100) if pressure_balance_normal > 0 else 0
        
        # Adaptation statistics
        total_adaptations = data['optimized']['adaptations'][-1] if data['optimized']['adaptations'] else 0
        adaptation_rate = total_adaptations / 5  # per minute
        
        print(f"📈 PERFORMANCE COMPARISON:")
        print(f"   Waiting Time   - Normal: {normal_avg_wait:5.1f}s | Optimized: {optimized_avg_wait:5.1f}s | Improvement: {wait_improvement:5.1f}%")
        print(f"   Vehicle Count  - Normal: {normal_avg_vehicles:5.1f} | Optimized: {optimized_avg_vehicles:5.1f}")
        print(f"   NS Pressure    - Normal: {normal_avg_ns:5.1f} | Optimized: {optimized_avg_ns:5.1f}")
        print(f"   EW Pressure    - Normal: {normal_avg_ew:5.1f} | Optimized: {optimized_avg_ew:5.1f}")
        print(f"   Balance Improve: {balance_improvement:5.1f}% (Lower pressure imbalance is better)")
        print(f"   Adaptations    : {total_adaptations} total ({adaptation_rate:.1f} per minute)")
        
        # Performance rating
        if wait_improvement > 10:
            rating = "🟢 EXCELLENT"
        elif wait_improvement > 5:
            rating = "🟡 GOOD"
        elif wait_improvement > 0:
            rating = "🟠 FAIR"
        else:
            rating = "🔴 NEEDS WORK"
        
        print(f"   Overall Rating: {rating}")
        
        # Create comparison graph
        self._create_optimization_graph(data)
        
        # Algorithm performance metrics
        performance = self.traffic_controller.get_performance_summary()
        print(f"\n🔧 ALGORITHM METRICS:")
        print(f"   Adaptations Made: {performance.get('total_adaptations', 0)}")
        print(f"   Phase History: {len(performance.get('phase_history', {}))}")
        
        return {
            'wait_improvement': wait_improvement,
            'balance_improvement': balance_improvement,
            'total_adaptations': total_adaptations,
            'rating': rating
        }
    
    def _create_optimization_graph(self, data):
        """Create optimization comparison graph"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Convert to minutes
        normal_time = [t/60 for t in data['normal']['timestamps']]
        optimized_time = [t/60 for t in data['optimized']['timestamps']]
        
        # Waiting time comparison
        ax1.plot(normal_time, data['normal']['waiting_times'], 'r-', 
               label='Normal Mode', linewidth=2, alpha=0.8)
        ax1.plot(optimized_time, data['optimized']['waiting_times'], 'g-', 
               label='Optimized Adaptive', linewidth=2, alpha=0.8)
        ax1.set_title('Waiting Time Comparison')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Waiting Time (s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Vehicle count comparison
        ax2.plot(normal_time, data['normal']['vehicle_counts'], 'r-', 
               label='Normal Mode', linewidth=2, alpha=0.8)
        ax2.plot(optimized_time, data['optimized']['vehicle_counts'], 'g-', 
               label='Optimized Adaptive', linewidth=2, alpha=0.8)
        ax2.set_title('Vehicle Count Comparison')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Vehicle Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # NS Pressure comparison
        ax3.plot(normal_time, data['normal']['pressures_ns'], 'r-', 
               label='Normal NS', linewidth=2, alpha=0.8)
        ax3.plot(optimized_time, data['optimized']['pressures_ns'], 'g-', 
               label='Optimized NS', linewidth=2, alpha=0.8)
        ax3.set_title('North-South Pressure Comparison')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Traffic Pressure')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # EW Pressure comparison
        ax4.plot(normal_time, data['normal']['pressures_ew'], 'r-', 
               label='Normal EW', linewidth=2, alpha=0.8)
        ax4.plot(optimized_time, data['optimized']['pressures_ew'], 'g-', 
               label='Optimized EW', linewidth=2, alpha=0.8)
        ax4.set_title('East-West Pressure Comparison')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Traffic Pressure')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Optimized Adaptive Algorithm Performance Test', fontsize=14)
        plt.tight_layout()
        plt.savefig('optimized_algorithm_test.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run optimization test"""
    try:
        test = OptimizedAlgorithmTest()
        test.run_comparison_test()
    except Exception as e:
        print(f"❌ Error in optimization test: {e}")

if __name__ == "__main__":
    main()