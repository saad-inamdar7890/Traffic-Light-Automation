
import os
import sys
import traci

# ✅ Set SUMO_HOME if not already set in Environment Variables
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

# Import our modular components
from dynamic_flow_manager import DynamicFlowManager
from dynamic_traffic_light import AdaptiveTrafficController
from results_analyzer import TrafficAnalyzer

class TrafficSimulation:
    def __init__(self, config_file="demo.sumocfg", junction_id="J4"):
        """Initialize the traffic simulation with all components"""
        self.config_file = config_file
        self.junction_id = junction_id
        self.sumo_cmd = ["sumo-gui", "-c", config_file]
        
        # Initialize components
        self.flow_manager = DynamicFlowManager()
        self.traffic_controller = AdaptiveTrafficController(junction_id)
        self.analyzer = TrafficAnalyzer()
        
        # Simulation parameters
        self.simulation_duration = 1800  # 30 minutes
        self.analysis_interval = 300     # Display detailed analysis every 5 minutes
        self.status_interval = 60        # Brief status every minute
        
    def run_simulation(self):
        """Run the complete dynamic traffic simulation"""
        print("🚦 Starting Integrated Dynamic Traffic Light Simulation")
        print("🔧 Components: Dynamic Flows + Adaptive Control + Real-time Analysis")
        print("⏱️  Duration: 30 minutes simulation time")
        print(f"📊 Analysis every {self.analysis_interval/60} minutes, Status every {self.status_interval} seconds")
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        
        try:
            for step in range(self.simulation_duration):
                # Advance simulation
                traci.simulationStep()
                
                # Update dynamic flows
                flow_updated = self.flow_manager.update_flow_rates(step)
                if flow_updated:
                    self.flow_manager.apply_flow_changes()
                
                # Collect traffic data
                step_data = self.analyzer.collect_traffic_metrics(step, traci)
                
                if step_data:
                    # Apply adaptive traffic light control
                    tl_result = self.traffic_controller.apply_adaptive_control(step_data, step)
                    
                    # Update pressure data for analysis
                    if tl_result and 'analysis' in tl_result:
                        analysis = tl_result['analysis']
                        self.analyzer.update_pressure_data(
                            analysis.get('ns_pressure', 0),
                            analysis.get('ew_pressure', 0)
                        )
                    
                    # Display detailed analysis
                    if step % self.analysis_interval == 0 and step > 0:
                        self.analyzer.display_real_time_analysis(step, step_data, tl_result)
                    
                    # Brief status update
                    if step % self.status_interval == 0:
                        self._display_brief_status(step, step_data, tl_result)
                
                # Check for early termination
                if step > 600 and len(traci.vehicle.getIDList()) == 0:
                    print(f"\n✅ No vehicles remaining at step {step} - Ending simulation early")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  Simulation interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Simulation error: {e}")
        
        finally:
            self._generate_final_report()
            traci.close()
            print("\n🎉 Dynamic Traffic Light Simulation Complete!")
    
    def _display_brief_status(self, step, step_data, tl_result):
        """Display brief status information"""
        hour = (step / 3600) % 24
        
        # Traffic light info
        if tl_result and 'tl_state' in tl_result:
            tl_state = tl_result['tl_state']
            phase_info = f"Phase {tl_state['phase']} ({tl_state['phase_name'][:8]})"
        else:
            phase_info = "Phase N/A"
        
        # Pressure info
        if tl_result and 'analysis' in tl_result:
            analysis = tl_result['analysis']
            ns_pressure = analysis.get('ns_pressure', 0)
            ew_pressure = analysis.get('ew_pressure', 0)
        else:
            ns_pressure = ew_pressure = 0
        
        print(f"Step {step:4d} | Time: {hour:4.1f}h | Vehicles: {step_data['total_vehicles']:3d} | "
              f"NS: {ns_pressure:5.1f} | EW: {ew_pressure:5.1f} | {phase_info}")
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("📋 GENERATING COMPREHENSIVE FINAL REPORT")
        print("="*80)
        
        # Get summary data from all components
        flow_summary = self.flow_manager.get_flow_summary()
        tl_performance = self.traffic_controller.get_performance_summary()
        
        # Generate comprehensive report
        final_metrics = self.analyzer.generate_comprehensive_report(
            flow_summary, tl_performance
        )
        
        # Additional insights
        print(f"\n🔍 SIMULATION INSIGHTS:")
        self._provide_insights(final_metrics, flow_summary, tl_performance)
        
        # Export data summary
        data_summary = self.analyzer.export_data_summary()
        print(f"\n📊 DATA COLLECTION SUMMARY:")
        print(f"   Waiting time samples: {data_summary['raw_data_counts']['waiting_time_samples']}")
        print(f"   Speed samples: {data_summary['raw_data_counts']['speed_samples']}")
        print(f"   Pressure measurements: {data_summary['raw_data_counts']['pressure_samples']}")
    
    def _provide_insights(self, metrics, flow_summary, tl_performance):
        """Provide actionable insights based on simulation results"""
        insights = []
        
        # Waiting time insights
        if metrics['waiting_times']:
            avg_waiting = metrics['waiting_times']['average']
            if avg_waiting > 60:
                insights.append("🔴 High average waiting time - consider optimizing signal timing")
            elif avg_waiting < 15:
                insights.append("🟢 Excellent waiting times - current system performing well")
        
        # Pressure balance insights
        if metrics['pressure_analysis']:
            ns_avg = metrics['pressure_analysis']['ns_avg']
            ew_avg = metrics['pressure_analysis']['ew_avg']
            if abs(ns_avg - ew_avg) > ns_avg * 0.3:
                insights.append("⚖️  Significant pressure imbalance - adjust base cycle times")
        
        # Flow efficiency insights
        if flow_summary:
            low_efficiency_flows = [
                f_id for f_id, data in flow_summary.items() 
                if data['efficiency'] < 30
            ]
            if low_efficiency_flows:
                insights.append(f"📉 Low efficiency flows detected: {', '.join(low_efficiency_flows)}")
        
        # Display insights
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        if not insights:
            print("   ✅ No major issues detected - system operating efficiently")

def main():
    """Main function to run the simulation"""
    try:
        # Create and run simulation
        simulation = TrafficSimulation()
        simulation.run_simulation()
        
    except EnvironmentError as e:
        print(f"❌ Environment Error: {e}")
        print("💡 Make sure SUMO is installed and SUMO_HOME is set correctly")
        
    except FileNotFoundError:
        print("❌ SUMO configuration files not found")
        print("💡 Make sure demo.sumocfg and related files are in the current directory")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
