"""
Dynamic Timing Test - Validate Fast Cycle Optimization
=====================================================

Tests the new dynamic timing features:
1. Dynamic minimum timing based on actual traffic needs
2. Fast-cycle optimization for low traffic scenarios
3. Cycle time reduction when all lanes are low traffic

Scenarios:
- Ultra Low Traffic: 1-2 vehicles per direction (should get 8-10s green)
- Light Traffic: 3-5 vehicles per direction (should get 10-12s green)  
- Empty Intersection: 0 vehicles (should get ultra-fast 8s cycles)
- Mixed: Some empty, some light (should optimize accordingly)
"""

import os
import sys
import traci
import time
import json
from datetime import datetime

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from edge_traffic_controller import EdgeTrafficController

class DynamicTimingTester:
    def __init__(self):
        self.results = {}
        self.test_scenarios = [
            {
                'name': 'Ultra Low Traffic',
                'description': '1-2 vehicles per direction - should get dynamic min 8-10s',
                'flows': {
                    'north': 60,   # 1 vehicle per minute
                    'south': 60,   
                    'east': 120,   # 2 vehicles per minute
                    'west': 60
                }
            },
            {
                'name': 'Light Traffic',
                'description': '3-5 vehicles per direction - should get 10-12s green',
                'flows': {
                    'north': 300,  # 5 vehicles per minute
                    'south': 180,  # 3 vehicles per minute
                    'east': 240,   # 4 vehicles per minute
                    'west': 300
                }
            },
            {
                'name': 'Empty Intersection',
                'description': '0 vehicles - should get ultra-fast 8s cycles',
                'flows': {
                    'north': 0,
                    'south': 0,
                    'east': 0,
                    'west': 0
                }
            },
            {
                'name': 'Mixed Low Traffic',
                'description': 'Some empty, some light - should optimize accordingly',
                'flows': {
                    'north': 0,    # Empty
                    'south': 60,   # Ultra low
                    'east': 300,   # Light
                    'west': 0      # Empty
                }
            }
        ]
        
    def generate_route_file(self, scenario, duration=1800):
        """Generate route file for specific traffic scenario"""
        flows = scenario['flows']
        
        route_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Vehicle Types -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89" />
    <vType id="motorcycle" accel="3.0" decel="5.0" sigma="0.3" length="2.5" maxSpeed="16.67" />
    
    <!-- Routes for all directions -->
    <route id="north_to_south" edges="E1 -E1" />
    <route id="south_to_north" edges="-E1 E1" />
    <route id="east_to_west" edges="E0 -E0" />
    <route id="west_to_east" edges="-E0 E0" />
'''
        
        # Add flows only if they have traffic
        if flows['north'] > 0:
            route_content += f'    <flow id="flow_north" route="north_to_south" begin="0" end="{duration}" vehsPerHour="{flows["north"]}" type="car" />\n'
        
        if flows['south'] > 0:
            route_content += f'    <flow id="flow_south" route="south_to_north" begin="0" end="{duration}" vehsPerHour="{flows["south"]}" type="car" />\n'
        
        if flows['east'] > 0:
            route_content += f'    <flow id="flow_east" route="east_to_west" begin="0" end="{duration}" vehsPerHour="{flows["east"]}" type="car" />\n'
        
        if flows['west'] > 0:
            route_content += f'    <flow id="flow_west" route="west_to_east" begin="0" end="{duration}" vehsPerHour="{flows["west"]}" type="car" />\n'
        
        route_content += '</routes>\n'
        
        filename = f"dynamic_timing_{scenario['name'].lower().replace(' ', '_')}.rou.xml"
        filepath = os.path.join(current_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(route_content)
        
        return filename
    
    def generate_config_file(self, route_file, scenario_name):
        """Generate SUMO configuration file"""
        config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="../demo.net.xml"/>
        <route-files value="{route_file}"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="1800"/>
        <step-length value="1"/>
    </time>
    
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>
    
    <traci_server>
        <remote-port value="8813"/>
    </traci_server>
</configuration>'''
        
        filename = f"dynamic_config_{scenario_name.lower().replace(' ', '_')}.sumocfg"
        filepath = os.path.join(current_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(config_content)
        
        return filename
    
    def run_scenario_test(self, scenario):
        """Run a single scenario test and collect dynamic timing data"""
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {scenario['name']}")
        print(f"📝 {scenario['description']}")
        print(f"🚗 Traffic Flows: {scenario['flows']}")
        print(f"{'='*60}")
        
        # Generate files
        route_file = self.generate_route_file(scenario)
        config_file = self.generate_config_file(route_file, scenario['name'])
        
        # Start SUMO
        sumo_cmd = [
            "sumo", "-c", config_file,
            "--step-length", "1",
            "--no-warnings", "true"
        ]
        
        try:
            traci.start(sumo_cmd)
            controller = EdgeTrafficController()
            
            # Test metrics
            timing_data = []
            cycle_times = []
            adaptations = 0
            fast_cycles = 0
            
            print(f"\n⏱️  Starting 30-minute simulation...")
            
            for step in range(0, 1800):  # 30 minutes
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                
                # Apply edge algorithm every 15 seconds
                if step % 15 == 0:
                    result = controller.apply_edge_algorithm(current_time)
                    
                    if result:
                        adaptations += 1
                        timing_plan = result['timing_plan']
                        
                        # Record timing data
                        timing_record = {
                            'time': current_time,
                            'ns_green': timing_plan['north_south_green'],
                            'ew_green': timing_plan['east_west_green'],
                            'cycle_time': timing_plan['cycle_time'],
                            'scenario_category': timing_plan.get('scenario_category'),
                            'avg_weight': timing_plan.get('avg_weight'),
                            'fast_cycle_applied': timing_plan.get('fast_cycle_applied', False),
                            'fast_cycle_type': timing_plan.get('fast_cycle_type'),
                            'cycle_time_reduction': timing_plan.get('cycle_time_reduction', 0),
                            'ns_min_time': timing_plan.get('ns_min_time'),
                            'ew_min_time': timing_plan.get('ew_min_time')
                        }
                        
                        timing_data.append(timing_record)
                        cycle_times.append(timing_plan['cycle_time'])
                        
                        if timing_plan.get('fast_cycle_applied', False):
                            fast_cycles += 1
                
                # Progress updates
                if step % 300 == 0 and step > 0:
                    mins = step // 60
                    print(f"   ⏱️  {mins} minutes completed | Adaptations: {adaptations} | Fast Cycles: {fast_cycles}")
            
            # Calculate results
            avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
            min_cycle_time = min(cycle_times) if cycle_times else 0
            max_cycle_time = max(cycle_times) if cycle_times else 0
            
            fast_cycle_percentage = (fast_cycles / adaptations * 100) if adaptations > 0 else 0
            
            scenario_results = {
                'scenario': scenario,
                'total_adaptations': adaptations,
                'fast_cycles_applied': fast_cycles,
                'fast_cycle_percentage': fast_cycle_percentage,
                'avg_cycle_time': avg_cycle_time,
                'min_cycle_time': min_cycle_time,
                'max_cycle_time': max_cycle_time,
                'timing_data': timing_data
            }
            
            print(f"\n📊 RESULTS FOR {scenario['name']}:")
            print(f"   Total Adaptations: {adaptations}")
            print(f"   Fast Cycles Applied: {fast_cycles} ({fast_cycle_percentage:.1f}%)")
            print(f"   Average Cycle Time: {avg_cycle_time:.1f}s")
            print(f"   Cycle Time Range: {min_cycle_time:.1f}s - {max_cycle_time:.1f}s")
            
            return scenario_results
            
        except Exception as e:
            print(f"❌ Error in scenario {scenario['name']}: {e}")
            return None
        finally:
            try:
                traci.close()
            except:
                pass
    
    def run_all_tests(self):
        """Run all dynamic timing test scenarios"""
        print("🚀 DYNAMIC TIMING OPTIMIZATION TEST SUITE")
        print("==========================================")
        print("Testing new features:")
        print("✅ Dynamic minimum timing based on actual traffic")
        print("✅ Fast-cycle optimization for low traffic")
        print("✅ Cycle time reduction for better flow")
        
        all_results = []
        
        for scenario in self.test_scenarios:
            result = self.run_scenario_test(scenario)
            if result:
                all_results.append(result)
                self.results[scenario['name']] = result
        
        # Generate comprehensive report
        self.generate_comprehensive_report(all_results)
        
        print("\n🎉 DYNAMIC TIMING TESTS COMPLETED!")
        print("📁 Check dynamic_timing_results/ for detailed analysis")
    
    def generate_comprehensive_report(self, results):
        """Generate detailed analysis report"""
        os.makedirs('dynamic_timing_results', exist_ok=True)
        
        report_content = f"""DYNAMIC TIMING OPTIMIZATION TEST REPORT
{'='*50}

Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Test Duration: 30 minutes per scenario
Algorithm: Dynamic minimum timing + Fast-cycle optimization

SCENARIO ANALYSIS:
{'-'*50}

"""
        
        for result in results:
            scenario = result['scenario']
            report_content += f"""
Scenario: {scenario['name']}
Description: {scenario['description']}
Traffic Flows: {scenario['flows']}

Results:
  - Total Adaptations: {result['total_adaptations']}
  - Fast Cycles Applied: {result['fast_cycles_applied']} ({result['fast_cycle_percentage']:.1f}%)
  - Average Cycle Time: {result['avg_cycle_time']:.1f}s
  - Cycle Time Range: {result['min_cycle_time']:.1f}s - {result['max_cycle_time']:.1f}s

Performance:
"""
            
            # Analyze performance based on expectations
            if scenario['name'] == 'Ultra Low Traffic':
                expected_cycle = "20-30s"
                if result['avg_cycle_time'] < 30:
                    report_content += "  ✅ EXCELLENT: Achieved fast cycles for ultra-low traffic\n"
                else:
                    report_content += "  ⚠️  NEEDS IMPROVEMENT: Cycles still too long for ultra-low traffic\n"
                    
            elif scenario['name'] == 'Empty Intersection':
                expected_cycle = "16-20s"  
                if result['avg_cycle_time'] < 20:
                    report_content += "  ✅ EXCELLENT: Ultra-fast cycles for empty intersection\n"
                else:
                    report_content += "  ⚠️  NEEDS IMPROVEMENT: Should achieve ultra-fast cycles when empty\n"
            
            report_content += f"  - Expected cycle time: {expected_cycle if 'expected_cycle' in locals() else 'Variable'}\n"
            report_content += f"  - Fast cycle optimization rate: {result['fast_cycle_percentage']:.1f}%\n\n"
        
        # Overall analysis
        avg_fast_cycle_rate = sum(r['fast_cycle_percentage'] for r in results) / len(results)
        
        report_content += f"""
OVERALL PERFORMANCE SUMMARY:
{'-'*50}
Average Fast Cycle Rate: {avg_fast_cycle_rate:.1f}%
Dynamic Timing Status: {'✅ WORKING' if avg_fast_cycle_rate > 50 else '⚠️ NEEDS TUNING'}

RECOMMENDATIONS:
{'-'*50}
"""
        
        if avg_fast_cycle_rate > 70:
            report_content += "✅ Dynamic timing is working excellently!\n"
            report_content += "✅ Fast-cycle optimization is being applied appropriately.\n"
            report_content += "✅ Algorithm successfully reduces waste in low-traffic scenarios.\n"
        elif avg_fast_cycle_rate > 50:
            report_content += "✅ Dynamic timing is working well.\n"
            report_content += "🔧 Consider more aggressive optimization for very low traffic.\n"
        else:
            report_content += "⚠️  Dynamic timing needs adjustment.\n"
            report_content += "🔧 Review traffic categorization thresholds.\n"
            report_content += "🔧 Consider more aggressive fast-cycle optimization.\n"
        
        # Save report
        with open('dynamic_timing_results/test_report.txt', 'w') as f:
            f.write(report_content)
        
        # Save detailed data
        with open('dynamic_timing_results/detailed_data.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📊 Comprehensive report saved to dynamic_timing_results/")

if __name__ == "__main__":
    tester = DynamicTimingTester()
    tester.run_all_tests()