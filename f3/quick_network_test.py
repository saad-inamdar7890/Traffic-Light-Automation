#!/usr/bin/env python3
"""
Quick Test Script for Complex Network
=====================================

This script performs basic validation of the complex network setup
and runs a quick simulation to verify everything is working.
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET

def validate_network_files():
    """Validate that all required network files exist and are properly formatted."""
    
    print("🔍 Validating Complex Network Files...")
    
    required_files = [
        "complex_network.net.xml",
        "complex_routes.rou.xml", 
        "complex_simulation.sumocfg",
        "complex_gui_settings.xml"
    ]
    
    validation_results = {}
    
    for file in required_files:
        file_path = os.path.join(os.getcwd(), file)
        if os.path.exists(file_path):
            try:
                # Try to parse XML files
                if file.endswith('.xml'):
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    validation_results[file] = f"✅ Valid XML ({root.tag})"
                else:
                    validation_results[file] = "✅ File exists"
            except ET.ParseError as e:
                validation_results[file] = f"❌ XML Parse Error: {e}"
            except Exception as e:
                validation_results[file] = f"⚠️ Error: {e}"
        else:
            validation_results[file] = "❌ File not found"
    
    print("\\n📋 Validation Results:")
    for file, result in validation_results.items():
        print(f"   {file}: {result}")
    
    return all("✅" in result for result in validation_results.values())

def analyze_network_structure():
    """Analyze the network structure and provide summary."""
    
    print("\\n🏗️ Analyzing Network Structure...")
    
    try:
        # Parse network file
        tree = ET.parse("complex_network.net.xml")
        root = tree.getroot()
        
        # Count elements
        junctions = root.findall('junction')
        edges = root.findall('edge')
        tl_logic = root.findall('tlLogic')
        
        # Analyze junctions
        junction_types = {}
        traffic_light_junctions = []
        
        for junction in junctions:
            j_type = junction.get('type', 'unknown')
            if j_type not in junction_types:
                junction_types[j_type] = 0
            junction_types[j_type] += 1
            
            if j_type == 'traffic_light':
                traffic_light_junctions.append(junction.get('id'))
        
        # Analyze edges
        edge_priorities = {}
        total_lanes = 0
        
        for edge in edges:
            priority = int(edge.get('priority', 0))
            lanes = int(edge.get('numLanes', 1))
            total_lanes += lanes
            
            if priority not in edge_priorities:
                edge_priorities[priority] = 0
            edge_priorities[priority] += 1
        
        print(f"\\n📊 Network Summary:")
        print(f"   Total Junctions: {len(junctions)}")
        print(f"   Total Edges: {len(edges)}")
        print(f"   Total Lanes: {total_lanes}")
        print(f"   Traffic Light Programs: {len(tl_logic)}")
        
        print(f"\\n🚦 Junction Types:")
        for j_type, count in junction_types.items():
            print(f"   {j_type}: {count}")
        
        print(f"\\n🛣️ Road Priorities:")
        for priority, count in sorted(edge_priorities.items(), reverse=True):
            print(f"   Priority {priority}: {count} roads")
        
        print(f"\\n🚥 Traffic Light Intersections:")
        for tl_id in traffic_light_junctions:
            print(f"   • {tl_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing network: {e}")
        return False

def analyze_traffic_patterns():
    """Analyze traffic route patterns."""
    
    print("\\n🚗 Analyzing Traffic Patterns...")
    
    try:
        tree = ET.parse("complex_routes.rou.xml")
        root = tree.getroot()
        
        # Count vehicle types
        vtypes = root.findall('vType')
        vehicle_types = {vt.get('id'): vt.get('maxSpeed', 'unknown') for vt in vtypes}
        
        # Count routes
        routes = root.findall('route')
        route_count = len(routes)
        
        # Count flows
        flows = root.findall('flow')
        flow_count = len(flows)
        
        # Analyze flows by vehicle type
        flow_by_type = {}
        total_vehicles_per_hour = 0
        
        for flow in flows:
            vtype = flow.get('type', 'unknown')
            veh_per_hour = int(flow.get('vehsPerHour', 0))
            
            if vtype not in flow_by_type:
                flow_by_type[vtype] = 0
            flow_by_type[vtype] += veh_per_hour
            total_vehicles_per_hour += veh_per_hour
        
        print(f"\\n📈 Traffic Pattern Summary:")
        print(f"   Vehicle Types Defined: {len(vehicle_types)}")
        print(f"   Routes Defined: {route_count}")
        print(f"   Traffic Flows: {flow_count}")
        print(f"   Total Vehicles/Hour: {total_vehicles_per_hour}")
        
        print(f"\\n🚙 Vehicle Types:")
        for vtype, max_speed in vehicle_types.items():
            print(f"   {vtype}: Max Speed {max_speed}")
        
        print(f"\\n📊 Traffic Volume by Vehicle Type:")
        for vtype, volume in flow_by_type.items():
            percentage = (volume / total_vehicles_per_hour) * 100 if total_vehicles_per_hour > 0 else 0
            print(f"   {vtype}: {volume} veh/h ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing traffic patterns: {e}")
        return False

def run_quick_test():
    """Run a quick test simulation."""
    
    print("\\n🧪 Running Quick Test Simulation...")
    
    try:
        # Create a quick test configuration
        test_config = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="complex_network.net.xml"/>
        <route-files value="complex_routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="300"/>
        <step-length value="1"/>
    </time>
    <processing>
        <max-depart-delay value="60"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <duration-log.disable value="true"/>
    </report>
</configuration>"""
        
        # Write test config
        with open("quick_test.sumocfg", "w") as f:
            f.write(test_config)
        
        # Run quick test
        cmd = ["sumo", "-c", "quick_test.sumocfg", "--no-warnings"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ Quick test simulation completed successfully!")
            print(f"   Simulation ran for 5 minutes of virtual time")
            
            # Clean up
            if os.path.exists("quick_test.sumocfg"):
                os.remove("quick_test.sumocfg")
            
            return True
        else:
            print(f"   ❌ Test simulation failed:")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Test simulation timed out (may indicate issues)")
        return False
    except Exception as e:
        print(f"   ❌ Error running test: {e}")
        return False

def main():
    """Main test function."""
    
    print("🌆 COMPLEX NETWORK VALIDATION & TESTING")
    print("=" * 50)
    
    # Change to f3 directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Run all validation steps
    all_good = True
    
    # 1. Validate files
    if not validate_network_files():
        all_good = False
        print("\\n❌ File validation failed!")
    
    # 2. Analyze network structure
    if not analyze_network_structure():
        all_good = False
        print("\\n❌ Network structure analysis failed!")
    
    # 3. Analyze traffic patterns
    if not analyze_traffic_patterns():
        all_good = False
        print("\\n❌ Traffic pattern analysis failed!")
    
    # 4. Run quick test (only if previous steps passed)
    if all_good:
        if not run_quick_test():
            all_good = False
            print("\\n❌ Quick test simulation failed!")
    
    # Final results
    print("\\n" + "=" * 50)
    if all_good:
        print("🎉 ALL TESTS PASSED!")
        print("\\n✅ Your complex network is ready to use!")
        print("\\n💡 Next steps:")
        print("   1. Launch SUMO GUI: sumo-gui -c complex_simulation.sumocfg")
        print("   2. Run full analysis: python complex_network_analyzer.py")
        print("   3. Experiment with different scenarios")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\\n🔧 Please check the error messages above and fix issues before proceeding.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()