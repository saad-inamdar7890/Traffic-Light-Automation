"""
Dynamic Traffic Flow Generator for K1 Network
Generates route files based on configurable scenarios
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple

class TrafficFlowGenerator:
    def __init__(self, network_name="k1"):
        self.network_name = network_name
        
        # Define entry points (where vehicles enter the network)
        self.entry_edges = ["-E15", "-E9", "-E10", "-E11", "-E12", "-E13", "-E24", "-E21", "-E6"]
        
        # Define exit points (where vehicles leave the network)
        self.exit_edges = ["E24", "E21", "E10", "E11", "E9", "E12", "E13", "E15", "E6"]
        
        # Define key routes (entry -> exit with via edges)
        self.key_routes = {
            # From Southwest (-E15)
            "sw_to_se": ("-E15", "E24", ["E4", "-E5", "E16"]),
            "sw_to_e": ("-E15", "E21", ["-E18", "E19", "E20"]),
            "sw_to_n": ("-E15", "E10", ["-E18", "-E17"]),
            "sw_to_nw": ("-E15", "E11", ["-E18", "E19", "-E0"]),
            "sw_to_center": ("-E15", "E12", ["-E18", "E19", "E20", "-E22"]),
            
            # From West (-E9)
            "w_to_se": ("-E9", "E24", ["E17", "E19", "E0.55", "E16"]),
            "w_to_e_north": ("-E9", "E13", ["E8", "E8.59"]),
            "w_to_nw": ("-E9", "E11", ["E8"]),
            "w_to_s": ("-E9", "E6", ["E17"]),
            "w_to_e": ("-E9", "E21", []),
            
            # From North (-E10, -E11)
            "n_to_w": ("-E10", "E9", []),
            "n_to_s": ("-E10", "E6", []),
            "n_to_e_north": ("-E10", "E13", []),
            "n_to_center": ("-E10", "E12", []),
            "n_to_se": ("-E10", "E24", []),
            "nw_to_e": ("-E11", "E21", []),
            
            # From East (-E12, -E13)
            "e_center_to_sw": ("-E12", "E15", []),
            "e_center_to_s": ("-E12", "E6", []),
            "e_north_to_w": ("-E13", "E9", []),
            "e_north_to_sw": ("-E13", "E15", []),
            
            # From Southeast (-E24)
            "se_to_center": ("-E24", "E12", ["-E23"]),
            "se_to_sw": ("-E24", "E15", []),
            
            # From East edge (-E21)
            "e_edge_to_sw": ("-E21", "E15", []),
            "e_edge_to_n": ("-E21", "E10", []),
            "e_edge_to_center": ("-E21", "E12", []),
            "e_edge_to_s": ("-E21", "E6", []),
            
            # From South (-E6)
            "s_to_nw": ("-E6", "E11", []),
            "s_to_e": ("-E6", "E21", []),
            "s_to_w": ("-E6", "E9", []),
        }
        
        # Define scenarios
        self.scenarios = self._define_scenarios()
        
    def _define_scenarios(self) -> Dict:
        """Define different traffic scenarios"""
        
        return {
            "uniform_light": {
                "name": "Uniform Light Traffic",
                "description": "Equal distribution, light traffic",
                "duration": 3600,
                "default_flow": 100,
                "flow_multipliers": {}  # All routes use default
            },
            
            "uniform_medium": {
                "name": "Uniform Medium Traffic",
                "description": "Equal distribution, medium traffic",
                "duration": 3600,
                "default_flow": 200,
                "flow_multipliers": {}
            },
            
            "uniform_heavy": {
                "name": "Uniform Heavy Traffic",
                "description": "Equal distribution, heavy traffic",
                "duration": 3600,
                "default_flow": 300,
                "flow_multipliers": {}
            },
            
            "morning_rush": {
                "name": "Morning Rush Hour",
                "description": "Heavy traffic from residential areas to commercial/industrial",
                "duration": 7200,  # 2 hours
                "default_flow": 100,
                "flow_multipliers": {
                    # High flow from Southwest (residential) to East/Center (commercial)
                    "sw_to_se": 3.0,
                    "sw_to_e": 3.0,
                    "sw_to_center": 2.5,
                    # High flow from North to Center/East
                    "n_to_e_north": 2.5,
                    "n_to_center": 2.0,
                    # Lower reverse flows
                    "e_center_to_sw": 0.5,
                    "e_north_to_sw": 0.5,
                }
            },
            
            "evening_rush": {
                "name": "Evening Rush Hour",
                "description": "Heavy traffic from commercial areas back to residential",
                "duration": 7200,  # 2 hours
                "default_flow": 100,
                "flow_multipliers": {
                    # High flow from East/Center to Southwest (back home)
                    "e_center_to_sw": 3.0,
                    "e_north_to_sw": 3.0,
                    "se_to_sw": 2.5,
                    # High flow from Center to North
                    "n_to_center": 0.5,  # Reverse direction is low
                    # Lower morning rush routes
                    "sw_to_se": 0.5,
                    "sw_to_e": 0.5,
                }
            },
            
            "custom_24h": {
                "name": "24-Hour Cycle",
                "description": "Full day with varying traffic patterns",
                "duration": 86400,  # 24 hours
                "time_periods": [
                    # Night (00:00-06:00) - Very light
                    {"start": 0, "end": 21600, "multiplier": 0.2},
                    # Morning Rush (06:00-09:00) - Heavy
                    {"start": 21600, "end": 32400, "multiplier": 3.0},
                    # Midday (09:00-17:00) - Medium
                    {"start": 32400, "end": 61200, "multiplier": 1.0},
                    # Evening Rush (17:00-19:00) - Heavy
                    {"start": 61200, "end": 68400, "multiplier": 3.0},
                    # Evening (19:00-22:00) - Light
                    {"start": 68400, "end": 79200, "multiplier": 0.6},
                    # Night (22:00-24:00) - Very light
                    {"start": 79200, "end": 86400, "multiplier": 0.2},
                ],
                "default_flow": 150,
                "flow_multipliers": {}
            }
        }
    
    def generate_routes_file(self, scenario_name: str, output_file: str = None):
        """Generate routes XML file for a given scenario"""
        
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(self.scenarios.keys())}")
        
        scenario = self.scenarios[scenario_name]
        
        # Create root element
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
        
        # Add comment
        comment_text = f"\n{scenario['name']}\n{scenario['description']}\nGenerated by dynamic_flow_generator.py\n"
        comment = ET.Comment(comment_text)
        root.append(comment)
        
        # Add vehicle type distribution (placeholder)
        vtype_dist = ET.SubElement(root, "vTypeDistribution")
        vtype_dist.set("id", "td_0")
        
        # Generate flows
        flow_id = 0
        
        if "time_periods" in scenario:
            # 24-hour scenario with time periods
            for period in scenario["time_periods"]:
                for route_name, (from_edge, to_edge, via_edges) in self.key_routes.items():
                    flow_rate = scenario["default_flow"] * period["multiplier"]
                    flow_rate = int(scenario.get("flow_multipliers", {}).get(route_name, 1.0) * flow_rate)
                    
                    if flow_rate > 0:
                        flow = ET.SubElement(root, "flow")
                        flow.set("id", f"f_{flow_id}")
                        flow.set("begin", f"{period['start']:.2f}")
                        flow.set("end", f"{period['end']:.2f}")
                        flow.set("from", from_edge)
                        flow.set("to", to_edge)
                        if via_edges:
                            flow.set("via", " ".join(via_edges))
                        flow.set("vehsPerHour", str(flow_rate))
                        flow_id += 1
        else:
            # Single period scenario
            duration = scenario["duration"]
            default_flow = scenario["default_flow"]
            
            for route_name, (from_edge, to_edge, via_edges) in self.key_routes.items():
                multiplier = scenario.get("flow_multipliers", {}).get(route_name, 1.0)
                flow_rate = int(default_flow * multiplier)
                
                if flow_rate > 0:
                    flow = ET.SubElement(root, "flow")
                    flow.set("id", f"f_{flow_id}")
                    flow.set("begin", "0.00")
                    flow.set("from", from_edge)
                    flow.set("to", to_edge)
                    if via_edges:
                        flow.set("via", " ".join(via_edges))
                    flow.set("end", f"{duration:.2f}")
                    flow.set("vehsPerHour", str(flow_rate))
                    flow_id += 1
        
        # Pretty print XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        
        # Remove extra blank lines
        xml_lines = [line for line in xml_str.split('\n') if line.strip()]
        xml_str = '\n'.join(xml_lines)
        
        # Write to file
        if output_file is None:
            output_file = f"{self.network_name}_routes_24h.rou.xml"
        
        with open(output_file, 'w', encoding='UTF-8') as f:
            f.write(xml_str)
        
        print(f"✓ Generated {output_file}")
        print(f"  Scenario: {scenario['name']}")
        print(f"  Total flows: {flow_id}")
        print(f"  Duration: {scenario['duration']} seconds")
        
        return output_file
    
    def list_scenarios(self):
        """Print all available scenarios"""
        print("\n=== Available Traffic Scenarios ===\n")
        for name, scenario in self.scenarios.items():
            print(f"{name}:")
            print(f"  Name: {scenario['name']}")
            print(f"  Description: {scenario['description']}")
            print(f"  Duration: {scenario['duration']} seconds ({scenario['duration']/3600:.1f} hours)")
            print(f"  Default flow: {scenario['default_flow']} veh/hour")
            if scenario.get('flow_multipliers'):
                print(f"  Custom routes: {len(scenario['flow_multipliers'])} modified flows")
            print()
    
    def add_custom_scenario(self, name: str, description: str, duration: int, 
                           default_flow: int, flow_multipliers: Dict[str, float] = None):
        """Add a custom scenario"""
        self.scenarios[name] = {
            "name": name.replace("_", " ").title(),
            "description": description,
            "duration": duration,
            "default_flow": default_flow,
            "flow_multipliers": flow_multipliers or {}
        }
        print(f"✓ Added custom scenario: {name}")


# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dynamic traffic flows for K1 network")
    parser.add_argument("--scenario", "-s", type=str, 
                       help="Scenario name (use --list to see available scenarios)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available scenarios")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file name (default: k1_routes_24h.rou.xml)")
    
    args = parser.parse_args()
    
    generator = TrafficFlowGenerator()
    
    if args.list:
        generator.list_scenarios()
    elif args.scenario:
        generator.generate_routes_file(args.scenario, args.output)
    else:
        print("Usage examples:")
        print("  python dynamic_flow_generator.py --list")
        print("  python dynamic_flow_generator.py --scenario uniform_medium")
        print("  python dynamic_flow_generator.py --scenario morning_rush -o morning_routes.rou.xml")
        print("\nQuick test:")
        generator.list_scenarios()
