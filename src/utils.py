"""
Traffic Utilities Module
========================
Helper functions and utilities for traffic simulation management,
SUMO configuration, route file generation, and common operations.

Features:
- Dynamic route file generation for different traffic scenarios
- SUMO configuration management
- File I/O operations and path management
- Common utility functions for simulation setup
"""

import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import subprocess
import json


class RouteGenerator:
    """
    Dynamic route file generator for creating custom traffic patterns
    for different test scenarios.
    """
    
    def __init__(self):
        """Initialize the route generator."""
        self.vehicle_types = {
            'car': {'id': 'car', 'vClass': 'passenger', 'length': '4.5', 'maxSpeed': '15.0'},
            'truck': {'id': 'truck', 'vClass': 'truck', 'length': '8.0', 'maxSpeed': '12.0'}
        }
        
        self.default_routes = {
            'ns_straight': {'edges': 'E1 -E0.254', 'color': 'blue'},
            'sn_straight': {'edges': '-E1 E0.319', 'color': 'green'},
            'ew_straight': {'edges': 'E0 -E1.238', 'color': 'red'},
            'we_straight': {'edges': '-E0 E1.200', 'color': 'yellow'}
        }
    
    def generate_route_file(self, output_path: str, scenario_config: Dict[str, Any]) -> bool:
        """
        Generate a SUMO route file based on scenario configuration.
        
        Args:
            output_path: Path where the route file will be saved
            scenario_config: Dictionary containing traffic flow configuration
            
        Returns:
            True if route file generated successfully
        """
        try:
            # Create root element
            routes = ET.Element('routes')
            
            # Add vehicle types
            for vtype_id, vtype_attrs in self.vehicle_types.items():
                vtype = ET.SubElement(routes, 'vType')
                for attr, value in vtype_attrs.items():
                    vtype.set(attr, value)
            
            # Add route definitions
            for route_id, route_info in self.default_routes.items():
                route = ET.SubElement(routes, 'route')
                route.set('id', route_id)
                route.set('edges', route_info['edges'])
                route.set('color', route_info['color'])
            
            # Add traffic flows based on scenario
            flow_configs = scenario_config.get('flows', {})
            
            for flow_name, flow_config in flow_configs.items():
                flow = ET.SubElement(routes, 'flow')
                
                # Set flow attributes
                flow.set('id', f'flow_{flow_name}')
                flow.set('route', flow_config.get('route', 'ns_straight'))
                flow.set('begin', str(flow_config.get('begin', 0)))
                flow.set('end', str(flow_config.get('end', 3600)))
                flow.set('vehsPerHour', str(flow_config.get('vehicles_per_hour', 360)))
                flow.set('type', flow_config.get('vehicle_type', 'car'))
                
                # Optional attributes
                if 'departLane' in flow_config:
                    flow.set('departLane', flow_config['departLane'])
                if 'departSpeed' in flow_config:
                    flow.set('departSpeed', flow_config['departSpeed'])
            
            # Write to file
            tree = ET.ElementTree(routes)
            ET.indent(tree, space="  ", level=0)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            print(f"✅ Route file generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error generating route file: {e}")
            return False
    
    def create_scenario_flows(self, scenario_name: str) -> Dict[str, Any]:
        """
        Create predefined traffic flow configurations for different scenarios.
        
        Args:
            scenario_name: Name of the scenario to create
            
        Returns:
            Flow configuration dictionary
        """
        scenarios = {
            'balanced': {
                'description': 'Balanced traffic in all directions',
                'flows': {
                    'ns_flow': {
                        'route': 'ns_straight',
                        'vehicles_per_hour': 400,
                        'begin': 0,
                        'end': 3600
                    },
                    'sn_flow': {
                        'route': 'sn_straight',
                        'vehicles_per_hour': 400,
                        'begin': 0,
                        'end': 3600
                    },
                    'ew_flow': {
                        'route': 'ew_straight',
                        'vehicles_per_hour': 400,
                        'begin': 0,
                        'end': 3600
                    },
                    'we_flow': {
                        'route': 'we_straight',
                        'vehicles_per_hour': 400,
                        'begin': 0,
                        'end': 3600
                    }
                }
            },
            
            'heavy_ns': {
                'description': 'Heavy North-South traffic, light East-West',
                'flows': {
                    'ns_flow': {
                        'route': 'ns_straight',
                        'vehicles_per_hour': 800,
                        'begin': 0,
                        'end': 3600
                    },
                    'sn_flow': {
                        'route': 'sn_straight',
                        'vehicles_per_hour': 800,
                        'begin': 0,
                        'end': 3600
                    },
                    'ew_flow': {
                        'route': 'ew_straight',
                        'vehicles_per_hour': 200,
                        'begin': 0,
                        'end': 3600
                    },
                    'we_flow': {
                        'route': 'we_straight',
                        'vehicles_per_hour': 200,
                        'begin': 0,
                        'end': 3600
                    }
                }
            },
            
            'heavy_ew': {
                'description': 'Heavy East-West traffic, light North-South',
                'flows': {
                    'ns_flow': {
                        'route': 'ns_straight',
                        'vehicles_per_hour': 200,
                        'begin': 0,
                        'end': 3600
                    },
                    'sn_flow': {
                        'route': 'sn_straight',
                        'vehicles_per_hour': 200,
                        'begin': 0,
                        'end': 3600
                    },
                    'ew_flow': {
                        'route': 'ew_straight',
                        'vehicles_per_hour': 800,
                        'begin': 0,
                        'end': 3600
                    },
                    'we_flow': {
                        'route': 'we_straight',
                        'vehicles_per_hour': 800,
                        'begin': 0,
                        'end': 3600
                    }
                }
            },
            
            'rush_hour': {
                'description': 'Rush hour simulation with varying flows',
                'flows': {
                    'ns_flow': {
                        'route': 'ns_straight',
                        'vehicles_per_hour': 600,
                        'begin': 0,
                        'end': 3600
                    },
                    'sn_flow': {
                        'route': 'sn_straight',
                        'vehicles_per_hour': 600,
                        'begin': 0,
                        'end': 3600
                    },
                    'ew_flow': {
                        'route': 'ew_straight',
                        'vehicles_per_hour': 600,
                        'begin': 0,
                        'end': 3600
                    },
                    'we_flow': {
                        'route': 'we_straight',
                        'vehicles_per_hour': 600,
                        'begin': 0,
                        'end': 3600
                    }
                }
            }
        }
        
        return scenarios.get(scenario_name, scenarios['balanced'])


class SUMOConfigManager:
    """
    SUMO configuration file manager for creating and modifying
    simulation configuration files.
    """
    
    def __init__(self):
        """Initialize the SUMO config manager."""
        self.default_config = {
            'net-file': 'demo.net.xml',
            'route-files': 'demo.rou.xml',
            'begin': '0',
            'end': '3600',
            'step-length': '1',
            'default.speeddev': '0.1',
            'pedestrian.model': 'none'
        }
    
    def create_config_file(self, output_path: str, config_params: Optional[Dict[str, str]] = None) -> bool:
        """
        Create a SUMO configuration file.
        
        Args:
            output_path: Path where the config file will be saved
            config_params: Custom configuration parameters
            
        Returns:
            True if config file created successfully
        """
        try:
            # Merge default config with custom parameters
            final_config = self.default_config.copy()
            if config_params:
                final_config.update(config_params)
            
            # Create XML structure
            configuration = ET.Element('configuration')
            
            # Input section
            input_section = ET.SubElement(configuration, 'input')
            for param in ['net-file', 'route-files']:
                if param in final_config:
                    elem = ET.SubElement(input_section, param)
                    elem.set('value', final_config[param])
            
            # Time section
            time_section = ET.SubElement(configuration, 'time')
            for param in ['begin', 'end', 'step-length']:
                if param in final_config:
                    elem = ET.SubElement(time_section, param)
                    elem.set('value', final_config[param])
            
            # Processing section
            processing_section = ET.SubElement(configuration, 'processing')
            for param in ['default.speeddev', 'pedestrian.model']:
                if param in final_config:
                    elem = ET.SubElement(processing_section, param)
                    elem.set('value', final_config[param])
            
            # Write to file
            tree = ET.ElementTree(configuration)
            ET.indent(tree, space="  ", level=0)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            print(f"✅ SUMO config file created: {output_path}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error creating SUMO config file: {e}")
            return False


class FileManager:
    """
    File and directory management utilities for the traffic simulation system.
    """
    
    @staticmethod
    def ensure_directory(path: str) -> bool:
        """
        Ensure that a directory exists, creating it if necessary.
        
        Args:
            path: Directory path to create
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print(f"⚠️  Error creating directory {path}: {e}")
            return False
    
    @staticmethod
    def get_temp_filename(prefix: str = "temp", suffix: str = ".xml") -> str:
        """
        Generate a temporary filename.
        
        Args:
            prefix: Filename prefix
            suffix: Filename suffix/extension
            
        Returns:
            Temporary filename
        """
        return tempfile.mktemp(prefix=prefix, suffix=suffix)
    
    @staticmethod
    def cleanup_temp_files(file_list: List[str]) -> None:
        """
        Clean up temporary files.
        
        Args:
            file_list: List of temporary files to remove
        """
        for file_path in file_list:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️  Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"⚠️  Error cleaning up {file_path}: {e}")
    
    @staticmethod
    def save_results_json(data: Dict[str, Any], output_path: str) -> bool:
        """
        Save results data to a JSON file.
        
        Args:
            data: Data to save
            output_path: Output file path
            
        Returns:
            True if saved successfully
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"💾 Results saved to {output_path}")
            return True
        except Exception as e:
            print(f"⚠️  Error saving results: {e}")
            return False
    
    @staticmethod
    def load_results_json(input_path: str) -> Optional[Dict[str, Any]]:
        """
        Load results data from a JSON file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Loaded data or None if error
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            print(f"📂 Results loaded from {input_path}")
            return data
        except Exception as e:
            print(f"⚠️  Error loading results: {e}")
            return None


class SimulationUtils:
    """
    Utility functions for SUMO simulation management and control.
    """
    
    @staticmethod
    def check_sumo_installation() -> bool:
        """
        Check if SUMO is properly installed and accessible.
        
        Returns:
            True if SUMO is available
        """
        try:
            result = subprocess.run(['sumo', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ SUMO installation verified")
                return True
            else:
                print("⚠️  SUMO not found or not working properly")
                return False
        except Exception as e:
            print(f"⚠️  Error checking SUMO installation: {e}")
            return False
    
    @staticmethod
    def get_simulation_summary(stats: Dict[str, Any]) -> str:
        """
        Generate a formatted summary of simulation statistics.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Formatted summary string
        """
        summary = f"""
🚦 SIMULATION SUMMARY
{'='*40}
Duration: {stats.get('duration', 0):.0f} seconds
Total Vehicles: {stats.get('total_vehicles', 0)}
Avg Waiting Time: {stats.get('avg_waiting_time', 0):.2f}s
Avg Speed: {stats.get('avg_speed', 0):.2f} m/s
Throughput: {stats.get('throughput', 0):.0f} vehicles/hour
Efficiency Score: {stats.get('efficiency_score', 0):.1f}%
Algorithm: {stats.get('algorithm', 'Unknown')}
Scenario: {stats.get('scenario', 'Unknown')}
"""
        return summary
    
    @staticmethod
    def calculate_improvement_percentage(adaptive_value: float, normal_value: float) -> float:
        """
        Calculate percentage improvement of adaptive over normal algorithm.
        
        Args:
            adaptive_value: Performance value for adaptive algorithm
            normal_value: Performance value for normal algorithm
            
        Returns:
            Improvement percentage (positive = better, negative = worse)
        """
        if normal_value == 0:
            return 0.0
        
        # For waiting time, lower is better
        improvement = ((normal_value - adaptive_value) / normal_value) * 100
        return improvement
    
    @staticmethod
    def generate_random_seed() -> int:
        """
        Generate a random seed for reproducible simulations.
        
        Returns:
            Random seed value
        """
        return random.randint(1, 1000000)
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format time in seconds to a human-readable format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def validate_network_file(net_file_path: str) -> bool:
        """
        Validate that a SUMO network file exists and is readable.
        
        Args:
            net_file_path: Path to the network file
            
        Returns:
            True if file is valid
        """
        try:
            if not os.path.exists(net_file_path):
                print(f"⚠️  Network file not found: {net_file_path}")
                return False
            
            # Try to parse the XML
            tree = ET.parse(net_file_path)
            root = tree.getroot()
            
            if root.tag != 'net':
                print(f"⚠️  Invalid network file format: {net_file_path}")
                return False
            
            print(f"✅ Network file validated: {net_file_path}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error validating network file: {e}")
            return False


# Convenience functions for easy access
def create_scenario_route_file(scenario_name: str, output_path: str) -> bool:
    """
    Convenience function to create a route file for a specific scenario.
    
    Args:
        scenario_name: Name of the scenario
        output_path: Output file path
        
    Returns:
        True if successful
    """
    generator = RouteGenerator()
    scenario_config = generator.create_scenario_flows(scenario_name)
    return generator.generate_route_file(output_path, scenario_config)


def create_sumo_config(output_path: str, net_file: str, route_file: str, 
                      duration: int = 3600) -> bool:
    """
    Convenience function to create a SUMO configuration file.
    
    Args:
        output_path: Output config file path
        net_file: Network file path
        route_file: Route file path
        duration: Simulation duration in seconds
        
    Returns:
        True if successful
    """
    manager = SUMOConfigManager()
    config_params = {
        'net-file': net_file,
        'route-files': route_file,
        'end': str(duration)
    }
    return manager.create_config_file(output_path, config_params)