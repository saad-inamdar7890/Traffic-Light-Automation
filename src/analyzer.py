"""
Traffic Analyzer Module
======================
Comprehensive traffic metrics collection and analysis functionality
for monitoring and evaluating traffic light performance.

Features:
- Real-time traffic data collection from SUMO
- Performance metrics calculation
- Statistical analysis and reporting
- Data validation and cleaning
"""

import traci
import statistics
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


class TrafficAnalyzer:
    """
    Comprehensive traffic data analyzer for collecting and processing
    traffic metrics from SUMO simulations.
    """
    
    def __init__(self):
        """Initialize the traffic analyzer."""
        self.data_history = []
        self.collection_interval = 1  # Collect data every second
        self.last_collection_time = 0
        
    def collect_traffic_metrics(self, current_time: int, traci_connection) -> Optional[Dict[str, Any]]:
        """
        Collect comprehensive traffic metrics from the simulation.
        
        Args:
            current_time: Current simulation time
            traci_connection: TraCI connection object
            
        Returns:
            Dictionary containing all traffic metrics or None if collection fails
        """
        try:
            # Get all vehicles in the simulation
            vehicle_ids = traci_connection.vehicle.getIDList()
            
            if not vehicle_ids:
                return {
                    'timestamp': current_time,
                    'total_vehicles': 0,
                    'avg_waiting_time': 0.0,
                    'avg_speed': 0.0,
                    'total_waiting_time': 0.0,
                    'edge_data': {},
                    'junction_data': {},
                    'traffic_light_data': {}
                }
            
            # Collect vehicle-level data
            vehicle_data = {}
            total_waiting_time = 0.0
            total_speed = 0.0
            
            for veh_id in vehicle_ids:
                try:
                    waiting_time = traci_connection.vehicle.getWaitingTime(veh_id)
                    speed = traci_connection.vehicle.getSpeed(veh_id)
                    edge_id = traci_connection.vehicle.getRoadID(veh_id)
                    position = traci_connection.vehicle.getPosition(veh_id)
                    lane_id = traci_connection.vehicle.getLaneID(veh_id)
                    
                    vehicle_data[veh_id] = {
                        'waiting_time': waiting_time,
                        'speed': speed,
                        'edge_id': edge_id,
                        'position': position,
                        'lane_id': lane_id
                    }
                    
                    total_waiting_time += waiting_time
                    total_speed += speed
                    
                except Exception as e:
                    # Skip vehicles that may have been removed
                    continue
            
            # Calculate aggregate metrics
            num_vehicles = len(vehicle_data)
            avg_waiting_time = total_waiting_time / num_vehicles if num_vehicles > 0 else 0.0
            avg_speed = total_speed / num_vehicles if num_vehicles > 0 else 0.0
            
            # Collect edge-level data
            edge_data = self._collect_edge_data(traci_connection, vehicle_data)
            
            # Collect junction data
            junction_data = self._collect_junction_data(traci_connection)
            
            # Collect traffic light data
            traffic_light_data = self._collect_traffic_light_data(traci_connection)
            
            metrics = {
                'timestamp': current_time,
                'total_vehicles': num_vehicles,
                'avg_waiting_time': avg_waiting_time,
                'avg_speed': avg_speed,
                'total_waiting_time': total_waiting_time,
                'vehicle_data': vehicle_data,
                'edge_data': edge_data,
                'junction_data': junction_data,
                'traffic_light_data': traffic_light_data
            }
            
            # Store in history
            self.data_history.append(metrics)
            self.last_collection_time = current_time
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error collecting traffic metrics: {e}")
            return None
    
    def _collect_edge_data(self, traci_connection, vehicle_data: Dict) -> Dict[str, Any]:
        """
        Collect edge-level traffic data.
        
        Args:
            traci_connection: TraCI connection
            vehicle_data: Vehicle data dictionary
            
        Returns:
            Edge data dictionary
        """
        edge_data = {}
        
        try:
            # Get all edges
            edge_ids = traci_connection.edge.getIDList()
            
            for edge_id in edge_ids:
                try:
                    # Get edge metrics
                    vehicle_count = traci_connection.edge.getLastStepVehicleNumber(edge_id)
                    mean_speed = traci_connection.edge.getLastStepMeanSpeed(edge_id)
                    occupancy = traci_connection.edge.getLastStepOccupancy(edge_id)
                    waiting_time = traci_connection.edge.getWaitingTime(edge_id)
                    
                    # Get vehicles on this edge
                    edge_vehicles = []
                    for veh_id, veh_data in vehicle_data.items():
                        if veh_data['edge_id'] == edge_id:
                            edge_vehicles.append(veh_data)
                    
                    edge_data[edge_id] = {
                        'vehicle_count': vehicle_count,
                        'mean_speed': mean_speed,
                        'occupancy': occupancy,
                        'waiting_time': waiting_time,
                        'vehicles': edge_vehicles,
                        'avg_waiting_time': statistics.mean([v['waiting_time'] for v in edge_vehicles]) if edge_vehicles else 0.0
                    }
                    
                except Exception as e:
                    # Skip problematic edges
                    continue
                    
        except Exception as e:
            print(f"⚠️  Error collecting edge data: {e}")
        
        return edge_data
    
    def _collect_junction_data(self, traci_connection) -> Dict[str, Any]:
        """
        Collect junction-level traffic data.
        
        Args:
            traci_connection: TraCI connection
            
        Returns:
            Junction data dictionary
        """
        junction_data = {}
        
        try:
            junction_ids = traci_connection.junction.getIDList()
            
            for junction_id in junction_ids:
                try:
                    position = traci_connection.junction.getPosition(junction_id)
                    shape = traci_connection.junction.getShape(junction_id)
                    
                    junction_data[junction_id] = {
                        'position': position,
                        'shape': shape
                    }
                    
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"⚠️  Error collecting junction data: {e}")
        
        return junction_data
    
    def _collect_traffic_light_data(self, traci_connection) -> Dict[str, Any]:
        """
        Collect traffic light status and timing data.
        
        Args:
            traci_connection: TraCI connection
            
        Returns:
            Traffic light data dictionary
        """
        traffic_light_data = {}
        
        try:
            tl_ids = traci_connection.trafficlight.getIDList()
            
            for tl_id in tl_ids:
                try:
                    current_phase = traci_connection.trafficlight.getPhase(tl_id)
                    phase_duration = traci_connection.trafficlight.getPhaseDuration(tl_id)
                    next_switch = traci_connection.trafficlight.getNextSwitch(tl_id)
                    current_time = traci_connection.simulation.getTime()
                    remaining_duration = next_switch - current_time
                    
                    program = traci_connection.trafficlight.getProgram(tl_id)
                    red_yellow_green_state = traci_connection.trafficlight.getRedYellowGreenState(tl_id)
                    
                    traffic_light_data[tl_id] = {
                        'current_phase': current_phase,
                        'phase_duration': phase_duration,
                        'remaining_duration': remaining_duration,
                        'program': program,
                        'state': red_yellow_green_state,
                        'next_switch': next_switch
                    }
                    
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"⚠️  Error collecting traffic light data: {e}")
        
        return traffic_light_data
    
    def calculate_performance_metrics(self, time_window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics from collected data.
        
        Args:
            time_window: Time window for calculation (None for all data)
            
        Returns:
            Performance metrics dictionary
        """
        if not self.data_history:
            return {}
        
        # Determine data subset
        if time_window is None:
            data_subset = self.data_history
        else:
            current_time = self.data_history[-1]['timestamp']
            data_subset = [d for d in self.data_history 
                          if d['timestamp'] > current_time - time_window]
        
        if not data_subset:
            return {}
        
        # Calculate metrics
        waiting_times = [d['avg_waiting_time'] for d in data_subset]
        speeds = [d['avg_speed'] for d in data_subset]
        vehicle_counts = [d['total_vehicles'] for d in data_subset]
        
        metrics = {
            'avg_waiting_time': statistics.mean(waiting_times),
            'max_waiting_time': max(waiting_times),
            'min_waiting_time': min(waiting_times),
            'waiting_time_std': statistics.stdev(waiting_times) if len(waiting_times) > 1 else 0.0,
            
            'avg_speed': statistics.mean(speeds),
            'max_speed': max(speeds),
            'min_speed': min(speeds),
            'speed_std': statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
            
            'avg_vehicle_count': statistics.mean(vehicle_counts),
            'max_vehicle_count': max(vehicle_counts),
            'min_vehicle_count': min(vehicle_counts),
            'vehicle_count_std': statistics.stdev(vehicle_counts) if len(vehicle_counts) > 1 else 0.0,
            
            'total_vehicle_time': sum(d['total_waiting_time'] for d in data_subset),
            'data_points': len(data_subset),
            'time_span': data_subset[-1]['timestamp'] - data_subset[0]['timestamp']
        }
        
        # Calculate derived metrics
        if metrics['time_span'] > 0:
            metrics['throughput'] = sum(vehicle_counts) / (metrics['time_span'] / 3600.0)  # vehicles per hour
            metrics['efficiency_score'] = max(0, 100 - metrics['avg_waiting_time'])
            metrics['consistency_score'] = max(0, 100 - metrics['waiting_time_std'])
        
        return metrics
    
    def calculate_directional_metrics(self, ns_edges: List[str], ew_edges: List[str]) -> Dict[str, Any]:
        """
        Calculate directional traffic metrics for NS and EW directions.
        
        Args:
            ns_edges: List of North-South edge IDs
            ew_edges: List of East-West edge IDs
            
        Returns:
            Directional metrics dictionary
        """
        if not self.data_history:
            return {}
        
        latest_data = self.data_history[-1]
        edge_data = latest_data.get('edge_data', {})
        
        # Calculate NS metrics
        ns_vehicles = 0
        ns_waiting_time = 0.0
        ns_speed = 0.0
        
        for edge_id in ns_edges:
            if edge_id in edge_data:
                edge_info = edge_data[edge_id]
                ns_vehicles += edge_info['vehicle_count']
                ns_waiting_time += edge_info['waiting_time']
                if edge_info['vehicles']:
                    ns_speed += statistics.mean([v['speed'] for v in edge_info['vehicles']])
        
        # Calculate EW metrics
        ew_vehicles = 0
        ew_waiting_time = 0.0
        ew_speed = 0.0
        
        for edge_id in ew_edges:
            if edge_id in edge_data:
                edge_info = edge_data[edge_id]
                ew_vehicles += edge_info['vehicle_count']
                ew_waiting_time += edge_info['waiting_time']
                if edge_info['vehicles']:
                    ew_speed += statistics.mean([v['speed'] for v in edge_info['vehicles']])
        
        return {
            'ns_metrics': {
                'vehicles': ns_vehicles,
                'waiting_time': ns_waiting_time,
                'avg_speed': ns_speed / len(ns_edges) if ns_edges else 0.0,
                'pressure': ns_vehicles * 5 + ns_waiting_time * 2
            },
            'ew_metrics': {
                'vehicles': ew_vehicles,
                'waiting_time': ew_waiting_time,
                'avg_speed': ew_speed / len(ew_edges) if ew_edges else 0.0,
                'pressure': ew_vehicles * 5 + ew_waiting_time * 2
            },
            'balance_ratio': ns_vehicles / max(ew_vehicles, 1),
            'pressure_difference': abs(ns_vehicles - ew_vehicles)
        }
    
    def get_time_series_data(self, metric: str, time_window: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Get time series data for a specific metric.
        
        Args:
            metric: Metric name to extract
            time_window: Time window for data (None for all)
            
        Returns:
            List of (timestamp, value) tuples
        """
        if not self.data_history:
            return []
        
        # Determine data subset
        if time_window is None:
            data_subset = self.data_history
        else:
            current_time = self.data_history[-1]['timestamp']
            data_subset = [d for d in self.data_history 
                          if d['timestamp'] > current_time - time_window]
        
        time_series = []
        for data_point in data_subset:
            if metric in data_point:
                time_series.append((data_point['timestamp'], data_point[metric]))
        
        return time_series
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report of traffic analysis.
        
        Returns:
            Formatted summary report string
        """
        if not self.data_history:
            return "No traffic data collected."
        
        metrics = self.calculate_performance_metrics()
        latest_data = self.data_history[-1]
        
        report = f"""
📊 TRAFFIC ANALYSIS SUMMARY REPORT
{'='*50}

🕐 Time Period: {self.data_history[0]['timestamp']}s - {latest_data['timestamp']}s
📈 Data Points: {len(self.data_history)}
⏱️  Duration: {metrics.get('time_span', 0):.0f} seconds

🚗 VEHICLE METRICS:
   Current Vehicles: {latest_data['total_vehicles']}
   Average Vehicles: {metrics.get('avg_vehicle_count', 0):.1f}
   Peak Vehicles: {metrics.get('max_vehicle_count', 0)}

⏰ WAITING TIME METRICS:
   Current Avg Wait: {latest_data['avg_waiting_time']:.1f}s
   Overall Avg Wait: {metrics.get('avg_waiting_time', 0):.1f}s
   Peak Waiting Time: {metrics.get('max_waiting_time', 0):.1f}s
   Waiting Consistency: {metrics.get('consistency_score', 0):.1f}%

🏃 SPEED METRICS:
   Current Avg Speed: {latest_data['avg_speed']:.1f} m/s
   Overall Avg Speed: {metrics.get('avg_speed', 0):.1f} m/s
   Speed Variability: {metrics.get('speed_std', 0):.1f} m/s

📊 PERFORMANCE INDICATORS:
   Traffic Efficiency: {metrics.get('efficiency_score', 0):.1f}%
   System Throughput: {metrics.get('throughput', 0):.0f} vehicles/hour
   Overall Performance: {'🟢 Good' if metrics.get('avg_waiting_time', 100) < 15 else '🟡 Fair' if metrics.get('avg_waiting_time', 100) < 30 else '🔴 Poor'}

🚦 EDGE ANALYSIS:
"""
        
        # Add edge-specific information
        edge_data = latest_data.get('edge_data', {})
        for edge_id, edge_info in edge_data.items():
            if edge_info['vehicle_count'] > 0:
                report += f"   {edge_id}: {edge_info['vehicle_count']} vehicles, {edge_info['avg_waiting_time']:.1f}s avg wait\n"
        
        return report
    
    def reset_data(self):
        """Reset all collected data."""
        self.data_history = []
        self.last_collection_time = 0
        print("📊 Traffic analyzer data reset")
    
    def export_data(self, filename: str) -> bool:
        """
        Export collected data to a file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if export successful
        """
        try:
            import json
            
            export_data = {
                'metadata': {
                    'collection_interval': self.collection_interval,
                    'total_data_points': len(self.data_history),
                    'export_time': self.last_collection_time
                },
                'performance_metrics': self.calculate_performance_metrics(),
                'data_history': self.data_history
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"📁 Traffic data exported to {filename}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error exporting data: {e}")
            return False