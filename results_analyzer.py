"""
Results and Analytics Module
Handles data collection, analysis, and report generation
"""

import statistics
from collections import defaultdict

class TrafficAnalyzer:
    def __init__(self):
        """Initialize the traffic analyzer"""
        self.traffic_data = {
            'waiting_times': defaultdict(list),
            'queue_lengths': defaultdict(list),
            'avg_speeds': defaultdict(list),
            'pressure_history': defaultdict(list),
            'flow_rates': defaultdict(list),
            'throughput': defaultdict(int)
        }
        
        self.edges = ["-E0", "-E0.254", "-E1", "-E1.238", "E0", "E0.319", "E1", "E1.200"]
        
    def collect_traffic_metrics(self, step, traci):
        """Collect comprehensive traffic metrics from SUMO"""
        try:
            vehicle_ids = traci.vehicle.getIDList()
            
            if not vehicle_ids:
                return None
            
            step_data = {
                'step': step,
                'total_vehicles': len(vehicle_ids),
                'waiting_vehicles': 0,
                'edge_data': {},
                'individual_vehicles': []
            }
            
            # Collect edge data
            for edge_id in self.edges:
                try:
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                    avg_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                    waiting_time = traci.edge.getWaitingTime(edge_id)
                    
                    step_data['edge_data'][edge_id] = {
                        'vehicle_count': vehicle_count,
                        'avg_speed': avg_speed,
                        'waiting_time': waiting_time
                    }
                    
                    # Update global tracking
                    self.traffic_data['queue_lengths'][edge_id].append(vehicle_count)
                    self.traffic_data['avg_speeds'][edge_id].append(avg_speed)
                    
                except Exception as e:
                    print(f"Error collecting data for edge {edge_id}: {e}")
            
            # Collect individual vehicle data
            total_waiting = 0
            for veh_id in vehicle_ids:
                try:
                    waiting = traci.vehicle.getWaitingTime(veh_id)
                    speed = traci.vehicle.getSpeed(veh_id)
                    edge = traci.vehicle.getRoadID(veh_id)
                    
                    step_data['individual_vehicles'].append({
                        'id': veh_id,
                        'waiting': waiting,
                        'speed': speed,
                        'edge': edge
                    })
                    
                    if waiting > 0:
                        step_data['waiting_vehicles'] += 1
                        total_waiting += waiting
                        self.traffic_data['waiting_times'][edge].append(waiting)
                        
                except Exception as e:
                    continue
            
            # Calculate average waiting time
            step_data['avg_waiting_time'] = total_waiting / len(vehicle_ids) if vehicle_ids else 0
            
            return step_data
            
        except Exception as e:
            print(f"Error in data collection: {e}")
            return None
    
    def update_pressure_data(self, ns_pressure, ew_pressure):
        """Update pressure history data"""
        self.traffic_data['pressure_history']['ns'].append(ns_pressure)
        self.traffic_data['pressure_history']['ew'].append(ew_pressure)
    
    def calculate_performance_metrics(self):
        """Calculate overall performance metrics"""
        metrics = {}
        
        # Waiting time analysis
        all_waiting_times = []
        for edge_times in self.traffic_data['waiting_times'].values():
            all_waiting_times.extend(edge_times)
        
        if all_waiting_times:
            metrics['waiting_times'] = {
                'average': statistics.mean(all_waiting_times),
                'median': statistics.median(all_waiting_times),
                'max': max(all_waiting_times),
                'min': min(all_waiting_times),
                'std_dev': statistics.stdev(all_waiting_times) if len(all_waiting_times) > 1 else 0
            }
            
            # Performance rating
            avg_waiting = metrics['waiting_times']['average']
            if avg_waiting < 20:
                metrics['performance_rating'] = "🟢 EXCELLENT"
            elif avg_waiting < 40:
                metrics['performance_rating'] = "🟡 GOOD"
            elif avg_waiting < 80:
                metrics['performance_rating'] = "🟠 FAIR"
            else:
                metrics['performance_rating'] = "🔴 POOR"
        else:
            metrics['waiting_times'] = None
            metrics['performance_rating'] = "📊 NO DATA"
        
        # Traffic pressure analysis
        if (self.traffic_data['pressure_history']['ns'] and 
            self.traffic_data['pressure_history']['ew']):
            
            ns_pressures = self.traffic_data['pressure_history']['ns']
            ew_pressures = self.traffic_data['pressure_history']['ew']
            
            metrics['pressure_analysis'] = {
                'ns_avg': statistics.mean(ns_pressures),
                'ns_peak': max(ns_pressures),
                'ew_avg': statistics.mean(ew_pressures),
                'ew_peak': max(ew_pressures),
                'total_measurements': len(ns_pressures)
            }
            
            # Balance recommendation
            ns_avg = metrics['pressure_analysis']['ns_avg']
            ew_avg = metrics['pressure_analysis']['ew_avg']
            
            if ns_avg > ew_avg * 1.2:
                metrics['balance_recommendation'] = "🔴 Consider longer NS green phases"
            elif ew_avg > ns_avg * 1.2:
                metrics['balance_recommendation'] = "🔴 Consider longer EW green phases"
            else:
                metrics['balance_recommendation'] = "🟢 Current timing appears balanced"
        else:
            metrics['pressure_analysis'] = None
            metrics['balance_recommendation'] = "📊 NO DATA"
        
        return metrics
    
    def calculate_edge_efficiency(self):
        """Calculate efficiency metrics for each edge"""
        edge_metrics = {}
        
        for edge in self.edges:
            if (edge in self.traffic_data['avg_speeds'] and 
                self.traffic_data['avg_speeds'][edge]):
                
                avg_vehicles = statistics.mean(self.traffic_data['queue_lengths'][edge])
                avg_speed = statistics.mean(self.traffic_data['avg_speeds'][edge])
                peak_queue = max(self.traffic_data['queue_lengths'][edge])
                throughput = len(self.traffic_data['avg_speeds'][edge])
                
                # Calculate efficiency (speed / vehicles ratio)
                efficiency = avg_speed / max(1, avg_vehicles)
                
                edge_metrics[edge] = {
                    'avg_vehicles': avg_vehicles,
                    'avg_speed': avg_speed,
                    'peak_queue': peak_queue,
                    'throughput': throughput,
                    'efficiency': efficiency
                }
        
        return edge_metrics
    
    def display_real_time_analysis(self, step, step_data, tl_analysis):
        """Display real-time traffic analysis"""
        hour = (step / 3600) % 24
        
        print(f"\n{'='*90}")
        print(f"🚦 REAL-TIME TRAFFIC ANALYSIS - Step {step} (Hour: {hour:.1f})")
        print(f"{'='*90}")
        
        # Network status
        print(f"📊 Network Status:")
        print(f"   Total Vehicles: {step_data['total_vehicles']}")
        print(f"   Vehicles Waiting: {step_data['waiting_vehicles']}")
        print(f"   Average Waiting Time: {step_data['avg_waiting_time']:.2f}s")
        
        # Traffic light analysis
        if tl_analysis and 'analysis' in tl_analysis:
            analysis = tl_analysis['analysis']
            tl_state = tl_analysis['tl_state']
            
            print(f"   Current Phase: {tl_state['phase']} ({tl_state['phase_name']})")
            print(f"   Remaining Time: {tl_state['remaining']:.1f}s")
            
            print(f"\n🎯 Traffic Pressure Analysis:")
            print(f"   North-South Pressure: {analysis.get('ns_pressure', 0):.1f}")
            print(f"   East-West Pressure: {analysis.get('ew_pressure', 0):.1f}")
            print(f"   Total Pressure: {analysis.get('total_pressure', 0):.1f}")
            
            print(f"\n💡 Adaptive Recommendations:")
            print(f"   Suggested NS Time: {analysis.get('suggested_ns_time', 0)}s")
            print(f"   Suggested EW Time: {analysis.get('suggested_ew_time', 0)}s")
            print(f"   Priority: {analysis.get('priority_message', 'Unknown')}")
        
        # Edge performance
        print(f"\n📈 Edge Performance:")
        print(f"{'Edge':<12} {'Vehicles':<10} {'Speed':<10} {'Waiting':<10}")
        print("-" * 90)
        
        for edge_id, data in step_data['edge_data'].items():
            if data['vehicle_count'] > 0:
                print(f"{edge_id:<12} {data['vehicle_count']:<10} "
                      f"{data['avg_speed']:<10.2f} {data['waiting_time']:<10.2f}")
    
    def generate_comprehensive_report(self, flow_summary=None, tl_performance=None):
        """Generate comprehensive final report"""
        print(f"\n{'='*100}")
        print(f"🎯 COMPREHENSIVE TRAFFIC SIMULATION REPORT")
        print(f"{'='*100}")
        
        # Performance metrics
        metrics = self.calculate_performance_metrics()
        
        if metrics['waiting_times']:
            wt = metrics['waiting_times']
            print(f"⏱️  WAITING TIME PERFORMANCE:")
            print(f"   Average: {wt['average']:.2f}s | Median: {wt['median']:.2f}s")
            print(f"   Max: {wt['max']:.2f}s | Min: {wt['min']:.2f}s")
            print(f"   Standard Deviation: {wt['std_dev']:.2f}s")
            print(f"   Overall Rating: {metrics['performance_rating']}")
        
        # Pressure analysis
        if metrics['pressure_analysis']:
            pa = metrics['pressure_analysis']
            print(f"\n📊 TRAFFIC PRESSURE ANALYSIS:")
            print(f"   North-South: Avg {pa['ns_avg']:.1f} | Peak {pa['ns_peak']:.1f}")
            print(f"   East-West: Avg {pa['ew_avg']:.1f} | Peak {pa['ew_peak']:.1f}")
            print(f"   Total Measurements: {pa['total_measurements']}")
            print(f"   Recommendation: {metrics['balance_recommendation']}")
        
        # Edge efficiency
        edge_metrics = self.calculate_edge_efficiency()
        if edge_metrics:
            print(f"\n🚗 EDGE EFFICIENCY ANALYSIS:")
            print(f"{'Edge':<12} {'Avg Vehicles':<15} {'Avg Speed':<12} {'Peak Queue':<12} {'Efficiency':<12}")
            print("-" * 100)
            
            for edge, data in edge_metrics.items():
                print(f"{edge:<12} {data['avg_vehicles']:<15.1f} "
                      f"{data['avg_speed']:<12.2f} {data['peak_queue']:<12} "
                      f"{data['efficiency']:<12.2f}")
        
        # Flow summary
        if flow_summary:
            print(f"\n🔄 DYNAMIC FLOW ANALYSIS:")
            print("Flow rates were dynamically adjusted throughout simulation:")
            for flow_id, data in flow_summary.items():
                print(f"   {flow_id}: {data['from']} → {data['to']} = "
                      f"{data['current_rate']} veh/h ({data['efficiency']:.1f}% of base)")
        
        # Traffic light performance
        if tl_performance:
            print(f"\n🚦 ADAPTIVE TRAFFIC LIGHT PERFORMANCE:")
            print(f"Total adaptive considerations: {tl_performance['total_adaptations']}")
            
            for phase_key, data in tl_performance.get('phase_history', {}).items():
                print(f"   {phase_key}: Avg pressure {data['avg_pressure']:.1f}, "
                      f"Peak {data['max_pressure']:.1f} ({data['measurements']} measurements)")
        
        print(f"\n{'='*100}")
        return metrics
    
    def export_data_summary(self):
        """Export summary data for external analysis"""
        summary = {
            'metrics': self.calculate_performance_metrics(),
            'edge_efficiency': self.calculate_edge_efficiency(),
            'raw_data_counts': {
                'waiting_time_samples': sum(len(times) for times in self.traffic_data['waiting_times'].values()),
                'speed_samples': sum(len(speeds) for speeds in self.traffic_data['avg_speeds'].values()),
                'queue_samples': sum(len(queues) for queues in self.traffic_data['queue_lengths'].values()),
                'pressure_samples': len(self.traffic_data['pressure_history']['ns'])
            }
        }
        return summary
