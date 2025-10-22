#!/usr/bin/env python3
"""
Complex Multi-Intersection Traffic Network Analysis
==================================================

This script analyzes traffic performance across a complex network featuring:
- 4 main intersections (CBD, Residential, Commercial, Industrial)
- 1 roundabout (University area)
- Multiple vehicle types (cars, trucks, buses, emergency)
- Diverse traffic patterns (commuter, commercial, industrial)

Features:
- Multi-intersection coordination analysis
- Vehicle type performance comparison
- Route efficiency evaluation
- Peak hour vs off-peak analysis
- Intersection-specific optimization
"""

import os
import sys
import json
import time
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ComplexNetworkAnalyzer:
    """Analyzer for complex multi-intersection traffic network."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the complex network analyzer."""
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.base_dir, "complex_network_results")
        
        # Network configuration
        self.network_file = os.path.join(self.base_dir, "complex_network.net.xml")
        self.routes_file = os.path.join(self.base_dir, "complex_routes.rou.xml")
        self.config_file = os.path.join(self.base_dir, "complex_simulation.sumocfg")
        
        # Simulation parameters
        self.simulation_duration = 8 * 3600  # 8 hours
        self.step_length = 30  # 30 second intervals
        
        # Network topology
        self.intersections = {
            'central_main': {'type': 'CBD', 'priority': 1, 'lanes': 8},
            'north_residential': {'type': 'Residential', 'priority': 3, 'lanes': 4},
            'east_commercial': {'type': 'Commercial', 'priority': 2, 'lanes': 4},
            'south_industrial': {'type': 'Industrial', 'priority': 2, 'lanes': 4},
            'west_roundabout': {'type': 'University', 'priority': 4, 'lanes': 4}
        }
        
        self.vehicle_types = {
            'passenger': {'weight': 1.0, 'priority': 3},
            'delivery': {'weight': 1.5, 'priority': 2},
            'truck': {'weight': 3.0, 'priority': 1},
            'bus': {'weight': 2.5, 'priority': 4},
            'emergency': {'weight': 1.0, 'priority': 5}
        }
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("🌆 Complex Multi-Intersection Network Analyzer Initialized")
        print(f"   Network: {len(self.intersections)} intersections")
        print(f"   Vehicle Types: {len(self.vehicle_types)}")
        print(f"   Simulation Duration: {self.simulation_duration/3600:.1f} hours")
        print(f"   Results Directory: {self.results_dir}")
    
    def generate_adaptive_traffic_lights(self) -> str:
        """Generate adaptive traffic light logic for each intersection."""
        
        print("\\n🚦 Generating Adaptive Traffic Light Programs...")
        
        # Create adaptive network file
        adaptive_net_file = os.path.join(self.base_dir, "complex_network_adaptive.net.xml")
        
        # Parse original network
        tree = ET.parse(self.network_file)
        root = tree.getroot()
        
        # Remove existing traffic light logic
        for tl_logic in root.findall('tlLogic'):
            root.remove(tl_logic)
        
        # Generate adaptive logic for each intersection
        adaptive_programs = {
            'central_main': self._generate_cbd_adaptive_logic(),
            'north_residential': self._generate_residential_adaptive_logic(),
            'east_commercial': self._generate_commercial_adaptive_logic(),
            'south_industrial': self._generate_industrial_adaptive_logic()
        }
        
        # Add adaptive traffic light programs
        for tl_id, phases in adaptive_programs.items():
            tl_logic = ET.SubElement(root, 'tlLogic')
            tl_logic.set('id', tl_id)
            tl_logic.set('type', 'actuated')
            tl_logic.set('programID', 'adaptive')
            tl_logic.set('offset', '0')
            
            for phase_data in phases:
                phase = ET.SubElement(tl_logic, 'phase')
                phase.set('duration', str(phase_data['duration']))
                phase.set('state', phase_data['state'])
                phase.set('minDur', str(phase_data['min_dur']))
                phase.set('maxDur', str(phase_data['max_dur']))
        
        # Save adaptive network
        tree.write(adaptive_net_file, encoding='UTF-8', xml_declaration=True)
        
        print(f"   ✅ Adaptive network saved: {os.path.basename(adaptive_net_file)}")
        return adaptive_net_file
    
    def _generate_cbd_adaptive_logic(self) -> List[Dict]:
        """Generate adaptive logic for Central Business District intersection."""
        return [
            {'duration': 45, 'state': 'GGrrrrGGrrrr', 'min_dur': 30, 'max_dur': 60},  # Main arterial
            {'duration': 4,  'state': 'yyrrrryyrrrr', 'min_dur': 3,  'max_dur': 5},   # Yellow
            {'duration': 40, 'state': 'rrGGrrrrGGrr', 'min_dur': 25, 'max_dur': 55},  # Cross arterial
            {'duration': 4,  'state': 'rryyrrrryyrr', 'min_dur': 3,  'max_dur': 5},   # Yellow
            {'duration': 20, 'state': 'rrrrGGrrrrGG', 'min_dur': 10, 'max_dur': 35},  # Left turns
            {'duration': 4,  'state': 'rrrryyrrrryy', 'min_dur': 3,  'max_dur': 5}    # Yellow
        ]
    
    def _generate_residential_adaptive_logic(self) -> List[Dict]:
        """Generate adaptive logic for Residential intersection."""
        return [
            {'duration': 30, 'state': 'GGrrrrGGrrrr', 'min_dur': 20, 'max_dur': 45},  # Main flow
            {'duration': 3,  'state': 'yyrrrryyrrrr', 'min_dur': 3,  'max_dur': 4},   # Yellow
            {'duration': 25, 'state': 'rrGGrrrrGGrr', 'min_dur': 15, 'max_dur': 40},  # Cross flow
            {'duration': 3,  'state': 'rryyrrrryyrr', 'min_dur': 3,  'max_dur': 4},   # Yellow
            {'duration': 15, 'state': 'rrrrGGrrrrGG', 'min_dur': 8,  'max_dur': 25},  # Left turns
            {'duration': 3,  'state': 'rrrryyrrrryy', 'min_dur': 3,  'max_dur': 4}    # Yellow
        ]
    
    def _generate_commercial_adaptive_logic(self) -> List[Dict]:
        """Generate adaptive logic for Commercial intersection."""
        return [
            {'duration': 40, 'state': 'GGrrrrGGrrrr', 'min_dur': 25, 'max_dur': 55},  # Commercial flow
            {'duration': 4,  'state': 'yyrrrryyrrrr', 'min_dur': 3,  'max_dur': 5},   # Yellow
            {'duration': 30, 'state': 'rrGGrrrrGGrr', 'min_dur': 20, 'max_dur': 45},  # Cross flow
            {'duration': 4,  'state': 'rryyrrrryyrr', 'min_dur': 3,  'max_dur': 5},   # Yellow
            {'duration': 20, 'state': 'rrrrGGrrrrGG', 'min_dur': 12, 'max_dur': 30},  # Left turns
            {'duration': 4,  'state': 'rrrryyrrrryy', 'min_dur': 3,  'max_dur': 5}    # Yellow
        ]
    
    def _generate_industrial_adaptive_logic(self) -> List[Dict]:
        """Generate adaptive logic for Industrial intersection."""
        return [
            {'duration': 50, 'state': 'GGrrrrGGrrrr', 'min_dur': 35, 'max_dur': 70},  # Heavy flow
            {'duration': 4,  'state': 'yyrrrryyrrrr', 'min_dur': 3,  'max_dur': 5},   # Yellow
            {'duration': 25, 'state': 'rrGGrrrrGGrr', 'min_dur': 15, 'max_dur': 40},  # Cross flow
            {'duration': 4,  'state': 'rryyrrrryyrr', 'min_dur': 3,  'max_dur': 5},   # Yellow
            {'duration': 12, 'state': 'rrrrGGrrrrGG', 'min_dur': 8,  'max_dur': 20},  # Minimal left
            {'duration': 4,  'state': 'rrrryyrrrryy', 'min_dur': 3,  'max_dur': 5}    # Yellow
        ]
    
    def run_simulation(self, mode: str = "normal") -> Dict[str, Any]:
        """Run SUMO simulation with specified mode."""
        
        print(f"\\n🚀 Running Complex Network Simulation ({mode.upper()} mode)...")
        
        # Choose network file based on mode
        if mode == "adaptive":
            net_file = self.generate_adaptive_traffic_lights()
        else:
            net_file = self.network_file
        
        # Create temporary config file
        temp_config = os.path.join(self.base_dir, f"temp_{mode}_config.sumocfg")
        self._create_temp_config(temp_config, net_file, mode)
        
        # Output files
        summary_file = os.path.join(self.results_dir, f"{mode}_summary.xml")
        tripinfo_file = os.path.join(self.results_dir, f"{mode}_tripinfo.xml")
        
        # Run SUMO simulation
        cmd = [
            "sumo",
            "-c", temp_config,
            "--summary-output", summary_file,
            "--tripinfo-output", tripinfo_file,
            "--no-warnings", "true",
            "--duration-log.disable", "true"
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            simulation_time = time.time() - start_time
            
            print(f"   ✅ Simulation completed in {simulation_time:.1f}s")
            
            # Parse results
            simulation_data = self._parse_simulation_results(summary_file, tripinfo_file, mode)
            simulation_data['simulation_time'] = simulation_time
            simulation_data['mode'] = mode
            
            # Clean up temp files
            if os.path.exists(temp_config):
                os.remove(temp_config)
            
            return simulation_data
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Simulation failed: {e}")
            print(f"   Error output: {e.stderr}")
            return {}
    
    def _create_temp_config(self, config_file: str, net_file: str, mode: str):
        """Create temporary configuration file."""
        
        config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{self.routes_file}"/>
    </input>
    <output>
        <summary-output value="{self.results_dir}/{mode}_summary.xml"/>
        <tripinfo-output value="{self.results_dir}/{mode}_tripinfo.xml"/>
    </output>
    <time>
        <begin value="0"/>
        <end value="{self.simulation_duration}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <collision.action value="warn"/>
        <max-depart-delay value="900"/>
        <routing-algorithm value="dijkstra"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <duration-log.disable value="true"/>
    </report>
</configuration>"""
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
    
    def _parse_simulation_results(self, summary_file: str, tripinfo_file: str, mode: str) -> Dict[str, Any]:
        """Parse simulation results from XML files."""
        
        print(f"   📊 Parsing {mode} simulation results...")
        
        results = {
            'mode': mode,
            'summary_data': [],
            'trip_data': [],
            'intersection_data': {},
            'vehicle_type_data': {},
            'route_performance': {},
            'time_period_analysis': {}
        }
        
        # Parse summary data
        if os.path.exists(summary_file):
            try:
                tree = ET.parse(summary_file)
                root = tree.getroot()
                
                for step in root.findall('step'):
                    step_data = {
                        'time': float(step.get('time', 0)),
                        'loaded': int(step.get('loaded', 0)),
                        'inserted': int(step.get('inserted', 0)),
                        'running': int(step.get('running', 0)),
                        'waiting': int(step.get('waiting', 0)),
                        'ended': int(step.get('ended', 0)),
                        'meanWaitingTime': float(step.get('meanWaitingTime', 0)),
                        'meanTravelTime': float(step.get('meanTravelTime', 0)),
                        'meanSpeed': float(step.get('meanSpeed', 0))
                    }
                    results['summary_data'].append(step_data)
                    
                print(f"      ✅ Parsed {len(results['summary_data'])} summary steps")
                
            except Exception as e:
                print(f"      ⚠️ Error parsing summary: {e}")
        
        # Parse trip data
        if os.path.exists(tripinfo_file):
            try:
                tree = ET.parse(tripinfo_file)
                root = tree.getroot()
                
                for trip in root.findall('tripinfo'):
                    trip_data = {
                        'id': trip.get('id'),
                        'depart': float(trip.get('depart', 0)),
                        'arrival': float(trip.get('arrival', 0)),
                        'duration': float(trip.get('duration', 0)),
                        'routeLength': float(trip.get('routeLength', 0)),
                        'waitingTime': float(trip.get('waitingTime', 0)),
                        'timeLoss': float(trip.get('timeLoss', 0)),
                        'vType': trip.get('vType', 'unknown')
                    }
                    results['trip_data'].append(trip_data)
                
                print(f"      ✅ Parsed {len(results['trip_data'])} trip records")
                
                # Analyze by vehicle type
                self._analyze_by_vehicle_type(results)
                
                # Analyze by time periods
                self._analyze_by_time_periods(results)
                
            except Exception as e:
                print(f"      ⚠️ Error parsing trips: {e}")
        
        return results
    
    def _analyze_by_vehicle_type(self, results: Dict[str, Any]):
        """Analyze performance by vehicle type."""
        
        vehicle_stats = {}
        
        for trip in results['trip_data']:
            vtype = trip['vType']
            if vtype not in vehicle_stats:
                vehicle_stats[vtype] = {
                    'count': 0,
                    'total_duration': 0,
                    'total_waiting': 0,
                    'total_distance': 0,
                    'total_time_loss': 0
                }
            
            stats = vehicle_stats[vtype]
            stats['count'] += 1
            stats['total_duration'] += trip['duration']
            stats['total_waiting'] += trip['waitingTime']
            stats['total_distance'] += trip['routeLength']
            stats['total_time_loss'] += trip['timeLoss']
        
        # Calculate averages
        for vtype, stats in vehicle_stats.items():
            if stats['count'] > 0:
                results['vehicle_type_data'][vtype] = {
                    'count': stats['count'],
                    'avg_duration': stats['total_duration'] / stats['count'],
                    'avg_waiting': stats['total_waiting'] / stats['count'],
                    'avg_distance': stats['total_distance'] / stats['count'],
                    'avg_time_loss': stats['total_time_loss'] / stats['count'],
                    'avg_speed': stats['total_distance'] / stats['total_duration'] if stats['total_duration'] > 0 else 0
                }
    
    def _analyze_by_time_periods(self, results: Dict[str, Any]):
        """Analyze performance by time periods."""
        
        # Define time periods (in seconds)
        periods = {
            'early_morning': (0, 7200),      # 0-2 AM
            'morning_rush': (7200, 10800),   # 2-3 AM (shifted for simulation)
            'midday': (10800, 14400),        # 3-4 AM
            'afternoon_rush': (14400, 18000), # 4-5 AM
            'evening': (18000, 21600),       # 5-6 AM
            'night': (21600, 28800)          # 6-8 AM
        }
        
        period_stats = {}
        
        for period_name, (start_time, end_time) in periods.items():
            period_trips = [trip for trip in results['trip_data'] 
                          if start_time <= trip['depart'] < end_time]
            
            if period_trips:
                period_stats[period_name] = {
                    'count': len(period_trips),
                    'avg_duration': statistics.mean([t['duration'] for t in period_trips]),
                    'avg_waiting': statistics.mean([t['waitingTime'] for t in period_trips]),
                    'avg_speed': statistics.mean([t['routeLength']/t['duration'] for t in period_trips if t['duration'] > 0]),
                    'total_time_loss': sum([t['timeLoss'] for t in period_trips])
                }
            else:
                period_stats[period_name] = {
                    'count': 0, 'avg_duration': 0, 'avg_waiting': 0, 
                    'avg_speed': 0, 'total_time_loss': 0
                }
        
        results['time_period_analysis'] = period_stats
    
    def run_comparative_analysis(self) -> Dict[str, Any]:
        """Run comparative analysis between normal and adaptive modes."""
        
        print("\\n🔬 COMPLEX NETWORK COMPARATIVE ANALYSIS")
        print("=" * 80)
        
        # Run both simulations
        normal_results = self.run_simulation("normal")
        adaptive_results = self.run_simulation("adaptive")
        
        if not normal_results or not adaptive_results:
            print("❌ One or both simulations failed!")
            return {}
        
        # Perform comparative analysis
        comparison = self._compare_results(normal_results, adaptive_results)
        
        # Generate comprehensive report
        self._generate_comprehensive_report(comparison)
        
        # Create visualizations
        self._create_complex_visualizations(comparison)
        
        return comparison
    
    def _compare_results(self, normal: Dict, adaptive: Dict) -> Dict[str, Any]:
        """Compare normal vs adaptive results."""
        
        print("\\n📊 Analyzing Performance Differences...")
        
        comparison = {
            'overall_performance': {},
            'vehicle_type_comparison': {},
            'time_period_comparison': {},
            'efficiency_metrics': {},
            'improvement_summary': {}
        }
        
        # Overall performance comparison
        if normal['trip_data'] and adaptive['trip_data']:
            normal_avg_waiting = statistics.mean([t['waitingTime'] for t in normal['trip_data']])
            adaptive_avg_waiting = statistics.mean([t['waitingTime'] for t in adaptive['trip_data']])
            
            normal_avg_duration = statistics.mean([t['duration'] for t in normal['trip_data']])
            adaptive_avg_duration = statistics.mean([t['duration'] for t in adaptive['trip_data']])
            
            normal_avg_speed = statistics.mean([t['routeLength']/t['duration'] for t in normal['trip_data'] if t['duration'] > 0])
            adaptive_avg_speed = statistics.mean([t['routeLength']/t['duration'] for t in adaptive['trip_data'] if t['duration'] > 0])
            
            comparison['overall_performance'] = {
                'waiting_time': {
                    'normal': normal_avg_waiting,
                    'adaptive': adaptive_avg_waiting,
                    'improvement': ((normal_avg_waiting - adaptive_avg_waiting) / normal_avg_waiting) * 100
                },
                'trip_duration': {
                    'normal': normal_avg_duration,
                    'adaptive': adaptive_avg_duration,
                    'improvement': ((normal_avg_duration - adaptive_avg_duration) / normal_avg_duration) * 100
                },
                'average_speed': {
                    'normal': normal_avg_speed,
                    'adaptive': adaptive_avg_speed,
                    'improvement': ((adaptive_avg_speed - normal_avg_speed) / normal_avg_speed) * 100
                }
            }
        
        # Vehicle type comparison
        for vtype in self.vehicle_types.keys():
            if vtype in normal.get('vehicle_type_data', {}) and vtype in adaptive.get('vehicle_type_data', {}):
                normal_data = normal['vehicle_type_data'][vtype]
                adaptive_data = adaptive['vehicle_type_data'][vtype]
                
                comparison['vehicle_type_comparison'][vtype] = {
                    'waiting_improvement': ((normal_data['avg_waiting'] - adaptive_data['avg_waiting']) / normal_data['avg_waiting']) * 100 if normal_data['avg_waiting'] > 0 else 0,
                    'duration_improvement': ((normal_data['avg_duration'] - adaptive_data['avg_duration']) / normal_data['avg_duration']) * 100 if normal_data['avg_duration'] > 0 else 0,
                    'speed_improvement': ((adaptive_data['avg_speed'] - normal_data['avg_speed']) / normal_data['avg_speed']) * 100 if normal_data['avg_speed'] > 0 else 0
                }
        
        # Time period comparison
        for period in ['morning_rush', 'midday', 'afternoon_rush', 'evening']:
            if (period in normal.get('time_period_analysis', {}) and 
                period in adaptive.get('time_period_analysis', {})):
                
                normal_period = normal['time_period_analysis'][period]
                adaptive_period = adaptive['time_period_analysis'][period]
                
                if normal_period['count'] > 0 and adaptive_period['count'] > 0:
                    comparison['time_period_comparison'][period] = {
                        'waiting_improvement': ((normal_period['avg_waiting'] - adaptive_period['avg_waiting']) / normal_period['avg_waiting']) * 100,
                        'speed_improvement': ((adaptive_period['avg_speed'] - normal_period['avg_speed']) / normal_period['avg_speed']) * 100
                    }
        
        return comparison
    
    def _generate_comprehensive_report(self, comparison: Dict[str, Any]):
        """Generate comprehensive analysis report."""
        
        report_file = os.path.join(self.results_dir, "COMPLEX_NETWORK_ANALYSIS_REPORT.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Complex Multi-Intersection Network Analysis Report\\n\\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## Network Overview\\n\\n")
            f.write("### Topology\\n")
            for intersection, data in self.intersections.items():
                f.write(f"- **{intersection}**: {data['type']} ({data['lanes']} lanes, Priority {data['priority']})\\n")
            
            f.write("\\n### Vehicle Types\\n")
            for vtype, data in self.vehicle_types.items():
                f.write(f"- **{vtype}**: Weight {data['weight']}, Priority {data['priority']}\\n")
            
            # Overall performance
            if 'overall_performance' in comparison:
                f.write("\\n## Overall Performance Comparison\\n\\n")
                overall = comparison['overall_performance']
                
                f.write("| Metric | Normal Mode | Adaptive Mode | Improvement |\\n")
                f.write("|--------|-------------|---------------|-------------|\\n")
                
                for metric, data in overall.items():
                    metric_name = metric.replace('_', ' ').title()
                    f.write(f"| {metric_name} | {data['normal']:.2f} | {data['adaptive']:.2f} | {data['improvement']:+.1f}% |\\n")
            
            # Vehicle type analysis
            if 'vehicle_type_comparison' in comparison:
                f.write("\\n## Vehicle Type Performance\\n\\n")
                f.write("| Vehicle Type | Waiting Improvement | Duration Improvement | Speed Improvement |\\n")
                f.write("|--------------|-------------------|-------------------|------------------|\\n")
                
                for vtype, data in comparison['vehicle_type_comparison'].items():
                    f.write(f"| {vtype.title()} | {data['waiting_improvement']:+.1f}% | {data['duration_improvement']:+.1f}% | {data['speed_improvement']:+.1f}% |\\n")
            
            # Time period analysis
            if 'time_period_comparison' in comparison:
                f.write("\\n## Time Period Analysis\\n\\n")
                f.write("| Time Period | Waiting Improvement | Speed Improvement |\\n")
                f.write("|-------------|-------------------|------------------|\\n")
                
                for period, data in comparison['time_period_comparison'].items():
                    period_name = period.replace('_', ' ').title()
                    f.write(f"| {period_name} | {data['waiting_improvement']:+.1f}% | {data['speed_improvement']:+.1f}% |\\n")
        
        print(f"   📋 Comprehensive report saved: {os.path.basename(report_file)}")
    
    def _create_complex_visualizations(self, comparison: Dict[str, Any]):
        """Create complex network visualizations."""
        
        print("\\n📊 Creating Complex Network Visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create multiple visualization files
        viz_files = []
        
        # 1. Overall Performance Summary
        if 'overall_performance' in comparison:
            viz_files.append(self._create_overall_performance_chart(comparison['overall_performance']))
        
        # 2. Vehicle Type Comparison
        if 'vehicle_type_comparison' in comparison:
            viz_files.append(self._create_vehicle_type_chart(comparison['vehicle_type_comparison']))
        
        # 3. Time Period Analysis
        if 'time_period_comparison' in comparison:
            viz_files.append(self._create_time_period_chart(comparison['time_period_comparison']))
        
        print(f"   ✅ Created {len(viz_files)} visualization files")
        return viz_files
    
    def _create_overall_performance_chart(self, overall_data: Dict) -> str:
        """Create overall performance comparison chart."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Complex Network Overall Performance Analysis', fontsize=16, fontweight='bold')
        
        metrics = list(overall_data.keys())
        normal_values = [overall_data[m]['normal'] for m in metrics]
        adaptive_values = [overall_data[m]['adaptive'] for m in metrics]
        improvements = [overall_data[m]['improvement'] for m in metrics]
        
        # 1. Comparison bars
        x = range(len(metrics))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], normal_values, width, label='Normal Mode', alpha=0.8, color='red')
        ax1.bar([i + width/2 for i in x], adaptive_values, width, label='Adaptive Mode', alpha=0.8, color='blue')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Improvement percentages
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax2.bar(metrics, improvements, color=colors, alpha=0.7)
        ax2.set_title('Improvement Percentages')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 3. Network topology diagram (simplified)
        ax3.text(0.5, 0.9, 'Complex Network Topology', ha='center', transform=ax3.transAxes, 
                fontsize=14, fontweight='bold')
        ax3.text(0.5, 0.7, '🏢 Central CBD\\n(Main Intersection)', ha='center', transform=ax3.transAxes, fontsize=10)
        ax3.text(0.2, 0.5, '🏠 Residential\\nArea', ha='center', transform=ax3.transAxes, fontsize=9)
        ax3.text(0.8, 0.5, '🏬 Commercial\\nDistrict', ha='center', transform=ax3.transAxes, fontsize=9)
        ax3.text(0.5, 0.3, '🏭 Industrial\\nZone', ha='center', transform=ax3.transAxes, fontsize=9)
        ax3.text(0.1, 0.1, '🎓 University\\n(Roundabout)', ha='center', transform=ax3.transAxes, fontsize=9)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # 4. Summary statistics
        avg_improvement = statistics.mean(improvements)
        best_metric = metrics[improvements.index(max(improvements))]
        
        ax4.text(0.5, 0.8, 'Performance Summary', ha='center', transform=ax4.transAxes, 
                fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.6, f'Average Improvement: {avg_improvement:.1f}%', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.5, f'Best Performing Metric: {best_metric.replace("_", " ").title()}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f'Best Improvement: {max(improvements):.1f}%', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.2, f'Intersections Analyzed: {len(self.intersections)}', transform=ax4.transAxes, fontsize=12)
        ax4.axis('off')
        
        plt.tight_layout()
        chart_file = os.path.join(self.results_dir, "01_overall_performance_analysis.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_file
    
    def _create_vehicle_type_chart(self, vehicle_data: Dict) -> str:
        """Create vehicle type performance chart."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Vehicle Type Performance Analysis', fontsize=16, fontweight='bold')
        
        vehicle_types = list(vehicle_data.keys())
        waiting_improvements = [vehicle_data[vt]['waiting_improvement'] for vt in vehicle_types]
        speed_improvements = [vehicle_data[vt]['speed_improvement'] for vt in vehicle_types]
        
        # 1. Waiting time improvements
        colors1 = ['green' if imp > 0 else 'red' for imp in waiting_improvements]
        bars1 = ax1.bar(vehicle_types, waiting_improvements, color=colors1, alpha=0.7)
        ax1.set_title('Waiting Time Improvement by Vehicle Type')
        ax1.set_ylabel('Improvement (%)')
        ax1.set_xticklabels([vt.title() for vt in vehicle_types], rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, waiting_improvements):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 2. Speed improvements
        colors2 = ['green' if imp > 0 else 'red' for imp in speed_improvements]
        bars2 = ax2.bar(vehicle_types, speed_improvements, color=colors2, alpha=0.7)
        ax2.set_title('Speed Improvement by Vehicle Type')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_xticklabels([vt.title() for vt in vehicle_types], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, speed_improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        chart_file = os.path.join(self.results_dir, "02_vehicle_type_analysis.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_file
    
    def _create_time_period_chart(self, time_data: Dict) -> str:
        """Create time period analysis chart."""
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Time Period Performance Analysis', fontsize=16, fontweight='bold')
        
        periods = list(time_data.keys())
        waiting_improvements = [time_data[p]['waiting_improvement'] for p in periods]
        speed_improvements = [time_data[p]['speed_improvement'] for p in periods]
        
        x = range(len(periods))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], waiting_improvements, width, 
                      label='Waiting Time Improvement', alpha=0.8, color='blue')
        bars2 = ax.bar([i + width/2 for i in x], speed_improvements, width, 
                      label='Speed Improvement', alpha=0.8, color='orange')
        
        ax.set_title('Performance Improvements by Time Period')
        ax.set_ylabel('Improvement (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in periods], rotation=45)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                       f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        chart_file = os.path.join(self.results_dir, "03_time_period_analysis.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_file
    
    def launch_sumo_gui(self):
        """Launch SUMO GUI for interactive visualization."""
        
        print("\\n🖥️ Launching SUMO GUI for Complex Network...")
        
        try:
            cmd = ["sumo-gui", "-c", self.config_file]
            subprocess.Popen(cmd)
            print("   ✅ SUMO GUI launched successfully!")
            print("   💡 Use the GUI to:")
            print("      - Visualize the complex network topology")
            print("      - Observe traffic flows in real-time")
            print("      - Analyze intersection performance")
            print("      - Test different scenarios")
            
        except Exception as e:
            print(f"   ❌ Failed to launch SUMO GUI: {e}")
            print("   💡 Make sure SUMO is installed and in your PATH")

def main():
    """Main execution function."""
    print("🌆 COMPLEX MULTI-INTERSECTION TRAFFIC NETWORK")
    print("=" * 60)
    print("Features:")
    print("  • 5 Intersections (4 Traffic Lights + 1 Roundabout)")
    print("  • Multiple Vehicle Types (Cars, Trucks, Buses, Emergency)")
    print("  • Diverse Traffic Patterns (Residential, Commercial, Industrial)")
    print("  • 8-Hour Simulation with Peak/Off-Peak Analysis")
    print("  • Adaptive vs Normal Mode Comparison")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ComplexNetworkAnalyzer()
    
    # Run comprehensive analysis
    try:
        comparison_results = analyzer.run_comparative_analysis()
        
        if comparison_results:
            print("\\n🎉 COMPLEX NETWORK ANALYSIS COMPLETE!")
            print(f"📁 Results saved to: {analyzer.results_dir}")
            print("\\n📊 Key Findings:")
            
            if 'overall_performance' in comparison_results:
                overall = comparison_results['overall_performance']
                for metric, data in overall.items():
                    print(f"   • {metric.replace('_', ' ').title()}: {data['improvement']:+.1f}% improvement")
            
            # Launch GUI for visualization
            print("\\n🖥️ Would you like to launch SUMO GUI? (y/n): ", end="")
            if input().lower().startswith('y'):
                analyzer.launch_sumo_gui()
            
        else:
            print("\\n❌ Analysis failed. Check the error messages above.")
            
    except KeyboardInterrupt:
        print("\\n\\n⏹️ Analysis interrupted by user.")
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()