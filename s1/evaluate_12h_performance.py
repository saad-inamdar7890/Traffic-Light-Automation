#!/usr/bin/env python3
"""
Comprehensive Evaluation: Fixed-Time vs MAPPO for 12-Hour Scenarios
====================================================================

This script performs a detailed comparison between Fixed-Time and MAPPO
traffic control over 12-hour periods (6am-6pm), providing:

1. Side-by-side simulation runs
2. Hourly performance breakdown
3. Rush hour vs off-peak analysis
4. Time-of-day adaptation metrics
5. Detailed visualizations

Key Metrics:
- Total/Average waiting time
- Network throughput
- Queue lengths
- Vehicle completion rate
- Rush hour performance

Usage:
    python evaluate_12h_performance.py --checkpoint checkpoint_12h_weekday_ep50 --scenario weekday
    python evaluate_12h_performance.py --checkpoint checkpoint_12h_weekday_ep50 --scenario weekday --quick
"""

import os
import sys
import argparse
import json
import zipfile
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

# Add parent directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import traci
except ImportError:
    print("Error: SUMO TraCI not found. Please install SUMO and set SUMO_HOME.")
    sys.exit(1)

from mappo_k1_implementation import MAPPOConfig, MAPPOAgent, K1Environment

# Time period definitions for 12-hour scenario
TIME_PERIODS = {
    'early_morning': (0, 7200, '6am-8am'),
    'morning_peak': (7200, 10800, '8am-9am'),
    'late_morning': (10800, 21600, '9am-12pm'),
    'lunch': (21600, 28800, '12pm-2pm'),
    'early_afternoon': (28800, 36000, '2pm-4pm'),
    'late_afternoon': (36000, 43200, '4pm-6pm'),
}


def get_time_period(step: int) -> str:
    """Get the time period name for a given simulation step."""
    for period_name, (start, end, _) in TIME_PERIODS.items():
        if start <= step < end:
            return period_name
    return 'late_afternoon'


def get_sim_time_str(step: int) -> str:
    """Convert simulation step to time string (starting from 6am)."""
    total_minutes = step // 60
    hours = 6 + (total_minutes // 60)
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


class FixedTimeController:
    """Fixed-time traffic light controller for comparison."""
    
    def __init__(self, cycle_length=90, green_split=0.5):
        self.cycle_length = cycle_length
        self.green_split = green_split
        self.current_step = 0
    
    def step(self):
        """Advance simulation by one step with fixed-time control."""
        self.current_step += 1
        # Fixed-time control is handled by SUMO's default TLS programs


def run_fixed_time_simulation(config_file: str, max_steps: int = 43200) -> dict:
    """Run simulation with fixed-time control."""
    print("\n" + "=" * 60)
    print("Running Fixed-Time Baseline (12 hours)")
    print("=" * 60)
    
    metrics = {
        'total_waiting_time': 0,
        'total_vehicles_completed': 0,
        'total_vehicles_entered': 0,
        'hourly_data': defaultdict(lambda: {
            'waiting_time': 0,
            'vehicles_completed': 0,
            'queue_length': 0,
            'step_count': 0
        }),
        'period_data': defaultdict(lambda: {
            'waiting_time': 0,
            'vehicles_completed': 0,
            'queue_length': 0,
            'step_count': 0
        }),
    }
    
    try:
        traci.start(['sumo', '-c', config_file, '--no-warnings', '--no-step-log'])
        
        step = 0
        while step < max_steps:
            traci.simulationStep()
            step += 1
            
            # Get current period
            period = get_time_period(step)
            hour = step // 3600
            
            # Collect metrics
            vehicle_ids = traci.vehicle.getIDList()
            waiting_time = sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids)
            queue_length = len([v for v in vehicle_ids if traci.vehicle.getSpeed(v) < 0.1])
            
            metrics['total_waiting_time'] += waiting_time
            metrics['hourly_data'][hour]['waiting_time'] += waiting_time
            metrics['hourly_data'][hour]['queue_length'] += queue_length
            metrics['hourly_data'][hour]['step_count'] += 1
            
            metrics['period_data'][period]['waiting_time'] += waiting_time
            metrics['period_data'][period]['queue_length'] += queue_length
            metrics['period_data'][period]['step_count'] += 1
            
            # Track completed vehicles
            departed = traci.simulation.getDepartedNumber()
            arrived = traci.simulation.getArrivedNumber()
            metrics['total_vehicles_entered'] += departed
            metrics['total_vehicles_completed'] += arrived
            metrics['hourly_data'][hour]['vehicles_completed'] += arrived
            metrics['period_data'][period]['vehicles_completed'] += arrived
            
            # Progress logging
            if step % 3600 == 0:
                sim_time = get_sim_time_str(step)
                print(f"  {sim_time}: Waiting={waiting_time:.0f}, Queue={queue_length}, "
                      f"Vehicles={len(vehicle_ids)}")
        
    finally:
        traci.close()
    
    # Calculate averages
    for hour, data in metrics['hourly_data'].items():
        if data['step_count'] > 0:
            data['avg_waiting_time'] = data['waiting_time'] / data['step_count']
            data['avg_queue_length'] = data['queue_length'] / data['step_count']
    
    for period, data in metrics['period_data'].items():
        if data['step_count'] > 0:
            data['avg_waiting_time'] = data['waiting_time'] / data['step_count']
            data['avg_queue_length'] = data['queue_length'] / data['step_count']
    
    metrics['avg_waiting_time'] = metrics['total_waiting_time'] / max_steps
    
    print(f"\nFixed-Time Results:")
    print(f"  Total waiting time: {metrics['total_waiting_time']:.0f}")
    print(f"  Average waiting time: {metrics['avg_waiting_time']:.2f}")
    print(f"  Vehicles completed: {metrics['total_vehicles_completed']}")
    
    return metrics


def run_mappo_simulation(checkpoint_path: str, config_file: str, max_steps: int = 43200) -> dict:
    """Run simulation with MAPPO control."""
    print("\n" + "=" * 60)
    print("Running MAPPO Control (12 hours)")
    print("=" * 60)
    
    import torch
    
    # Load checkpoint
    config = MAPPOConfig()
    config.STEPS_PER_EPISODE = max_steps
    config.SUMO_CONFIG = config_file
    config.USE_REALISTIC_24H_TRAFFIC = False
    
    env = K1Environment(config)
    
    agent = MAPPOAgent(config)
    
    # Handle zip files
    actual_path = checkpoint_path
    temp_dir = None
    
    if checkpoint_path.endswith('.zip') or Path(checkpoint_path + '.zip').exists():
        zip_path = checkpoint_path if checkpoint_path.endswith('.zip') else checkpoint_path + '.zip'
        if Path(zip_path).exists():
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)
            extracted_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
            actual_path = str(extracted_dirs[0]) if extracted_dirs else temp_dir
    
    agent.load_checkpoint(actual_path)
    print(f"  Loaded checkpoint: {checkpoint_path}")
    
    metrics = {
        'total_waiting_time': 0,
        'total_vehicles_completed': 0,
        'total_vehicles_entered': 0,
        'total_reward': 0,
        'hourly_data': defaultdict(lambda: {
            'waiting_time': 0,
            'vehicles_completed': 0,
            'queue_length': 0,
            'reward': 0,
            'step_count': 0
        }),
        'period_data': defaultdict(lambda: {
            'waiting_time': 0,
            'vehicles_completed': 0,
            'queue_length': 0,
            'reward': 0,
            'step_count': 0
        }),
    }
    
    try:
        local_states, global_state = env.reset()
        
        step = 0
        while step < max_steps:
            step += 1
            
            # Get actions from MAPPO
            with torch.no_grad():
                actions, _, _ = agent.select_actions(local_states, deterministic=True)
            
            # Environment step
            step_result = env.step(actions)
            if len(step_result) == 4:
                next_local_states, next_global_state, rewards, done = step_result
            else:
                next_local_states, next_global_state, rewards, done, _ = step_result
            
            step_reward = sum(rewards) / len(rewards)
            metrics['total_reward'] += step_reward
            
            # Get current period
            period = get_time_period(step)
            hour = step // 3600
            
            # Collect metrics via TraCI
            vehicle_ids = traci.vehicle.getIDList()
            waiting_time = sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids)
            queue_length = len([v for v in vehicle_ids if traci.vehicle.getSpeed(v) < 0.1])
            
            metrics['total_waiting_time'] += waiting_time
            metrics['hourly_data'][hour]['waiting_time'] += waiting_time
            metrics['hourly_data'][hour]['queue_length'] += queue_length
            metrics['hourly_data'][hour]['reward'] += step_reward
            metrics['hourly_data'][hour]['step_count'] += 1
            
            metrics['period_data'][period]['waiting_time'] += waiting_time
            metrics['period_data'][period]['queue_length'] += queue_length
            metrics['period_data'][period]['reward'] += step_reward
            metrics['period_data'][period]['step_count'] += 1
            
            # Track completed vehicles
            departed = traci.simulation.getDepartedNumber()
            arrived = traci.simulation.getArrivedNumber()
            metrics['total_vehicles_entered'] += departed
            metrics['total_vehicles_completed'] += arrived
            metrics['hourly_data'][hour]['vehicles_completed'] += arrived
            metrics['period_data'][period]['vehicles_completed'] += arrived
            
            # Progress logging
            if step % 3600 == 0:
                sim_time = get_sim_time_str(step)
                hour_reward = metrics['hourly_data'][hour]['reward']
                print(f"  {sim_time}: Waiting={waiting_time:.0f}, Queue={queue_length}, "
                      f"Reward={hour_reward:+.2f}")
            
            local_states = next_local_states
            global_state = next_global_state
            
            if done:
                break
    
    finally:
        env.close()
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Calculate averages
    for hour, data in metrics['hourly_data'].items():
        if data['step_count'] > 0:
            data['avg_waiting_time'] = data['waiting_time'] / data['step_count']
            data['avg_queue_length'] = data['queue_length'] / data['step_count']
    
    for period, data in metrics['period_data'].items():
        if data['step_count'] > 0:
            data['avg_waiting_time'] = data['waiting_time'] / data['step_count']
            data['avg_queue_length'] = data['queue_length'] / data['step_count']
    
    metrics['avg_waiting_time'] = metrics['total_waiting_time'] / max_steps
    
    print(f"\nMAPPO Results:")
    print(f"  Total waiting time: {metrics['total_waiting_time']:.0f}")
    print(f"  Average waiting time: {metrics['avg_waiting_time']:.2f}")
    print(f"  Total reward: {metrics['total_reward']:+.2f}")
    print(f"  Vehicles completed: {metrics['total_vehicles_completed']}")
    
    return metrics


def compare_results(fixed_metrics: dict, mappo_metrics: dict) -> dict:
    """Compare fixed-time and MAPPO metrics."""
    
    comparison = {
        'overall': {
            'fixed_waiting': fixed_metrics['avg_waiting_time'],
            'mappo_waiting': mappo_metrics['avg_waiting_time'],
            'improvement': (fixed_metrics['avg_waiting_time'] - mappo_metrics['avg_waiting_time']) / 
                          fixed_metrics['avg_waiting_time'] * 100 if fixed_metrics['avg_waiting_time'] > 0 else 0,
            'fixed_completed': fixed_metrics['total_vehicles_completed'],
            'mappo_completed': mappo_metrics['total_vehicles_completed'],
        },
        'by_period': {},
        'by_hour': {},
    }
    
    # Compare by period
    for period in TIME_PERIODS.keys():
        fixed_data = fixed_metrics['period_data'].get(period, {})
        mappo_data = mappo_metrics['period_data'].get(period, {})
        
        fixed_avg = fixed_data.get('avg_waiting_time', 0)
        mappo_avg = mappo_data.get('avg_waiting_time', 0)
        
        comparison['by_period'][period] = {
            'fixed_waiting': fixed_avg,
            'mappo_waiting': mappo_avg,
            'improvement': (fixed_avg - mappo_avg) / fixed_avg * 100 if fixed_avg > 0 else 0,
            'time_range': TIME_PERIODS[period][2],
        }
    
    # Compare by hour
    for hour in range(12):
        fixed_data = fixed_metrics['hourly_data'].get(hour, {})
        mappo_data = mappo_metrics['hourly_data'].get(hour, {})
        
        fixed_avg = fixed_data.get('avg_waiting_time', 0)
        mappo_avg = mappo_data.get('avg_waiting_time', 0)
        
        comparison['by_hour'][hour] = {
            'fixed_waiting': fixed_avg,
            'mappo_waiting': mappo_avg,
            'improvement': (fixed_avg - mappo_avg) / fixed_avg * 100 if fixed_avg > 0 else 0,
            'sim_time': f"{6+hour:02d}:00-{7+hour:02d}:00",
        }
    
    return comparison


def print_comparison_report(comparison: dict, fixed_metrics: dict, mappo_metrics: dict):
    """Print detailed comparison report."""
    
    print("\n" + "=" * 70)
    print("12-HOUR PERFORMANCE COMPARISON: Fixed-Time vs MAPPO")
    print("=" * 70)
    
    # Overall summary
    print("\n📊 OVERALL SUMMARY")
    print("-" * 50)
    overall = comparison['overall']
    print(f"  Average Waiting Time:")
    print(f"    Fixed-Time: {overall['fixed_waiting']:.2f} sec")
    print(f"    MAPPO:      {overall['mappo_waiting']:.2f} sec")
    print(f"    Improvement: {overall['improvement']:+.1f}%")
    print(f"\n  Vehicles Completed:")
    print(f"    Fixed-Time: {overall['fixed_completed']}")
    print(f"    MAPPO:      {overall['mappo_completed']}")
    
    # By time period
    print("\n📈 PERFORMANCE BY TIME PERIOD")
    print("-" * 50)
    print(f"{'Period':<20} {'Time':<12} {'Fixed':>10} {'MAPPO':>10} {'Improv':>10}")
    print("-" * 62)
    
    for period, data in comparison['by_period'].items():
        print(f"{period:<20} {data['time_range']:<12} "
              f"{data['fixed_waiting']:>10.2f} {data['mappo_waiting']:>10.2f} "
              f"{data['improvement']:>+9.1f}%")
    
    # By hour
    print("\n📈 PERFORMANCE BY HOUR")
    print("-" * 50)
    print(f"{'Hour':<15} {'Fixed Wait':>12} {'MAPPO Wait':>12} {'Improvement':>12}")
    print("-" * 51)
    
    for hour, data in sorted(comparison['by_hour'].items()):
        print(f"{data['sim_time']:<15} "
              f"{data['fixed_waiting']:>12.2f} {data['mappo_waiting']:>12.2f} "
              f"{data['improvement']:>+11.1f}%")
    
    # Rush hour analysis
    print("\n🚗 RUSH HOUR ANALYSIS")
    print("-" * 50)
    
    morning_rush = comparison['by_period'].get('morning_peak', {})
    late_afternoon = comparison['by_period'].get('late_afternoon', {})
    
    print(f"  Morning Peak (8-9am):")
    print(f"    Fixed: {morning_rush.get('fixed_waiting', 0):.2f} sec")
    print(f"    MAPPO: {morning_rush.get('mappo_waiting', 0):.2f} sec")
    print(f"    Improvement: {morning_rush.get('improvement', 0):+.1f}%")
    
    print(f"\n  Late Afternoon (4-6pm):")
    print(f"    Fixed: {late_afternoon.get('fixed_waiting', 0):.2f} sec")
    print(f"    MAPPO: {late_afternoon.get('mappo_waiting', 0):.2f} sec")
    print(f"    Improvement: {late_afternoon.get('improvement', 0):+.1f}%")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if overall['improvement'] > 10:
        print(f"✅ MAPPO significantly outperforms Fixed-Time by {overall['improvement']:.1f}%")
    elif overall['improvement'] > 0:
        print(f"✅ MAPPO slightly outperforms Fixed-Time by {overall['improvement']:.1f}%")
    elif overall['improvement'] > -10:
        print(f"⚠️ MAPPO performs similarly to Fixed-Time ({overall['improvement']:.1f}%)")
    else:
        print(f"❌ Fixed-Time outperforms MAPPO by {-overall['improvement']:.1f}%")


def save_results(comparison: dict, fixed_metrics: dict, mappo_metrics: dict, 
                 output_dir: Path, scenario: str):
    """Save results to files."""
    
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    results = {
        'scenario': scenario,
        'duration': '12_hours',
        'timestamp': datetime.now().isoformat(),
        'comparison': comparison,
        'fixed_metrics': {
            'avg_waiting_time': fixed_metrics['avg_waiting_time'],
            'total_vehicles_completed': fixed_metrics['total_vehicles_completed'],
        },
        'mappo_metrics': {
            'avg_waiting_time': mappo_metrics['avg_waiting_time'],
            'total_reward': mappo_metrics['total_reward'],
            'total_vehicles_completed': mappo_metrics['total_vehicles_completed'],
        },
    }
    
    results_file = output_dir / f'evaluation_12h_{scenario}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    return results_file


def main():
    parser = argparse.ArgumentParser(description='Evaluate MAPPO vs Fixed-Time for 12-hour scenarios')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to MAPPO checkpoint')
    parser.add_argument('--scenario', type=str, default='weekday',
                        choices=['weekday', 'weekend', 'friday', 'event'],
                        help='Traffic scenario')
    parser.add_argument('--quick', action='store_true',
                        help='Quick evaluation (first 3 hours only)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_12h',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup paths
    config_file = str(SCRIPT_DIR / f'k1_12h_{args.scenario}.sumocfg')
    route_file = SCRIPT_DIR / f'k1_routes_12h_{args.scenario}.rou.xml'
    
    # Check if route file exists
    if not route_file.exists():
        print(f"Route file not found: {route_file}")
        print("Generating routes...")
        import subprocess
        subprocess.run([sys.executable, str(SCRIPT_DIR / 'generate_12h_routes.py'), 
                       '--scenario', args.scenario])
    
    # Set max steps
    max_steps = 10800 if args.quick else 43200  # 3 hours or 12 hours
    
    print("\n" + "=" * 70)
    print("12-Hour Performance Evaluation")
    print("=" * 70)
    print(f"Scenario: {args.scenario}")
    print(f"Duration: {'3 hours (quick mode)' if args.quick else '12 hours'}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {config_file}")
    
    # Run fixed-time simulation
    fixed_metrics = run_fixed_time_simulation(config_file, max_steps)
    
    # Run MAPPO simulation
    mappo_metrics = run_mappo_simulation(args.checkpoint, config_file, max_steps)
    
    # Compare results
    comparison = compare_results(fixed_metrics, mappo_metrics)
    
    # Print report
    print_comparison_report(comparison, fixed_metrics, mappo_metrics)
    
    # Save results
    output_dir = SCRIPT_DIR / args.output_dir
    save_results(comparison, fixed_metrics, mappo_metrics, output_dir, args.scenario)


if __name__ == '__main__':
    main()
