#!/usr/bin/env python3
"""
Comprehensive Evaluation: Fixed-Time vs MAPPO for 24-Hour Scenarios
=====================================================================

This script performs a detailed comparison between Fixed-Time and MAPPO
traffic control over full 24-hour periods, providing:

1. Side-by-side simulation runs
2. Hourly performance breakdown
3. Rush hour vs off-peak analysis
4. Time-of-day adaptation metrics
5. Detailed visualizations
6. Statistical analysis with confidence intervals

Key Metrics:
- Total/Average waiting time
- Network throughput
- Queue lengths
- Vehicle completion rate
- Rush hour performance
- Adaptation speed

Usage:
    python evaluate_24h_performance.py --checkpoint checkpoint_24h_weekday_final --scenario weekday
    python evaluate_24h_performance.py --checkpoint checkpoint_24h_weekday_final --scenario weekday --quick
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
import csv

import numpy as np

# Add parent directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import traci
except ImportError:
    print("Error: SUMO TraCI not found. Please install SUMO and set SUMO_HOME.")
    sys.exit(1)

# Import MAPPO components
from mappo_k1_implementation import MAPPOConfig, MAPPOAgent

# Try matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

JUNCTION_IDS = ["J0", "J1", "J5", "J6", "J7", "J10", "J11", "J12", "J22"]

SCENARIOS_24H = {
    'weekday': 'k1_24h_weekday.sumocfg',
    'weekend': 'k1_24h_weekend.sumocfg',
    'friday': 'k1_24h_friday.sumocfg',
    'event': 'k1_24h_event.sumocfg',
}

# Time periods for analysis (matching route generation)
TIME_PERIODS = [
    (0,     21600, 'Night (12am-6am)'),
    (21600, 32400, 'Morning Rush (6am-9am)'),
    (32400, 50400, 'Midday (9am-2pm)'),
    (50400, 61200, 'Afternoon (2pm-5pm)'),
    (61200, 72000, 'Evening Rush (5pm-8pm)'),
    (72000, 86400, 'Late Night (8pm-12am)'),
]

# Quick mode: Run only 6 hours (first period + morning rush)
QUICK_MODE_DURATION = 32400  # 9 hours


# =============================================================================
# FIXED-TIME CONTROLLER
# =============================================================================

class FixedTimeController24H:
    """
    Time-adaptive fixed-time controller for 24-hour scenarios.
    
    Uses different phase timings for different times of day:
    - Night: Longer cycles, favor main roads
    - Rush hour: Shorter cycles, balanced timing
    - Off-peak: Standard timing
    """
    
    def __init__(self, junction_ids):
        self.junction_ids = junction_ids
        self.current_phases = {}
        self.time_in_phase = {}
        self.num_phases = {}
        self.current_step = 0
    
    def get_phase_duration(self, step):
        """
        Get phase duration based on time of day.
        
        Returns appropriate cycle time for current simulation time.
        """
        hour = (step // 3600) % 24
        
        if 0 <= hour < 6:
            # Night: longer cycles
            return 45
        elif 6 <= hour < 9 or 17 <= hour < 20:
            # Rush hours: shorter, more responsive
            return 25
        else:
            # Standard daytime
            return 30
    
    def initialize(self):
        """Initialize controller after SUMO starts."""
        for jid in self.junction_ids:
            try:
                programs = traci.trafficlight.getAllProgramLogics(jid)
                if programs:
                    self.num_phases[jid] = len(programs[0].phases)
                else:
                    self.num_phases[jid] = 4
            except Exception:
                self.num_phases[jid] = 4
            
            self.current_phases[jid] = 0
            self.time_in_phase[jid] = 0
    
    def step(self):
        """Execute one step of time-adaptive fixed-time control."""
        phase_duration = self.get_phase_duration(self.current_step)
        
        for jid in self.junction_ids:
            self.time_in_phase[jid] += 1
            
            if self.time_in_phase[jid] >= phase_duration:
                next_phase = (self.current_phases[jid] + 1) % self.num_phases[jid]
                try:
                    traci.trafficlight.setPhase(jid, next_phase)
                    self.current_phases[jid] = next_phase
                    self.time_in_phase[jid] = 0
                except Exception:
                    pass
        
        self.current_step += 1


# =============================================================================
# MAPPO CONTROLLER
# =============================================================================

class MAPPOController24H:
    """
    MAPPO controller for 24-hour evaluation.
    """
    
    def __init__(self, checkpoint_path, device='cpu'):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.agent = None
        self.junction_ids = JUNCTION_IDS
        self.current_phases = {}
        self.time_in_phase = {}
        self.neighbors = MAPPOConfig.NEIGHBORS
        self.vehicle_weights = MAPPOConfig.VEHICLE_WEIGHTS
    
    def initialize(self):
        """Load checkpoint and initialize."""
        config = MAPPOConfig()
        config.DEVICE = self.device
        self.agent = MAPPOAgent(config)
        
        print(f"Loading checkpoint: {self.checkpoint_path}")
        self.agent.load_checkpoint(
            str(self.checkpoint_path),
            load_buffer=False,
            load_optimizer=False
        )
        self.agent.epsilon = 0.0
        
        for jid in self.junction_ids:
            self.current_phases[jid] = 0
            self.time_in_phase[jid] = 0
    
    def get_local_state(self, junction_id):
        """Get local state for a junction."""
        state = []
        state.append(self.current_phases[junction_id])
        
        incoming_lanes = traci.trafficlight.getControlledLanes(junction_id)
        
        queues = {'n': 0, 's': 0, 'e': 0, 'w': 0}
        weighted_vehicles = {'n': 0, 's': 0, 'e': 0, 'w': 0}
        occupancy = {'n': 0, 's': 0, 'e': 0, 'w': 0}
        
        for lane_id in incoming_lanes:
            if '_n' in lane_id or 'n_' in lane_id:
                direction = 'n'
            elif '_s' in lane_id or 's_' in lane_id:
                direction = 's'
            elif '_e' in lane_id or 'e_' in lane_id:
                direction = 'e'
            elif '_w' in lane_id or 'w_' in lane_id:
                direction = 'w'
            else:
                continue
            
            queues[direction] += traci.lane.getLastStepHaltingNumber(lane_id)
            
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                veh_type = traci.vehicle.getTypeID(veh_id)
                base_type = veh_type.split('_')[0].lower()
                weighted_vehicles[direction] += self.vehicle_weights.get(base_type, 1.0)
            
            occupancy[direction] = max(occupancy[direction], 
                                       traci.lane.getLastStepOccupancy(lane_id) / 100.0)
        
        for d in ['n', 's', 'e', 'w']:
            state.append(queues[d])
        for d in ['n', 's', 'e', 'w']:
            state.append(weighted_vehicles[d])
        for d in ['n', 's', 'e', 'w']:
            state.append(occupancy[d])
        
        state.append(self.time_in_phase[junction_id])
        
        neighbors = self.neighbors[junction_id]
        for neighbor_id in neighbors[:2]:
            state.append(self.current_phases.get(neighbor_id, 0))
        
        while len(state) < 16:
            state.append(0)
        
        return np.array(state[:16], dtype=np.float32)
    
    def step(self):
        """Execute one step using MAPPO policy."""
        local_states = [self.get_local_state(jid) for jid in self.junction_ids]
        actions, _, _ = self.agent.select_actions(local_states)
        
        for i, jid in enumerate(self.junction_ids):
            action = actions[i]
            
            if action == 1:  # Switch phase
                try:
                    programs = traci.trafficlight.getAllProgramLogics(jid)
                    num_phases = len(programs[0].phases) if programs else 4
                    next_phase = (self.current_phases[jid] + 1) % num_phases
                    traci.trafficlight.setPhase(jid, next_phase)
                    self.current_phases[jid] = next_phase
                    self.time_in_phase[jid] = 0
                except Exception:
                    pass
            
            self.time_in_phase[jid] += 1


# =============================================================================
# METRICS COLLECTION
# =============================================================================

def collect_metrics(junction_ids):
    """Collect traffic metrics from SUMO."""
    total_waiting = 0
    total_queue = 0
    lane_count = 0
    
    for jid in junction_ids:
        try:
            lanes = traci.trafficlight.getControlledLanes(jid)
            for lane_id in lanes:
                total_waiting += traci.lane.getWaitingTime(lane_id)
                total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
                lane_count += 1
        except Exception:
            pass
    
    return {
        'total_waiting_time': total_waiting,
        'total_vehicles': traci.vehicle.getIDCount(),
        'total_queue': total_queue,
        'avg_queue': total_queue / max(1, lane_count),
        'arrived': traci.simulation.getArrivedNumber(),
        'departed': traci.simulation.getDepartedNumber(),
    }


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_24h_simulation(sumo_config, controller, duration, seed=42):
    """
    Run a full 24-hour simulation.
    
    Returns detailed metrics including hourly breakdowns.
    """
    sumo_binary = shutil.which('sumo') or 'sumo'
    
    sumo_cmd = [
        sumo_binary,
        '-c', str(sumo_config),
        '--start',
        '--quit-on-end',
        '--no-warnings',
        '--no-step-log',
        '--seed', str(seed),
    ]
    
    print(f"  Starting SUMO simulation (duration: {duration}s = {duration/3600:.1f}h)...")
    traci.start(sumo_cmd)
    controller.initialize()
    
    # Metrics storage
    metrics = {
        'step': [],
        'total_waiting_time': [],
        'total_vehicles': [],
        'avg_queue': [],
        'cumulative_arrived': [],
    }
    
    # Hourly metrics
    hourly_metrics = defaultdict(lambda: {
        'waiting_times': [],
        'queues': [],
        'throughput': 0,
    })
    
    cumulative_arrived = 0
    step = 0
    start_time = datetime.now()
    
    try:
        while step < duration:
            controller.step()
            traci.simulationStep()
            
            m = collect_metrics(JUNCTION_IDS)
            cumulative_arrived += m['arrived']
            
            metrics['step'].append(step)
            metrics['total_waiting_time'].append(m['total_waiting_time'])
            metrics['total_vehicles'].append(m['total_vehicles'])
            metrics['avg_queue'].append(m['avg_queue'])
            metrics['cumulative_arrived'].append(cumulative_arrived)
            
            # Hourly tracking
            hour = step // 3600
            hourly_metrics[hour]['waiting_times'].append(m['total_waiting_time'])
            hourly_metrics[hour]['queues'].append(m['avg_queue'])
            hourly_metrics[hour]['throughput'] += m['arrived']
            
            # Progress every hour
            if step % 3600 == 0 and step > 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                eta = (elapsed / step) * (duration - step)
                print(f"    Hour {step//3600:2d}/{duration//3600}: "
                      f"Waiting={m['total_waiting_time']:.0f}s, "
                      f"Queue={m['avg_queue']:.1f}, "
                      f"Vehicles={m['total_vehicles']}, "
                      f"ETA={eta/60:.1f}min")
            
            step += 1
            
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"  Simulation ended at step {step}")
                break
                
    except Exception as e:
        print(f"  Error at step {step}: {e}")
    finally:
        if traci.isLoaded():
            traci.close()
    
    # Process hourly metrics
    processed_hourly = {}
    for hour, data in hourly_metrics.items():
        processed_hourly[hour] = {
            'avg_waiting': np.mean(data['waiting_times']) if data['waiting_times'] else 0,
            'avg_queue': np.mean(data['queues']) if data['queues'] else 0,
            'throughput': data['throughput'],
        }
    
    metrics['hourly'] = processed_hourly
    return metrics


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def analyze_period_performance(metrics, period_start, period_end):
    """Analyze performance for a specific time period."""
    steps = np.array(metrics['step'])
    mask = (steps >= period_start) & (steps < period_end)
    
    return {
        'avg_waiting': np.mean(np.array(metrics['total_waiting_time'])[mask]),
        'avg_queue': np.mean(np.array(metrics['avg_queue'])[mask]),
        'throughput': np.array(metrics['cumulative_arrived'])[mask][-1] - 
                      (np.array(metrics['cumulative_arrived'])[mask][0] if len(np.array(metrics['cumulative_arrived'])[mask]) > 0 else 0),
    }


def generate_24h_plots(fixed_metrics, mappo_metrics, output_dir):
    """Generate comprehensive 24-hour comparison plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Waiting Time Over 24 Hours
    ax1 = fig.add_subplot(3, 2, 1)
    steps_hours = np.array(fixed_metrics['step']) / 3600
    ax1.plot(steps_hours, fixed_metrics['total_waiting_time'], 'r-', alpha=0.6, label='Fixed-Time')
    ax1.plot(steps_hours, mappo_metrics['total_waiting_time'], 'b-', alpha=0.6, label='MAPPO')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Total Waiting Time (s)')
    ax1.set_title('Waiting Time Over 24 Hours')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(6, 9, alpha=0.2, color='orange', label='Morning Rush')
    ax1.axvspan(17, 20, alpha=0.2, color='red', label='Evening Rush')
    
    # 2. Average Queue Length
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(steps_hours, fixed_metrics['avg_queue'], 'r-', alpha=0.6, label='Fixed-Time')
    ax2.plot(steps_hours, mappo_metrics['avg_queue'], 'b-', alpha=0.6, label='MAPPO')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Avg Queue Length')
    ax2.set_title('Queue Length Over 24 Hours')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(6, 9, alpha=0.2, color='orange')
    ax2.axvspan(17, 20, alpha=0.2, color='red')
    
    # 3. Cumulative Throughput
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(steps_hours, fixed_metrics['cumulative_arrived'], 'r-', label='Fixed-Time')
    ax3.plot(steps_hours, mappo_metrics['cumulative_arrived'], 'b-', label='MAPPO')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Cumulative Vehicles Arrived')
    ax3.set_title('Cumulative Throughput')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Vehicles in Network
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(steps_hours, fixed_metrics['total_vehicles'], 'r-', alpha=0.6, label='Fixed-Time')
    ax4.plot(steps_hours, mappo_metrics['total_vehicles'], 'b-', alpha=0.6, label='MAPPO')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Active Vehicles')
    ax4.set_title('Network Load Over 24 Hours')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Hourly Performance Bar Chart
    ax5 = fig.add_subplot(3, 2, 5)
    hours = list(range(24))
    fixed_hourly_wait = [fixed_metrics['hourly'].get(h, {}).get('avg_waiting', 0) for h in hours]
    mappo_hourly_wait = [mappo_metrics['hourly'].get(h, {}).get('avg_waiting', 0) for h in hours]
    
    x = np.arange(len(hours))
    width = 0.35
    ax5.bar(x - width/2, fixed_hourly_wait, width, label='Fixed-Time', color='red', alpha=0.7)
    ax5.bar(x + width/2, mappo_hourly_wait, width, label='MAPPO', color='blue', alpha=0.7)
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Avg Waiting Time')
    ax5.set_title('Hourly Average Waiting Time')
    ax5.set_xticks(x[::2])
    ax5.set_xticklabels([f'{h}:00' for h in hours[::2]], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Period Comparison
    ax6 = fig.add_subplot(3, 2, 6)
    periods = ['Night\n(12am-6am)', 'Morning Rush\n(6am-9am)', 'Midday\n(9am-2pm)', 
               'Afternoon\n(2pm-5pm)', 'Evening Rush\n(5pm-8pm)', 'Late Night\n(8pm-12am)']
    
    fixed_period_wait = []
    mappo_period_wait = []
    
    for start, end, _ in TIME_PERIODS:
        fixed_period_wait.append(analyze_period_performance(fixed_metrics, start, end)['avg_waiting'])
        mappo_period_wait.append(analyze_period_performance(mappo_metrics, start, end)['avg_waiting'])
    
    x = np.arange(len(periods))
    ax6.bar(x - width/2, fixed_period_wait, width, label='Fixed-Time', color='red', alpha=0.7)
    ax6.bar(x + width/2, mappo_period_wait, width, label='MAPPO', color='blue', alpha=0.7)
    ax6.set_xlabel('Time Period')
    ax6.set_ylabel('Avg Waiting Time')
    ax6.set_title('Performance by Time Period')
    ax6.set_xticks(x)
    ax6.set_xticklabels(periods, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_24h_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots to: {output_dir / 'evaluation_24h_comparison.png'}")


def save_24h_results(fixed_metrics, mappo_metrics, output_dir, args):
    """Save detailed 24-hour evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_len = min(len(fixed_metrics['step']), len(mappo_metrics['step']))
    
    # Save detailed CSV
    csv_path = output_dir / 'metrics_24h.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'hour', 'fixed_waiting', 'fixed_queue', 'fixed_throughput',
                        'mappo_waiting', 'mappo_queue', 'mappo_throughput'])
        
        for i in range(min_len):
            writer.writerow([
                fixed_metrics['step'][i],
                fixed_metrics['step'][i] // 3600,
                fixed_metrics['total_waiting_time'][i],
                fixed_metrics['avg_queue'][i],
                fixed_metrics['cumulative_arrived'][i],
                mappo_metrics['total_waiting_time'][i],
                mappo_metrics['avg_queue'][i],
                mappo_metrics['cumulative_arrived'][i],
            ])
    
    # Calculate overall statistics
    fixed_avg_wait = np.mean(fixed_metrics['total_waiting_time'][:min_len])
    mappo_avg_wait = np.mean(mappo_metrics['total_waiting_time'][:min_len])
    fixed_avg_queue = np.mean(fixed_metrics['avg_queue'][:min_len])
    mappo_avg_queue = np.mean(mappo_metrics['avg_queue'][:min_len])
    fixed_throughput = fixed_metrics['cumulative_arrived'][min_len-1]
    mappo_throughput = mappo_metrics['cumulative_arrived'][min_len-1]
    
    # Period-specific analysis
    period_results = {}
    for start, end, name in TIME_PERIODS:
        fixed_period = analyze_period_performance(fixed_metrics, start, end)
        mappo_period = analyze_period_performance(mappo_metrics, start, end)
        
        wait_improvement = (fixed_period['avg_waiting'] - mappo_period['avg_waiting']) / max(1, fixed_period['avg_waiting']) * 100
        
        period_results[name] = {
            'fixed': fixed_period,
            'mappo': mappo_period,
            'waiting_time_improvement_pct': round(wait_improvement, 2),
        }
    
    # Rush hour specific analysis
    morning_rush_improvement = period_results['Morning Rush (6am-9am)']['waiting_time_improvement_pct']
    evening_rush_improvement = period_results['Evening Rush (5pm-8pm)']['waiting_time_improvement_pct']
    
    # Build summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'scenario': args.scenario,
        'duration_hours': args.duration / 3600,
        'checkpoint': args.checkpoint,
        'overall_results': {
            'fixed_time': {
                'avg_waiting_time': round(fixed_avg_wait, 2),
                'avg_queue_length': round(fixed_avg_queue, 2),
                'total_throughput': fixed_throughput,
            },
            'mappo': {
                'avg_waiting_time': round(mappo_avg_wait, 2),
                'avg_queue_length': round(mappo_avg_queue, 2),
                'total_throughput': mappo_throughput,
            },
            'improvement': {
                'waiting_time_reduction_pct': round((fixed_avg_wait - mappo_avg_wait) / max(1, fixed_avg_wait) * 100, 2),
                'queue_reduction_pct': round((fixed_avg_queue - mappo_avg_queue) / max(0.01, fixed_avg_queue) * 100, 2),
                'throughput_increase_pct': round((mappo_throughput - fixed_throughput) / max(1, fixed_throughput) * 100, 2),
            }
        },
        'period_analysis': period_results,
        'rush_hour_analysis': {
            'morning_rush_improvement_pct': morning_rush_improvement,
            'evening_rush_improvement_pct': evening_rush_improvement,
            'avg_rush_hour_improvement_pct': round((morning_rush_improvement + evening_rush_improvement) / 2, 2),
        }
    }
    
    # Save JSON
    json_path = output_dir / 'summary_24h.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print detailed summary
    print("\n" + "="*80)
    print("24-HOUR EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nScenario: {args.scenario}")
    print(f"Duration: {args.duration/3600:.1f} hours")
    print(f"Checkpoint: {args.checkpoint}")
    
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"\n{'Metric':<30} {'Fixed-Time':>15} {'MAPPO':>15} {'Improvement':>15}")
    print("-"*80)
    print(f"{'Avg Waiting Time (s)':<30} {fixed_avg_wait:>15.2f} {mappo_avg_wait:>15.2f} "
          f"{summary['overall_results']['improvement']['waiting_time_reduction_pct']:>14.1f}%")
    print(f"{'Avg Queue Length':<30} {fixed_avg_queue:>15.2f} {mappo_avg_queue:>15.2f} "
          f"{summary['overall_results']['improvement']['queue_reduction_pct']:>14.1f}%")
    print(f"{'Total Throughput':<30} {fixed_throughput:>15} {mappo_throughput:>15} "
          f"{summary['overall_results']['improvement']['throughput_increase_pct']:>14.1f}%")
    
    print(f"\n{'='*80}")
    print("TIME PERIOD ANALYSIS")
    print(f"{'='*80}")
    print(f"\n{'Period':<25} {'Fixed Wait':>12} {'MAPPO Wait':>12} {'Improvement':>12}")
    print("-"*65)
    
    for name, data in period_results.items():
        print(f"{name:<25} {data['fixed']['avg_waiting']:>12.1f} {data['mappo']['avg_waiting']:>12.1f} "
              f"{data['waiting_time_improvement_pct']:>11.1f}%")
    
    print(f"\n{'='*80}")
    print("RUSH HOUR PERFORMANCE")
    print(f"{'='*80}")
    print(f"\nMorning Rush (6am-9am): {morning_rush_improvement:+.1f}% waiting time reduction")
    print(f"Evening Rush (5pm-8pm): {evening_rush_improvement:+.1f}% waiting time reduction")
    print(f"Average Rush Hour Improvement: {summary['rush_hour_analysis']['avg_rush_hour_improvement_pct']:+.1f}%")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Fixed-Time vs MAPPO on 24-hour scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_24h_performance.py --checkpoint checkpoint_24h_weekday_final --scenario weekday
  python evaluate_24h_performance.py --checkpoint checkpoint.zip --scenario weekday --quick
  python evaluate_24h_performance.py --checkpoint checkpoint_24h_weekday_final --scenario event
"""
    )
    
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to MAPPO checkpoint')
    parser.add_argument('--scenario', '-s', type=str, default='weekday',
                       choices=['weekday', 'weekend', 'friday', 'event'],
                       help='Traffic scenario (default: weekday)')
    parser.add_argument('--duration', '-d', type=int, default=86400,
                       help='Duration in seconds (default: 86400 = 24h)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick mode: only 9 hours (morning rush)')
    parser.add_argument('--output', '-o', type=str, default='evaluation_24h_results',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.duration = QUICK_MODE_DURATION
        print("Quick mode: Running 9-hour evaluation (includes morning rush)")
    
    print("="*80)
    print("24-HOUR TRAFFIC CONTROL EVALUATION")
    print("="*80)
    
    # Resolve paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = SCRIPT_DIR / checkpoint_path
    
    # Handle zip files
    temp_dir = None
    if checkpoint_path.suffix == '.zip':
        print(f"\nExtracting checkpoint from: {checkpoint_path}")
        temp_dir = tempfile.mkdtemp(prefix='mappo_24h_')
        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
            zf.extractall(temp_dir)
        extracted = list(Path(temp_dir).iterdir())
        checkpoint_path = extracted[0] if len(extracted) == 1 and extracted[0].is_dir() else Path(temp_dir)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Get SUMO config
    sumo_config = SCRIPT_DIR / SCENARIOS_24H.get(args.scenario, f'k1_24h_{args.scenario}.sumocfg')
    if not sumo_config.exists():
        print(f"Error: SUMO config not found: {sumo_config}")
        print(f"Run: python generate_24h_routes.py --scenario {args.scenario}")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Duration: {args.duration}s ({args.duration/3600:.1f}h)")
    print(f"  Output: {args.output}")
    
    # Run Fixed-Time
    print("\n" + "="*60)
    print("RUNNING FIXED-TIME BASELINE")
    print("="*60)
    
    fixed_controller = FixedTimeController24H(JUNCTION_IDS)
    fixed_metrics = run_24h_simulation(sumo_config, fixed_controller, args.duration, args.seed)
    print(f"  Fixed-Time completed: {len(fixed_metrics['step'])} steps")
    
    # Run MAPPO
    print("\n" + "="*60)
    print("RUNNING MAPPO EVALUATION")
    print("="*60)
    
    mappo_controller = MAPPOController24H(checkpoint_path)
    mappo_metrics = run_24h_simulation(sumo_config, mappo_controller, args.duration, args.seed)
    print(f"  MAPPO completed: {len(mappo_metrics['step'])} steps")
    
    # Save and visualize results
    output_dir = SCRIPT_DIR / args.output
    save_24h_results(fixed_metrics, mappo_metrics, output_dir, args)
    generate_24h_plots(fixed_metrics, mappo_metrics, output_dir)
    
    # Cleanup
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
