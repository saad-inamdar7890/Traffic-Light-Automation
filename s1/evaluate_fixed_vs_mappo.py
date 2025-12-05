#!/usr/bin/env python3
"""
Evaluate Fixed-Time vs MAPPO Traffic Light Control
===================================================

This script runs two back-to-back simulations on the same 6-hour traffic scenario:
1. Fixed-Time Baseline: Cycles phases with a constant green duration
2. MAPPO Policy: Uses trained actors from a checkpoint

Outputs:
- CSV with per-step metrics for both controllers
- Comparison plots (waiting time, throughput, queue length)
- JSON summary with aggregate statistics

Usage:
    python evaluate_fixed_vs_mappo.py --checkpoint checkpoint_time_20251129_103621 --scenario weekday
    python evaluate_fixed_vs_mappo.py --checkpoint checkpoint_time_20251129_103621 --scenario weekday --duration 3600
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
import traci

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import MAPPO components
from mappo_k1_implementation import MAPPOConfig, MAPPOAgent

# Try to import matplotlib (optional for headless environments)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will not be generated.")


# =============================================================================
# CONFIGURATION
# =============================================================================

JUNCTION_IDS = ["J0", "J1", "J5", "J6", "J7", "J10", "J11", "J12", "J22"]

SCENARIOS = {
    'weekday': 'k1_6h_weekday.sumocfg',
    'weekend': 'k1_6h_weekend.sumocfg',
    'event': 'k1_6h_event.sumocfg',
    '3h': 'k1_3h_varying.sumocfg',
    'realistic': 'k1_6h_evaluation.sumocfg',  # Uses validated 24h_realistic routes
}

# Default fixed phase duration (seconds per phase)
DEFAULT_FIXED_PHASE_DURATION = 30


# =============================================================================
# FIXED-TIME CONTROLLER
# =============================================================================

class FixedTimeController:
    """
    Simple fixed-time traffic light controller.
    
    Cycles through phases with a constant duration for each phase.
    """
    
    def __init__(self, junction_ids, phase_duration=30):
        """
        Args:
            junction_ids: List of junction IDs to control
            phase_duration: Seconds to stay in each phase before switching
        """
        self.junction_ids = junction_ids
        self.phase_duration = phase_duration
        
        # Track current phase and time in phase for each junction
        self.current_phases = {}
        self.time_in_phase = {}
        self.num_phases = {}
    
    def initialize(self):
        """Initialize controller after SUMO starts."""
        for jid in self.junction_ids:
            try:
                programs = traci.trafficlight.getAllProgramLogics(jid)
                if programs:
                    self.num_phases[jid] = len(programs[0].phases)
                else:
                    self.num_phases[jid] = 4  # Default
            except Exception:
                self.num_phases[jid] = 4
            
            self.current_phases[jid] = 0
            self.time_in_phase[jid] = 0
    
    def step(self):
        """
        Execute one step of fixed-time control.
        
        Increments time counters and switches phases when duration exceeded.
        """
        for jid in self.junction_ids:
            self.time_in_phase[jid] += 1
            
            # Switch phase if duration exceeded
            if self.time_in_phase[jid] >= self.phase_duration:
                next_phase = (self.current_phases[jid] + 1) % self.num_phases[jid]
                try:
                    traci.trafficlight.setPhase(jid, next_phase)
                    self.current_phases[jid] = next_phase
                    self.time_in_phase[jid] = 0
                except Exception:
                    pass  # Ignore errors


# =============================================================================
# MAPPO CONTROLLER (using trained checkpoint)
# =============================================================================

class MAPPOController:
    """
    MAPPO-based traffic light controller using trained actors.
    """
    
    def __init__(self, checkpoint_path, device='cpu'):
        """
        Args:
            checkpoint_path: Path to checkpoint directory with actor weights
            device: 'cpu' or 'cuda'
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.agent = None
        self.junction_ids = JUNCTION_IDS
        
        # State tracking (mirrors K1Environment)
        self.current_phases = {}
        self.time_in_phase = {}
        self.neighbors = MAPPOConfig.NEIGHBORS
        self.vehicle_weights = MAPPOConfig.VEHICLE_WEIGHTS
    
    def initialize(self):
        """Load checkpoint and initialize controller."""
        # Create agent
        config = MAPPOConfig()
        config.DEVICE = self.device
        self.agent = MAPPOAgent(config)
        
        # Load checkpoint
        print(f"Loading MAPPO checkpoint from: {self.checkpoint_path}")
        self.agent.load_checkpoint(
            str(self.checkpoint_path),
            load_buffer=False,
            load_optimizer=False
        )
        
        # Disable exploration
        self.agent.epsilon = 0.0
        
        # Initialize phase tracking
        for jid in self.junction_ids:
            self.current_phases[jid] = 0
            self.time_in_phase[jid] = 0
    
    def get_local_state(self, junction_id):
        """Get local state for a junction (same as K1Environment)."""
        state = []
        
        # 1. Current phase
        state.append(self.current_phases[junction_id])
        
        # 2. Get incoming lanes
        incoming_lanes = traci.trafficlight.getControlledLanes(junction_id)
        
        queues = {'n': 0, 's': 0, 'e': 0, 'w': 0}
        weighted_vehicles = {'n': 0, 's': 0, 'e': 0, 'w': 0}
        occupancy = {'n': 0, 's': 0, 'e': 0, 'w': 0}
        
        for lane_id in incoming_lanes:
            # Determine direction
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
            
            # Queue length
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            queues[direction] += queue
            
            # Weighted vehicle count
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            weighted_count = 0
            for veh_id in vehicle_ids:
                veh_type = traci.vehicle.getTypeID(veh_id)
                base_type = veh_type.split('_')[0].lower()
                weight = self.vehicle_weights.get(base_type, 1.0)
                weighted_count += weight
            weighted_vehicles[direction] += weighted_count
            
            # Occupancy
            occ = traci.lane.getLastStepOccupancy(lane_id) / 100.0
            occupancy[direction] = max(occupancy[direction], occ)
        
        # Add to state
        for d in ['n', 's', 'e', 'w']:
            state.append(queues[d])
        for d in ['n', 's', 'e', 'w']:
            state.append(weighted_vehicles[d])
        for d in ['n', 's', 'e', 'w']:
            state.append(occupancy[d])
        
        # Time in phase
        state.append(self.time_in_phase[junction_id])
        
        # Neighbor phases
        neighbors = self.neighbors[junction_id]
        for neighbor_id in neighbors[:2]:
            neighbor_phase = self.current_phases.get(neighbor_id, 0)
            state.append(neighbor_phase)
        
        # Pad
        while len(state) < 16:
            state.append(0)
        
        return np.array(state[:16], dtype=np.float32)
    
    def get_local_states(self):
        """Get local states for all junctions."""
        return [self.get_local_state(jid) for jid in self.junction_ids]
    
    def step(self):
        """Execute one step using MAPPO policy."""
        # Get states
        local_states = self.get_local_states()
        
        # Get actions from trained policy
        actions, _, _ = self.agent.select_actions(local_states)
        
        # Execute actions
        for i, jid in enumerate(self.junction_ids):
            action = actions[i]
            
            if action == 0:
                # Keep current phase
                pass
            elif action == 1:
                # Switch to next phase
                try:
                    programs = traci.trafficlight.getAllProgramLogics(jid)
                    num_phases = len(programs[0].phases) if programs else 4
                    next_phase = (self.current_phases[jid] + 1) % num_phases
                    traci.trafficlight.setPhase(jid, next_phase)
                    self.current_phases[jid] = next_phase
                    self.time_in_phase[jid] = 0
                except Exception:
                    pass
            elif action == 2:
                # Extend current phase
                pass
            
            # Update time in phase
            self.time_in_phase[jid] += 1


# =============================================================================
# METRICS COLLECTION
# =============================================================================

def collect_metrics(junction_ids):
    """
    Collect current traffic metrics from SUMO.
    
    Returns:
        dict with metrics
    """
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
    
    # Get network-wide metrics
    total_vehicles = traci.vehicle.getIDCount()
    arrived = traci.simulation.getArrivedNumber()
    departed = traci.simulation.getDepartedNumber()
    
    return {
        'total_waiting_time': total_waiting,
        'total_vehicles': total_vehicles,
        'total_queue': total_queue,
        'avg_queue': total_queue / max(1, lane_count),
        'arrived': arrived,
        'departed': departed,
        'lane_count': lane_count,
    }


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_simulation(sumo_config, controller, duration, seed=None, use_precomputed=False):
    """
    Run a simulation with the given controller.
    
    Args:
        sumo_config: Path to .sumocfg file
        controller: FixedTimeController or MAPPOController
        duration: Simulation duration in seconds
        seed: Random seed (optional)
        use_precomputed: Whether to use precomputed routes
    
    Returns:
        dict with time series metrics
    """
    # Build SUMO command
    sumo_binary = shutil.which('sumo') or 'sumo'
    
    sumo_cmd = [
        sumo_binary,
        '-c', str(sumo_config),
        '--start',
        '--quit-on-end',
        '--no-warnings',
        '--no-step-log',
    ]
    
    if seed is not None:
        sumo_cmd.extend(['--seed', str(seed)])
    
    # Modify route file if using precomputed
    if use_precomputed:
        # Parse config to find route file and swap to precomputed version
        import xml.etree.ElementTree as ET
        config_path = Path(sumo_config)
        tree = ET.parse(config_path)
        root = tree.getroot()
        
        route_elem = root.find('.//route-files')
        if route_elem is not None:
            route_file = route_elem.get('value', '')
            precomputed = route_file.replace('.rou.xml', '_precomputed.rou.xml')
            precomputed_path = config_path.parent / precomputed
            
            if precomputed_path.exists():
                print(f"  Using precomputed routes: {precomputed}")
                sumo_cmd.extend(['--route-files', str(precomputed_path)])
            else:
                print(f"  Warning: Precomputed routes not found: {precomputed_path}")
                print(f"  Run: python generate_6h_routes.py --duarouter")
    
    # Start SUMO
    print(f"  Starting SUMO simulation...")
    traci.start(sumo_cmd)
    
    # Initialize controller
    controller.initialize()
    
    # Metrics storage
    metrics = {
        'step': [],
        'total_waiting_time': [],
        'total_vehicles': [],
        'avg_queue': [],
        'cumulative_arrived': [],
        'cumulative_departed': [],
    }
    
    cumulative_arrived = 0
    cumulative_departed = 0
    
    # Run simulation
    step = 0
    try:
        while step < duration:
            # Controller step
            controller.step()
            
            # SUMO step
            traci.simulationStep()
            
            # Collect metrics
            m = collect_metrics(JUNCTION_IDS)
            
            cumulative_arrived += m['arrived']
            cumulative_departed += m['departed']
            
            metrics['step'].append(step)
            metrics['total_waiting_time'].append(m['total_waiting_time'])
            metrics['total_vehicles'].append(m['total_vehicles'])
            metrics['avg_queue'].append(m['avg_queue'])
            metrics['cumulative_arrived'].append(cumulative_arrived)
            metrics['cumulative_departed'].append(cumulative_departed)
            
            # Progress
            if step % 1000 == 0:
                print(f"    Step {step}/{duration} ({100*step/duration:.1f}%)")
            
            step += 1
            
            # Check if simulation ended early
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"  Simulation ended early at step {step}")
                break
                
    except traci.exceptions.FatalTraCIError as e:
        print(f"  SUMO error at step {step}: {e}")
    finally:
        if traci.isLoaded():
            traci.close()
    
    return metrics


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(fixed_metrics, mappo_metrics, output_dir):
    """Generate comparison plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy for easier processing
    fixed_steps = np.array(fixed_metrics['step'])
    mappo_steps = np.array(mappo_metrics['step'])
    
    # Use shorter length if different
    min_len = min(len(fixed_steps), len(mappo_steps))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fixed-Time vs MAPPO Traffic Control Comparison', fontsize=14, fontweight='bold')
    
    # 1. Total Waiting Time
    ax = axes[0, 0]
    ax.plot(fixed_steps[:min_len], fixed_metrics['total_waiting_time'][:min_len], 
            label='Fixed-Time', color='red', alpha=0.7)
    ax.plot(mappo_steps[:min_len], mappo_metrics['total_waiting_time'][:min_len], 
            label='MAPPO', color='blue', alpha=0.7)
    ax.set_xlabel('Simulation Step (seconds)')
    ax.set_ylabel('Total Waiting Time (s)')
    ax.set_title('Total Waiting Time Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Average Queue Length
    ax = axes[0, 1]
    ax.plot(fixed_steps[:min_len], fixed_metrics['avg_queue'][:min_len], 
            label='Fixed-Time', color='red', alpha=0.7)
    ax.plot(mappo_steps[:min_len], mappo_metrics['avg_queue'][:min_len], 
            label='MAPPO', color='blue', alpha=0.7)
    ax.set_xlabel('Simulation Step (seconds)')
    ax.set_ylabel('Avg Queue Length (vehicles)')
    ax.set_title('Average Queue Length Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative Throughput (Arrived Vehicles)
    ax = axes[1, 0]
    ax.plot(fixed_steps[:min_len], fixed_metrics['cumulative_arrived'][:min_len], 
            label='Fixed-Time', color='red', alpha=0.7)
    ax.plot(mappo_steps[:min_len], mappo_metrics['cumulative_arrived'][:min_len], 
            label='MAPPO', color='blue', alpha=0.7)
    ax.set_xlabel('Simulation Step (seconds)')
    ax.set_ylabel('Cumulative Vehicles Arrived')
    ax.set_title('Cumulative Throughput Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Vehicles in Network
    ax = axes[1, 1]
    ax.plot(fixed_steps[:min_len], fixed_metrics['total_vehicles'][:min_len], 
            label='Fixed-Time', color='red', alpha=0.7)
    ax.plot(mappo_steps[:min_len], mappo_metrics['total_vehicles'][:min_len], 
            label='MAPPO', color='blue', alpha=0.7)
    ax.set_xlabel('Simulation Step (seconds)')
    ax.set_ylabel('Vehicles in Network')
    ax.set_title('Active Vehicles Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'comparison_plots.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plots to: {plot_path}")
    
    # Bar chart summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate averages
    fixed_avg_wait = np.mean(fixed_metrics['total_waiting_time'][:min_len])
    mappo_avg_wait = np.mean(mappo_metrics['total_waiting_time'][:min_len])
    fixed_avg_queue = np.mean(fixed_metrics['avg_queue'][:min_len])
    mappo_avg_queue = np.mean(mappo_metrics['avg_queue'][:min_len])
    fixed_throughput = fixed_metrics['cumulative_arrived'][min_len-1] if min_len > 0 else 0
    mappo_throughput = mappo_metrics['cumulative_arrived'][min_len-1] if min_len > 0 else 0
    
    metrics_names = ['Avg Waiting\nTime (s)', 'Avg Queue\nLength', 'Total\nThroughput']
    fixed_values = [fixed_avg_wait, fixed_avg_queue, fixed_throughput / 100]  # Scale throughput
    mappo_values = [mappo_avg_wait, mappo_avg_queue, mappo_throughput / 100]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fixed_values, width, label='Fixed-Time', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, mappo_values, width, label='MAPPO', color='blue', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('Performance Summary (MAPPO vs Fixed-Time)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    summary_path = output_dir / 'summary_bar_chart.png'
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"Saved summary chart to: {summary_path}")


def save_results(fixed_metrics, mappo_metrics, output_dir, args):
    """Save metrics to CSV and JSON summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_len = min(len(fixed_metrics['step']), len(mappo_metrics['step']))
    
    # Save CSV
    csv_path = output_dir / 'metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('step,fixed_waiting,fixed_queue,fixed_throughput,fixed_vehicles,')
        f.write('mappo_waiting,mappo_queue,mappo_throughput,mappo_vehicles\n')
        
        for i in range(min_len):
            f.write(f"{fixed_metrics['step'][i]},")
            f.write(f"{fixed_metrics['total_waiting_time'][i]:.2f},")
            f.write(f"{fixed_metrics['avg_queue'][i]:.2f},")
            f.write(f"{fixed_metrics['cumulative_arrived'][i]},")
            f.write(f"{fixed_metrics['total_vehicles'][i]},")
            f.write(f"{mappo_metrics['total_waiting_time'][i]:.2f},")
            f.write(f"{mappo_metrics['avg_queue'][i]:.2f},")
            f.write(f"{mappo_metrics['cumulative_arrived'][i]},")
            f.write(f"{mappo_metrics['total_vehicles'][i]}\n")
    
    print(f"Saved metrics CSV to: {csv_path}")
    
    # Calculate summary statistics
    fixed_avg_wait = np.mean(fixed_metrics['total_waiting_time'][:min_len])
    mappo_avg_wait = np.mean(mappo_metrics['total_waiting_time'][:min_len])
    fixed_avg_queue = np.mean(fixed_metrics['avg_queue'][:min_len])
    mappo_avg_queue = np.mean(mappo_metrics['avg_queue'][:min_len])
    fixed_throughput = fixed_metrics['cumulative_arrived'][min_len-1] if min_len > 0 else 0
    mappo_throughput = mappo_metrics['cumulative_arrived'][min_len-1] if min_len > 0 else 0
    
    # Improvement percentages
    wait_improvement = (fixed_avg_wait - mappo_avg_wait) / max(1, fixed_avg_wait) * 100
    queue_improvement = (fixed_avg_queue - mappo_avg_queue) / max(0.01, fixed_avg_queue) * 100
    throughput_improvement = (mappo_throughput - fixed_throughput) / max(1, fixed_throughput) * 100
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'scenario': args.scenario,
        'duration': args.duration,
        'fixed_phase_duration': args.fixed_phase_duration,
        'checkpoint': args.checkpoint,
        'results': {
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
                'waiting_time_reduction_pct': round(wait_improvement, 2),
                'queue_length_reduction_pct': round(queue_improvement, 2),
                'throughput_increase_pct': round(throughput_improvement, 2),
            }
        }
    }
    
    json_path = output_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nScenario: {args.scenario}")
    print(f"Duration: {args.duration} seconds ({args.duration/3600:.1f} hours)")
    print(f"Fixed phase duration: {args.fixed_phase_duration}s")
    print(f"\n{'Metric':<25} {'Fixed-Time':>15} {'MAPPO':>15} {'Improvement':>15}")
    print("-"*70)
    print(f"{'Avg Waiting Time (s)':<25} {fixed_avg_wait:>15.2f} {mappo_avg_wait:>15.2f} {wait_improvement:>14.1f}%")
    print(f"{'Avg Queue Length':<25} {fixed_avg_queue:>15.2f} {mappo_avg_queue:>15.2f} {queue_improvement:>14.1f}%")
    print(f"{'Total Throughput':<25} {fixed_throughput:>15} {mappo_throughput:>15} {throughput_improvement:>14.1f}%")
    print("="*60)
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare Fixed-Time vs MAPPO traffic light control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_fixed_vs_mappo.py --checkpoint checkpoint_time_20251129_103621 --scenario weekday
  python evaluate_fixed_vs_mappo.py --checkpoint checkpoint_time_20251129_103621 --scenario weekday --duration 3600
  python evaluate_fixed_vs_mappo.py --checkpoint checkpoint_time_20251129_103621.zip --scenario weekend --use-precomputed
"""
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to MAPPO checkpoint (directory or .zip file)'
    )
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default='realistic',
        choices=['weekday', 'weekend', 'event', '3h', 'realistic'],
        help='Traffic scenario to evaluate (default: realistic - uses validated routes)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=21600,
        help='Simulation duration in seconds (default: 21600 = 6 hours)'
    )
    parser.add_argument(
        '--fixed-phase-duration', '-f',
        type=int,
        default=DEFAULT_FIXED_PHASE_DURATION,
        help=f'Fixed phase duration in seconds (default: {DEFAULT_FIXED_PHASE_DURATION})'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results',
        help='Output directory for results (default: evaluation_results)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--use-precomputed',
        action='store_true',
        help='Use precomputed routes (recommended for stability)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("FIXED-TIME vs MAPPO TRAFFIC CONTROL EVALUATION")
    print("="*60)
    
    # Resolve paths
    script_dir = SCRIPT_DIR
    checkpoint_path = Path(args.checkpoint)
    
    # Handle relative paths
    if not checkpoint_path.is_absolute():
        checkpoint_path = script_dir / checkpoint_path
    
    # Handle zip files
    temp_dir = None
    if checkpoint_path.suffix == '.zip':
        print(f"\nExtracting checkpoint from: {checkpoint_path}")
        temp_dir = tempfile.mkdtemp(prefix='mappo_checkpoint_')
        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        # Find the actual checkpoint directory inside
        extracted_items = list(Path(temp_dir).iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            checkpoint_path = extracted_items[0]
        else:
            checkpoint_path = Path(temp_dir)
        
        print(f"Extracted to: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Get SUMO config
    sumo_config = script_dir / SCENARIOS[args.scenario]
    if not sumo_config.exists():
        print(f"Error: SUMO config not found: {sumo_config}")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Scenario: {args.scenario}")
    print(f"  SUMO Config: {sumo_config}")
    print(f"  Duration: {args.duration}s ({args.duration/3600:.1f}h)")
    print(f"  Fixed Phase Duration: {args.fixed_phase_duration}s")
    print(f"  Use Precomputed Routes: {args.use_precomputed}")
    print(f"  Random Seed: {args.seed}")
    print(f"  Output Directory: {args.output}")
    
    # Run Fixed-Time evaluation
    print("\n" + "="*60)
    print("RUNNING FIXED-TIME BASELINE")
    print("="*60)
    
    fixed_controller = FixedTimeController(JUNCTION_IDS, args.fixed_phase_duration)
    fixed_metrics = run_simulation(
        sumo_config, 
        fixed_controller, 
        args.duration, 
        seed=args.seed,
        use_precomputed=args.use_precomputed
    )
    
    print(f"  Fixed-Time completed: {len(fixed_metrics['step'])} steps")
    
    # Run MAPPO evaluation
    print("\n" + "="*60)
    print("RUNNING MAPPO EVALUATION")
    print("="*60)
    
    mappo_controller = MAPPOController(checkpoint_path)
    mappo_metrics = run_simulation(
        sumo_config, 
        mappo_controller, 
        args.duration, 
        seed=args.seed,
        use_precomputed=args.use_precomputed
    )
    
    print(f"  MAPPO completed: {len(mappo_metrics['step'])} steps")
    
    # Save results and generate plots
    output_dir = script_dir / args.output
    save_results(fixed_metrics, mappo_metrics, output_dir, args)
    plot_comparison(fixed_metrics, mappo_metrics, output_dir)
    
    # Cleanup temp directory
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
