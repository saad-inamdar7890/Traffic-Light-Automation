#!/usr/bin/env python3
"""
Evaluation Script for 1-Hour Heavy Traffic Scenario
=====================================================

Compare Fixed-Time vs MAPPO performance on heavy traffic.
Use this to validate model learning before scaling to longer scenarios.

Usage:
    python evaluate_1h_performance.py
    python evaluate_1h_performance.py --checkpoint mappo_1h_heavy_best.pt
    python evaluate_1h_performance.py --runs 5  # Multiple evaluation runs
"""

import sys
import json
import time
import zipfile
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import traci
except ImportError:
    print("Error: SUMO TraCI not found. Please install SUMO and set SUMO_HOME.")
    sys.exit(1)

from mappo_k1_implementation import MAPPOConfig, MAPPOAgent, K1Environment


def run_fixed_time_baseline(env, duration: int = 3600) -> dict:
    """Run simulation with fixed-time traffic light control."""
    print("  Running Fixed-Time baseline...")
    
    states = env.reset()
    total_reward = 0
    step = 0
    
    while True:
        # Fixed-time: no action changes (action 0 = maintain current phase)
        actions = [0] * env.num_agents
        states, rewards, done, info = env.step(actions)
        total_reward += np.mean(rewards)
        step += 1
        
        if done:
            break
    
    return {
        'total_reward': total_reward,
        'total_waiting_time': info.get('total_waiting_time', 0),
        'throughput': info.get('throughput', 0),
        'avg_speed': info.get('avg_speed', 0),
        'steps': step
    }


def run_mappo_control(env, agent, duration: int = 3600) -> dict:
    """Run simulation with MAPPO-controlled traffic lights."""
    print("  Running MAPPO control...")
    
    states = env.reset()
    total_reward = 0
    step = 0
    
    while True:
        # MAPPO agent selects optimal actions
        actions, _, _ = agent.select_actions(states, deterministic=True)
        states, rewards, done, info = env.step(actions)
        total_reward += np.mean(rewards)
        step += 1
        
        if done:
            break
    
    return {
        'total_reward': total_reward,
        'total_waiting_time': info.get('total_waiting_time', 0),
        'throughput': info.get('throughput', 0),
        'avg_speed': info.get('avg_speed', 0),
        'steps': step
    }


def evaluate_1h_scenario(
    checkpoint_path: str = None,
    num_runs: int = 3,
    use_gui: bool = False
):
    """
    Evaluate performance on 1-hour heavy traffic scenario.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        num_runs: Number of evaluation runs per method
        use_gui: Whether to show SUMO GUI
    """
    print("=" * 70)
    print("Performance Evaluation - 1-Hour Heavy Traffic")
    print("=" * 70)
    print(f"Evaluation runs: {num_runs}")
    print()
    
    # Setup config file
    config_file = SCRIPT_DIR / 'k1_1h_heavy.sumocfg'
    
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        print("Run 'python generate_1h_heavy.py' first")
        return
    
    # Setup checkpoint - support folder or zip
    checkpoint_dir = None
    if checkpoint_path is None:
        # Try common checkpoint locations
        candidates = [
            SCRIPT_DIR / 'checkpoints_1h_heavy' / 'best',
            SCRIPT_DIR / 'temp_checkpoint',
            SCRIPT_DIR / 'checkpoint_time_20251129_103621',
        ]
        for c in candidates:
            if c.exists():
                checkpoint_dir = c
                break
        # Also check for zip
        if not checkpoint_dir:
            zip_path = SCRIPT_DIR / 'checkpoint_time_20251129_103621.zip'
            if zip_path.exists():
                checkpoint_dir = SCRIPT_DIR / 'checkpoint_time_20251129_103621'
                if not checkpoint_dir.exists():
                    print(f"Extracting {zip_path.name}...")
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(checkpoint_dir)
    else:
        checkpoint_dir = Path(checkpoint_path)
        # Extract if zip
        if checkpoint_dir.suffix == '.zip' and checkpoint_dir.exists():
            extract_dir = checkpoint_dir.parent / checkpoint_dir.stem
            if not extract_dir.exists():
                print(f"Extracting {checkpoint_dir.name}...")
                with zipfile.ZipFile(checkpoint_dir, 'r') as zf:
                    zf.extractall(extract_dir)
            checkpoint_dir = extract_dir
    
    has_checkpoint = checkpoint_dir is not None and checkpoint_dir.exists()
    if not has_checkpoint:
        print(f"Warning: No checkpoint found")
        print("Only running Fixed-Time baseline")
    else:
        print(f"Checkpoint: {checkpoint_dir}")
    print()
    
    # Create MAPPOConfig
    mappo_config = MAPPOConfig()
    mappo_config.sumo_cfg = str(config_file)
    mappo_config.max_steps = 3600
    mappo_config.use_gui = use_gui
    
    # Results storage
    fixed_results = []
    mappo_results = []
    
    # Run Fixed-Time baseline
    print("-" * 70)
    print("Fixed-Time Traffic Control")
    print("-" * 70)
    
    for run in range(num_runs):
        print(f"\n[Run {run + 1}/{num_runs}]")
        env = K1Environment(mappo_config)
        result = run_fixed_time_baseline(env, 3600)
        fixed_results.append(result)
        env.close()
        
        print(f"  Reward: {result['total_reward']:.2f}")
        print(f"  Waiting Time: {result['total_waiting_time']:.0f}s")
        print(f"  Throughput: {result['throughput']}")
    
    # Run MAPPO if checkpoint exists
    if has_checkpoint:
        print("\n" + "-" * 70)
        print("MAPPO Adaptive Control")
        print("-" * 70)
        
        for run in range(num_runs):
            print(f"\n[Run {run + 1}/{num_runs}]")
            env = K1Environment(mappo_config)
            agent = MAPPOAgent(mappo_config)
            agent.load_checkpoint(str(checkpoint_dir))
            
            result = run_mappo_control(env, agent, 3600)
            mappo_results.append(result)
            env.close()
            
            print(f"  Reward: {result['total_reward']:.2f}")
            print(f"  Waiting Time: {result['total_waiting_time']:.0f}s")
            print(f"  Throughput: {result['throughput']}")
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    def calc_stats(results, metric):
        values = [r[metric] for r in results]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Fixed-Time stats
    ft_reward = calc_stats(fixed_results, 'total_reward')
    ft_waiting = calc_stats(fixed_results, 'total_waiting_time')
    ft_throughput = calc_stats(fixed_results, 'throughput')
    
    print("\nFixed-Time Control:")
    print(f"  Average Reward: {ft_reward['mean']:.2f} ± {ft_reward['std']:.2f}")
    print(f"  Waiting Time: {ft_waiting['mean']:.0f} ± {ft_waiting['std']:.0f} seconds")
    print(f"  Throughput: {ft_throughput['mean']:.0f} ± {ft_throughput['std']:.0f} vehicles")
    
    if has_checkpoint and mappo_results:
        # MAPPO stats
        mappo_reward = calc_stats(mappo_results, 'total_reward')
        mappo_waiting = calc_stats(mappo_results, 'total_waiting_time')
        mappo_throughput = calc_stats(mappo_results, 'throughput')
        
        print("\nMAPPO Control:")
        print(f"  Average Reward: {mappo_reward['mean']:.2f} ± {mappo_reward['std']:.2f}")
        print(f"  Waiting Time: {mappo_waiting['mean']:.0f} ± {mappo_waiting['std']:.0f} seconds")
        print(f"  Throughput: {mappo_throughput['mean']:.0f} ± {mappo_throughput['std']:.0f} vehicles")
        
        # Improvement metrics
        print("\n" + "-" * 70)
        print("IMPROVEMENT (MAPPO vs Fixed-Time)")
        print("-" * 70)
        
        reward_improvement = ((mappo_reward['mean'] - ft_reward['mean']) / abs(ft_reward['mean'])) * 100
        waiting_reduction = ((ft_waiting['mean'] - mappo_waiting['mean']) / ft_waiting['mean']) * 100
        throughput_improvement = ((mappo_throughput['mean'] - ft_throughput['mean']) / ft_throughput['mean']) * 100
        
        print(f"  Reward: {reward_improvement:+.1f}%")
        print(f"  Waiting Time: {waiting_reduction:+.1f}% reduction")
        print(f"  Throughput: {throughput_improvement:+.1f}%")
        
        if reward_improvement > 10:
            print("\n  ✓ Model shows significant improvement!")
            print("  → Ready to scale up to longer scenarios")
        elif reward_improvement > 0:
            print("\n  ~ Model shows some improvement")
            print("  → Consider more training before scaling up")
        else:
            print("\n  ✗ Model not improving over baseline")
            print("  → Need more training on 1-hour scenario")
    
    # Save results
    results_data = {
        'scenario': '1h_heavy',
        'timestamp': datetime.now().isoformat(),
        'num_runs': num_runs,
        'fixed_time': {
            'reward': ft_reward,
            'waiting_time': ft_waiting,
            'throughput': ft_throughput,
            'raw_results': fixed_results
        }
    }
    
    if mappo_results:
        results_data['mappo'] = {
            'reward': mappo_reward,
            'waiting_time': mappo_waiting,
            'throughput': mappo_throughput,
            'raw_results': mappo_results,
            'checkpoint': str(checkpoint_path)
        }
        results_data['improvement'] = {
            'reward_pct': reward_improvement,
            'waiting_reduction_pct': waiting_reduction,
            'throughput_pct': throughput_improvement
        }
    
    results_file = SCRIPT_DIR / f"eval_results_1h_heavy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=float)
    
    print(f"\nResults saved to: {results_file.name}")
    print("=" * 70)
    
    return results_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate 1-hour heavy traffic scenario')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--runs', type=int, default=3, help='Number of evaluation runs')
    parser.add_argument('--gui', action='store_true', help='Enable SUMO GUI')
    
    args = parser.parse_args()
    
    evaluate_1h_scenario(
        checkpoint_path=args.checkpoint,
        num_runs=args.runs,
        use_gui=args.gui
    )


if __name__ == '__main__':
    main()
