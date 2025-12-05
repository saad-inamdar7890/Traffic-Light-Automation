"""
Mixed Scenario Training for MAPPO
==================================

This script trains the MAPPO model on randomly sampled scenarios each episode.
This improves robustness by exposing the model to:
- Normal weekday/weekend traffic
- Special events
- Stress scenarios (gridlock, incidents, spikes, night surge)

Usage:
    python train_mixed_scenarios.py --run-episodes 100 --max-hours 5
    python train_mixed_scenarios.py --resume-checkpoint ./mappo_models/checkpoint_100 --run-episodes 50
"""

import os
import sys
import random
import subprocess
import argparse
from datetime import datetime


# Available scenarios with weights (higher = more likely to be selected)
SCENARIOS = {
    'weekday': 3,      # Most common - normal weekday traffic
    'weekend': 2,      # Weekend traffic patterns
    'event': 1,        # Special event traffic
    'gridlock': 1,     # Extreme congestion stress test
    'incident': 1,     # Asymmetric flow (incident simulation)
    'spike': 1,        # Sudden demand surge
    'night_surge': 1,  # Late-night event
}


def weighted_random_scenario():
    """Select a scenario with weighted probability"""
    choices = []
    for scenario, weight in SCENARIOS.items():
        choices.extend([scenario] * weight)
    return random.choice(choices)


def train_single_episode(checkpoint_path, scenario, episode_num):
    """Train a single episode on a specific scenario"""
    cmd = [
        sys.executable,
        'mappo_k1_implementation.py',
        '--scenario', scenario,
        '--run-episodes', '1',
    ]
    
    if checkpoint_path:
        cmd.extend(['--resume-checkpoint', checkpoint_path])
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_num}: Training on '{scenario}' scenario")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def find_latest_checkpoint():
    """Find the most recent checkpoint in mappo_models directory"""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mappo_models')
    
    if not os.path.exists(model_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path) and 'checkpoint' in item:
            # Get modification time
            mtime = os.path.getmtime(item_path)
            checkpoints.append((item_path, mtime))
    
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints[0][0]


def main():
    parser = argparse.ArgumentParser(description='Mixed scenario training for MAPPO')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--run-episodes', type=int, default=100,
                        help='Number of episodes to run (default: 100)')
    parser.add_argument('--max-hours', type=float, default=None,
                        help='Maximum training time in hours')
    parser.add_argument('--sequential', action='store_true',
                        help='Run scenarios sequentially instead of randomly')
    args = parser.parse_args()
    
    print("="*60)
    print("MAPPO Mixed Scenario Training")
    print("="*60)
    print(f"Episodes to run: {args.run_episodes}")
    print(f"Available scenarios: {list(SCENARIOS.keys())}")
    if args.max_hours:
        print(f"Time limit: {args.max_hours} hours")
    print("="*60)
    
    # Use the built-in mixed mode instead of running separate processes
    cmd = [
        sys.executable,
        'mappo_k1_implementation.py',
        '--scenario', 'mixed',
        '--run-episodes', str(args.run_episodes),
    ]
    
    if args.resume_checkpoint:
        cmd.extend(['--resume-checkpoint', args.resume_checkpoint])
    
    if args.max_hours:
        cmd.extend(['--max-hours', str(args.max_hours)])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ Mixed scenario training completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Training failed or was interrupted")
        print("="*60)
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
