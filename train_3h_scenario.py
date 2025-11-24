"""
3-Hour Time-Varying Traffic Training Script
===========================================

This script helps train the MAPPO agent on a more realistic 3-hour scenario
with time-varying traffic patterns (light → building → rush → peak → declining).

Benefits of 3-hour training:
1. More robust learning: Model experiences traffic transitions within each episode
2. Faster than 24-hour: ~20-30 min/episode vs ~2-3 hours
3. Pattern diversity: 5 distinct traffic periods per episode
4. Better generalization: Learns to adapt to changing conditions

Usage:
------
# Test scenario first (1 episode):
python train_3h_scenario.py --test

# Start fresh training:
python train_3h_scenario.py --max-hours 3

# Resume from checkpoint (recommended after 80 episodes of 1-hour training):
python train_3h_scenario.py --resume-checkpoint mappo_models/checkpoint_time_20251123_135403 --max-hours 3

# Quick test run (10 episodes):
python train_3h_scenario.py --max-episodes 10
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Train MAPPO on 3-hour time-varying traffic scenario')
    parser.add_argument('--test', action='store_true', help='Test scenario with 1 episode')
    parser.add_argument('--resume-checkpoint', type=str, help='Path to checkpoint folder to resume from')
    parser.add_argument('--max-hours', type=float, default=3.0, help='Maximum training time in hours')
    parser.add_argument('--max-episodes', type=int, help='Maximum number of episodes (overrides max-hours)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Force device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Verify files exist
    required_files = [
        's1/k1_3h_varying.sumocfg',
        's1/k1_routes_3h_varying.rou.xml',
        's1/k1.net.xml',
        's1/mappo_k1_implementation.py'
    ]
    
    print("=" * 70)
    print("3-Hour Time-Varying Traffic Training")
    print("=" * 70)
    
    for f in required_files:
        if os.path.exists(f):
            print(f"✓ Found: {f}")
        else:
            print(f"✗ Missing: {f}")
            print(f"\nError: Required file not found: {f}")
            return 1
    
    print("\n" + "=" * 70)
    print("Traffic Pattern Details:")
    print("=" * 70)
    print("Period 1 (0-36 min):     60-80 veh/h     [Light traffic]")
    print("Period 2 (36-72 min):    120-180 veh/h   [Building up]")
    print("Period 3 (72-108 min):   250-350 veh/h   [Rush hour begins]")
    print("Period 4 (108-144 min):  400-500 veh/h   [Peak congestion]")
    print("Period 5 (144-180 min):  200-280 veh/h   [Declining]")
    print("\nTotal episode duration: 3 hours (10,800 seconds)")
    print("Estimated time per episode: ~20-30 minutes on GPU")
    print("=" * 70)
    
    # Build command
    cmd = [sys.executable, 's1/mappo_k1_implementation.py']
    
    if args.test:
        print("\n🔍 TEST MODE: Running 1 episode to verify scenario")
        cmd.extend(['--num-episodes', '1'])
    elif args.max_episodes:
        print(f"\n🎯 TRAINING MODE: {args.max_episodes} episodes")
        cmd.extend(['--num-episodes', str(args.max_episodes)])
    else:
        print(f"\n🎯 TRAINING MODE: {args.max_hours} hours")
        cmd.extend(['--max-hours', str(args.max_hours)])
    
    if args.resume_checkpoint:
        if not os.path.exists(args.resume_checkpoint):
            print(f"\n❌ Error: Checkpoint not found: {args.resume_checkpoint}")
            return 1
        print(f"📂 Resuming from: {args.resume_checkpoint}")
        cmd.extend(['--resume-checkpoint', args.resume_checkpoint])
    else:
        print("🆕 Starting fresh training")
    
    if args.device:
        cmd.extend(['--device', args.device])
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 0

if __name__ == '__main__':
    exit(main())
