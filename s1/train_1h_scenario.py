#!/usr/bin/env python3
"""
Training Script for 1-Hour Heavy Traffic Scenario
==================================================

Intensive training on heavy traffic to help model learn route patterns.
Perfect for initial training before scaling to longer scenarios.

Features:
- 1-hour episodes (3600 steps)
- Heavy traffic load (~8000 vehicles/hour)
- More frequent checkpoints for faster iteration
- All 39 validated routes
- Resume from existing checkpoints (temp_checkpoint folder)

Usage:
    python train_1h_scenario.py
    python train_1h_scenario.py --epochs 20 --episodes 15
    python train_1h_scenario.py --resume temp_checkpoint  # Resume from checkpoint folder
    python train_1h_scenario.py --resume checkpoint_time_20251129_103621  # Resume from zip
    python train_1h_scenario.py --gui  # With visualization
"""

import os
import sys
import json
import time
import pickle
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

# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

SCENARIO_CONFIG = {
    'name': '1h_heavy',
    'duration': 3600,       # 1 hour in seconds
    'max_steps': 3600,      # 3600 simulation steps (1 step/sec)
    'config_file': 'k1_1h_heavy.sumocfg',
    'description': '1-hour heavy traffic for intensive route learning'
}

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints_1h_heavy'


def extract_checkpoint_if_zip(checkpoint_path: Path) -> Path:
    """Extract checkpoint if it's a zip file, return folder path."""
    if checkpoint_path.suffix == '.zip':
        extract_dir = checkpoint_path.parent / checkpoint_path.stem
        if not extract_dir.exists():
            print(f"  Extracting {checkpoint_path.name}...")
            with zipfile.ZipFile(checkpoint_path, 'r') as zf:
                zf.extractall(extract_dir)
        return extract_dir
    return checkpoint_path


def find_checkpoint(resume_arg: str) -> Path:
    """Find checkpoint folder from various input formats."""
    if not resume_arg:
        return None
    
    # Direct path
    path = Path(resume_arg)
    if path.exists():
        return extract_checkpoint_if_zip(path)
    
    # Check in SCRIPT_DIR
    path = SCRIPT_DIR / resume_arg
    if path.exists():
        return extract_checkpoint_if_zip(path)
    
    # Try with .zip extension
    path = SCRIPT_DIR / f"{resume_arg}.zip"
    if path.exists():
        return extract_checkpoint_if_zip(path)
    
    # Check in checkpoint directory
    path = DEFAULT_CHECKPOINT_DIR / resume_arg
    if path.exists():
        return extract_checkpoint_if_zip(path)
    
    # Special names
    if resume_arg.lower() == 'latest':
        # Find most recent checkpoint folder or zip
        checkpoints = list(SCRIPT_DIR.glob('checkpoint_time_*.zip'))
        checkpoints += list(SCRIPT_DIR.glob('temp_checkpoint'))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            return extract_checkpoint_if_zip(latest)
    
    return None


def train_1h_scenario(
    epochs: int = 15,
    episodes_per_epoch: int = 10,
    use_gui: bool = False,
    checkpoint_dir: str = None,
    resume_from: str = None
):
    """
    Train MAPPO agent on 1-hour heavy traffic scenario.
    
    Args:
        epochs: Number of training epochs
        episodes_per_epoch: Episodes per epoch
        use_gui: Whether to show SUMO GUI
        checkpoint_dir: Directory for checkpoints
        resume_from: Path to checkpoint folder to resume from
    """
    print("=" * 70)
    print("MAPPO Training - 1-Hour Heavy Traffic Scenario")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Episodes per epoch: {episodes_per_epoch}")
    print(f"Total episodes: {epochs * episodes_per_epoch}")
    print(f"Episode length: {SCENARIO_CONFIG['duration']} seconds")
    print(f"GUI: {'Enabled' if use_gui else 'Disabled'}")
    print()
    
    # Setup paths
    config_file = SCRIPT_DIR / SCENARIO_CONFIG['config_file']
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
    else:
        checkpoint_path = DEFAULT_CHECKPOINT_DIR
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Verify config file exists
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        print("Run 'python generate_1h_heavy.py' first to create route files")
        return
    
    print(f"SUMO Config: {config_file}")
    print(f"Checkpoints: {checkpoint_path}")
    print()
    
    # Create MAPPOConfig
    mappo_config = MAPPOConfig()
    mappo_config.SUMO_CONFIG = str(config_file)  # Override to 1h heavy config
    mappo_config.STEPS_PER_EPISODE = SCENARIO_CONFIG['max_steps']
    mappo_config.USE_REALISTIC_24H_TRAFFIC = False  # Use our custom config, not k1_realistic
    mappo_config.use_gui = use_gui
    
    # Initialize environment
    env = K1Environment(mappo_config)
    
    print(f"Environment initialized:")
    print(f"  - Traffic lights: {len(env.junction_ids)}")
    print(f"  - Local state dim: {mappo_config.LOCAL_STATE_DIM}")
    print(f"  - Action dim: {mappo_config.ACTION_DIM}")
    print()
    
    # Initialize agent
    agent = MAPPOAgent(mappo_config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    resumed_history = None
    
    if resume_from:
        resume_path = find_checkpoint(resume_from)
        
        if resume_path and resume_path.exists():
            print(f"Loading checkpoint from: {resume_path}")
            try:
                # Load checkpoint (folder with actor_*.pth, critic.pth, etc.)
                agent.load_checkpoint(str(resume_path))
                print(f"✓ Checkpoint loaded successfully!")
                
                # Try to load training state for epoch info
                train_state_file = resume_path / 'train_state.pkl'
                if train_state_file.exists():
                    with open(train_state_file, 'rb') as f:
                        train_state = pickle.load(f)
                    episode_count = train_state.get('episode_count', 0)
                    start_epoch = episode_count // episodes_per_epoch
                    print(f"  Previous episodes: {episode_count}")
                    print(f"  Continuing from epoch: {start_epoch}")
                
                # Try to load training history
                history_files = sorted(checkpoint_path.glob('training_history_*.json'))
                if history_files:
                    try:
                        with open(history_files[-1], 'r') as f:
                            resumed_history = json.load(f)
                        print(f"  Loaded training history: {history_files[-1].name}")
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Starting training from scratch...")
                start_epoch = 0
        else:
            print(f"Warning: Checkpoint not found: {resume_from}")
            print("Starting training from scratch...")
    
    # Training metrics - restore from history if resuming
    if resumed_history:
        training_history = resumed_history
        training_history['resumed_at'] = datetime.now().isoformat()
        training_history['resumed_from_epoch'] = start_epoch
        best_avg_reward = training_history.get('best_reward', float('-inf'))
        print(f"  Previous best reward: {best_avg_reward:.2f} (Epoch {training_history.get('best_epoch', 0)})")
    else:
        training_history = {
            'scenario': SCENARIO_CONFIG['name'],
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'episode_rewards': [],
            'episode_waiting_times': [],
            'episode_throughputs': [],
            'best_reward': float('-inf'),
            'best_epoch': 0
        }
        best_avg_reward = float('-inf')
    
    # Training loop
    print("\nStarting training...")
    print("-" * 70)
    
    total_episodes = start_epoch * episodes_per_epoch
    training_start = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        epoch_rewards = []
        epoch_waiting_times = []
        epoch_throughputs = []
        
        print(f"\n[Epoch {epoch + 1}/{epochs}]")
        
        for episode in range(episodes_per_epoch):
            total_episodes += 1
            episode_start = time.time()
            
            # Reset environment
            states = env.reset()
            episode_reward = 0
            step = 0
            
            # Episode loop
            while True:
                # Agent selects actions
                actions, log_probs, values = agent.select_actions(states)
                
                # Environment step
                next_states, rewards, done, info = env.step(actions)
                
                # Store transition
                agent.store_transition(states, actions, rewards, next_states, done, log_probs, values)
                
                # Update metrics
                episode_reward += np.mean(rewards)
                states = next_states
                step += 1
                
                if done:
                    break
            
            # Get episode statistics
            waiting_time = info.get('total_waiting_time', 0)
            throughput = info.get('throughput', 0)
            
            epoch_rewards.append(episode_reward)
            epoch_waiting_times.append(waiting_time)
            epoch_throughputs.append(throughput)
            training_history['episode_rewards'].append(episode_reward)
            training_history['episode_waiting_times'].append(waiting_time)
            training_history['episode_throughputs'].append(throughput)
            
            episode_time = time.time() - episode_start
            
            print(f"  Episode {episode + 1}/{episodes_per_epoch}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Wait={waiting_time:.0f}s, "
                  f"Throughput={throughput}, "
                  f"Steps={step}, "
                  f"Time={episode_time:.1f}s")
            
            # Update agent
            if len(agent.buffer) >= mappo_config.UPDATE_FREQUENCY:
                loss = agent.update()
                if loss:
                    print(f"    -> Policy update, Loss: {loss:.4f}")
        
        # Epoch summary
        avg_reward = np.mean(epoch_rewards)
        avg_waiting = np.mean(epoch_waiting_times)
        avg_throughput = np.mean(epoch_throughputs)
        epoch_time = time.time() - epoch_start
        
        training_history['epochs'].append({
            'epoch': epoch + 1,
            'avg_reward': avg_reward,
            'avg_waiting_time': avg_waiting,
            'avg_throughput': avg_throughput,
            'time': epoch_time
        })
        
        print(f"\n  Epoch {epoch + 1} Summary:")
        print(f"    Avg Reward: {avg_reward:.2f}")
        print(f"    Avg Waiting Time: {avg_waiting:.0f}s")
        print(f"    Avg Throughput: {avg_throughput:.0f}")
        print(f"    Epoch Time: {epoch_time:.1f}s")
        
        # Save checkpoints
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            training_history['best_reward'] = avg_reward
            training_history['best_epoch'] = epoch + 1
            
            # Save best model
            best_path = checkpoint_path / "best"
            agent.save_checkpoint(str(best_path))
            print(f"    ★ New best model saved: {best_path}")
        
        # Regular checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_name = f"epoch_{epoch + 1}"
            ckpt_path = checkpoint_path / ckpt_name
            agent.save_checkpoint(str(ckpt_path))
            print(f"    → Checkpoint saved: {ckpt_name}")
    
    # Training complete
    total_time = time.time() - training_start
    training_history['end_time'] = datetime.now().isoformat()
    training_history['total_time'] = total_time
    
    # Save final model
    final_path = checkpoint_path / "final"
    agent.save_checkpoint(str(final_path))
    
    # Save training history
    history_file = checkpoint_path / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Close environment
    env.close()
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Time: {total_time / 60:.1f} minutes")
    print(f"Avg Time/Episode: {total_time / max(1, total_episodes - start_epoch * episodes_per_epoch):.1f}s")
    print()
    print(f"Best Average Reward: {training_history['best_reward']:.2f} (Epoch {training_history['best_epoch']})")
    print()
    print("Saved Files:")
    print(f"  - Best Model: {checkpoint_path / 'best'}")
    print(f"  - Final Model: {final_path}")
    print(f"  - Training History: {history_file.name}")
    print()
    print("To resume training:")
    print(f"  python train_1h_scenario.py --resume {checkpoint_path / 'best'}")
    print()
    print("Next Steps:")
    print("  1. Evaluate: python evaluate_1h_performance.py")
    print("  2. Scale up to longer scenarios")
    print("=" * 70)
    
    return training_history


def main():
    parser = argparse.ArgumentParser(
        description='Train MAPPO on 1-hour heavy traffic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python train_1h_scenario.py                           # Start fresh training
  python train_1h_scenario.py --resume temp_checkpoint  # Resume from folder
  python train_1h_scenario.py --resume checkpoint_time_20251129_103621.zip
  python train_1h_scenario.py --resume latest           # Auto-find latest
  python train_1h_scenario.py --epochs 20 --episodes 15
        '''
    )
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs (default: 15)')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per epoch (default: 10)')
    parser.add_argument('--gui', action='store_true', help='Enable SUMO GUI')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--resume', type=str, metavar='CHECKPOINT',
                       help='Resume from checkpoint folder: path, "latest", or zip filename')
    
    args = parser.parse_args()
    
    train_1h_scenario(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        use_gui=args.gui,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume
    )


if __name__ == '__main__':
    main()
