#!/usr/bin/env python3
"""
Training Script for 12-Hour Heavy Traffic Scenario (Kaggle Ready)
==================================================================

Direct training on 12-hour heavy traffic for model adaptation.
This is the SIMPLE version that matches the Kaggle deployment fixes.

Duration: 12 simulation hours per episode (43,200 steps)
Traffic: ~8,240 vehicles/hour (heavy sustained load)
Total vehicles: ~98,880 per episode

Usage (Kaggle):
    !python train_12h_heavy.py --resume "/kaggle/input/traffic18" --epochs 5 --episodes 2

Usage (Local):
    python train_12h_heavy.py --epochs 5 --episodes 2
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
# 12-HOUR HEAVY SCENARIO CONFIGURATION
# =============================================================================

SCENARIO_CONFIG = {
    'name': '12h_heavy',
    'duration_hours': 12,
    'duration': 43200,       # 12 hours in seconds
    'max_steps': 43200,      # 43200 simulation steps (1 step/sec)
    'config_file': 'k1_12h_heavy.sumocfg',
    'description': '12-hour heavy traffic for extended adaptation training'
}

DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints_12h_heavy'


def find_checkpoint(resume_arg: str) -> Path:
    """Find checkpoint folder from various input formats."""
    if not resume_arg:
        return None
    
    path = Path(resume_arg)
    if path.exists():
        if path.suffix == '.zip':
            extract_dir = path.parent / path.stem
            if not extract_dir.exists():
                print(f"  Extracting {path.name}...")
                with zipfile.ZipFile(path, 'r') as zf:
                    zf.extractall(extract_dir)
            return extract_dir
        return path
    
    # Check in SCRIPT_DIR
    for check_path in [SCRIPT_DIR / resume_arg, 
                       SCRIPT_DIR / f"{resume_arg}.zip",
                       DEFAULT_CHECKPOINT_DIR / resume_arg]:
        if check_path.exists():
            if check_path.suffix == '.zip':
                extract_dir = check_path.parent / check_path.stem
                if not extract_dir.exists():
                    with zipfile.ZipFile(check_path, 'r') as zf:
                        zf.extractall(extract_dir)
                return extract_dir
            return check_path
    
    return None


def train_12h_heavy(
    epochs: int = 5,
    episodes_per_epoch: int = 2,
    use_gui: bool = False,
    checkpoint_dir: str = None,
    resume_from: str = None
):
    """
    Train MAPPO agent on 12-hour heavy traffic scenario.
    """
    print("=" * 70)
    print("MAPPO Training - 12-Hour Heavy Traffic Scenario")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Episodes per epoch: {episodes_per_epoch}")
    print(f"Total episodes: {epochs * episodes_per_epoch}")
    print(f"Episode length: {SCENARIO_CONFIG['duration']:,} seconds ({SCENARIO_CONFIG['duration_hours']} hours)")
    print(f"Traffic: ~8,240 vehicles/hour (~98,880 per episode)")
    print()
    
    # Setup paths
    config_file = SCRIPT_DIR / SCENARIO_CONFIG['config_file']
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        print("Run 'python generate_12h_heavy.py' first to create route files")
        return
    
    print(f"SUMO Config: {config_file}")
    print(f"Checkpoints: {checkpoint_path}")
    
    # Create MAPPOConfig
    mappo_config = MAPPOConfig()
    mappo_config.SUMO_CONFIG = str(config_file)
    mappo_config.STEPS_PER_EPISODE = SCENARIO_CONFIG['max_steps']
    mappo_config.USE_REALISTIC_24H_TRAFFIC = False
    mappo_config.use_gui = use_gui
    
    # Initialize environment
    env = K1Environment(mappo_config)
    
    print(f"\nEnvironment initialized:")
    print(f"  - Traffic lights: {len(env.junction_ids)}")
    print(f"  - Update frequency: {mappo_config.UPDATE_FREQUENCY}")
    print(f"  - Updates per episode: ~{SCENARIO_CONFIG['max_steps'] // mappo_config.UPDATE_FREQUENCY}")
    
    # Initialize agent
    agent = MAPPOAgent(mappo_config)
    
    # Resume from checkpoint
    start_epoch = 0
    if resume_from:
        resume_path = find_checkpoint(resume_from)
        if resume_path and resume_path.exists():
            print(f"\nLoading checkpoint from: {resume_path}")
            try:
                agent.load_checkpoint(str(resume_path))
                print(f"✓ Checkpoint loaded successfully!")
                
                train_state_file = resume_path / 'train_state.pkl'
                if train_state_file.exists():
                    with open(train_state_file, 'rb') as f:
                        train_state = pickle.load(f)
                    start_epoch = train_state.get('episode_count', 0) // episodes_per_epoch
                    print(f"  Continuing from epoch: {start_epoch}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                start_epoch = 0
        else:
            print(f"Warning: Checkpoint not found: {resume_from}")
    
    # Training history
    training_history = {
        'scenario': SCENARIO_CONFIG['name'],
        'start_time': datetime.now().isoformat(),
        'epochs': [],
        'episode_rewards': [],
        'best_reward': float('-inf'),
        'best_epoch': 0
    }
    best_avg_reward = float('-inf')
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING 12-HOUR TRAINING")
    print("=" * 70)
    
    total_episodes = start_epoch * episodes_per_epoch
    training_start = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        epoch_rewards = []
        
        print(f"\n{'='*70}")
        print(f"[EPOCH {epoch + 1}/{epochs}]")
        print(f"{'='*70}")
        
        for episode in range(episodes_per_epoch):
            total_episodes += 1
            episode_start = time.time()
            
            print(f"\n  [Episode {episode + 1}/{episodes_per_epoch}] Starting 12-hour simulation...")
            
            # Reset environment
            local_states, global_state = env.reset()
            episode_reward = 0
            step = 0
            update_count = 0
            max_steps = SCENARIO_CONFIG['max_steps']
            current_hour_reward = 0
            
            # Episode loop
            while step < max_steps:
                # Select actions
                actions, log_probs, entropies = agent.select_actions(local_states)
                
                # Environment step
                next_local_states, next_global_state, rewards, done = env.step(actions)
                
                # Store transition
                agent.buffer.store(local_states, global_state, actions, rewards, log_probs, entropies, done)
                
                # Update agent (GPU usage)
                if len(agent.buffer) >= mappo_config.UPDATE_FREQUENCY:
                    agent.update()
                    update_count += 1
                
                # Update metrics
                reward_value = np.mean(rewards)
                episode_reward += reward_value
                current_hour_reward += reward_value
                
                # Progress every hour
                if (step + 1) % 3600 == 0:
                    sim_hour = (step + 1) // 3600
                    elapsed = time.time() - episode_start
                    steps_per_sec = (step + 1) / elapsed
                    eta_minutes = (max_steps - step - 1) / steps_per_sec / 60
                    
                    print(f"    Hour {sim_hour:2d}/12: Reward={current_hour_reward:8.2f}, "
                          f"Total={episode_reward:10.2f}, "
                          f"Updates={update_count:4d}, "
                          f"ETA={eta_minutes:.1f}min")
                    current_hour_reward = 0
                
                local_states = next_local_states
                global_state = next_global_state
                step += 1
                
                if done:
                    print(f"    [Simulation ended early at step {step}]")
                    break
            
            agent.decay_epsilon()
            
            epoch_rewards.append(episode_reward)
            training_history['episode_rewards'].append(episode_reward)
            
            episode_time = time.time() - episode_start
            print(f"\n  Episode {episode + 1} Complete:")
            print(f"    Total Reward: {episode_reward:.2f}")
            print(f"    Steps: {step:,}, Updates: {update_count}")
            print(f"    Time: {episode_time / 60:.1f} minutes")
            
            # Save checkpoint after each episode
            temp_ckpt = checkpoint_path / "temp_checkpoint"
            agent.save_checkpoint(str(temp_ckpt))
            with open(temp_ckpt / 'train_state.pkl', 'wb') as f:
                pickle.dump({'episode_count': total_episodes, 'epoch': epoch}, f)
        
        # Epoch summary
        avg_reward = np.mean(epoch_rewards)
        epoch_time = time.time() - epoch_start
        
        training_history['epochs'].append({
            'epoch': epoch + 1,
            'avg_reward': avg_reward,
            'time': epoch_time
        })
        
        print(f"\n  Epoch {epoch + 1} Summary:")
        print(f"    Avg Reward: {avg_reward:.2f}")
        print(f"    Epoch Time: {epoch_time / 60:.1f} minutes")
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            training_history['best_reward'] = avg_reward
            training_history['best_epoch'] = epoch + 1
            best_path = checkpoint_path / "best"
            agent.save_checkpoint(str(best_path))
            print(f"    ★ New best model saved!")
        
        # Save epoch checkpoint
        agent.save_checkpoint(str(checkpoint_path / f"epoch_{epoch + 1}"))
    
    # Training complete
    total_time = time.time() - training_start
    training_history['end_time'] = datetime.now().isoformat()
    training_history['total_time'] = total_time
    
    # Save final
    agent.save_checkpoint(str(checkpoint_path / "final"))
    with open(checkpoint_path / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    env.close()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Time: {total_time / 60:.1f} minutes")
    print(f"Best Reward: {training_history['best_reward']:.2f} (Epoch {training_history['best_epoch']})")
    print(f"\nCheckpoints saved to: {checkpoint_path}")
    
    return training_history


def main():
    parser = argparse.ArgumentParser(description='Train MAPPO on 12-hour heavy traffic')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--episodes', type=int, default=2, help='Episodes per epoch (default: 2)')
    parser.add_argument('--gui', action='store_true', help='Enable SUMO GUI')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train_12h_heavy(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        use_gui=args.gui,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume
    )


if __name__ == '__main__':
    main()
