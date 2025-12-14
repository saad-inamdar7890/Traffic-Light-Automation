#!/usr/bin/env python3
"""
Training Script for 6-Hour Original Routes Scenario (Kaggle Ready)
===================================================================

Train MAPPO on the 6-hour scenario using ALL 39 original routes.
This ensures the model is trained on the SAME routes used for evaluation.

Duration: 6 simulation hours per episode (21,600 steps)
Routes: All 39 original routes with time-varying traffic
Traffic Pattern: Morning peak + Evening peak

Usage (Kaggle):
    !python train_6h_original.py --resume "/kaggle/input/traffic18" --epochs 10 --episodes 3

Usage (Local):
    python train_6h_original.py --epochs 10 --episodes 3
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
# 6-HOUR ORIGINAL ROUTES SCENARIO CONFIGURATION
# =============================================================================

SCENARIO_CONFIG = {
    'name': '6h_original',
    'duration_hours': 6,
    'duration': 21600,       # 6 hours in seconds
    'max_steps': 21600,      # 21600 simulation steps (1 step/sec)
    'config_file': 'k1_6h_original.sumocfg',
    'description': '6-hour training with all 39 original routes'
}

# Time periods for logging
TIME_PERIODS = [
    (0, 3600, 'Early Morning', '6am-7am'),
    (3600, 7200, 'Morning Peak', '7am-8am'),
    (7200, 10800, 'Midday', '8am-9am'),
    (10800, 14400, 'Afternoon', '9am-10am'),
    (14400, 18000, 'Evening Peak', '10am-11am'),
    (18000, 21600, 'Night', '11am-12pm'),
]

DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints_6h_original'


class Config6H(MAPPOConfig):
    """Extended configuration for 6-hour training with optimal settings."""
    
    # 6-hour episode = 21,600 steps
    STEPS_PER_EPISODE = 21600
    
    # Training parameters - balanced settings
    UPDATE_FREQUENCY = 64         # More frequent updates
    PPO_EPOCHS = 4                # Safe middle ground
    
    # Exploration settings
    EPSILON_START = 0.25
    EPSILON_END = 0.02
    EPSILON_DECAY = 0.995
    
    # Learning rates
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 8e-4


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


def get_time_period_name(step: int) -> str:
    """Get the current time period name based on simulation step."""
    for start, end, name, _ in TIME_PERIODS:
        if start <= step < end:
            return name
    return "Unknown"


def train_6h_original(
    epochs: int = 10,
    episodes_per_epoch: int = 3,
    use_gui: bool = False,
    checkpoint_dir: str = None,
    resume_from: str = None
):
    """
    Train MAPPO agent on 6-hour original routes scenario.
    """
    print("=" * 70)
    print("MAPPO Training - 6-Hour Original Routes Scenario")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Episodes per epoch: {episodes_per_epoch}")
    print(f"Total episodes: {epochs * episodes_per_epoch}")
    print(f"Episode length: {SCENARIO_CONFIG['duration']:,} seconds ({SCENARIO_CONFIG['duration_hours']} hours)")
    print(f"Routes: All 39 original routes with time-varying traffic")
    print()
    
    # Setup paths
    config_file = SCRIPT_DIR / SCENARIO_CONFIG['config_file']
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        print("Run 'python generate_6h_original_routes.py' first to create route files")
        return
    
    print(f"SUMO Config: {config_file}")
    print(f"Checkpoints: {checkpoint_path}")
    
    # Create config with optimal settings
    config = Config6H()
    config.SUMO_CONFIG = str(config_file)
    config.STEPS_PER_EPISODE = SCENARIO_CONFIG['max_steps']
    config.USE_REALISTIC_24H_TRAFFIC = False
    config.use_gui = use_gui
    
    # Initialize environment
    env = K1Environment(config)
    
    print(f"\nEnvironment initialized:")
    print(f"  - Traffic lights: {len(env.junction_ids)}")
    print(f"  - Update frequency: {config.UPDATE_FREQUENCY}")
    print(f"  - PPO epochs: {config.PPO_EPOCHS}")
    print(f"  - Updates per episode: ~{SCENARIO_CONFIG['max_steps'] // config.UPDATE_FREQUENCY}")
    
    # Initialize agent
    agent = MAPPOAgent(config)
    
    # Resume from checkpoint
    start_epoch = 0
    if resume_from:
        resume_path = find_checkpoint(resume_from)
        if resume_path and resume_path.exists():
            print(f"\nLoading checkpoint from: {resume_path}")
            try:
                agent.load_checkpoint(str(resume_path))
                print(f"✓ Checkpoint loaded successfully!")
                
                # Force optimal config values after loading
                agent.config.UPDATE_FREQUENCY = config.UPDATE_FREQUENCY
                agent.config.PPO_EPOCHS = config.PPO_EPOCHS
                print(f"  Forced UPDATE_FREQUENCY={config.UPDATE_FREQUENCY}, PPO_EPOCHS={config.PPO_EPOCHS}")
                
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
        'period_rewards': [],
        'best_reward': float('-inf'),
        'best_epoch': 0
    }
    best_avg_reward = float('-inf')
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING 6-HOUR TRAINING (Original Routes)")
    print("=" * 70)
    print("Time Periods:")
    for start, end, name, time_str in TIME_PERIODS:
        print(f"  {name}: {time_str} (steps {start}-{end})")
    print("-" * 70)
    
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
            
            print(f"\n  [Episode {episode + 1}/{episodes_per_epoch}] Starting 6-hour simulation...")
            
            # Reset environment
            local_states, global_state = env.reset()
            episode_reward = 0
            step = 0
            update_count = 0
            max_steps = SCENARIO_CONFIG['max_steps']
            
            # Track rewards by period
            period_rewards = {name: 0.0 for _, _, name, _ in TIME_PERIODS}
            current_period_reward = 0
            current_period_idx = 0
            
            # Episode loop
            while step < max_steps:
                # Select actions
                actions, log_probs, entropies = agent.select_actions(local_states)
                
                # Environment step
                next_local_states, next_global_state, rewards, done = env.step(actions)
                
                # Store transition
                agent.buffer.store(local_states, global_state, actions, rewards, log_probs, entropies, done)
                
                # Update agent (GPU usage)
                if len(agent.buffer) >= config.UPDATE_FREQUENCY:
                    agent.update()
                    update_count += 1
                
                # Update metrics
                reward_value = np.mean(rewards)
                episode_reward += reward_value
                current_period_reward += reward_value
                
                # Check for period change and log
                period_start, period_end, period_name, _ = TIME_PERIODS[current_period_idx]
                if step >= period_end and current_period_idx < len(TIME_PERIODS) - 1:
                    # Save period reward
                    period_rewards[period_name] = current_period_reward
                    
                    # Get vehicle count
                    try:
                        vehicle_count = traci.vehicle.getIDCount() if traci.isLoaded() else 0
                    except:
                        vehicle_count = 0
                    
                    elapsed = time.time() - episode_start
                    print(f"    {period_name}: Reward={current_period_reward:8.2f}, "
                          f"Vehicles={vehicle_count:3d}, "
                          f"Updates={update_count:3d}, "
                          f"Time={elapsed:.1f}s")
                    
                    # Move to next period
                    current_period_idx += 1
                    current_period_reward = 0
                
                local_states = next_local_states
                global_state = next_global_state
                step += 1
                
                if done:
                    print(f"    [Simulation ended early at step {step}]")
                    break
            
            # Final period
            if current_period_idx < len(TIME_PERIODS):
                period_name = TIME_PERIODS[current_period_idx][2]
                period_rewards[period_name] = current_period_reward
            
            agent.decay_epsilon()
            
            epoch_rewards.append(episode_reward)
            training_history['episode_rewards'].append(episode_reward)
            training_history['period_rewards'].append(period_rewards)
            
            episode_time = time.time() - episode_start
            
            print(f"\n  Episode {episode + 1} Complete:")
            print(f"    Total Reward: {episode_reward:.2f}")
            print(f"    Steps: {step:,}, Updates: {update_count}")
            print(f"    Time: {episode_time / 60:.1f} minutes")
            print(f"    Period breakdown:")
            for _, _, name, _ in TIME_PERIODS:
                print(f"      {name}: {period_rewards.get(name, 0):.2f}")
            
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
    final_path = checkpoint_path / "final"
    agent.save_checkpoint(str(final_path))
    
    history_file = checkpoint_path / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    env.close()
    
    # =========================================================================
    # KAGGLE: Save and zip checkpoint for download
    # =========================================================================
    try:
        import shutil
        kaggle_output = Path('/kaggle/working')
        
        if kaggle_output.exists():
            print(f"\n{'='*70}")
            print("KAGGLE: Saving final checkpoint for download...")
            print(f"{'='*70}")
            
            # Copy final model
            kaggle_final = kaggle_output / 'mappo_6h_final'
            if kaggle_final.exists():
                shutil.rmtree(kaggle_final)
            shutil.copytree(str(final_path), kaggle_final)
            print(f"✓ Final model copied to {kaggle_final}")
            
            # Create zip
            zip_name = f"mappo_6h_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.make_archive(str(kaggle_output / zip_name), 'zip', str(checkpoint_path))
            print(f"✓ Created {zip_name}.zip")
            
            # Simple named zip
            shutil.make_archive(str(kaggle_output / 'mappo_6h_trained'), 'zip', str(checkpoint_path))
            print(f"✓ Created mappo_6h_trained.zip")
            
            print(f"\nDOWNLOAD: /kaggle/working/mappo_6h_trained.zip")
    except Exception as e:
        print(f"Note: Kaggle save skipped ({e})")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Time: {total_time / 60:.1f} minutes")
    print(f"Best Reward: {training_history['best_reward']:.2f} (Epoch {training_history['best_epoch']})")
    print(f"\nCheckpoints saved to: {checkpoint_path}")
    print(f"\nTo evaluate:")
    print(f"  python evaluate_fixed_vs_mappo.py --checkpoint {checkpoint_path / 'best'} --scenario original")
    
    return training_history


def main():
    parser = argparse.ArgumentParser(
        description='Train MAPPO on 6-hour original routes scenario',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python train_6h_original.py                              # Start fresh
  python train_6h_original.py --epochs 10 --episodes 3     # 30 total episodes  
  python train_6h_original.py --resume temp_checkpoint     # Resume training
  python train_6h_original.py --resume /kaggle/input/traffic18  # Resume on Kaggle

For Kaggle:
  !python train_6h_original.py --resume "/kaggle/input/traffic18" --epochs 10 --episodes 3
        '''
    )
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--episodes', type=int, default=3, help='Episodes per epoch (default: 3)')
    parser.add_argument('--gui', action='store_true', help='Enable SUMO GUI')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train_6h_original(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        use_gui=args.gui,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume
    )


if __name__ == '__main__':
    main()
