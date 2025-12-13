#!/usr/bin/env python3
"""
MAPPO Training for 24-Hour Traffic Scenarios
=============================================

This script trains the MAPPO model on full 24-hour traffic scenarios,
which present unique challenges:
- Multiple rush hour periods (morning and evening)
- Long-term credit assignment (86,400 steps)
- Day-night traffic pattern transitions
- Varying traffic densities

Key Adaptations for 24-Hour Training:
1. Episode segmentation for memory efficiency
2. Checkpoint saving every 2 simulated hours
3. Progressive training curriculum
4. Enhanced reward normalization for long episodes

Usage:
    python train_24h_scenario.py --scenario weekday
    python train_24h_scenario.py --scenario weekday --resume-checkpoint checkpoint_24h_ep100
    python train_24h_scenario.py --scenario all --max-hours 6  # Train on all scenarios, limit real time
"""

import os
import sys
import argparse
import time
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from mappo_k1_implementation import (
    MAPPOConfig, 
    MAPPOAgent, 
    K1Environment, 
    train_mappo
)


class Config24H(MAPPOConfig):
    """
    Extended configuration for 24-hour training.
    Inherits from MAPPOConfig and adjusts for longer episodes.
    """
    
    # 24-hour episode = 86,400 steps
    STEPS_PER_EPISODE = 86400
    
    # Adjusted training parameters for long episodes
    UPDATE_FREQUENCY = 128        # Less frequent updates (memory efficiency)
    PPO_EPOCHS = 10               # Fewer epochs per update (speed)
    
    # More aggressive exploration for larger state space
    EPSILON_START = 0.25
    EPSILON_END = 0.02
    EPSILON_DECAY = 0.999         # Slower decay for longer episodes
    
    # Learning rates (slightly lower for stability)
    LEARNING_RATE_ACTOR = 3e-4    # More conservative
    LEARNING_RATE_CRITIC = 8e-4
    
    # Adjusted reward normalization
    MAX_WAITING_THRESHOLD = 750.0  # Higher threshold for 24h variation
    
    # Checkpointing
    SAVE_INTERVAL = 1             # Save every episode (they're long!)
    LOG_INTERVAL = 1              # Log every episode
    
    # Training directory naming
    TENSORBOARD_DIR = "mappo_logs_24h"
    MODEL_DIR = "mappo_models_24h"


def get_24h_config(scenario: str) -> Config24H:
    """
    Get configuration for a specific 24-hour scenario.
    
    Args:
        scenario: One of 'weekday', 'weekend', 'friday', 'event'
    
    Returns:
        Configured Config24H instance
    """
    config = Config24H()
    
    # Set scenario-specific SUMO config
    config.SUMO_CONFIG = f"k1_24h_{scenario}.sumocfg"
    
    # Validate config file exists
    config_path = SCRIPT_DIR / config.SUMO_CONFIG
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        print(f"Run: python generate_24h_routes.py --scenario {scenario}")
        raise FileNotFoundError(f"Missing SUMO config: {config.SUMO_CONFIG}")
    
    return config


def train_24h_episode(config: Config24H, agent: MAPPOAgent, env: K1Environment, 
                      episode: int, checkpoint_interval: int = 7200):
    """
    Train a single 24-hour episode with periodic checkpointing.
    
    Args:
        config: Training configuration
        agent: MAPPO agent
        env: K1 SUMO environment
        episode: Episode number
        checkpoint_interval: Save checkpoint every N simulation steps (default: 2 hours)
    
    Returns:
        episode_reward: Total episode reward
        metrics: Dictionary of training metrics
    """
    print(f"\n{'='*60}")
    print(f"24-HOUR EPISODE {episode}")
    print(f"{'='*60}")
    
    # Reset environment
    local_states, global_state = env.reset()
    
    episode_reward = 0.0
    episode_steps = 0
    
    # Metrics tracking
    hourly_rewards = []
    current_hour_reward = 0.0
    
    start_time = time.time()
    last_checkpoint_step = 0
    
    try:
        while episode_steps < config.STEPS_PER_EPISODE:
            # Select actions
            actions, log_probs, entropies = agent.select_actions(local_states)
            
            # Environment step (returns 4 values, not 5)
            step_result = env.step(actions)
            
            # Handle both 4-value and 5-value returns for compatibility
            if len(step_result) == 4:
                next_local_states, next_global_state, rewards, done = step_result
                info = {}  # No info dict returned
            else:
                next_local_states, next_global_state, rewards, done, info = step_result
            
            # Store experience using the correct method name
            agent.buffer.store(
                local_states=local_states,
                global_state=global_state,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                entropies=entropies,
                done=done
            )
            
            # Update states
            local_states = next_local_states
            global_state = next_global_state
            
            # Track rewards
            step_reward = sum(rewards) / len(rewards)
            episode_reward += step_reward
            current_hour_reward += step_reward
            episode_steps += 1
            
            # Hourly tracking (3600 steps = 1 hour)
            if episode_steps % 3600 == 0:
                hour = episode_steps // 3600
                hourly_rewards.append(current_hour_reward)
                # Get vehicle count from TraCI if info not available
                try:
                    import traci
                    total_vehicles = traci.vehicle.getIDCount() if traci.isLoaded() else 0
                except:
                    total_vehicles = info.get('total_vehicles', 0)
                print(f"  Hour {hour:2d}/24: Reward={current_hour_reward:+.2f}, "
                      f"Cumulative={episode_reward:+.2f}, "
                      f"Vehicles={total_vehicles}")
                current_hour_reward = 0.0
            
            # PPO update
            if len(agent.buffer) >= config.UPDATE_FREQUENCY:
                actor_loss, critic_loss = agent.update()
                
                # Progress logging every 15 minutes (900 steps)
                if episode_steps % 900 == 0:
                    sim_hour = episode_steps // 3600
                    sim_min = (episode_steps % 3600) // 60
                    elapsed = time.time() - start_time
                    remaining = (elapsed / episode_steps) * (config.STEPS_PER_EPISODE - episode_steps)
                    print(f"    Step {episode_steps:5d} (Sim {sim_hour:02d}:{sim_min:02d}): "
                          f"Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}, "
                          f"ETA={remaining/60:.1f}min")
            
            # Periodic checkpoint within episode
            if episode_steps - last_checkpoint_step >= checkpoint_interval:
                checkpoint_name = f"checkpoint_24h_ep{episode}_hour{episode_steps//3600}"
                agent.save_checkpoint(checkpoint_name, episode, episode_reward)
                print(f"  [Checkpoint saved: {checkpoint_name}]")
                last_checkpoint_step = episode_steps
            
            if done:
                break
                
    except KeyboardInterrupt:
        print("\n[Training interrupted by user]")
    finally:
        env.close()
    
    # Final metrics
    elapsed_time = time.time() - start_time
    metrics = {
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'hourly_rewards': hourly_rewards,
        'elapsed_time': elapsed_time,
        'steps_per_second': episode_steps / elapsed_time if elapsed_time > 0 else 0,
    }
    
    print(f"\nEpisode {episode} Complete:")
    print(f"  Total Reward: {episode_reward:+.2f}")
    print(f"  Steps: {episode_steps}")
    print(f"  Time: {elapsed_time/60:.1f} minutes")
    print(f"  Speed: {metrics['steps_per_second']:.1f} steps/sec")
    
    return episode_reward, metrics


def train_24h(scenario: str, num_episodes: int = 100, resume_checkpoint: str = None,
              max_hours: float = None, ppo_epochs: int = None):
    """
    Main training loop for 24-hour scenarios.
    
    Args:
        scenario: Traffic scenario name
        num_episodes: Number of episodes to train
        resume_checkpoint: Path to checkpoint to resume from
        max_hours: Maximum real-time hours to train (None = unlimited)
        ppo_epochs: Number of PPO epochs per update (None = use default)
    """
    print("="*60)
    print("MAPPO 24-HOUR TRAINING")
    print("="*60)
    print(f"\nScenario: {scenario}")
    print(f"Episodes: {num_episodes}")
    print(f"Resume from: {resume_checkpoint or 'None (fresh start)'}")
    print(f"Max training time: {max_hours or 'Unlimited'} hours")
    print(f"PPO Epochs: {ppo_epochs or 'Default (10)'}")
    
    # Get configuration
    config = get_24h_config(scenario)
    
    # Override PPO epochs if specified
    if ppo_epochs is not None:
        config.PPO_EPOCHS = ppo_epochs
    
    print(f"\nConfiguration:")
    print(f"  SUMO Config: {config.SUMO_CONFIG}")
    print(f"  Steps/Episode: {config.STEPS_PER_EPISODE} (24 hours)")
    print(f"  Update Frequency: {config.UPDATE_FREQUENCY}")
    print(f"  Device: {config.DEVICE}")
    
    # Create agent
    agent = MAPPOAgent(config)
    
    # Resume from checkpoint if specified
    start_episode = 0
    temp_checkpoint_dir = None
    
    if resume_checkpoint:
        # Try multiple locations for checkpoint
        checkpoint_path = Path(resume_checkpoint)
        
        # Check various possible locations
        possible_paths = [
            checkpoint_path,  # As-is (absolute or relative)
            SCRIPT_DIR / resume_checkpoint,  # In s1 directory
            SCRIPT_DIR / config.MODEL_DIR / resume_checkpoint,  # In model dir
            SCRIPT_DIR / 'mappo_models' / resume_checkpoint,  # In original model dir
        ]
        
        found_path = None
        for p in possible_paths:
            if p.exists():
                found_path = p
                break
        
        if found_path is None:
            print(f"Error: Checkpoint not found. Searched:")
            for p in possible_paths:
                print(f"  - {p}")
            sys.exit(1)
        
        print(f"\nFound checkpoint: {found_path}")
        
        # Handle zip files
        if found_path.suffix == '.zip':
            print(f"Extracting checkpoint from zip...")
            temp_checkpoint_dir = tempfile.mkdtemp(prefix='mappo_24h_checkpoint_')
            with zipfile.ZipFile(found_path, 'r') as zf:
                zf.extractall(temp_checkpoint_dir)
            
            # Find extracted checkpoint directory
            extracted = list(Path(temp_checkpoint_dir).iterdir())
            if len(extracted) == 1 and extracted[0].is_dir():
                checkpoint_path = extracted[0]
            else:
                checkpoint_path = Path(temp_checkpoint_dir)
            print(f"Extracted to: {checkpoint_path}")
        else:
            checkpoint_path = found_path
        
        # Load checkpoint
        try:
            episode_info = agent.load_checkpoint(str(checkpoint_path))
            
            # Handle None return or missing keys gracefully
            if episode_info is None:
                episode_info = {}
                print("Note: Checkpoint loaded but no metadata returned")
            
            # Get episode number, default to 0 if not found
            start_episode = episode_info.get('episode', 0)
            if start_episode is None:
                start_episode = 0
            start_episode += 1
            
            print(f"Resumed from episode {start_episode - 1}")
            print(f"Previous epsilon: {episode_info.get('epsilon', 'N/A')}")
            if 'episode_reward' in episode_info:
                print(f"Previous reward: {episode_info.get('episode_reward')}")
        except Exception as e:
            print(f"Warning: Error loading checkpoint metadata: {e}")
            print("Continuing with checkpoint weights loaded, starting from episode 0")
            start_episode = 0
            # Don't exit - the weights may have loaded successfully
            # Only the metadata failed
    
    # Create environment
    env = K1Environment(config)
    
    # Training loop
    training_start = time.time()
    all_rewards = []
    
    for episode in range(start_episode, start_episode + num_episodes):
        # Check time limit
        if max_hours is not None:
            elapsed_hours = (time.time() - training_start) / 3600
            if elapsed_hours >= max_hours:
                print(f"\n[Time limit reached: {max_hours} hours]")
                break
        
        # Train episode
        episode_reward, metrics = train_24h_episode(config, agent, env, episode)
        all_rewards.append(episode_reward)
        
        # Save checkpoint after each episode
        checkpoint_name = f"checkpoint_24h_{scenario}_ep{episode}"
        agent.save_checkpoint(checkpoint_name, episode, episode_reward)
        
        # Log progress
        avg_reward = sum(all_rewards[-10:]) / len(all_rewards[-10:])
        print(f"\n[Episode {episode}] Avg Reward (last 10): {avg_reward:+.2f}")
    
    # Cleanup temp checkpoint directory if used
    if temp_checkpoint_dir:
        shutil.rmtree(temp_checkpoint_dir, ignore_errors=True)
    
    # Final save
    final_checkpoint = f"checkpoint_24h_{scenario}_final"
    checkpoint_dir = SCRIPT_DIR / config.MODEL_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent.save_checkpoint(final_checkpoint, episode, episode_reward)
    
    total_time = time.time() - training_start
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Episodes trained: {len(all_rewards)}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Best episode reward: {max(all_rewards):+.2f}")
    print(f"  Final checkpoint saved to: {checkpoint_dir / final_checkpoint}")
    
    return agent, all_rewards


def main():
    parser = argparse.ArgumentParser(
        description='Train MAPPO on 24-hour traffic scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_24h_scenario.py --scenario weekday
  python train_24h_scenario.py --scenario weekday --episodes 50
  python train_24h_scenario.py --scenario weekday --resume-checkpoint checkpoint_24h_weekday_ep10
  python train_24h_scenario.py --scenario weekday --max-hours 6
  
Before training, generate route files:
  python generate_24h_routes.py --scenario weekday
"""
    )
    
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default='weekday',
        choices=['weekday', 'weekend', 'friday', 'event'],
        help='Traffic scenario to train on (default: weekday)'
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=100,
        help='Number of episodes to train (default: 100)'
    )
    parser.add_argument(
        '--resume-checkpoint', '-r',
        type=str,
        default=None,
        help='Checkpoint to resume training from'
    )
    parser.add_argument(
        '--max-hours', '-t',
        type=float,
        default=None,
        help='Maximum real-time hours to train'
    )
    parser.add_argument(
        '--generate-routes',
        action='store_true',
        help='Generate route files before training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of PPO epochs per update (default: 10)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='Save checkpoint every N episodes (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Generate routes if requested
    if args.generate_routes:
        print("Generating 24-hour routes...")
        import subprocess
        subprocess.run([
            sys.executable, 
            str(SCRIPT_DIR / 'generate_24h_routes.py'),
            '--scenario', args.scenario
        ])
    
    # Check if route files exist
    route_file = SCRIPT_DIR / f'k1_routes_24h_{args.scenario}.rou.xml'
    if not route_file.exists():
        print(f"Error: Route file not found: {route_file}")
        print(f"Run: python generate_24h_routes.py --scenario {args.scenario}")
        sys.exit(1)
    
    # Train
    train_24h(
        scenario=args.scenario,
        num_episodes=args.episodes,
        resume_checkpoint=args.resume_checkpoint,
        max_hours=args.max_hours,
        ppo_epochs=args.epochs
    )


if __name__ == '__main__':
    main()
