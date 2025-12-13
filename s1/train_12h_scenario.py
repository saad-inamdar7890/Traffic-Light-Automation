#!/usr/bin/env python3
"""
MAPPO Training for 12-Hour Traffic Scenarios
=============================================

This script trains the MAPPO model on 12-hour traffic scenarios (6am-6pm),
capturing key traffic patterns with half the training time of 24-hour scenarios:
- Morning rush hour (8am-9am)
- Midday traffic
- Building toward evening rush (4pm-6pm)

Key Features:
1. 43,200 steps per episode (12 hours)
2. Checkpoint saving every 2 simulated hours
3. Resume from checkpoint support
4. Configurable PPO epochs

Usage:
    python train_12h_scenario.py --scenario weekday
    python train_12h_scenario.py --scenario weekday --resume-checkpoint checkpoint_12h_ep10
    python train_12h_scenario.py --scenario weekday --epochs 15 --episodes 20
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


class Config12H(MAPPOConfig):
    """
    Extended configuration for 12-hour training.
    Inherits from MAPPOConfig and adjusts for 12-hour episodes.
    """
    
    # 12-hour episode = 43,200 steps (6am to 6pm)
    STEPS_PER_EPISODE = 43200
    
    # Training parameters
    UPDATE_FREQUENCY = 64         # More frequent updates (was 128)
    PPO_EPOCHS = 4                # Safe middle ground (not too aggressive)
    
    # Exploration settings
    EPSILON_START = 0.25
    EPSILON_END = 0.02
    EPSILON_DECAY = 0.998         # Faster decay than 24h
    
    # Learning rates
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 8e-4
    
    # Reward normalization
    MAX_WAITING_THRESHOLD = 600.0  # Adjusted for 12h
    
    # Checkpointing
    SAVE_INTERVAL = 1             # Save every episode
    LOG_INTERVAL = 1              # Log every episode
    
    # Training directory naming
    TENSORBOARD_DIR = "mappo_logs_12h"
    MODEL_DIR = "mappo_models_12h"


def get_time_period(step: int) -> str:
    """Get the time period name for a given simulation step."""
    if step < 7200:      # 6am-8am
        return "early_morning"
    elif step < 10800:   # 8am-9am
        return "morning_peak"
    elif step < 21600:   # 9am-12pm
        return "late_morning"
    elif step < 28800:   # 12pm-2pm
        return "lunch"
    elif step < 36000:   # 2pm-4pm
        return "early_afternoon"
    else:                # 4pm-6pm
        return "late_afternoon"


def get_sim_time_str(step: int) -> str:
    """Convert simulation step to time string (starting from 6am)."""
    total_minutes = step // 60
    hours = 6 + (total_minutes // 60)  # Start from 6am
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def train_12h_episode(config, agent, env, episode: int, ppo_epochs: int = None):
    """
    Train a single 12-hour episode.
    
    Args:
        config: Training configuration
        agent: MAPPO agent
        env: K1 environment
        episode: Current episode number
        ppo_epochs: Override PPO epochs if specified
    
    Returns:
        Tuple of (episode_reward, metrics_dict)
    """
    import torch
    
    # Override PPO epochs if specified
    if ppo_epochs is not None:
        agent.config.PPO_EPOCHS = ppo_epochs
    
    # Reset environment
    local_states, global_state = env.reset()
    
    episode_reward = 0.0
    episode_steps = 0
    current_hour_reward = 0.0
    
    # Tracking metrics
    hourly_rewards = []
    period_rewards = {period: 0.0 for period in [
        'early_morning', 'morning_peak', 'late_morning', 
        'lunch', 'early_afternoon', 'late_afternoon'
    ]}
    period_steps = {period: 0 for period in period_rewards.keys()}
    
    start_time = time.time()
    last_checkpoint_step = 0
    checkpoint_interval = 7200  # Every 2 simulated hours
    
    print(f"\n{'='*60}")
    print(f"Episode {episode}: 12-Hour Training (6am-6pm)")
    print(f"{'='*60}")
    print(f"Steps per episode: {config.STEPS_PER_EPISODE:,}")
    print(f"Update frequency: {config.UPDATE_FREQUENCY}")
    print(f"PPO epochs: {agent.config.PPO_EPOCHS}")
    
    try:
        while episode_steps < config.STEPS_PER_EPISODE:
            episode_steps += 1
            
            # Get current period
            current_period = get_time_period(episode_steps)
            
            # Select actions
            with torch.no_grad():
                actions, log_probs, entropies = agent.select_actions(local_states)
            
            # Environment step
            step_result = env.step(actions)
            if len(step_result) == 4:
                next_local_states, next_global_state, rewards, done = step_result
                info = {}
            else:
                next_local_states, next_global_state, rewards, done, info = step_result
            
            # Calculate step reward
            step_reward = sum(rewards) / len(rewards)
            episode_reward += step_reward
            current_hour_reward += step_reward
            
            # Track by period
            period_rewards[current_period] += step_reward
            period_steps[current_period] += 1
            
            # Store in buffer
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
            
            # Hourly logging
            hour = episode_steps // 3600
            if episode_steps % 3600 == 0 and hour > 0:
                hourly_rewards.append(current_hour_reward)
                sim_time = get_sim_time_str(episode_steps)
                try:
                    import traci
                    total_vehicles = traci.vehicle.getIDCount() if traci.isLoaded() else 0
                except:
                    total_vehicles = info.get('total_vehicles', 0)
                print(f"  {sim_time} (Hour {hour}/12): Reward={current_hour_reward:+.2f}, "
                      f"Cumulative={episode_reward:+.2f}, "
                      f"Vehicles={total_vehicles}")
                current_hour_reward = 0.0
            
            # PPO update
            if len(agent.buffer) >= config.UPDATE_FREQUENCY:
                update_result = agent.update()
                
                # Progress logging every 15 minutes (900 steps)
                if episode_steps % 900 == 0:
                    sim_time = get_sim_time_str(episode_steps)
                    elapsed = time.time() - start_time
                    remaining = (elapsed / episode_steps) * (config.STEPS_PER_EPISODE - episode_steps) if episode_steps > 0 else 0
                    print(f"    Step {episode_steps:5d} ({sim_time}): "
                          f"Reward={current_hour_reward:+.2f}, "
                          f"ETA={remaining/60:.1f}min")
            
            # Periodic checkpoint within episode (every 2 sim hours)
            if episode_steps - last_checkpoint_step >= checkpoint_interval:
                sim_hour = 6 + (episode_steps // 3600)  # Convert to actual hour
                checkpoint_name = f"checkpoint_12h_ep{episode}_hour{sim_hour}"
                checkpoint_path = str(SCRIPT_DIR / config.MODEL_DIR / checkpoint_name)
                agent.save_checkpoint(checkpoint_path)
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
    
    # Calculate period averages
    period_avg_rewards = {}
    for period in period_rewards:
        if period_steps[period] > 0:
            period_avg_rewards[period] = period_rewards[period] / period_steps[period]
        else:
            period_avg_rewards[period] = 0.0
    
    metrics = {
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'elapsed_time': elapsed_time,
        'steps_per_second': episode_steps / elapsed_time if elapsed_time > 0 else 0,
        'hourly_rewards': hourly_rewards,
        'period_rewards': period_rewards,
        'period_avg_rewards': period_avg_rewards,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Episode {episode} Summary")
    print(f"{'='*60}")
    print(f"Total Reward: {episode_reward:+.2f}")
    print(f"Steps: {episode_steps:,}")
    print(f"Time: {elapsed_time/60:.1f} minutes")
    print(f"Speed: {metrics['steps_per_second']:.1f} steps/sec")
    print(f"\nPeriod Performance:")
    for period, reward in period_avg_rewards.items():
        steps = period_steps[period]
        print(f"  {period:20s}: avg={reward:+.4f} (steps={steps:,})")
    
    return episode_reward, metrics


def train_12h(
    scenario: str = 'weekday',
    episodes: int = 50,
    resume_checkpoint: str = None,
    ppo_epochs: int = None
):
    """
    Main training function for 12-hour scenarios.
    
    Args:
        scenario: Scenario name ('weekday', 'weekend', 'friday', 'event')
        episodes: Number of episodes to train
        resume_checkpoint: Path to checkpoint to resume from
        ppo_epochs: Number of PPO epochs per update
    """
    print("\n" + "=" * 70)
    print("MAPPO 12-Hour Scenario Training")
    print("=" * 70)
    
    # Setup config
    config = Config12H()
    
    # Override PPO epochs if specified
    if ppo_epochs is not None:
        config.PPO_EPOCHS = ppo_epochs
        print(f"Using {ppo_epochs} PPO epochs per update")
    
    # Setup paths
    route_file = SCRIPT_DIR / f'k1_routes_12h_{scenario}.rou.xml'
    config_file = SCRIPT_DIR / f'k1_12h_{scenario}.sumocfg'
    
    print(f"\nScenario: {scenario}")
    print(f"Episodes: {episodes}")
    print(f"Steps per episode: {config.STEPS_PER_EPISODE:,} (12 hours)")
    print(f"Route file: {route_file}")
    
    # Check if route file exists, generate if not
    if not route_file.exists():
        print(f"\nRoute file not found. Generating...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / 'generate_12h_routes.py'), '--scenario', scenario],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error generating routes: {result.stderr}")
            return
        print("Route file generated successfully.")
    
    # Verify files exist
    if not route_file.exists():
        print(f"Error: Route file not found: {route_file}")
        return
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return
    
    print(f"✓ Using route file: {route_file}")
    
    # Set the SUMO config in the config object
    config.SUMO_CONFIG = str(config_file)
    config.USE_REALISTIC_24H_TRAFFIC = False  # We use our own config file
    
    # Create environment
    print("\nInitializing environment...")
    env = K1Environment(config)
    
    # Create agent
    print("Initializing MAPPO agent...")
    agent = MAPPOAgent(config)
    
    # Resume from checkpoint if specified
    start_episode = 1
    if resume_checkpoint:
        print(f"\nLooking for checkpoint: {resume_checkpoint}")
        
        # Try different paths
        checkpoint_paths = [
            Path(resume_checkpoint),
            SCRIPT_DIR / resume_checkpoint,
            SCRIPT_DIR / config.MODEL_DIR / resume_checkpoint,
            Path(resume_checkpoint) / 'checkpoint.zip' if Path(resume_checkpoint).is_dir() else None,
        ]
        
        # Also check for zip files
        for base_path in [Path(resume_checkpoint), SCRIPT_DIR / resume_checkpoint]:
            if not str(base_path).endswith('.zip'):
                checkpoint_paths.append(Path(str(base_path) + '.zip'))
        
        checkpoint_found = False
        for cp_path in checkpoint_paths:
            if cp_path is None:
                continue
            if cp_path.exists():
                print(f"  Found checkpoint at: {cp_path}")
                
                # Handle zip files
                if str(cp_path).endswith('.zip'):
                    temp_dir = tempfile.mkdtemp()
                    try:
                        with zipfile.ZipFile(cp_path, 'r') as zf:
                            zf.extractall(temp_dir)
                        
                        # Find the actual checkpoint directory
                        extracted_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
                        if extracted_dirs:
                            actual_path = extracted_dirs[0]
                        else:
                            actual_path = Path(temp_dir)
                        
                        metadata = agent.load_checkpoint(str(actual_path))
                        if metadata:
                            start_episode = metadata.get('episode', 0) + 1
                            print(f"  Resuming from episode {start_episode}")
                        checkpoint_found = True
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    metadata = agent.load_checkpoint(str(cp_path))
                    if metadata:
                        start_episode = metadata.get('episode', 0) + 1
                        print(f"  Resuming from episode {start_episode}")
                    # Force config values after loading checkpoint
                    agent.config.UPDATE_FREQUENCY = config.UPDATE_FREQUENCY
                    agent.config.PPO_EPOCHS = config.PPO_EPOCHS
                    print(f"  Forced UPDATE_FREQUENCY={config.UPDATE_FREQUENCY}, PPO_EPOCHS={config.PPO_EPOCHS}")
                    checkpoint_found = True
                break
        
        if not checkpoint_found:
            print(f"  Warning: Checkpoint not found, starting from scratch")
    
    # Create model directory
    model_dir = SCRIPT_DIR / config.MODEL_DIR
    model_dir.mkdir(exist_ok=True)
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting Training: Episodes {start_episode} to {start_episode + episodes - 1}")
    print(f"{'='*70}")
    
    all_rewards = []
    all_metrics = []
    
    for episode in range(start_episode, start_episode + episodes):
        episode_reward, metrics = train_12h_episode(config, agent, env, episode, ppo_epochs)
        all_rewards.append(episode_reward)
        all_metrics.append(metrics)
        
        # Save checkpoint after each episode
        checkpoint_name = f"checkpoint_12h_{scenario}_ep{episode}"
        checkpoint_path = str(SCRIPT_DIR / config.MODEL_DIR / checkpoint_name)
        agent.save_checkpoint(checkpoint_path)
        print(f"\n[Checkpoint saved: {checkpoint_name}]")
        
        # Decay epsilon
        agent.decay_epsilon()
        print(f"Epsilon: {agent.epsilon:.4f}")
        
        # Re-initialize environment for next episode
        if episode < start_episode + episodes - 1:
            env = K1Environment(config)
    
    # Final summary
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Episodes trained: {episodes}")
    print(f"Average reward: {sum(all_rewards)/len(all_rewards):+.2f}")
    print(f"Best reward: {max(all_rewards):+.2f}")
    print(f"Final checkpoint: checkpoint_12h_{scenario}_ep{start_episode + episodes - 1}")
    
    # Save final model
    final_name = f"mappo_12h_{scenario}_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    final_path = str(SCRIPT_DIR / config.MODEL_DIR / final_name)
    agent.save_checkpoint(final_path)
    print(f"Final model saved: {final_name}")


def main():
    parser = argparse.ArgumentParser(description='Train MAPPO on 12-hour traffic scenarios')
    parser.add_argument('--scenario', type=str, default='weekday',
                        choices=['weekday', 'weekend', 'friday', 'event'],
                        help='Traffic scenario to train on')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes to train')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of PPO epochs per update (overrides config)')
    
    args = parser.parse_args()
    
    train_12h(
        scenario=args.scenario,
        episodes=args.episodes,
        resume_checkpoint=args.resume_checkpoint,
        ppo_epochs=args.epochs
    )


if __name__ == '__main__':
    main()
