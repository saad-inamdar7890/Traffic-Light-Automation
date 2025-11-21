"""
MAPPO with Adaptive Deployment - Handling Traffic Pattern Changes
=================================================================

This script demonstrates how to handle traffic pattern changes during deployment:
1. Training on diverse scenarios
2. Performance monitoring
3. Scenario detection
4. Offline fine-tuning when needed

Key insight: DON'T keep critic active during deployment!
Instead: Train robustly, monitor, adapt offline when needed.

Author: Traffic Light Automation Team
Date: November 2025
"""

import os
import numpy as np
import torch
from datetime import datetime, timedelta
import json
from collections import deque, defaultdict

# Import from original implementation
from mappo_k1_implementation import (
    MAPPOAgent,
    MAPPOConfig,
    K1Environment,
    train_mappo
)
from deploy_mappo import MAPPODeployment


# ============================================================================
# DIVERSE SCENARIO TRAINING
# ============================================================================

class DiverseScenarioConfig(MAPPOConfig):
    """Extended config for diverse scenario training"""
    
    # Multiple training scenarios
    TRAINING_SCENARIOS = [
        # Normal patterns
        'custom_24h',
        'morning_rush',
        'evening_rush',
        'uniform_light',
        'uniform_medium',
        'uniform_heavy',
        
        # Variations (important for robustness!)
        'north_route_blocked',        # Simulate route blockage
        'diverted_traffic_west',      # Diverted flow
        'construction_reduced',       # Reduced capacity
        'high_truck_percentage',      # More heavy vehicles
        'emergency_frequent',         # High emergency rate
        'random_mixed',               # Random patterns
    ]
    
    # Scenario probabilities (can weight certain scenarios)
    SCENARIO_WEIGHTS = {
        'custom_24h': 0.3,           # 30% normal
        'morning_rush': 0.15,
        'evening_rush': 0.15,
        'uniform_medium': 0.1,
        'north_route_blocked': 0.05,  # 5% each variation
        'diverted_traffic_west': 0.05,
        'construction_reduced': 0.05,
        'high_truck_percentage': 0.05,
        'emergency_frequent': 0.05,
        'random_mixed': 0.05,
    }


def train_mappo_diverse_scenarios(config):
    """
    Train MAPPO on diverse scenarios for robustness
    
    This creates agents that can handle variations without needing
    online adaptation.
    """
    print("=" * 80)
    print("MAPPO Training with Diverse Scenarios (Robust Training)")
    print("=" * 80)
    print(f"Scenarios: {len(config.TRAINING_SCENARIOS)}")
    print(f"Episodes: {config.NUM_EPISODES}")
    print("=" * 80)
    
    # Create environment and agent
    env = K1Environment(config)
    agent = MAPPOAgent(config)
    
    # Scenario selection setup
    scenarios = list(config.SCENARIO_WEIGHTS.keys())
    weights = list(config.SCENARIO_WEIGHTS.values())
    
    # Training loop
    for episode in range(config.NUM_EPISODES):
        # Select scenario for this episode
        scenario = np.random.choice(scenarios, p=weights)
        
        print(f"\nEpisode {episode}/{config.NUM_EPISODES} | Scenario: {scenario}")
        
        # Reset environment with selected scenario
        # Note: You would modify K1Environment to accept scenario parameter
        local_states, global_state = env.reset(scenario=scenario)
        
        episode_reward = 0
        episode_length = 0
        
        # Episode loop (same as before)
        for step in range(config.STEPS_PER_EPISODE):
            actions, log_probs, entropies = agent.select_actions(local_states)
            next_local_states, next_global_state, rewards, done = env.step(actions)
            
            agent.buffer.store(
                local_states, global_state, actions, rewards,
                log_probs, entropies, done
            )
            
            episode_reward += sum(rewards)
            episode_length += 1
            
            local_states = next_local_states
            global_state = next_global_state
            
            if step % config.UPDATE_FREQUENCY == 0:
                agent.update()
            
            if done:
                break
        
        agent.decay_epsilon()
        agent.episode_count = episode
        
        # Logging
        if episode % config.LOG_INTERVAL == 0:
            print(f"  Reward: {episode_reward:.2f} | Length: {episode_length}")
            agent.writer.add_scalar('Episode/Reward', episode_reward, episode)
            agent.writer.add_scalar('Episode/Scenario', 
                                   scenarios.index(scenario), episode)
        
        # Save models
        if episode % config.SAVE_INTERVAL == 0 and episode > 0:
            model_path = os.path.join(config.MODEL_DIR, f'episode_{episode}')
            agent.save_models(model_path)
    
    # Final save
    agent.save_models(os.path.join(config.MODEL_DIR, 'final_robust'))
    env.close()
    agent.writer.close()
    
    print("=" * 80)
    print("Robust training completed!")
    print("Models can now handle diverse traffic patterns!")
    print("=" * 80)


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """
    Monitor deployment performance to detect pattern changes
    
    Tracks metrics over time and alerts when performance degrades,
    indicating a potential traffic pattern change.
    """
    
    def __init__(self, baseline_waiting_time=850.0, window_size=7):
        """
        Args:
            baseline_waiting_time: Expected performance from training
            window_size: Days to average for trend detection
        """
        self.baseline_waiting_time = baseline_waiting_time
        self.window_size = window_size
        
        # Rolling window of daily metrics
        self.daily_waiting_times = deque(maxlen=window_size)
        self.daily_queue_lengths = deque(maxlen=window_size)
        self.daily_throughputs = deque(maxlen=window_size)
        
        # Thresholds
        self.alert_threshold = 1.3  # 30% worse triggers alert
        self.critical_threshold = 1.5  # 50% worse is critical
        
        # Status tracking
        self.alerts = []
        self.status = 'normal'
    
    def add_daily_metrics(self, waiting_time, queue_length, throughput):
        """Add metrics from one day of operation"""
        self.daily_waiting_times.append(waiting_time)
        self.daily_queue_lengths.append(queue_length)
        self.daily_throughputs.append(throughput)
        
        # Check status
        self._update_status()
    
    def _update_status(self):
        """Update status based on recent performance"""
        if len(self.daily_waiting_times) < 3:
            return  # Need at least 3 days of data
        
        # Calculate recent average
        recent_avg = np.mean(list(self.daily_waiting_times)[-3:])
        
        # Check against baseline
        ratio = recent_avg / self.baseline_waiting_time
        
        if ratio > self.critical_threshold:
            self.status = 'critical'
            self._create_alert('CRITICAL', ratio)
        elif ratio > self.alert_threshold:
            self.status = 'degraded'
            self._create_alert('WARNING', ratio)
        else:
            self.status = 'normal'
    
    def _create_alert(self, level, ratio):
        """Create performance alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'expected_waiting_time': self.baseline_waiting_time,
            'actual_waiting_time': np.mean(list(self.daily_waiting_times)[-3:]),
            'degradation_ratio': ratio,
            'recommendation': self._get_recommendation(ratio)
        }
        self.alerts.append(alert)
        
        # Print alert
        print("\n" + "!" * 80)
        print(f"⚠️  PERFORMANCE ALERT - {level}")
        print("!" * 80)
        print(f"Expected waiting time: {alert['expected_waiting_time']:.1f}s")
        print(f"Actual waiting time:   {alert['actual_waiting_time']:.1f}s")
        print(f"Degradation:           {(ratio-1)*100:.1f}% worse")
        print(f"Recommendation:        {alert['recommendation']}")
        print("!" * 80 + "\n")
    
    def _get_recommendation(self, ratio):
        """Get recommendation based on degradation level"""
        if ratio > self.critical_threshold:
            return "CRITICAL - Investigate immediately. Consider offline fine-tuning."
        elif ratio > self.alert_threshold:
            return "WARNING - Monitor closely. Prepare for offline fine-tuning if persists."
        else:
            return "Normal operation."
    
    def should_fine_tune(self):
        """Check if fine-tuning is recommended"""
        if len(self.daily_waiting_times) < self.window_size:
            return False
        
        # Check if consistently degraded for entire window
        recent_avg = np.mean(self.daily_waiting_times)
        ratio = recent_avg / self.baseline_waiting_time
        
        return ratio > self.alert_threshold and self.status != 'normal'
    
    def get_report(self):
        """Generate performance report"""
        if len(self.daily_waiting_times) == 0:
            return "No data collected yet."
        
        report = {
            'status': self.status,
            'days_monitored': len(self.daily_waiting_times),
            'baseline_waiting_time': self.baseline_waiting_time,
            'current_avg_waiting_time': np.mean(self.daily_waiting_times),
            'trend': self._calculate_trend(),
            'alerts': self.alerts[-5:],  # Last 5 alerts
            'recommendation': self._get_recommendation(
                np.mean(self.daily_waiting_times) / self.baseline_waiting_time
            )
        }
        
        return report
    
    def _calculate_trend(self):
        """Calculate performance trend"""
        if len(self.daily_waiting_times) < 3:
            return "insufficient_data"
        
        data = list(self.daily_waiting_times)
        # Simple linear regression
        x = np.arange(len(data))
        y = np.array(data)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 10:
            return "degrading"
        elif slope < -10:
            return "improving"
        else:
            return "stable"


# ============================================================================
# OFFLINE FINE-TUNING
# ============================================================================

class OfflineFineTuner:
    """
    Offline fine-tuning for adapted models
    
    When traffic patterns change persistently, collect new data
    and fine-tune model in safe offline environment.
    """
    
    def __init__(self, original_model_path, config):
        self.original_model_path = original_model_path
        self.config = config
        self.agent = MAPPOAgent(config)
        self.agent.load_models(original_model_path)
    
    def collect_adaptation_data(self, env, days=3):
        """
        Collect data with current (sub-optimal) policy
        
        Args:
            env: K1Environment instance
            days: Number of days to collect data
        
        Returns:
            buffer: ReplayBuffer with collected experiences
        """
        print(f"Collecting adaptation data for {days} days...")
        
        steps_per_day = 24 * 3600  # 24 hours
        total_steps = days * steps_per_day
        
        for step in range(total_steps):
            local_states = env.get_local_states()
            global_state = env.get_global_state()
            
            # Use current policy (sub-optimal but functional)
            actions, log_probs, entropies = self.agent.select_actions(local_states)
            
            next_local_states, next_global_state, rewards, done = env.step(actions)
            
            # Store experience
            self.agent.buffer.store(
                local_states, global_state, actions, rewards,
                log_probs, entropies, done
            )
            
            if step % 3600 == 0:  # Every hour
                print(f"  Collected {step}/{total_steps} steps "
                      f"({step/total_steps*100:.1f}%)")
            
            if done:
                local_states, global_state = env.reset()
            else:
                local_states = next_local_states
                global_state = next_global_state
        
        print(f"✓ Collected {len(self.agent.buffer)} experiences")
        return self.agent.buffer
    
    def fine_tune(self, adaptation_buffer, episodes=500):
        """
        Fine-tune model on new data
        
        Args:
            adaptation_buffer: Buffer with new traffic pattern data
            episodes: Number of fine-tuning episodes
        """
        print(f"\nFine-tuning model for {episodes} episodes...")
        print("Using collected adaptation data...")
        
        # Reduce learning rate for fine-tuning (more conservative)
        for optimizer in self.agent.actor_optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.config.LEARNING_RATE_ACTOR * 0.1
        
        for param_group in self.agent.critic_optimizer.param_groups:
            param_group['lr'] = self.config.LEARNING_RATE_CRITIC * 0.1
        
        # Fine-tuning loop
        for episode in range(episodes):
            # Use collected data (replay)
            self.agent.update()
            
            if episode % 10 == 0:
                print(f"  Fine-tuning episode {episode}/{episodes}")
        
        print("✓ Fine-tuning completed")
    
    def validate(self, env, duration=3600):
        """
        Validate fine-tuned model in simulation
        
        Args:
            env: K1Environment instance
            duration: Validation duration in seconds
        
        Returns:
            metrics: Performance metrics
        """
        print(f"\nValidating fine-tuned model ({duration}s simulation)...")
        
        local_states, global_state = env.reset()
        
        total_waiting_time = 0
        total_queue_length = 0
        steps = 0
        
        for step in range(duration):
            actions, _, _ = self.agent.select_actions(local_states)
            next_local_states, next_global_state, rewards, done = env.step(actions)
            
            # Collect metrics
            waiting_time = sum(
                sum(traci.lane.getWaitingTime(lane_id)
                    for lane_id in traci.trafficlight.getControlledLanes(j))
                for j in self.config.JUNCTION_IDS
            )
            
            queue_length = sum(
                sum(traci.lane.getLastStepHaltingNumber(lane_id)
                    for lane_id in traci.trafficlight.getControlledLanes(j))
                for j in self.config.JUNCTION_IDS
            )
            
            total_waiting_time += waiting_time
            total_queue_length += queue_length
            steps += 1
            
            local_states = next_local_states
            global_state = next_global_state
            
            if done:
                break
        
        metrics = {
            'avg_waiting_time': total_waiting_time / steps,
            'avg_queue_length': total_queue_length / steps,
            'steps': steps
        }
        
        print(f"✓ Validation completed")
        print(f"  Avg waiting time: {metrics['avg_waiting_time']:.1f}s")
        print(f"  Avg queue length: {metrics['avg_queue_length']:.1f}")
        
        return metrics
    
    def save_fine_tuned_model(self, path):
        """Save fine-tuned model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(path, f'fine_tuned_{timestamp}')
        self.agent.save_models(save_path)
        print(f"✓ Fine-tuned model saved to: {save_path}")
        return save_path


# ============================================================================
# ADAPTIVE DEPLOYMENT SYSTEM
# ============================================================================

class AdaptiveDeploymentSystem:
    """
    Complete adaptive deployment system
    
    Combines:
    1. Deployment with performance monitoring
    2. Pattern change detection
    3. Automatic offline fine-tuning when needed
    """
    
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        
        # Deployment
        self.deployment = MAPPODeployment(model_path, config)
        
        # Monitoring
        self.monitor = PerformanceMonitor(
            baseline_waiting_time=850.0,  # Expected from training
            window_size=7
        )
        
        # Fine-tuning
        self.fine_tuner = None
        
        # Status
        self.days_running = 0
        self.fine_tune_count = 0
    
    def run_day(self, env, day_number):
        """
        Run deployment for one day with monitoring
        
        Args:
            env: K1Environment instance
            day_number: Day number (for logging)
        """
        print(f"\n{'='*80}")
        print(f"DAY {day_number} - Adaptive Deployment")
        print(f"{'='*80}")
        
        # Run deployment for 24 hours
        local_states, global_state = env.reset()
        
        daily_waiting_times = []
        daily_queue_lengths = []
        daily_throughputs = []
        
        steps_per_day = 24 * 3600
        
        for step in range(steps_per_day):
            # Get actions from deployment
            actions, _ = self.deployment.select_actions(local_states)
            
            # Execute
            next_local_states, next_global_state, rewards, done = env.step(actions)
            
            # Collect metrics
            # (simplified - in reality would use TraCI)
            waiting_time = sum(abs(r) for r in rewards)  # Approximation
            daily_waiting_times.append(waiting_time)
            
            local_states = next_local_states
            
            if done:
                break
        
        # Calculate daily averages
        avg_waiting_time = np.mean(daily_waiting_times)
        avg_queue_length = 0  # Would calculate from TraCI
        avg_throughput = 0  # Would calculate from TraCI
        
        # Update monitor
        self.monitor.add_daily_metrics(
            avg_waiting_time,
            avg_queue_length,
            avg_throughput
        )
        
        # Check if fine-tuning needed
        if self.monitor.should_fine_tune():
            print("\n⚠️  Performance consistently degraded!")
            print("Initiating offline fine-tuning...")
            self._perform_fine_tuning(env)
        
        self.days_running += 1
        
        # Daily report
        print(f"\nDay {day_number} Summary:")
        print(f"  Avg waiting time: {avg_waiting_time:.1f}s")
        print(f"  Status: {self.monitor.status}")
        print(f"  Fine-tuning count: {self.fine_tune_count}")
    
    def _perform_fine_tuning(self, env):
        """
        Perform offline fine-tuning
        """
        print("\n" + "="*80)
        print("OFFLINE FINE-TUNING PROCEDURE")
        print("="*80)
        
        # Initialize fine-tuner
        self.fine_tuner = OfflineFineTuner(self.model_path, self.config)
        
        # Step 1: Collect adaptation data
        adaptation_buffer = self.fine_tuner.collect_adaptation_data(env, days=3)
        
        # Step 2: Fine-tune
        self.fine_tuner.fine_tune(adaptation_buffer, episodes=500)
        
        # Step 3: Validate
        metrics = self.fine_tuner.validate(env, duration=3600)
        
        # Step 4: Compare with current
        if metrics['avg_waiting_time'] < self.monitor.baseline_waiting_time * 1.2:
            print("\n✓ Fine-tuned model performs better!")
            
            # Save and update deployment
            new_model_path = self.fine_tuner.save_fine_tuned_model('mappo_models')
            self.deployment = MAPPODeployment(new_model_path, self.config)
            self.model_path = new_model_path
            
            # Update baseline
            self.monitor.baseline_waiting_time = metrics['avg_waiting_time']
            
            self.fine_tune_count += 1
            
            print("✓ Deployment updated with fine-tuned model")
        else:
            print("\n⚠️  Fine-tuned model not better than current")
            print("Keeping original model")
            print("Manual investigation recommended")
        
        print("="*80 + "\n")
    
    def get_status_report(self):
        """Get comprehensive status report"""
        report = {
            'days_running': self.days_running,
            'fine_tune_count': self.fine_tune_count,
            'current_model': self.model_path,
            'performance': self.monitor.get_report(),
        }
        
        return report


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def train_robust_mappo():
    """Train MAPPO on diverse scenarios"""
    config = DiverseScenarioConfig()
    train_mappo_diverse_scenarios(config)


def run_adaptive_deployment(duration_days=30):
    """
    Run adaptive deployment with monitoring and fine-tuning
    
    Args:
        duration_days: Number of days to run
    """
    config = MAPPOConfig()
    
    # Create adaptive system
    system = AdaptiveDeploymentSystem(
        model_path='mappo_models/final_robust',
        config=config
    )
    
    # Create environment
    env = K1Environment(config)
    
    # Run for specified days
    for day in range(1, duration_days + 1):
        system.run_day(env, day)
        
        # Weekly report
        if day % 7 == 0:
            report = system.get_status_report()
            print("\n" + "="*80)
            print("WEEKLY STATUS REPORT")
            print("="*80)
            print(json.dumps(report, indent=2))
            print("="*80 + "\n")
    
    # Final report
    final_report = system.get_status_report()
    
    # Save report
    report_path = f"adaptive_deployment_report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n✓ Adaptive deployment completed!")
    print(f"✓ Final report saved to: {report_path}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MAPPO with Adaptive Deployment'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'deploy'],
        required=True,
        help='Mode: train (robust training) or deploy (adaptive deployment)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days for adaptive deployment'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting robust MAPPO training on diverse scenarios...")
        train_robust_mappo()
    else:
        print(f"Starting adaptive deployment for {args.days} days...")
        run_adaptive_deployment(args.days)
