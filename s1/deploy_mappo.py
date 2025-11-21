"""
MAPPO Deployment Script for K1 Network
======================================

Deploy trained MAPPO agents on K1 network for real-time traffic control.

This script demonstrates:
1. Loading trained models
2. Decentralized execution (local sensors only)
3. Real-time traffic control
4. Performance monitoring

Usage:
    python deploy_mappo.py --model mappo_models/final

Author: Traffic Light Automation Team
Date: November 2025
"""

import os
import sys
import argparse
import numpy as np
import torch
import traci
import time
from datetime import datetime
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# Import from training script
from mappo_k1_implementation import (
    ActorNetwork,
    MAPPOConfig,
    K1Environment
)


class MAPPODeployment:
    """
    MAPPO Deployment Manager
    
    Handles model loading and real-time inference for traffic control.
    """
    
    def __init__(self, model_path, config):
        self.config = config
        self.model_path = model_path
        
        # Load trained actors
        self.actors = self._load_actors()
        
        # Set to evaluation mode
        for actor in self.actors:
            actor.eval()
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        self.junction_metrics = {j: defaultdict(list) for j in config.JUNCTION_IDS}
    
    def _load_actors(self):
        """Load trained actor networks"""
        actors = []
        
        for i in range(self.config.NUM_JUNCTIONS):
            actor = ActorNetwork(
                state_dim=self.config.LOCAL_STATE_DIM,
                action_dim=self.config.ACTION_DIM,
                hidden_dims=self.config.ACTOR_HIDDEN
            )
            
            # Load weights
            model_file = os.path.join(self.model_path, f'actor_{i}.pth')
            if os.path.exists(model_file):
                actor.load_state_dict(torch.load(model_file))
                print(f"✓ Loaded actor {i} from {model_file}")
            else:
                print(f"✗ Warning: Actor {i} model not found at {model_file}")
            
            actors.append(actor)
        
        return actors
    
    def select_actions(self, local_states):
        """
        Select actions for all junctions (inference mode - no exploration)
        
        Args:
            local_states: List of local states (9 × 17)
        
        Returns:
            actions: List of actions (9 integers)
            action_probs: List of action probability distributions
        """
        actions = []
        action_probs_list = []
        
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                # Get action probabilities
                state_tensor = torch.FloatTensor(local_states[i]).unsqueeze(0)
                action_probs = actor(state_tensor).squeeze()
                
                # Select action with highest probability (greedy)
                action = torch.argmax(action_probs).item()
                
                actions.append(action)
                action_probs_list.append(action_probs.numpy())
        
        return actions, action_probs_list
    
    def run_deployment(self, duration_seconds=3600, scenario='custom_24h'):
        """
        Run deployment for specified duration
        
        Args:
            duration_seconds: Simulation duration in seconds
            scenario: Traffic scenario to use
        """
        print("=" * 80)
        print("MAPPO Deployment - Real-Time Traffic Control")
        print("=" * 80)
        print(f"Duration: {duration_seconds}s ({duration_seconds/3600:.1f} hours)")
        print(f"Scenario: {scenario}")
        print(f"Model: {self.model_path}")
        print("=" * 80)
        
        # Create environment
        env = K1Environment(self.config)
        
        # Reset environment
        local_states, global_state = env.reset()
        
        start_time = time.time()
        step = 0
        
        # Deployment loop
        while step < duration_seconds:
            # Select actions using trained policies
            actions, action_probs = self.select_actions(local_states)
            
            # Execute actions
            next_local_states, next_global_state, rewards, done = env.step(actions)
            
            # Collect metrics
            self._collect_metrics(env, step, actions, action_probs, rewards)
            
            # Move to next state
            local_states = next_local_states
            global_state = next_global_state
            step += 1
            
            # Progress update
            if step % 600 == 0:  # Every 10 minutes
                elapsed = time.time() - start_time
                progress = (step / duration_seconds) * 100
                print(f"Progress: {progress:.1f}% | "
                      f"Step: {step}/{duration_seconds} | "
                      f"Elapsed: {elapsed:.1f}s")
            
            if done:
                print("Simulation ended (no more vehicles)")
                break
        
        # Close environment
        env.close()
        
        # Generate report
        self._generate_report()
        
        print("=" * 80)
        print("Deployment completed!")
        print("=" * 80)
    
    def _collect_metrics(self, env, step, actions, action_probs, rewards):
        """Collect performance metrics"""
        # Network-wide metrics
        total_vehicles = traci.vehicle.getIDCount()
        total_waiting_time = 0
        total_queue_length = 0
        
        self.metrics['step'].append(step)
        self.metrics['total_vehicles'].append(total_vehicles)
        
        # Per-junction metrics
        for i, junction_id in enumerate(self.config.JUNCTION_IDS):
            incoming_lanes = traci.trafficlight.getControlledLanes(junction_id)
            
            # Waiting time
            waiting_time = sum(
                traci.lane.getWaitingTime(lane_id)
                for lane_id in incoming_lanes
            )
            total_waiting_time += waiting_time
            
            # Queue length
            queue_length = sum(
                traci.lane.getLastStepHaltingNumber(lane_id)
                for lane_id in incoming_lanes
            )
            total_queue_length += queue_length
            
            # Store per-junction metrics
            self.junction_metrics[junction_id]['waiting_time'].append(waiting_time)
            self.junction_metrics[junction_id]['queue_length'].append(queue_length)
            self.junction_metrics[junction_id]['action'].append(actions[i])
            self.junction_metrics[junction_id]['reward'].append(rewards[i])
            self.junction_metrics[junction_id]['action_prob'].append(action_probs[i])
        
        self.metrics['total_waiting_time'].append(total_waiting_time)
        self.metrics['total_queue_length'].append(total_queue_length)
    
    def _generate_report(self):
        """Generate deployment report with visualizations"""
        print("\nGenerating deployment report...")
        
        # Create report directory
        report_dir = f"deployment_reports/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Summary statistics
        summary = {
            'total_steps': len(self.metrics['step']),
            'avg_vehicles': np.mean(self.metrics['total_vehicles']),
            'avg_waiting_time': np.mean(self.metrics['total_waiting_time']),
            'avg_queue_length': np.mean(self.metrics['total_queue_length']),
            'peak_vehicles': np.max(self.metrics['total_vehicles']),
            'peak_waiting_time': np.max(self.metrics['total_waiting_time']),
            'peak_queue_length': np.max(self.metrics['total_queue_length']),
        }
        
        # Per-junction summary
        junction_summary = {}
        for junction_id in self.config.JUNCTION_IDS:
            junction_summary[junction_id] = {
                'avg_waiting_time': np.mean(self.junction_metrics[junction_id]['waiting_time']),
                'avg_queue_length': np.mean(self.junction_metrics[junction_id]['queue_length']),
                'avg_reward': np.mean(self.junction_metrics[junction_id]['reward']),
                'action_distribution': self._action_distribution(junction_id),
            }
        
        summary['junctions'] = junction_summary
        
        # Save summary as JSON
        with open(os.path.join(report_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("DEPLOYMENT SUMMARY")
        print("=" * 80)
        print(f"Total steps: {summary['total_steps']}")
        print(f"Average vehicles: {summary['avg_vehicles']:.1f}")
        print(f"Average waiting time: {summary['avg_waiting_time']:.1f}s")
        print(f"Average queue length: {summary['avg_queue_length']:.1f}")
        print(f"Peak vehicles: {summary['peak_vehicles']}")
        print(f"Peak waiting time: {summary['peak_waiting_time']:.1f}s")
        print(f"Peak queue length: {summary['peak_queue_length']:.1f}")
        print("=" * 80)
        
        # 2. Generate plots
        self._plot_network_metrics(report_dir)
        self._plot_junction_metrics(report_dir)
        self._plot_action_distributions(report_dir)
        
        print(f"\n✓ Report saved to: {report_dir}")
    
    def _action_distribution(self, junction_id):
        """Calculate action distribution for junction"""
        actions = self.junction_metrics[junction_id]['action']
        unique, counts = np.unique(actions, return_counts=True)
        
        distribution = {
            'keep': 0,
            'next_phase': 0,
            'extend': 0,
            'emergency': 0
        }
        
        action_names = ['keep', 'next_phase', 'extend', 'emergency']
        for action, count in zip(unique, counts):
            distribution[action_names[action]] = int(count)
        
        return distribution
    
    def _plot_network_metrics(self, report_dir):
        """Plot network-wide metrics over time"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        steps = np.array(self.metrics['step']) / 60  # Convert to minutes
        
        # Total vehicles
        axes[0].plot(steps, self.metrics['total_vehicles'], linewidth=1)
        axes[0].set_ylabel('Total Vehicles')
        axes[0].set_title('Network-Wide Metrics Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Total waiting time
        axes[1].plot(steps, self.metrics['total_waiting_time'], linewidth=1, color='orange')
        axes[1].set_ylabel('Total Waiting Time (s)')
        axes[1].grid(True, alpha=0.3)
        
        # Total queue length
        axes[2].plot(steps, self.metrics['total_queue_length'], linewidth=1, color='red')
        axes[2].set_ylabel('Total Queue Length')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'network_metrics.png'), dpi=150)
        plt.close()
        
        print("✓ Network metrics plot saved")
    
    def _plot_junction_metrics(self, report_dir):
        """Plot per-junction metrics"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Per-Junction Performance', fontsize=16)
        
        for i, junction_id in enumerate(self.config.JUNCTION_IDS):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            steps = np.arange(len(self.junction_metrics[junction_id]['waiting_time'])) / 60
            waiting_times = self.junction_metrics[junction_id]['waiting_time']
            
            ax.plot(steps, waiting_times, linewidth=0.8)
            ax.set_title(f'{junction_id}')
            ax.set_ylabel('Waiting Time (s)')
            if row == 2:
                ax.set_xlabel('Time (min)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'junction_metrics.png'), dpi=150)
        plt.close()
        
        print("✓ Junction metrics plot saved")
    
    def _plot_action_distributions(self, report_dir):
        """Plot action distribution for each junction"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle('Action Distribution per Junction', fontsize=16)
        
        action_names = ['Keep', 'Next Phase', 'Extend', 'Emergency']
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        
        for i, junction_id in enumerate(self.config.JUNCTION_IDS):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            distribution = self._action_distribution(junction_id)
            values = [distribution['keep'], distribution['next_phase'], 
                     distribution['extend'], distribution['emergency']]
            
            ax.bar(action_names, values, color=colors)
            ax.set_title(f'{junction_id}')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'action_distributions.png'), dpi=150)
        plt.close()
        
        print("✓ Action distribution plot saved")


def compare_with_baseline(mappo_model_path, config, duration=3600):
    """
    Compare MAPPO with baseline (fixed-time) control
    
    Args:
        mappo_model_path: Path to trained MAPPO models
        config: MAPPOConfig instance
        duration: Simulation duration in seconds
    """
    print("=" * 80)
    print("MAPPO vs Baseline Comparison")
    print("=" * 80)
    
    results = {}
    
    # 1. Run baseline (fixed-time)
    print("\n[1/2] Running baseline (fixed-time)...")
    env = K1Environment(config)
    local_states, global_state = env.reset()
    
    baseline_metrics = {
        'waiting_times': [],
        'queue_lengths': [],
        'vehicles': [],
    }
    
    for step in range(duration):
        # Fixed-time: do nothing, let SUMO's fixed timing handle it
        actions = [0] * config.NUM_JUNCTIONS  # Keep current phase
        next_local_states, next_global_state, rewards, done = env.step(actions)
        
        # Collect metrics
        total_waiting = sum(
            traci.lane.getWaitingTime(lane_id)
            for junction_id in config.JUNCTION_IDS
            for lane_id in traci.trafficlight.getControlledLanes(junction_id)
        )
        total_queue = sum(
            traci.lane.getLastStepHaltingNumber(lane_id)
            for junction_id in config.JUNCTION_IDS
            for lane_id in traci.trafficlight.getControlledLanes(junction_id)
        )
        total_vehicles = traci.vehicle.getIDCount()
        
        baseline_metrics['waiting_times'].append(total_waiting)
        baseline_metrics['queue_lengths'].append(total_queue)
        baseline_metrics['vehicles'].append(total_vehicles)
        
        local_states = next_local_states
        global_state = next_global_state
        
        if done:
            break
    
    env.close()
    
    results['baseline'] = {
        'avg_waiting_time': np.mean(baseline_metrics['waiting_times']),
        'avg_queue_length': np.mean(baseline_metrics['queue_lengths']),
        'avg_vehicles': np.mean(baseline_metrics['vehicles']),
    }
    
    print(f"✓ Baseline completed")
    print(f"  Avg waiting time: {results['baseline']['avg_waiting_time']:.1f}s")
    print(f"  Avg queue length: {results['baseline']['avg_queue_length']:.1f}")
    
    # 2. Run MAPPO
    print("\n[2/2] Running MAPPO...")
    deployment = MAPPODeployment(mappo_model_path, config)
    
    env = K1Environment(config)
    local_states, global_state = env.reset()
    
    mappo_metrics = {
        'waiting_times': [],
        'queue_lengths': [],
        'vehicles': [],
    }
    
    for step in range(duration):
        # MAPPO actions
        actions, _ = deployment.select_actions(local_states)
        next_local_states, next_global_state, rewards, done = env.step(actions)
        
        # Collect metrics
        total_waiting = sum(
            traci.lane.getWaitingTime(lane_id)
            for junction_id in config.JUNCTION_IDS
            for lane_id in traci.trafficlight.getControlledLanes(junction_id)
        )
        total_queue = sum(
            traci.lane.getLastStepHaltingNumber(lane_id)
            for junction_id in config.JUNCTION_IDS
            for lane_id in traci.trafficlight.getControlledLanes(junction_id)
        )
        total_vehicles = traci.vehicle.getIDCount()
        
        mappo_metrics['waiting_times'].append(total_waiting)
        mappo_metrics['queue_lengths'].append(total_queue)
        mappo_metrics['vehicles'].append(total_vehicles)
        
        local_states = next_local_states
        global_state = next_global_state
        
        if done:
            break
    
    env.close()
    
    results['mappo'] = {
        'avg_waiting_time': np.mean(mappo_metrics['waiting_times']),
        'avg_queue_length': np.mean(mappo_metrics['queue_lengths']),
        'avg_vehicles': np.mean(mappo_metrics['vehicles']),
    }
    
    print(f"✓ MAPPO completed")
    print(f"  Avg waiting time: {results['mappo']['avg_waiting_time']:.1f}s")
    print(f"  Avg queue length: {results['mappo']['avg_queue_length']:.1f}")
    
    # 3. Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    improvement_waiting = (
        (results['baseline']['avg_waiting_time'] - results['mappo']['avg_waiting_time'])
        / results['baseline']['avg_waiting_time'] * 100
    )
    improvement_queue = (
        (results['baseline']['avg_queue_length'] - results['mappo']['avg_queue_length'])
        / results['baseline']['avg_queue_length'] * 100
    )
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'MAPPO':<15} {'Improvement'}")
    print("-" * 80)
    print(f"{'Waiting Time (s)':<20} {results['baseline']['avg_waiting_time']:<15.1f} "
          f"{results['mappo']['avg_waiting_time']:<15.1f} {improvement_waiting:>+.1f}%")
    print(f"{'Queue Length':<20} {results['baseline']['avg_queue_length']:<15.1f} "
          f"{results['mappo']['avg_queue_length']:<15.1f} {improvement_queue:>+.1f}%")
    print(f"{'Vehicles':<20} {results['baseline']['avg_vehicles']:<15.1f} "
          f"{results['mappo']['avg_vehicles']:<15.1f}")
    
    print("=" * 80)
    
    # Save comparison
    comparison_dir = f"deployment_reports/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(comparison_dir, exist_ok=True)
    
    with open(os.path.join(comparison_dir, 'comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {comparison_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Deploy MAPPO for K1 traffic control')
    parser.add_argument('--model', type=str, default='mappo_models/final',
                       help='Path to trained model directory')
    parser.add_argument('--duration', type=int, default=3600,
                       help='Simulation duration in seconds')
    parser.add_argument('--scenario', type=str, default='custom_24h',
                       help='Traffic scenario to use')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with baseline')
    
    args = parser.parse_args()
    
    # Create configuration
    config = MAPPOConfig()
    
    if args.compare:
        # Run comparison
        compare_with_baseline(args.model, config, args.duration)
    else:
        # Run deployment
        deployment = MAPPODeployment(args.model, config)
        deployment.run_deployment(args.duration, args.scenario)


if __name__ == "__main__":
    main()
