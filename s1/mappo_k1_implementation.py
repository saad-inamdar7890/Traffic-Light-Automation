"""
MAPPO Implementation for K1 Traffic Network
===========================================

Multi-Agent Proximal Policy Optimization for coordinated traffic light control
on the K1 network with 9 junctions.

Features:
- Centralized Training, Decentralized Execution (CTDE)
- Realistic sensor inputs (vehicle types, queues, occupancy)
- Coordinated learning across 9 junctions
- PPO stability for training
- Deployable with local sensors only

Author: Traffic Light Automation Team
Date: November 2025
"""

import os
import sys
import time
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import traci
from datetime import datetime
import json
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

class MAPPOConfig:
    """Configuration for MAPPO training"""
    
    # Environment
    SUMO_CONFIG = "k1.sumocfg"
    NUM_JUNCTIONS = 9
    JUNCTION_IDS = ["J0", "J1", "J5", "J6", "J7", "J10", "J11", "J12", "J22"]
    
    # Network topology (neighbors for each junction)
    NEIGHBORS = {
        "J0": ["J6", "J11"],
        "J1": ["J5", "J10"],
        "J5": ["J1", "J10"],
        "J6": ["J0", "J11", "J7"],
        "J7": ["J6", "J22", "J12"],
        "J10": ["J1", "J5", "J11", "J12"],
        "J11": ["J0", "J6", "J10", "J22"],
        "J12": ["J7", "J10", "J22"],
        "J22": ["J7", "J11", "J12"]
    }
    
    # State dimensions
    LOCAL_STATE_DIM = 17   # Per junction
    GLOBAL_STATE_DIM = 155 # All junctions + network features
    ACTION_DIM = 4         # [keep, next_phase, extend, emergency]
    
    # Vehicle type weights (Passenger Car Equivalents - PCE)
    VEHICLE_WEIGHTS = {
        'passenger': 1.0,
        'delivery': 2.5,
        'truck': 5.0,
        'bus': 4.5,
        'emergency': 1.5
    }
    
    # Network architecture
    ACTOR_HIDDEN = [128, 64]
    CRITIC_HIDDEN = [256, 256, 128]
    
    # Training hyperparameters
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 1e-3
    GAMMA = 0.99              # Discount factor
    LAMBDA = 0.95             # GAE parameter
    CLIP_EPSILON = 0.2        # PPO clip parameter
    ENTROPY_COEF = 0.01       # Exploration bonus
    GRAD_CLIP = 0.5           # Gradient clipping
    
    # Training schedule
    STEPS_PER_EPISODE = 3600  # 1 hour simulation
    NUM_EPISODES = 5000
    UPDATE_FREQUENCY = 128     # Steps between updates
    PPO_EPOCHS = 10           # Updates per batch
    
    # Exploration
    EPSILON_START = 0.1
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    
    # Reward weights
    REWARD_WEIGHT_OWN = 0.6        # Own junction performance
    REWARD_WEIGHT_NEIGHBORS = 0.3   # Downstream impact
    REWARD_WEIGHT_NETWORK = 0.1    # Network-wide
    EMERGENCY_BONUS = 5.0
    DEADLOCK_PENALTY = -10.0
    
    # Logging
    LOG_INTERVAL = 10         # Episodes between logs
    SAVE_INTERVAL = 100       # Episodes between model saves
    TENSORBOARD_DIR = "mappo_logs"
    MODEL_DIR = "mappo_models"


# ============================================================================
# NEURAL NETWORKS
# ============================================================================

class ActorNetwork(nn.Module):
    """
    Actor Network - Policy for individual junction
    
    Takes LOCAL state (own junction + neighbors) and outputs action probabilities.
    Each junction has its own actor network.
    """
    
    def __init__(self, state_dim=17, action_dim=4, hidden_dims=[128, 64]):
        super(ActorNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: Local state tensor (batch_size, 17)
                   [current_phase, queue_n, queue_s, queue_e, queue_w,
                    weighted_veh_n, weighted_veh_s, weighted_veh_e, weighted_veh_w,
                    occupancy_n, occupancy_s, occupancy_e, occupancy_w,
                    time_in_phase, emergency, neighbor1_phase, neighbor2_phase]
        
        Returns:
            action_probs: Action probabilities (batch_size, 4)
        """
        logits = self.network(state)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs
    
    def get_action(self, state, epsilon=0.0):
        """
        Sample action from policy
        
        Args:
            state: Local state (17,)
            epsilon: Exploration rate (0.0 = no exploration)
        
        Returns:
            action: Sampled action (integer)
            log_prob: Log probability of action (scalar tensor)
            entropy: Policy entropy (scalar tensor)
        """
        if np.random.random() < epsilon:
            # Random exploration
            action = np.random.randint(0, 4)
            log_prob = torch.tensor(np.log(0.25), dtype=torch.float32)
            entropy = torch.tensor(np.log(4), dtype=torch.float32)
            return action, log_prob, entropy
        
        # Get action probabilities
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).squeeze()  # Ensure scalar
        entropy = dist.entropy().squeeze()  # Ensure scalar
        
        return action.item(), log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Critic Network - Value function for global state
    
    Takes GLOBAL state (all junctions + network features) and outputs value.
    Shared by all junctions - enables coordination learning.
    """
    
    def __init__(self, state_dim=155, hidden_dims=[256, 256, 128]):
        super(CriticNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: Global state tensor (batch_size, 155)
                   [All junction local states + network features]
        
        Returns:
            value: State value (batch_size, 1)
        """
        value = self.network(state)
        return value


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for MAPPO
    
    Stores trajectories during episode for batch training.
    """
    
    def __init__(self, num_agents=9):
        self.num_agents = num_agents
        self.clear()
    
    def clear(self):
        """Clear all stored experiences"""
        # Store per-agent data
        self.local_states = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.log_probs = [[] for _ in range(self.num_agents)]
        self.entropies = [[] for _ in range(self.num_agents)]
        
        # Store global data
        self.global_states = []
        self.dones = []
    
    def store(self, local_states, global_state, actions, rewards, log_probs, entropies, done):
        """
        Store one timestep of experience
        
        Args:
            local_states: List of local states (9 × 17)
            global_state: Global state (155,)
            actions: List of actions (9 integers)
            rewards: List of rewards (9 floats)
            log_probs: List of log probabilities (9 tensors)
            entropies: List of entropies (9 tensors)
            done: Episode done flag
        """
        for i in range(self.num_agents):
            self.local_states[i].append(local_states[i])
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
            self.log_probs[i].append(log_probs[i])
            self.entropies[i].append(entropies[i])
        
        self.global_states.append(global_state)
        self.dones.append(done)
    
    def get_batch(self):
        """
        Get all stored experiences as batch
        
        Returns:
            batch: Dictionary with tensors for training
        """
        batch = {
            'local_states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'entropies': [],
        }
        
        # Convert per-agent data to tensors (optimize by converting to numpy first)
        for i in range(self.num_agents):
            # Convert lists to numpy arrays first (much faster)
            batch['local_states'].append(torch.FloatTensor(np.array(self.local_states[i], dtype=np.float32)))
            batch['actions'].append(torch.LongTensor(np.array(self.actions[i], dtype=np.int64)))
            batch['rewards'].append(torch.FloatTensor(np.array(self.rewards[i], dtype=np.float32)))
            batch['log_probs'].append(torch.stack(self.log_probs[i]))
            batch['entropies'].append(torch.stack(self.entropies[i]))
        
        # Convert global data
        batch['global_states'] = torch.FloatTensor(np.array(self.global_states, dtype=np.float32))
        batch['dones'] = torch.FloatTensor(np.array(self.dones, dtype=np.float32))
        
        return batch
    
    def __len__(self):
        """Return number of timesteps stored"""
        return len(self.global_states)


# ============================================================================
# SUMO ENVIRONMENT WRAPPER
# ============================================================================

class K1Environment:
    """
    SUMO environment wrapper for K1 network
    
    Handles:
    - State extraction (realistic sensors)
    - Action execution (phase changes)
    - Reward computation (coordinated rewards)
    """
    
    def __init__(self, config):
        self.config = config
        self.junction_ids = config.JUNCTION_IDS
        self.neighbors = config.NEIGHBORS
        self.vehicle_weights = config.VEHICLE_WEIGHTS
        
        # Track metrics for rewards
        self.prev_waiting_times = defaultdict(float)
        self.prev_queue_lengths = defaultdict(float)
        self.emergency_cleared = defaultdict(bool)
        
        # Phase tracking
        self.current_phases = {}
        self.time_in_phase = {}
        
        # Initialize SUMO
        self._init_sumo()
    
    def _init_sumo(self):
        """Initialize SUMO simulation"""
        # Use headless SUMO (works on servers/Colab without display)
        # Check if DISPLAY is available, otherwise force headless
        import shutil
        if os.environ.get('DISPLAY') and shutil.which('sumo-gui'):
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"
        
        # Make config path absolute if relative
        config_path = self.config.SUMO_CONFIG
        if not os.path.isabs(config_path):
            # Try to find it relative to the script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
            if not os.path.exists(config_path):
                # Fallback: relative to current working directory
                config_path = self.config.SUMO_CONFIG
        
        sumo_cmd = [
            sumo_binary,
            "-c", config_path,
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--no-step-log",
        ]
        
        try:
            traci.start(sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO with config: {config_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Config file exists: {os.path.exists(config_path)}")
            print(f"SUMO binary: {sumo_binary}")
            raise e
        
        # Initialize phase tracking
        for junction_id in self.junction_ids:
            self.current_phases[junction_id] = 0
            self.time_in_phase[junction_id] = 0
    
    def reset(self):
        """Reset environment for new episode"""
        # Close existing connection
        if traci.isLoaded():
            traci.close()
        
        # Restart SUMO
        self._init_sumo()
        
        # Reset tracking
        self.prev_waiting_times.clear()
        self.prev_queue_lengths.clear()
        self.emergency_cleared.clear()
        self.current_phases = {j: 0 for j in self.junction_ids}
        self.time_in_phase = {j: 0 for j in self.junction_ids}
        
        # Get initial states
        local_states = self.get_local_states()
        global_state = self.get_global_state()
        
        return local_states, global_state
    
    def get_local_state(self, junction_id):
        """
        Get local state for specific junction (realistic sensors only)
        
        Returns:
            state: numpy array (17,)
                [current_phase, queue_n, queue_s, queue_e, queue_w,
                 weighted_veh_n, weighted_veh_s, weighted_veh_e, weighted_veh_w,
                 occupancy_n, occupancy_s, occupancy_e, occupancy_w,
                 time_in_phase, emergency, neighbor1_phase, neighbor2_phase]
        """
        state = []
        
        # 1. Current phase (0-7)
        current_phase = self.current_phases[junction_id]
        state.append(current_phase)
        
        # 2. Queue lengths per direction (realistic: induction loops)
        incoming_lanes = traci.trafficlight.getControlledLanes(junction_id)
        
        queues = {'n': 0, 's': 0, 'e': 0, 'w': 0}
        weighted_vehicles = {'n': 0, 's': 0, 'e': 0, 'w': 0}
        occupancy = {'n': 0, 's': 0, 'e': 0, 'w': 0}
        
        for lane_id in incoming_lanes:
            # Determine direction
            if '_n' in lane_id or 'n_' in lane_id:
                direction = 'n'
            elif '_s' in lane_id or 's_' in lane_id:
                direction = 's'
            elif '_e' in lane_id or 'e_' in lane_id:
                direction = 'e'
            elif '_w' in lane_id or 'w_' in lane_id:
                direction = 'w'
            else:
                continue
            
            # Queue length (halting vehicles)
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            queues[direction] += queue
            
            # Weighted vehicle count (by type)
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            weighted_count = 0
            for veh_id in vehicle_ids:
                veh_type = traci.vehicle.getTypeID(veh_id)
                # Extract base type (remove route suffix)
                base_type = veh_type.split('_')[0].lower()
                weight = self.vehicle_weights.get(base_type, 1.0)
                weighted_count += weight
            weighted_vehicles[direction] += weighted_count
            
            # Occupancy (0-1, percentage of lane occupied)
            occ = traci.lane.getLastStepOccupancy(lane_id) / 100.0
            occupancy[direction] = max(occupancy[direction], occ)
        
        # Add to state
        for direction in ['n', 's', 'e', 'w']:
            state.append(queues[direction])
        
        for direction in ['n', 's', 'e', 'w']:
            state.append(weighted_vehicles[direction])
        
        for direction in ['n', 's', 'e', 'w']:
            state.append(occupancy[direction])
        
        # 3. Time in current phase (seconds)
        state.append(self.time_in_phase[junction_id])
        
        # 4. Emergency vehicle present (0 or 1)
        emergency = 0
        for lane_id in incoming_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            for veh_id in vehicle_ids:
                veh_type = traci.vehicle.getTypeID(veh_id)
                if 'emergency' in veh_type.lower():
                    emergency = 1
                    break
            if emergency:
                break
        state.append(emergency)
        
        # 5. Neighbor junction phases (for coordination)
        neighbors = self.neighbors[junction_id]
        for neighbor_id in neighbors[:2]:  # Max 2 neighbors for fixed state size
            neighbor_phase = self.current_phases.get(neighbor_id, 0)
            state.append(neighbor_phase)
        
        # Pad if less than 2 neighbors
        while len(state) < 17:
            state.append(0)
        
        return np.array(state[:17], dtype=np.float32)
    
    def get_local_states(self):
        """Get local states for all junctions"""
        return [self.get_local_state(j) for j in self.junction_ids]
    
    def get_global_state(self):
        """
        Get global state (all junction info + network features)
        
        Returns:
            state: numpy array (155,)
        """
        global_state = []
        
        # 1. All local states concatenated (9 × 17 = 153)
        for junction_id in self.junction_ids:
            local_state = self.get_local_state(junction_id)
            global_state.extend(local_state)
        
        # 2. Network-wide features
        total_vehicles = traci.vehicle.getIDCount()
        global_state.append(total_vehicles)
        
        # Total waiting time across network
        total_waiting = sum(
            traci.lane.getWaitingTime(lane_id)
            for junction_id in self.junction_ids
            for lane_id in traci.trafficlight.getControlledLanes(junction_id)
        )
        global_state.append(total_waiting / 100.0)  # Normalize
        
        return np.array(global_state, dtype=np.float32)
    
    def step(self, actions):
        """
        Execute actions and simulate one step
        
        Args:
            actions: List of actions for each junction (9 integers)
        
        Returns:
            next_local_states: List of next local states
            next_global_state: Next global state
            rewards: List of rewards for each junction
            done: Episode done flag
        """
        # Execute actions
        for i, junction_id in enumerate(self.junction_ids):
            action = actions[i]
            self._execute_action(junction_id, action)
        
        # Simulate one step
        traci.simulationStep()
        
        # Update phase timers
        for junction_id in self.junction_ids:
            self.time_in_phase[junction_id] += 1
        
        # Get next states
        next_local_states = self.get_local_states()
        next_global_state = self.get_global_state()
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Check if done
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        return next_local_states, next_global_state, rewards, done
    
    def _execute_action(self, junction_id, action):
        """
        Execute action for junction
        
        Actions:
        0: Keep current phase
        1: Switch to next phase
        2: Extend current phase (add time)
        3: Emergency override (prioritize emergency vehicles)
        """
        if action == 0:
            # Keep current phase - do nothing
            pass
        
        elif action == 1:
            # Switch to next phase
            num_phases = len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
            next_phase = (self.current_phases[junction_id] + 1) % num_phases
            traci.trafficlight.setPhase(junction_id, next_phase)
            self.current_phases[junction_id] = next_phase
            self.time_in_phase[junction_id] = 0
        
        elif action == 2:
            # Extend current phase (stay in current for longer)
            # Just keep tracking time
            pass
        
        elif action == 3:
            # Emergency override - switch to phase that helps emergency
            # For now, just switch to next phase
            # In full implementation, would detect emergency direction
            num_phases = len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
            next_phase = (self.current_phases[junction_id] + 1) % num_phases
            traci.trafficlight.setPhase(junction_id, next_phase)
            self.current_phases[junction_id] = next_phase
            self.time_in_phase[junction_id] = 0
    
    def _compute_rewards(self):
        """
        Compute rewards for all junctions (coordinated rewards)
        
        Reward components:
        - Own junction performance (60%)
        - Downstream neighbor impact (30%)
        - Network-wide metrics (10%)
        - Bonuses/penalties
        """
        rewards = []
        
        for junction_id in self.junction_ids:
            reward = 0.0
            
            # 1. Own junction performance
            incoming_lanes = traci.trafficlight.getControlledLanes(junction_id)
            
            # Waiting time change
            current_waiting = sum(
                traci.lane.getWaitingTime(lane_id)
                for lane_id in incoming_lanes
            )
            waiting_change = self.prev_waiting_times.get(junction_id, current_waiting) - current_waiting
            own_reward = 0.01 * waiting_change  # Positive if waiting decreased
            
            self.prev_waiting_times[junction_id] = current_waiting
            
            # Throughput (vehicles that passed)
            throughput = sum(
                traci.lane.getLastStepVehicleNumber(lane_id)
                for lane_id in incoming_lanes
            )
            own_reward += 0.1 * throughput
            
            # 2. Downstream neighbor impact
            neighbor_reward = 0.0
            neighbors = self.neighbors[junction_id]
            if neighbors:
                for neighbor_id in neighbors:
                    neighbor_lanes = traci.trafficlight.getControlledLanes(neighbor_id)
                    neighbor_waiting = sum(
                        traci.lane.getWaitingTime(lane_id)
                        for lane_id in neighbor_lanes
                    )
                    neighbor_change = self.prev_waiting_times.get(neighbor_id, neighbor_waiting) - neighbor_waiting
                    neighbor_reward += 0.005 * neighbor_change
                
                neighbor_reward /= len(neighbors)
            
            # 3. Network-wide metrics
            total_vehicles = traci.vehicle.getIDCount()
            network_reward = -0.001 * total_vehicles  # Penalty for network congestion
            
            # 4. Bonuses/penalties
            bonus = 0.0
            
            # Emergency vehicle bonus
            if self.emergency_cleared.get(junction_id, False):
                bonus += self.config.EMERGENCY_BONUS
                self.emergency_cleared[junction_id] = False
            
            # Deadlock penalty (very long waiting times)
            if current_waiting > 500:  # 500 seconds = severe congestion
                bonus += self.config.DEADLOCK_PENALTY
            
            # Combine rewards
            total_reward = (
                self.config.REWARD_WEIGHT_OWN * own_reward +
                self.config.REWARD_WEIGHT_NEIGHBORS * neighbor_reward +
                self.config.REWARD_WEIGHT_NETWORK * network_reward +
                bonus
            )
            
            rewards.append(total_reward)
        
        return rewards
    
    def close(self):
        """Close SUMO connection"""
        if traci.isLoaded():
            traci.close()


# ============================================================================
# MAPPO AGENT
# ============================================================================

class MAPPOAgent:
    """
    MAPPO Agent - Manages actors, critic, and training
    """
    
    def __init__(self, config):
        self.config = config
        self.num_agents = config.NUM_JUNCTIONS
        
        # Create actors (one per junction)
        self.actors = [
            ActorNetwork(
                state_dim=config.LOCAL_STATE_DIM,
                action_dim=config.ACTION_DIM,
                hidden_dims=config.ACTOR_HIDDEN
            )
            for _ in range(self.num_agents)
        ]
        
        # Create shared critic
        self.critic = CriticNetwork(
            state_dim=config.GLOBAL_STATE_DIM,
            hidden_dims=config.CRITIC_HIDDEN
        )
        
        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=config.LEARNING_RATE_ACTOR)
            for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.LEARNING_RATE_CRITIC
        )
        
        # Exploration rate
        self.epsilon = config.EPSILON_START
        
        # Replay buffer
        self.buffer = ReplayBuffer(num_agents=self.num_agents)
        
        # Logging
        self.writer = SummaryWriter(config.TENSORBOARD_DIR)
        self.episode_count = 0
    
    def select_actions(self, local_states):
        """
        Select actions for all junctions
        
        Args:
            local_states: List of local states (9 × 17)
        
        Returns:
            actions: List of actions (9 integers)
            log_probs: List of log probabilities
            entropies: List of entropies
        """
        actions = []
        log_probs = []
        entropies = []
        
        for i, actor in enumerate(self.actors):
            action, log_prob, entropy = actor.get_action(local_states[i], self.epsilon)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        return actions, log_probs, entropies
    
    def compute_gae(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Tensor of rewards (T,)
            values: Tensor of values (T,)
            next_values: Tensor of next values (T,)
            dones: Tensor of done flags (T,)
        
        Returns:
            advantages: Tensor of advantages (T,)
            returns: Tensor of returns (T,)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Work backwards through episode
        for t in reversed(range(len(rewards))):
            # TD error
            delta = rewards[t] + self.config.GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.config.GAMMA * self.config.LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def update(self):
        """
        Update actors and critic using PPO
        """
        if len(self.buffer) < self.config.UPDATE_FREQUENCY:
            return
        
        # Get batch
        batch = self.buffer.get_batch()
        
        # Compute values for all timesteps
        with torch.no_grad():
            values = self.critic(batch['global_states']).squeeze()
            
            # Next values (shift by 1, last is 0)
            next_values = torch.cat([values[1:], torch.zeros(1)])
        
        # Update each actor
        actor_losses = []
        for i in range(self.num_agents):
            # Compute advantages for this agent
            advantages, returns = self.compute_gae(
                batch['rewards'][i],
                values,
                next_values,
                batch['dones']
            )
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Detach advantages to prevent gradient flow issues
            advantages = advantages.detach()
            
            # PPO update epochs
            for epoch in range(self.config.PPO_EPOCHS):
                # Get new action probabilities
                new_action_probs = self.actors[i](batch['local_states'][i])
                dist = Categorical(new_action_probs)
                new_log_probs = dist.log_prob(batch['actions'][i])
                
                # Compute ratio (detach old log_probs to avoid graph issues)
                ratio = torch.exp(new_log_probs - batch['log_probs'][i].detach())
                
                # PPO clipped objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.CLIP_EPSILON, 1 + self.config.CLIP_EPSILON) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Add entropy bonus
                entropy = dist.entropy().mean()
                actor_loss = actor_loss - self.config.ENTROPY_COEF * entropy
                
                # Update actor
                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.config.GRAD_CLIP)
                self.actor_optimizers[i].step()
            
            actor_losses.append(actor_loss.item())
        
        # Update critic
        critic_losses = []
        for epoch in range(self.config.PPO_EPOCHS):
            # Recompute advantages and returns
            advantages_list = []
            returns_list = []
            for i in range(self.num_agents):
                adv, ret = self.compute_gae(
                    batch['rewards'][i],
                    values,
                    next_values,
                    batch['dones']
                )
                advantages_list.append(adv)
                returns_list.append(ret)
            
            # Average returns across agents
            returns = torch.stack(returns_list).mean(dim=0)
            
            # Predict values
            predicted_values = self.critic(batch['global_states']).squeeze()
            
            # Critic loss
            critic_loss = F.mse_loss(predicted_values, returns)
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.GRAD_CLIP)
            self.critic_optimizer.step()
        
        critic_losses.append(critic_loss.item())
        
        # Log losses
        self.writer.add_scalar('Loss/Actor', np.mean(actor_losses), self.episode_count)
        self.writer.add_scalar('Loss/Critic', np.mean(critic_losses), self.episode_count)
        
        # Clear buffer
        self.buffer.clear()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.config.EPSILON_END, self.epsilon * self.config.EPSILON_DECAY)
    
    def save_models(self, path):
        """Save actor and critic models"""
        os.makedirs(path, exist_ok=True)
        
        # Save actors
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(path, f'actor_{i}.pth'))
        
        # Save critic
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        
        print(f"Models saved to {path}")

    def save_checkpoint(self, path, include_buffer=True):
        """Save full training checkpoint including optimizers and RNG states."""
        os.makedirs(path, exist_ok=True)

        # Save model weights
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(path, f'actor_{i}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))

        # Save optimizers
        for i, optim in enumerate(self.actor_optimizers):
            torch.save(optim.state_dict(), os.path.join(path, f'actor_optim_{i}.pth'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, 'critic_optim.pth'))

        # Save training state
        state = {
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
        }
        with open(os.path.join(path, 'train_state.pkl'), 'wb') as f:
            pickle.dump(state, f)

        # Save replay buffer optionally
        if include_buffer:
            try:
                with open(os.path.join(path, 'replay_buffer.pkl'), 'wb') as f:
                    pickle.dump(self.buffer, f)
            except Exception:
                # Fallback: don't block saving if buffer cannot be pickled
                print('Warning: replay buffer could not be saved.')

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, load_buffer=True, map_location=None):
        """Load full training checkpoint and optimizer states."""
        # Load model weights
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(os.path.join(path, f'actor_{i}.pth'), map_location=map_location))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth'), map_location=map_location))

        # Load optimizers if available
        for i, optim in enumerate(self.actor_optimizers):
            optim_path = os.path.join(path, f'actor_optim_{i}.pth')
            if os.path.exists(optim_path):
                optim.load_state_dict(torch.load(optim_path, map_location=map_location))
        critic_optim_path = os.path.join(path, 'critic_optim.pth')
        if os.path.exists(critic_optim_path):
            self.critic_optimizer.load_state_dict(torch.load(critic_optim_path, map_location=map_location))

        # Load training state
        state_path = os.path.join(path, 'train_state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            self.episode_count = state.get('episode_count', self.episode_count)
            self.epsilon = state.get('epsilon', self.epsilon)
            try:
                random.setstate(state.get('random_state'))
                np.random.set_state(state.get('np_random_state'))
                torch.set_rng_state(state.get('torch_random_state'))
            except Exception:
                print('Warning: could not restore RNG state exactly.')

        # Load replay buffer if requested and exists
        buffer_path = os.path.join(path, 'replay_buffer.pkl')
        if load_buffer and os.path.exists(buffer_path):
            try:
                with open(buffer_path, 'rb') as f:
                    self.buffer = pickle.load(f)
            except Exception:
                print('Warning: replay buffer could not be loaded.')

        print(f"Checkpoint loaded from {path}")
    
    def load_models(self, path):
        """Load actor and critic models"""
        # Load actors
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(os.path.join(path, f'actor_{i}.pth')))
        
        # Load critic
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
        
        print(f"Models loaded from {path}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_mappo(config, resume_checkpoint=None, max_hours=None):
    """
    Main training loop for MAPPO
    """
    print("=" * 80)
    print("MAPPO Training for K1 Traffic Network")
    print("=" * 80)
    print(f"Junctions: {config.NUM_JUNCTIONS}")
    print(f"Episodes: {config.NUM_EPISODES}")
    print(f"Steps per episode: {config.STEPS_PER_EPISODE}")
    print("=" * 80)
    
    # Create environment and agent
    env = K1Environment(config)
    agent = MAPPOAgent(config)
    
    # Resume from checkpoint if requested
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        agent.load_checkpoint(resume_checkpoint, load_buffer=True)

    # Training loop
    start_time = time.time()
    max_seconds = None
    if max_hours is not None:
        max_seconds = float(max_hours) * 3600.0

    for episode in range(agent.episode_count, config.NUM_EPISODES):
        # Reset environment
        local_states, global_state = env.reset()
        
        episode_reward = 0
        episode_length = 0
        
        # Episode loop
        for step in range(config.STEPS_PER_EPISODE):
            # Select actions
            actions, log_probs, entropies = agent.select_actions(local_states)
            
            # Execute actions
            next_local_states, next_global_state, rewards, done = env.step(actions)
            
            # Store experience
            agent.buffer.store(
                local_states,
                global_state,
                actions,
                rewards,
                log_probs,
                entropies,
                done
            )
            
            # Update
            episode_reward += sum(rewards)
            episode_length += 1
            
            # Move to next state
            local_states = next_local_states
            global_state = next_global_state
            
            # Update agent
            if step % config.UPDATE_FREQUENCY == 0:
                agent.update()
            
            if done:
                break
        
        # Decay exploration
        agent.decay_epsilon()
        agent.episode_count = episode
        
        # Logging
        if episode % config.LOG_INTERVAL == 0:
            print(f"Episode {episode}/{config.NUM_EPISODES} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Length: {episode_length} | "
                  f"Epsilon: {agent.epsilon:.4f}")
            
            agent.writer.add_scalar('Episode/Reward', episode_reward, episode)
            agent.writer.add_scalar('Episode/Length', episode_length, episode)
            agent.writer.add_scalar('Episode/Epsilon', agent.epsilon, episode)
        
        # Save models / checkpoints
        if episode % config.SAVE_INTERVAL == 0 and episode > 0:
            model_path = os.path.join(config.MODEL_DIR, f'episode_{episode}')
            agent.save_models(model_path)
            # Also save a full checkpoint
            try:
                agent.save_checkpoint(os.path.join(config.MODEL_DIR, f'checkpoint_{episode}'))
            except Exception:
                print('Warning: checkpoint save failed.')

        # Periodically check elapsed time for time-limited runs
        if max_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed >= max_seconds:
                # Save a checkpoint and exit gracefully
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                ckpt_path = os.path.join(config.MODEL_DIR, f'checkpoint_time_{timestamp}')
                print(f"Max time reached ({max_hours} hours). Saving checkpoint to {ckpt_path} and exiting.")
                agent.save_checkpoint(ckpt_path)
                env.close()
                agent.writer.close()
                return
    
    # Final save
    agent.save_models(os.path.join(config.MODEL_DIR, 'final'))
    
    # Close environment
    env.close()
    agent.writer.close()
    
    print("=" * 80)
    print("Training completed!")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MAPPO training for K1 network')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help='Path to checkpoint directory to resume from')
    parser.add_argument('--max-hours', type=float, default=None,
                        help='Run training for at most this many hours, then save checkpoint and exit')
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Override number of episodes in config')
    args = parser.parse_args()

    # Create configuration
    config = MAPPOConfig()
    if args.num_episodes is not None:
        config.NUM_EPISODES = args.num_episodes

    # Create directories
    os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # Train with optional resume and time limit
    train_mappo(config, resume_checkpoint=args.resume_checkpoint, max_hours=args.max_hours)
