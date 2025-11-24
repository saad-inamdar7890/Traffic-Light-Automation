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
    SUMO_CONFIG = "k1_3h_varying.sumocfg"  # 3-hour time-varying traffic scenario
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
    LOCAL_STATE_DIM = 16   # Per junction (reduced from 17, removed emergency flag)
    GLOBAL_STATE_DIM = 146 # All junctions + network features (9*16 + 2 = 146)
    ACTION_DIM = 3         # [keep, next_phase, extend]
    
    # Vehicle type weights (Passenger Car Equivalents - PCE)
    VEHICLE_WEIGHTS = {
        'passenger': 1.0,
        'delivery': 2.5,
        'truck': 5.0,
        'bus': 4.5
    }
    
    # Network architecture
    ACTOR_HIDDEN = [128, 64]
    CRITIC_HIDDEN = [256, 256, 128]
    
    # Training hyperparameters (Phase 1 optimized)
    LEARNING_RATE_ACTOR = 5e-4     # 3e-4 → 5e-4 (faster learning)
    LEARNING_RATE_CRITIC = 1e-3
    GAMMA = 0.99                   # Discount factor
    LAMBDA = 0.95                  # GAE parameter
    CLIP_EPSILON = 0.25            # 0.2 → 0.25 (less conservative updates)
    ENTROPY_COEF = 0.02            # 0.01 → 0.02 (better exploration)
    GRAD_CLIP = 0.5                # Gradient clipping
    
    # Training schedule
    STEPS_PER_EPISODE = 10800      # Simulation duration per episode in seconds (3 hours)
                                   # 3600 = 1 hour, 10800 = 3 hours, 86400 = 24 hours
    NUM_EPISODES = 5000
    UPDATE_FREQUENCY = 64          # 128 → 64 (2x more frequent updates)
    PPO_EPOCHS = 15                # 10 → 15 (more thorough optimization)
    
    # Traffic variation for robust training
    ENABLE_TRAFFIC_VARIATION = False  # Disabled: Using time-varying routes instead
    TRAFFIC_VARIATION_PERCENT = 0.30  # (Not used when time-varying routes enabled)
    
    # SUMO configuration
    USE_REALISTIC_24H_TRAFFIC = False  # True = Time-varying rush hours, False = Constant traffic
                                       # If True: uses k1_realistic.sumocfg with morning/evening peaks
                                       # If False: uses k1.sumocfg with constant traffic rates
    
    # Exploration (Phase 1 optimized)
    EPSILON_START = 0.2        # 0.1 → 0.2 (more initial exploration)
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.998      # 0.995 → 0.998 (slower decay, explore longer)
    
    # Reward weights (Phase 1 rebalanced)
    REWARD_WEIGHT_OWN = 0.5        # 0.6 → 0.5 (own junction)
    REWARD_WEIGHT_NEIGHBORS = 0.35  # 0.3 → 0.35 (stronger coordination)
    REWARD_WEIGHT_NETWORK = 0.15   # 0.1 → 0.15 (network awareness)
    DEADLOCK_PENALTY = -10.0
    GREEN_WAVE_BONUS = 0.3         # NEW: Reward for coordination
    
    # Reward normalization constants
    MAX_WAITING_THRESHOLD = 500.0  # For normalization
    MAX_THROUGHPUT_PER_STEP = 10.0 # Expected max vehicles/step
    
    # Logging
    LOG_INTERVAL = 10         # Episodes between logs
    SAVE_INTERVAL = 100       # Episodes between model saves
    TENSORBOARD_DIR = "mappo_logs"
    MODEL_DIR = "mappo_models"
    
    # Device (GPU/CPU) - Automatically detected
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_GPU = torch.cuda.is_available()  # Flag for easy checking


# ============================================================================
# NEURAL NETWORKS
# ============================================================================

class ActorNetwork(nn.Module):
    """
    Actor Network - Policy for individual junction
    
    Takes LOCAL state (own junction + neighbors) and outputs action probabilities.
    Each junction has its own actor network.
    """
    
    def __init__(self, state_dim=17, action_dim=4, hidden_dims=[128, 64], device='cpu'):
        super(ActorNetwork, self).__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        
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
            state: Local state tensor (batch_size, 16)
                   [current_phase, queue_n, queue_s, queue_e, queue_w,
                    weighted_veh_n, weighted_veh_s, weighted_veh_e, weighted_veh_w,
                    occupancy_n, occupancy_s, occupancy_e, occupancy_w,
                    time_in_phase, neighbor1_phase, neighbor2_phase]
        
        Returns:
            action_probs: Action probabilities (batch_size, 4)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
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
    
    def __init__(self, state_dim=155, hidden_dims=[256, 256, 128], device='cpu'):
        super(CriticNetwork, self).__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        
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
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
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
    
    def __init__(self, num_agents=9, device='cpu'):
        self.num_agents = num_agents
        self.device = torch.device(device) if isinstance(device, str) else device
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
        
        # Convert per-agent data to tensors and move to device
        for i in range(self.num_agents):
            # Convert lists to numpy arrays first (much faster), then to tensors on device
            batch['local_states'].append(torch.FloatTensor(np.array(self.local_states[i], dtype=np.float32)).to(self.device))
            batch['actions'].append(torch.LongTensor(np.array(self.actions[i], dtype=np.int64)).to(self.device))
            batch['rewards'].append(torch.FloatTensor(np.array(self.rewards[i], dtype=np.float32)).to(self.device))
            # Move log_probs and entropies to device before stacking
            batch['log_probs'].append(torch.stack([lp.detach().to(self.device) for lp in self.log_probs[i]]))
            batch['entropies'].append(torch.stack([ent.detach().to(self.device) for ent in self.entropies[i]]))
        
        # Convert global data and move to device
        batch['global_states'] = torch.FloatTensor(np.array(self.global_states, dtype=np.float32)).to(self.device)
        batch['dones'] = torch.FloatTensor(np.array(self.dones, dtype=np.float32)).to(self.device)
        
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
        
        # Traffic variation tracking
        self.current_traffic_multiplier = 1.0
        self.episode_count = 0
        
        # State tracking
        self.prev_waiting_times = {}
        self.prev_queue_lengths = {}
        self.current_phases = {}
        self.time_in_phase = {}
        
        # Phase 1: Neighbor coordination tracking
        self.neighbor_last_action = {}  # Track when neighbors changed phase
        self.steps_since_neighbor_change = {}  # For green wave detection
        
        # Track metrics for rewards
        self.prev_waiting_times = defaultdict(float)
        self.prev_queue_lengths = defaultdict(float)
        
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
        
        # Select config based on USE_REALISTIC_24H_TRAFFIC setting
        if self.config.USE_REALISTIC_24H_TRAFFIC:
            config_path = "k1_realistic.sumocfg"  # Time-varying traffic (rush hours)
        else:
            config_path = self.config.SUMO_CONFIG  # Constant traffic
        
        # Make config path absolute if relative
        if not os.path.isabs(config_path):
            # Try to find it relative to the script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
            if not os.path.exists(config_path):
                # Fallback: relative to current working directory
                config_path = config_path
        
        sumo_cmd = [
            sumo_binary,
            "-c", config_path,
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--no-step-log",
            "--threads", "8",  # Multi-threading for Ryzen 7 (8 cores)
        ]
        
        try:
            traci.start(sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO with config: {config_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Config file exists: {os.path.exists(config_path)}")
            print(f"SUMO binary: {sumo_binary}")
            raise e
        
        # Apply traffic variation by modifying vehicle type speeds if enabled
        # (This affects all vehicles spawned during this episode)
        if hasattr(self, 'current_traffic_multiplier') and self.current_traffic_multiplier != 1.0:
            try:
                # Modify flow density (indirect: via vType speed adjustments)
                # Lower speed = more congestion, higher speed = less congestion
                # Inverse relationship: more traffic (1.3x) → slower speeds (1/1.3x)
                speed_factor = 1.0 / self.current_traffic_multiplier
                
                for vtype_id in ['DEFAULT_VEHTYPE', 'passenger', 'bus', 'truck']:
                    try:
                        if vtype_id in traci.vehicletype.getIDList():
                            base_speed = traci.vehicletype.getMaxSpeed(vtype_id)
                            new_speed = base_speed * speed_factor
                            # Keep reasonable bounds (5-50 m/s)
                            new_speed = max(5.0, min(50.0, new_speed))
                            traci.vehicletype.setMaxSpeed(vtype_id, new_speed)
                    except:
                        pass
            except:
                pass  # Fail silently if variation can't be applied
        
        # Initialize phase tracking
        for junction_id in self.junction_ids:
            self.current_phases[junction_id] = 0
            self.time_in_phase[junction_id] = 0
    
    def reset(self, apply_variation=True):
        """Reset environment for new episode"""
        # Close existing connection
        if traci.isLoaded():
            traci.close()
        
        # Generate random traffic variation if enabled
        if apply_variation and self.config.ENABLE_TRAFFIC_VARIATION:
            # Random multiplier: 1.0 ± TRAFFIC_VARIATION_PERCENT
            variation = self.config.TRAFFIC_VARIATION_PERCENT
            self.current_traffic_multiplier = 1.0 + np.random.uniform(-variation, variation)
        else:
            self.current_traffic_multiplier = 1.0
        
        self.episode_count += 1
        
        # Restart SUMO
        self._init_sumo()
        
        # Reset tracking
        self.prev_waiting_times.clear()
        self.prev_queue_lengths.clear()
        self.current_phases = {j: 0 for j in self.junction_ids}
        self.time_in_phase = {j: 0 for j in self.junction_ids}
        self.neighbor_last_action = {j: 1 for j in self.junction_ids}  # 1 = keep
        self.steps_since_neighbor_change = {j: 999 for j in self.junction_ids}  # Large number
        
        # Get initial states
        local_states = self.get_local_states()
        global_state = self.get_global_state()
        
        return local_states, global_state
    
    def get_local_state(self, junction_id):
        """
        Get local state for specific junction (realistic sensors only)
        
        Returns:
            state: numpy array (16,)
                [current_phase, queue_n, queue_s, queue_e, queue_w,
                 weighted_veh_n, weighted_veh_s, weighted_veh_e, weighted_veh_w,
                 occupancy_n, occupancy_s, occupancy_e, occupancy_w,
                 time_in_phase, neighbor1_phase, neighbor2_phase]
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
        
        # 4. Neighbor junction phases (for coordination)
        neighbors = self.neighbors[junction_id]
        for neighbor_id in neighbors[:2]:  # Max 2 neighbors for fixed state size
            neighbor_phase = self.current_phases.get(neighbor_id, 0)
            state.append(neighbor_phase)
        
        # Pad if less than 2 neighbors
        while len(state) < 16:
            state.append(0)
        
        return np.array(state[:16], dtype=np.float32)
    
    def get_local_states(self):
        """Get local states for all junctions"""
        return [self.get_local_state(j) for j in self.junction_ids]
    
    def get_global_state(self):
        """
        Get global state (all junction info + network features)
        
        Returns:
            state: numpy array (146,)  # 9 junctions * 16 + 2 network features
        """
        global_state = []
        
        # 1. All local states concatenated (9 × 16 = 144)
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
        # Execute actions and track for green wave coordination
        for i, junction_id in enumerate(self.junction_ids):
            action = actions[i]
            self._execute_action(junction_id, action)
            
            # Track action for green wave bonus calculation
            self.neighbor_last_action[junction_id] = action
            
            # Update steps since change counter
            if action == 1:  # Phase change
                self.steps_since_neighbor_change[junction_id] = 0
            else:
                self.steps_since_neighbor_change[junction_id] += 1
        
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
    
    def _compute_rewards(self):
        """
        Phase 1 Improved Reward Computation
        
        Improvements:
        1. Normalized scales for all components
        2. Balanced weights (0.5 own, 0.35 neighbor, 0.15 network)
        3. Green wave coordination bonus
        4. Queue balance incentive
        
        Returns:
            rewards: List of rewards for each junction
        """
        rewards = []
        
        for junction_id in self.junction_ids:
            # Get controlled lanes
            incoming_lanes = traci.trafficlight.getControlledLanes(junction_id)
            
            # === 1. OWN JUNCTION PERFORMANCE (Normalized) ===
            
            # 1a. Waiting time change (normalized by max threshold)
            current_waiting = sum(
                traci.lane.getWaitingTime(lane_id)
                for lane_id in incoming_lanes
            )
            prev_waiting = self.prev_waiting_times.get(junction_id, current_waiting)
            waiting_reduction = (prev_waiting - current_waiting) / self.config.MAX_WAITING_THRESHOLD
            # Clip to reasonable range [-1, 1]
            waiting_reduction = np.clip(waiting_reduction, -1.0, 1.0)
            
            self.prev_waiting_times[junction_id] = current_waiting
            
            # 1b. Throughput (normalized by expected max)
            throughput = sum(
                traci.lane.getLastStepVehicleNumber(lane_id)
                for lane_id in incoming_lanes
            )
            throughput_normalized = throughput / self.config.MAX_THROUGHPUT_PER_STEP
            throughput_normalized = np.clip(throughput_normalized, 0.0, 2.0)
            
            # 1c. Queue balance (penalize imbalanced queues across directions)
            queues = []
            for lane_id in incoming_lanes:
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                queues.append(queue)
            
            if len(queues) > 1:
                queue_std = np.std(queues)
                queue_balance = -queue_std / 10.0  # Normalize by typical std
                queue_balance = np.clip(queue_balance, -1.0, 0.0)
            else:
                queue_balance = 0.0
            
            # Combine own metrics with balanced weights
            own_reward = (
                0.5 * waiting_reduction +     # Primary: reduce waiting
                0.3 * throughput_normalized +  # Secondary: increase flow
                0.2 * queue_balance           # Tertiary: balance queues
            )
            
            # === 2. NEIGHBOR COORDINATION (Stronger weight) ===
            
            neighbor_reward = 0.0
            neighbors = self.neighbors[junction_id]
            
            if neighbors:
                for neighbor_id in neighbors:
                    neighbor_lanes = traci.trafficlight.getControlledLanes(neighbor_id)
                    neighbor_waiting = sum(
                        traci.lane.getWaitingTime(lane_id)
                        for lane_id in neighbor_lanes
                    )
                    prev_neighbor_waiting = self.prev_waiting_times.get(neighbor_id, neighbor_waiting)
                    neighbor_waiting_reduction = (prev_neighbor_waiting - neighbor_waiting) / self.config.MAX_WAITING_THRESHOLD
                    neighbor_waiting_reduction = np.clip(neighbor_waiting_reduction, -1.0, 1.0)
                    
                    neighbor_reward += neighbor_waiting_reduction
                
                neighbor_reward /= len(neighbors)
            
            # === 3. NETWORK-WIDE METRICS (Stronger penalty) ===
            
            total_vehicles = traci.vehicle.getIDCount()
            network_reward = -0.01 * (total_vehicles / 100.0)  # 10x stronger than before
            network_reward = np.clip(network_reward, -2.0, 0.0)
            
            # === 4. GREEN WAVE COORDINATION BONUS ===
            
            green_wave_bonus = 0.0
            
            # Check if any neighbor recently changed phase (within last 5 steps)
            # and current junction is keeping (good coordination)
            for neighbor_id in neighbors:
                steps_since_change = self.steps_since_neighbor_change.get(neighbor_id, 999)
                if steps_since_change <= 5:  # Neighbor changed recently
                    # If we kept our phase (didn't change), reward coordination
                    if self.neighbor_last_action.get(junction_id, 1) == 0:  # action 0 = keep
                        green_wave_bonus += self.config.GREEN_WAVE_BONUS / len(neighbors)
            
            # === 5. DEADLOCK PENALTY ===
            
            deadlock_penalty = 0.0
            if current_waiting > self.config.MAX_WAITING_THRESHOLD:
                deadlock_penalty = self.config.DEADLOCK_PENALTY
            
            # === COMBINE ALL COMPONENTS ===
            
            total_reward = (
                self.config.REWARD_WEIGHT_OWN * own_reward +
                self.config.REWARD_WEIGHT_NEIGHBORS * neighbor_reward +
                self.config.REWARD_WEIGHT_NETWORK * network_reward +
                green_wave_bonus +
                deadlock_penalty
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
        self.device = config.DEVICE
        
        # Create actors (one per junction) and move to device
        self.actors = [
            ActorNetwork(
                state_dim=config.LOCAL_STATE_DIM,
                action_dim=config.ACTION_DIM,
                hidden_dims=config.ACTOR_HIDDEN,
                device=self.device
            ).to(self.device)
            for _ in range(self.num_agents)
        ]
        
        # Create shared critic and move to device
        self.critic = CriticNetwork(
            state_dim=config.GLOBAL_STATE_DIM,
            hidden_dims=config.CRITIC_HIDDEN,
            device=self.device
        ).to(self.device)
        
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
        
        # Replay buffer (pass device for proper tensor handling)
        self.buffer = ReplayBuffer(num_agents=self.num_agents, device=self.device)
        
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
            next_values = torch.cat([values[1:], torch.zeros(1, device=self.device)])
        
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
        # Use configured device if map_location not specified
        if map_location is None:
            map_location = self.device
        
        def _adapt_and_copy(src_tensor, dst_tensor):
            """Copy overlapping region from src_tensor into a new tensor shaped like dst_tensor."""
            # Ensure both are tensors
            if not isinstance(src_tensor, torch.Tensor):
                src_tensor = torch.tensor(src_tensor)
            if not isinstance(dst_tensor, torch.Tensor):
                dst_tensor = torch.tensor(dst_tensor)

            new_tensor = dst_tensor.clone().detach()

            # Build slices for overlapping region
            slices = []
            for s_sz, d_sz in zip(src_tensor.shape, dst_tensor.shape):
                m = min(s_sz, d_sz)
                slices.append(slice(0, m))

            # For extra dimensions in dst, we keep initialized values
            try:
                new_tensor[tuple(slices)] = src_tensor[tuple(slices)].to(new_tensor.dtype)
            except Exception:
                # As a fallback, attempt to copy flattened overlap
                flat_src = src_tensor.flatten()
                flat_dst = new_tensor.flatten()
                m = min(flat_src.numel(), flat_dst.numel())
                flat_dst[:m] = flat_src[:m].to(flat_dst.dtype)
                new_tensor = flat_dst.reshape(new_tensor.shape)

            return new_tensor

        # Load model weights and move to device (shape-tolerant)
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f'actor_{i}.pth')
            if not os.path.exists(actor_path):
                print(f"Warning: actor file missing: {actor_path}")
                continue

            try:
                loaded_sd = torch.load(actor_path, map_location=map_location)
            except Exception as e:
                print(f"Error loading actor state {actor_path}: {e}")
                continue

            # Current model state
            current_sd = actor.state_dict()
            adapted_sd = {}

            for k, v in current_sd.items():
                if k in loaded_sd:
                    src = loaded_sd[k]
                    dst = v
                    if isinstance(src, torch.Tensor) and src.shape == dst.shape:
                        adapted_sd[k] = src.to(dst.dtype)
                    else:
                        # Shapes differ - adapt by copying overlap
                        try:
                            adapted = _adapt_and_copy(src, dst)
                            print(f"Adapted parameter '{k}' for actor_{i}: {getattr(src, 'shape', None)} -> {dst.shape}")
                            adapted_sd[k] = adapted
                        except Exception as e:
                            print(f"Failed to adapt parameter '{k}' for actor_{i}: {e}")
                            adapted_sd[k] = dst
                else:
                    # Key missing in checkpoint, keep current init
                    adapted_sd[k] = v

            # Load adapted state dict
            try:
                actor.load_state_dict(adapted_sd)
            except Exception as e:
                print(f"Warning: actor_{i} load_state_dict raised: {e} - attempting non-strict load")
                actor.load_state_dict(adapted_sd, strict=False)

            actor.to(self.device)

        # Critic
        critic_path = os.path.join(path, 'critic.pth')
        if os.path.exists(critic_path):
            try:
                loaded_sd = torch.load(critic_path, map_location=map_location)
                current_sd = self.critic.state_dict()
                adapted_sd = {}
                for k, v in current_sd.items():
                    if k in loaded_sd:
                        src = loaded_sd[k]
                        dst = v
                        if isinstance(src, torch.Tensor) and src.shape == dst.shape:
                            adapted_sd[k] = src.to(dst.dtype)
                        else:
                            adapted = _adapt_and_copy(src, dst)
                            print(f"Adapted parameter '{k}' for critic: {getattr(src, 'shape', None)} -> {dst.shape}")
                            adapted_sd[k] = adapted
                    else:
                        adapted_sd[k] = v

                try:
                    self.critic.load_state_dict(adapted_sd)
                except Exception as e:
                    print(f"Warning: critic load_state_dict raised: {e} - attempting non-strict load")
                    self.critic.load_state_dict(adapted_sd, strict=False)

                self.critic.to(self.device)
            except Exception as e:
                print(f"Error loading critic state {critic_path}: {e}")
        else:
            print(f"Warning: critic file missing: {critic_path}")

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
                # Ensure device attribute exists (for buffers saved before device support)
                if not hasattr(self.buffer, 'device'):
                    self.buffer.device = self.device
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
    
    # GPU/CPU info
    print("\n🖥️  DEVICE INFORMATION:")
    print(f"  Device: {config.DEVICE}")
    if config.USE_GPU:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  GPU Memory: {gpu_memory:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        print("  ✅ Neural networks will use GPU acceleration")
    else:
        print("  ⚠️  No GPU available - using CPU only")
        print("  Note: SUMO simulation always uses CPU (expected)")
    
    print("=" * 80)
    
    # Create environment and agent
    print("\n[1/4] Initializing SUMO environment...")
    env = K1Environment(config)
    print("✓ SUMO environment initialized successfully")
    
    print("\n[2/4] Creating MAPPO agent (9 actors + 1 critic)...")
    agent = MAPPOAgent(config)
    print("✓ Agent created successfully")
    print(f"  - Actor networks: {config.NUM_JUNCTIONS} (on {config.DEVICE})")
    print(f"  - Critic network: 1 shared (on {config.DEVICE})")
    print(f"  - Total parameters: ~{sum(p.numel() for p in agent.actors[0].parameters()) * 9 + sum(p.numel() for p in agent.critic.parameters()):,}")
    
    # Resume from checkpoint if requested
    if resume_checkpoint:
        print(f"\n[3/4] Resuming training from checkpoint: {resume_checkpoint}")
        # Don't load buffer to avoid shape mismatches between old (17-dim) and new (16-dim) states
        agent.load_checkpoint(resume_checkpoint, load_buffer=False)
        print(f"✓ Checkpoint loaded - resuming from episode {agent.episode_count}")
        print(f"  ℹ️  Replay buffer cleared (prevents shape mismatch errors)")
    else:
        print("\n[3/4] Starting fresh training (no checkpoint)")

    # Training loop
    print("\n[4/4] Starting training loop...")
    start_time = time.time()
    max_seconds = None
    if max_hours is not None:
        max_seconds = float(max_hours) * 3600.0
        print(f"✓ Time limit: {max_hours} hours ({max_seconds:.0f} seconds)")
    else:
        print("✓ No time limit set")
    
    print("\n" + "="*80)
    print("TRAINING IN PROGRESS")
    print("="*80)

    for episode in range(agent.episode_count, config.NUM_EPISODES):
        episode_start_time = time.time()
        
        # Print episode header
        print(f"\n{'='*80}")
        print(f"Episode {episode}/{config.NUM_EPISODES} | Epsilon: {agent.epsilon:.4f}")
        print(f"{'='*80}")
        
        # Reset environment
        reset_start = time.time()
        print(f"[Step 1/4] Resetting SUMO environment...", end=" ", flush=True)
        local_states, global_state = env.reset()
        reset_time = time.time() - reset_start
        
        # Show traffic variation info
        if config.ENABLE_TRAFFIC_VARIATION:
            variation_pct = (env.current_traffic_multiplier - 1.0) * 100
            print(f" ✓ (Traffic: {variation_pct:+.1f}%) [{reset_time:.1f}s]")
        else:
            print(f" ✓ [{reset_time:.1f}s]")
        
        episode_reward = 0
        episode_length = 0
        
        sim_start = time.time()
        print(f"[Step 2/4] Running simulation ({config.STEPS_PER_EPISODE} steps)...")
        
        # Episode loop
        step_times = []
        update_times = []
        for step in range(config.STEPS_PER_EPISODE):
            step_start = time.time()
            
            # Progress indicator every 100 steps
            if step % 100 == 0 and step > 0:
                avg_step_time = np.mean(step_times[-100:]) if step_times else 0
                steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                elapsed_sim = time.time() - sim_start
                eta_sim = (config.STEPS_PER_EPISODE - step) * avg_step_time
                print(f"  Step {step}/{config.STEPS_PER_EPISODE} ({step/config.STEPS_PER_EPISODE*100:.1f}%) | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Speed: {steps_per_sec:.1f} steps/s | "
                      f"Elapsed: {elapsed_sim/60:.1f}m | ETA: {eta_sim/60:.1f}m", flush=True)
            
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
            
            # Track step time
            step_times.append(time.time() - step_start)
            
            # Update agent
            if step % config.UPDATE_FREQUENCY == 0 and len(agent.buffer) >= config.UPDATE_FREQUENCY:
                update_start = time.time()
                if step % 500 == 0:  # Less frequent update logging
                    print(f"  → Updating networks (step {step})...", end=" ", flush=True)
                agent.update()
                update_time = time.time() - update_start
                update_times.append(update_time)
                if step % 500 == 0:
                    print(f"✓ [{update_time:.2f}s]")
            
            if done:
                break
        
        sim_time = time.time() - sim_start
        
        # Episode summary
        episode_duration = time.time() - episode_start_time
        avg_step_time = np.mean(step_times) if step_times else 0
        total_update_time = sum(update_times)
        
        print(f"\n[Step 3/4] Episode completed in {episode_duration:.1f}s")
        print(f"  ⏱️  Timing breakdown:")
        print(f"     - Reset time: {reset_time:.1f}s")
        print(f"     - Simulation time: {sim_time:.1f}s ({sim_time/60:.2f} minutes)")
        print(f"     - Network updates: {total_update_time:.1f}s ({len(update_times)} updates)")
        print(f"     - Avg step time: {avg_step_time*1000:.1f}ms")
        print(f"     - Steps per second: {1.0/avg_step_time:.1f}")
        print(f"  📊 Episode metrics:")
        print(f"     - Total reward: {episode_reward:.2f}")
        print(f"     - Steps: {episode_length}/{config.STEPS_PER_EPISODE}")
        print(f"     - Avg reward/step: {episode_reward/max(episode_length,1):.4f}")
        
        # Decay exploration
        agent.decay_epsilon()
        agent.episode_count = episode
        
        # Logging
        print(f"[Step 4/4] Logging to TensorBoard...", end=" ", flush=True)
        agent.writer.add_scalar('Episode/Reward', episode_reward, episode)
        agent.writer.add_scalar('Episode/Length', episode_length, episode)
        agent.writer.add_scalar('Episode/Epsilon', agent.epsilon, episode)
        agent.writer.add_scalar('Episode/Duration', episode_duration, episode)
        print("✓")
        
        if episode % config.LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"\n📊 PROGRESS SUMMARY (Episode {episode})")
            print(f"  Elapsed time: {elapsed/3600:.2f}h ({elapsed:.0f}s)")
            print(f"  Avg episode time: {elapsed/(episode+1-agent.episode_count):.1f}s")
            print(f"  Episodes remaining: {config.NUM_EPISODES - episode}")
            print(f"  Est. time remaining: {(config.NUM_EPISODES - episode) * (elapsed/(episode+1-agent.episode_count))/3600:.2f}h")
        
        # Save models / checkpoints
        if episode % config.SAVE_INTERVAL == 0 and episode > 0:
            print(f"\n💾 SAVING CHECKPOINT (Episode {episode})...")
            model_path = os.path.join(config.MODEL_DIR, f'episode_{episode}')
            agent.save_models(model_path)
            print(f"  ✓ Models saved to: {model_path}")
            # Also save a full checkpoint
            try:
                checkpoint_path = os.path.join(config.MODEL_DIR, f'checkpoint_{episode}')
                agent.save_checkpoint(checkpoint_path)
                print(f"  ✓ Full checkpoint saved to: {checkpoint_path}")
            except Exception as e:
                print(f'  ⚠ Warning: checkpoint save failed: {e}')

        # Periodically check elapsed time for time-limited runs
        if max_seconds is not None:
            elapsed = time.time() - start_time
            remaining = max_seconds - elapsed
            if remaining < 300:  # Warn when <5 min remaining
                print(f"\n⏰ WARNING: Only {remaining/60:.1f} minutes remaining!")
            
            if elapsed >= max_seconds:
                # Save a checkpoint and exit gracefully
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                ckpt_path = os.path.join(config.MODEL_DIR, f'checkpoint_time_{timestamp}')
                print(f"\n{'='*80}")
                print(f"⏰ TIME LIMIT REACHED ({max_hours} hours)")
                print(f"{'='*80}")
                print(f"\n💾 Saving final checkpoint...")
                agent.save_checkpoint(ckpt_path)
                print(f"✓ Checkpoint saved to: {ckpt_path}")
                print(f"\n📊 TRAINING SUMMARY:")
                print(f"  Episodes completed: {episode + 1}")
                print(f"  Total time: {elapsed/3600:.2f}h")
                print(f"  Final episode: {episode}")
                print(f"\n✓ Training session completed successfully!")
                print(f"\nTo resume, run:")
                print(f"  python mappo_k1_implementation.py --resume-checkpoint {ckpt_path} --max-hours 3")
                print(f"\n{'='*80}\n")
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
