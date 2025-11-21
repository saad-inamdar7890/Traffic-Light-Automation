# MAPPO Architecture - Deep Dive Explanation

## Table of Contents
1. [What is MAPPO?](#what-is-mappo)
2. [Why MAPPO for Traffic Control?](#why-mappo-for-traffic-control)
3. [Core Concepts](#core-concepts)
4. [Architecture Components](#architecture-components)
5. [How Training Works](#how-training-works)
6. [How Coordination Emerges](#how-coordination-emerges)
7. [Step-by-Step Execution Flow](#step-by-step-execution-flow)

---

## What is MAPPO?

**MAPPO = Multi-Agent Proximal Policy Optimization**

It's a reinforcement learning algorithm designed for scenarios where **multiple agents** (traffic lights) need to learn to **coordinate** their actions.

### The Key Insight

```
Traditional Approach:
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Agent J0│  │Agent J11│  │Agent J12│
│ (alone) │  │ (alone) │  │ (alone) │
└─────────┘  └─────────┘  └─────────┘
     ↓            ↓            ↓
  Learns       Learns       Learns
independently independently independently
❌ No coordination! Agents don't help each other

MAPPO Approach:
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Agent J0│  │Agent J11│  │Agent J12│
│ Actor   │  │ Actor   │  │ Actor   │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┴────────────┘
                  │
        ┌─────────▼─────────┐
        │   Shared Critic   │
        │ (Sees everything) │
        └───────────────────┘
✅ Coordination! Critic teaches actors to help each other
```

---

## Why MAPPO for Traffic Control?

### Problem: Traffic Lights are NOT Independent

**Example scenario:**
1. **J0** (north entry) has morning rush → gives long green to south-bound traffic
2. All cars go south to **J11**
3. **J11** gets overwhelmed with queue from J0
4. Result: J0 solved its problem but created congestion at J11 ❌

**What we need:**
- J0 should consider: "Will J11 be able to handle this traffic?"
- J11 should prepare: "J0 is about to send traffic, I should clear space"
- **Coordination between junctions** 🎯

### MAPPO Solution

**During Training (learns coordination):**
```
Critic sees GLOBAL state:
- J0 has queue = 20 vehicles
- J11 has queue = 30 vehicles (already congested!)
- J0 wants to send more south

Critic learns: "This is BAD! J11 is already full"
↓
Actor J0 learns: "Don't send more traffic south when J11 is congested"
Actor J11 learns: "Clear traffic faster when J0 has queue"
```

**During Execution (uses local sensors only):**
```
J0 observes:
- Own queue: 20 vehicles
- J11's phase: Currently red for north-south
- J11's approximate queue: Medium (from traffic pattern)

J0's learned policy says: "Wait! J11 isn't ready. Give short green."
✅ J0 acts independently but with learned coordination
```

---

## Core Concepts

### 1. Centralized Training, Decentralized Execution (CTDE)

**Centralized Training:**
```python
# During training, critic sees EVERYTHING
global_state = {
    'J0': [queue_n, queue_s, queue_e, queue_w, ...],
    'J1': [queue_n, queue_s, queue_e, queue_w, ...],
    'J5': [queue_n, queue_s, queue_e, queue_w, ...],
    # ... all 9 junctions
    'network_total_vehicles': 450,
    'network_waiting_time': 2350.5,
    'traffic_flow_patterns': [...],
}

# Critic learns: "What's the value of this GLOBAL situation?"
value = critic(global_state)  # Uses all information
```

**Decentralized Execution:**
```python
# During deployment, each junction uses only LOCAL sensors
local_state_J0 = {
    'own_queue_n': 15,  # From induction loops
    'own_queue_s': 8,
    'own_vehicles_weighted': 45.5,  # From cameras
    'neighbor_J11_phase': 2,  # From communication
    # NO global network information!
}

# Actor acts independently
action = actor_J0(local_state_J0)  # Only local state
```

**Why this matters:**
- ✅ **Training:** Learn coordination with full information
- ✅ **Deployment:** Each junction is independent (realistic!)
- ✅ **Robustness:** If J11 fails, J0 still works
- ✅ **Scalability:** Can add/remove junctions easily

---

### 2. PPO (Proximal Policy Optimization)

**What PPO solves:**
Policy gradient methods can be unstable - one bad update can ruin everything.

**PPO's solution: Clipped updates**
```python
# Old policy (before update)
old_probability = 0.4  # 40% chance of action "extend green"

# New policy (after gradient update)
new_probability = 0.9  # 90% chance of action "extend green"

# Problem: This is TOO big a change! (0.4 → 0.9)
# Could destabilize training

# PPO clips the change:
ratio = new_probability / old_probability  # 0.9 / 0.4 = 2.25
clipped_ratio = clip(ratio, 0.8, 1.2)      # Clip to max 1.2

# Result: Update is limited to reasonable range
# Policy changes smoothly, training is stable ✅
```

---

### 3. Actor-Critic Architecture

**Actor (Policy Network):**
- **Job:** Decide what action to take
- **Input:** Local state (own junction + neighbors)
- **Output:** Probability distribution over actions
- **Example:** [Keep: 70%, Switch: 20%, Extend: 10%]

**Critic (Value Network):**
- **Job:** Evaluate how good a situation is
- **Input:** Global state (all junctions + network)
- **Output:** Expected future reward (value)
- **Example:** "This state will lead to average reward of 45.2"

**How they work together:**
```
Actor: "I think action A is good"
Critic: "Actually, that state had value 10, next state has value 15"
Actor: "Oh! Action A improved things by +5. I'll do that more often!"

vs.

Actor: "I think action B is good"
Critic: "That state had value 20, next state has value 5"
Actor: "Oops! Action B made things worse by -15. I'll avoid that!"
```

---

## Architecture Components

### Component 1: Actor Networks (9 independent)

```python
class ActorNetwork(nn.Module):
    """
    One actor per junction - learns individual policy
    
    Input: Local state (17 dimensions)
    Output: Action probabilities (4 actions)
    """
    def __init__(self):
        super().__init__()
        # Input layer: local state → hidden
        self.fc1 = nn.Linear(17, 128)
        # Hidden layers
        self.fc2 = nn.Linear(128, 64)
        # Output layer: hidden → action probabilities
        self.fc3 = nn.Linear(64, 4)
    
    def forward(self, local_state):
        """
        local_state shape: (batch_size, 17)
        
        Example local_state for J0:
        [
            2,        # current_phase (0-7)
            15, 8, 12, 6,  # queue_n, queue_s, queue_e, queue_w
            45.5, 24.0, 38.5, 19.0,  # weighted_vehicles (using vehicle types!)
            0.65, 0.40, 0.55, 0.30,  # occupancy (0-1)
            28,       # time_in_phase (seconds)
            0,        # emergency_vehicle (0 or 1)
            5, 3      # neighbor_J11_phase, neighbor_J6_phase
        ]
        """
        x = F.relu(self.fc1(local_state))  # (batch, 17) → (batch, 128)
        x = F.relu(self.fc2(x))            # (batch, 128) → (batch, 64)
        x = self.fc3(x)                    # (batch, 64) → (batch, 4)
        action_probs = F.softmax(x, dim=-1) # Convert to probabilities
        
        """
        Output example:
        [0.65, 0.25, 0.08, 0.02]
        Means:
        - Action 0 (keep current): 65% probability
        - Action 1 (next phase): 25% probability
        - Action 2 (extend phase): 8% probability
        - Action 3 (emergency override): 2% probability
        """
        return action_probs
```

**Why separate actors?**
- Each junction has different traffic patterns
- J0 (north entry) ≠ J22 (internal junction)
- Each needs specialized policy
- But they share learning through critic! 🔗

---

### Component 2: Critic Network (1 shared)

```python
class CriticNetwork(nn.Module):
    """
    ONE critic shared by all junctions - evaluates global state
    
    Input: Global state (155 dimensions)
    Output: Single value (expected future reward)
    """
    def __init__(self):
        super().__init__()
        # Bigger network - needs to understand complex global state
        self.fc1 = nn.Linear(155, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
    
    def forward(self, global_state):
        """
        global_state shape: (batch_size, 155)
        
        Example global_state:
        [
            # All 9 local states concatenated (9 × 17 = 153)
            J0_local_state (17 dims),
            J1_local_state (17 dims),
            J5_local_state (17 dims),
            J6_local_state (17 dims),
            J7_local_state (17 dims),
            J10_local_state (17 dims),
            J11_local_state (17 dims),
            J12_local_state (17 dims),
            J22_local_state (17 dims),
            
            # Network-wide features (2 dims)
            450,      # total_network_vehicles
            2350.5,   # total_network_waiting_time
        ]
        """
        x = F.relu(self.fc1(global_state))  # (batch, 155) → (batch, 256)
        x = F.relu(self.fc2(x))             # (batch, 256) → (batch, 256)
        x = F.relu(self.fc3(x))             # (batch, 256) → (batch, 128)
        value = self.fc4(x)                 # (batch, 128) → (batch, 1)
        
        """
        Output example: 45.2
        Means: "I expect future total reward of 45.2 from this state"
        
        This value considers:
        - Are queues balanced across network?
        - Will congestion spread?
        - Are junctions coordinating well?
        """
        return value
```

**Why shared critic?**
- Sees entire network → understands coordination
- Teaches actors: "Your action affects others!"
- Learns global objective: "Minimize network-wide waiting time"

---

### Component 3: Experience Buffer

```python
class ReplayBuffer:
    """
    Stores experiences during episode for batch training
    """
    def __init__(self):
        self.local_states = []   # List of [J0_state, J1_state, ..., J22_state]
        self.global_states = []  # List of global_state
        self.actions = []        # List of [J0_action, J1_action, ..., J22_action]
        self.rewards = []        # List of [J0_reward, J1_reward, ..., J22_reward]
        self.log_probs = []      # List of action log probabilities
        self.dones = []          # List of episode done flags
    
    def store(self, local_states, global_state, actions, rewards, log_probs, done):
        """
        Store one timestep of experience
        
        Example:
        local_states = [
            tensor([2, 15, 8, ...]),  # J0's local state
            tensor([5, 20, 10, ...]), # J1's local state
            # ... all 9 junctions
        ]
        global_state = tensor([2, 15, 8, ..., 450, 2350.5])  # All info
        actions = [0, 1, 0, 2, 0, 1, 0, 0, 1]  # Each junction's action
        rewards = [-2.5, -3.1, -1.8, ...]      # Each junction's reward
        """
        self.local_states.append(local_states)
        self.global_states.append(global_state)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.log_probs.append(log_probs)
        self.dones.append(done)
    
    def get_batch(self):
        """
        Return entire buffer as batch for training
        """
        return {
            'local_states': torch.stack(self.local_states),
            'global_states': torch.stack(self.global_states),
            'actions': torch.tensor(self.actions),
            'rewards': torch.tensor(self.rewards),
            'log_probs': torch.stack(self.log_probs),
            'dones': torch.tensor(self.dones),
        }
    
    def clear(self):
        """Clear buffer after training"""
        self.local_states = []
        self.global_states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
```

---

## How Training Works

### Step 1: Experience Collection

```python
# Initialize environment
env.reset()  # Start SUMO simulation

# Collect one episode (e.g., 3600 steps = 1 hour simulation)
for step in range(3600):
    # 1. Get states
    local_states = []
    for junction_id in range(9):
        local_state = get_local_state(junction_id)  # Only local sensors
        local_states.append(local_state)
    
    global_state = get_global_state()  # Combine all for critic
    
    # 2. Each actor selects action
    actions = []
    log_probs = []
    for i, actor in enumerate(actors):
        action_probs = actor(local_states[i])
        
        # Sample action from probability distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        
        actions.append(action)
        log_probs.append(dist.log_prob(action))
    
    # 3. Execute actions in environment
    next_states, rewards, done = env.step(actions)
    
    # 4. Store experience
    buffer.store(local_states, global_state, actions, rewards, log_probs, done)
    
    # 5. Move to next state
    if done:
        break
```

**What's happening:**
- Each junction observes its **local state** (realistic sensors)
- Each junction independently chooses an **action** (phase change)
- Environment executes all actions simultaneously
- Each junction receives a **reward** (based on traffic improvement)
- Experience is stored for later training

---

### Step 2: Advantage Calculation (GAE)

**Advantage = "How much better was this action than expected?"**

```python
def compute_gae(rewards, values, next_values, gamma=0.99, lambda_=0.95):
    """
    Generalized Advantage Estimation (GAE)
    
    Intuition:
    - If reward > expected, advantage is positive → reinforce action
    - If reward < expected, advantage is negative → avoid action
    """
    advantages = []
    gae = 0
    
    # Work backwards through episode
    for t in reversed(range(len(rewards))):
        # TD error: reward + discounted next value - current value
        delta = rewards[t] + gamma * next_values[t] - values[t]
        
        # Accumulate advantage with exponential decay
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages)

"""
Example:
Step 1: reward = -2.5, value = 45.2, next_value = 47.8
  delta = -2.5 + 0.99 * 47.8 - 45.2 = 0.622
  This action was GOOD! (improved state value)

Step 2: reward = -8.5, value = 47.8, next_value = 42.1
  delta = -8.5 + 0.99 * 42.1 - 47.8 = -14.621
  This action was BAD! (decreased state value)
"""
```

**Why GAE?**
- Balances bias vs variance
- More stable than simple advantage
- Uses both immediate reward and future value

---

### Step 3: Actor Update (PPO Clip)

```python
def update_actor(actor, optimizer, batch, advantages, clip_epsilon=0.2):
    """
    Update actor using PPO clipped objective
    """
    local_states = batch['local_states']
    actions = batch['actions']
    old_log_probs = batch['log_probs']
    
    # Get new action probabilities
    new_action_probs = actor(local_states)
    dist = Categorical(new_action_probs)
    new_log_probs = dist.log_prob(actions)
    
    # Compute probability ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    """
    Example:
    old_log_prob = -0.916 (old probability = 0.4)
    new_log_prob = -0.105 (new probability = 0.9)
    ratio = exp(-0.105 - (-0.916)) = exp(0.811) = 2.25
    
    This means: new policy is 2.25× more likely to take this action
    """
    
    # Clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    
    """
    If advantage > 0 (good action):
      surr1 = 2.25 * 5.0 = 11.25  (want to increase probability)
      surr2 = 1.2 * 5.0 = 6.0     (clipped to max 1.2)
      Use min = 6.0 (limit update)
    
    If advantage < 0 (bad action):
      surr1 = 2.25 * (-5.0) = -11.25  (want to decrease probability)
      surr2 = 1.2 * (-5.0) = -6.0     (clipped)
      Use min = -11.25 (allow full decrease)
    
    Result: Can quickly decrease bad actions, but slowly increase good ones
    → Prevents overconfidence, stabilizes training ✅
    """
    
    actor_loss = -torch.min(surr1, surr2).mean()
    
    # Add entropy bonus (encourages exploration)
    entropy = dist.entropy().mean()
    actor_loss = actor_loss - 0.01 * entropy
    
    # Update
    optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
    optimizer.step()
    
    return actor_loss.item()
```

---

### Step 4: Critic Update

```python
def update_critic(critic, optimizer, batch, advantages):
    """
    Update critic to better predict values
    """
    global_states = batch['global_states']
    rewards = batch['rewards']
    values = batch['values']
    
    # Compute target values (returns)
    returns = advantages + values
    
    """
    Example:
    advantage = 5.0 (action was better than expected)
    value = 45.2 (critic's prediction)
    return = 50.2 (actual observed return)
    
    Critic should learn: "I underestimated! Should have predicted 50.2"
    """
    
    # Get current predictions
    predicted_values = critic(global_states)
    
    # Value loss (MSE)
    critic_loss = F.mse_loss(predicted_values, returns)
    
    # Update
    optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
    optimizer.step()
    
    return critic_loss.item()
```

---

## How Coordination Emerges

### Example: Morning Rush (North → East flow)

**Training Episode 1 (No coordination yet):**

```
Step 100:
J0 state: queue_north = 25 vehicles (high!)
J0 action: Long green for south-bound (clear queue)
J0 reward: -25 (own queue reduced)

J11 state: queue_north = 35 vehicles (overloaded from J0!)
J11 action: Try to clear, but too much traffic
J11 reward: -35 (queue growing)

Critic sees global state:
  global_state = [J0 has 25, J11 has 35, ...]
  value = critic(global_state) = 15.2
  
Next state after actions:
  J0: queue = 10 (improved!)
  J11: queue = 50 (much worse!)
  next_value = critic(next_state) = 8.5
  
Critic learns: "State value DECREASED from 15.2 → 8.5"
  → This global situation is BAD
  → Even though J0 improved, J11 got worse
  → Net effect: negative

Actor J0 learns: "Wait, my action had advantage = -6.7 (negative!)"
  → "I thought clearing my queue was good..."
  → "But critic says global situation got worse"
  → "Maybe I shouldn't send so much traffic to J11?"
```

**Training Episode 1000 (Coordination learned!):**

```
Step 100:
J0 state: queue_north = 25 vehicles
J0 also observes: neighbor_J11_phase = 2 (J11 busy with east-west)

J0's learned policy: "J11 is busy, don't send more traffic"
J0 action: SHORT green for south-bound (controlled release)
J0 reward: -20 (queue reduced moderately)

J11 state: queue_north = 10 vehicles (manageable)
J11 action: Balance all directions
J11 reward: -10 (steady state)

Critic sees:
  global_state = [J0 has 20, J11 has 10, ...]
  value = 25.8
  
Next state:
  J0: queue = 15 (gradual improvement)
  J11: queue = 12 (still manageable)
  next_value = 28.5
  
Critic: "State value INCREASED from 25.8 → 28.5"
  → Global situation improved!
  → Both junctions balanced

Actor J0: "advantage = +2.7 (positive!)"
  → "My coordinated action was good!"
  → "Keep doing this - short greens when J11 is busy"

Actor J11: "advantage = +1.2 (positive!)"
  → "My balanced approach was good!"
  → "Keep managing traffic steadily"
```

**Result: Emergent Coordination! 🎯**
- J0 learned to **meter traffic** based on downstream conditions
- J11 learned to **prepare for upstream traffic**
- No explicit rule - learned through shared critic
- Works because critic sees global consequences

---

## Step-by-Step Execution Flow

### During Training:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. OBSERVATION PHASE                                        │
│                                                             │
│    Environment (SUMO) → TraCI → State Extraction           │
│                                                             │
│    For each junction:                                       │
│      - Read queue lengths (induction loops)                │
│      - Count vehicles by type (cameras)                    │
│      - Calculate occupancy (sensors)                       │
│      - Get neighbor phases (communication)                 │
│                                                             │
│    Output: 9 local states + 1 global state                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. ACTION SELECTION PHASE                                   │
│                                                             │
│    For each junction i:                                     │
│      actor_i(local_state_i) → action_probabilities         │
│      sample(action_probabilities) → action_i               │
│                                                             │
│    Output: [action_0, action_1, ..., action_8]             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. EXECUTION PHASE                                          │
│                                                             │
│    TraCI → SUMO: Apply all actions simultaneously          │
│    SUMO: Simulate 1 second of traffic                      │
│    SUMO → TraCI: Return new states                         │
│                                                             │
│    Calculate rewards:                                       │
│      - Waiting time change                                 │
│      - Throughput                                          │
│      - Neighbor impacts                                    │
│                                                             │
│    Output: 9 rewards, next states                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. STORAGE PHASE                                            │
│                                                             │
│    Store in buffer:                                         │
│      - Local states (9 × 17 dims)                          │
│      - Global state (155 dims)                             │
│      - Actions (9 integers)                                │
│      - Rewards (9 floats)                                  │
│      - Log probabilities                                   │
│                                                             │
│    Repeat steps 1-4 for entire episode (3600 steps)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. LEARNING PHASE (after episode ends)                      │
│                                                             │
│    A. Critic evaluates all states:                          │
│       values = critic(global_states)                        │
│                                                             │
│    B. Compute advantages:                                   │
│       advantages = GAE(rewards, values)                     │
│                                                             │
│    C. Update all 9 actors (PPO):                            │
│       For each actor_i:                                     │
│         - Compute ratio of new/old probabilities           │
│         - Clip ratio to prevent big updates                │
│         - Maximize: min(ratio * advantage, clipped)        │
│                                                             │
│    D. Update critic:                                        │
│       - Minimize: MSE(predicted_values, returns)           │
│                                                             │
│    E. Clear buffer, start new episode                       │
└─────────────────────────────────────────────────────────────┘
```

### During Deployment (Execution Only):

```
┌─────────────────────────────────────────────────────────────┐
│ REAL-TIME LOOP (every second)                               │
│                                                             │
│    1. Each junction reads LOCAL sensors:                    │
│       - Queue lengths (induction loops) ✅                 │
│       - Vehicle types (cameras) ✅                         │
│       - Occupancy (sensors) ✅                             │
│       - Neighbor phases (wireless comm) ✅                 │
│                                                             │
│    2. Each junction's actor network:                        │
│       action = actor_i(local_state_i)                      │
│       [All actors run in parallel/independently]            │
│                                                             │
│    3. Apply actions:                                        │
│       Execute phase changes                                 │
│                                                             │
│    Note: NO critic needed! ❌                               │
│    Note: NO global state needed! ❌                         │
│    Note: NO communication during action! ❌                 │
│                                                             │
│    → Fully decentralized execution ✅                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### 1. **Two Roles**
- **Actors:** Make decisions (one per junction)
- **Critic:** Evaluate decisions (shared, sees everything)

### 2. **Two Phases**
- **Training:** Critic teaches actors to coordinate (centralized)
- **Deployment:** Actors work independently (decentralized)

### 3. **Learning Process**
- Actors propose actions → Environment responds → Critic evaluates
- If global outcome good → Reinforce those actions
- If global outcome bad → Avoid those actions
- Over time: Coordination emerges!

### 4. **Why It Works**
- **Shared critic** sees consequences of all actions
- Teaches each actor: "Consider your impact on others"
- **Local actors** learn policies that help the network
- No explicit coordination rules - emerges from learning!

---

**Ready for the actual code implementation?** 
Next document: `mappo_k1_implementation.py` - Complete working code! 🚀
