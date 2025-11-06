"""
MAIRS Agent for integrated routing and scheduling decisions
Each switch has an agent that makes local routing and scheduling decisions
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx

class MAIRSAgent:
    """
    Individual agent for a TSN switch in MAIRS framework.
    
    Each agent makes local decisions about:
    1. Which neighboring switch to forward the flow to (routing)
    2. Which time slot to use for transmission (scheduling)
    """
    
    def __init__(self, agent_id, num_neighbors, max_slots, state_dim, 
                 hidden_dim=128, learning_rate=0.003):
        """
        Initialize MAIRS agent.
        
        Args:
            agent_id: Unique identifier for this agent (switch ID)
            num_neighbors: Number of neighboring switches
            max_slots: Maximum number of time slots in hyperperiod
            state_dim: Dimension of state observation
            hidden_dim: Hidden layer dimension for neural networks
            learning_rate: Learning rate for optimization
        """
        self.agent_id = agent_id
        self.num_neighbors = num_neighbors
        self.max_slots = max_slots
        self.state_dim = state_dim
        
        # Action space: (neighbor_id, slot_id)
        # Total actions = num_neighbors * max_slots
        # Ensure at least 1 action dimension
        self.action_dim = max(1, num_neighbors * max_slots)
        
        # Actor network (policy)
        self.actor = self._build_actor_network(hidden_dim)
        
        # Critic network (value function)
        self.critic = self._build_critic_network(hidden_dim)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
    def _build_actor_network(self, hidden_dim):
        """Build actor (policy) network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic_network(self, hidden_dim):
        """Build critic (value) network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_local_observation(self, network, current_switch, flow, visited_switches):
        """
        Generate local observation for current decision point.
        
        Args:
            network: TSNNetwork instance
            current_switch: Current switch ID
            flow: TTFlow instance
            visited_switches: Set of already visited switches (for loop detection)
            
        Returns:
            numpy array representing local state observation
        """
        obs = []
        
        # Flow features
        obs.append(flow.data_size / 1500.0)  # Normalized data size
        obs.append(flow.period / 6.0)  # Normalized period
        obs.append(flow.deadline / 6.0)  # Normalized deadline
        
        # Current position features
        obs.append(1.0 if current_switch == flow.source else 0.0)
        obs.append(1.0 if current_switch == flow.destination else 0.0)
        
        # Neighbor features
        switch = network.switches[current_switch]
        neighbors = switch.get_neighbors()
        
        for neighbor in neighbors:
            # Distance to destination
            try:
                dist = nx.shortest_path_length(network.graph, neighbor, flow.destination)
                obs.append(1.0 / (dist + 1))  # Inverse distance
            except (nx.NetworkXNoPath, nx.NetworkXError):
                obs.append(0.0)
            
            # Link utilization
            utilization = network.get_link_utilization(current_switch, neighbor)
            obs.append(utilization)
            
            # Already visited flag
            obs.append(1.0 if neighbor in visited_switches else 0.0)
        
        # Pad if fewer neighbors than max
        while len(obs) < self.state_dim:
            obs.append(0.0)
        
        return np.array(obs[:self.state_dim], dtype=np.float32)
    
    def select_action(self, state, action_mask=None, deterministic=False):
        """
        Select action based on current state.
        
        Args:
            state: State observation (numpy array or tensor)
            action_mask: Boolean mask for valid actions (1=valid, 0=invalid)
            deterministic: If True, select argmax; if False, sample from distribution
            
        Returns:
            Tuple of (action_index, action_log_prob)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Get action probabilities from actor
        with torch.no_grad():
            action_probs = self.actor(state)
        
        # Apply action mask if provided
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.FloatTensor(action_mask)
            
            # Mask invalid actions (set probability to 0)
            action_probs = action_probs * action_mask
            
            # Renormalize
            prob_sum = action_probs.sum()
            if prob_sum > 0:
                action_probs = action_probs / prob_sum
            else:
                # If all actions are masked, uniform over valid actions
                action_probs = action_mask / action_mask.sum()
        
        # Select action
        if deterministic:
            action = torch.argmax(action_probs)
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        # Calculate log probability
        log_prob = torch.log(action_probs[action] + 1e-10)
        
        return action.item(), log_prob.item()
    
    def decode_action(self, action_index):
        """
        Decode action index to (neighbor_index, slot_index).
        
        Args:
            action_index: Integer action index
            
        Returns:
            Tuple of (neighbor_index, slot_index)
        """
        if self.max_slots == 0:
            return 0, 0
        
        neighbor_idx = action_index // self.max_slots
        slot_idx = action_index % self.max_slots
        return neighbor_idx, slot_idx
    
    def create_action_mask(self, network, current_switch, flow, visited_switches):
        """
        Create action mask to filter out invalid actions.
        
        Invalid actions include:
        1. Routing loops (revisiting already visited switches)
        2. Slots already occupied on the link
        3. Actions that violate deadline constraints
        
        Args:
            network: TSNNetwork instance
            current_switch: Current switch ID
            flow: TTFlow instance
            visited_switches: Set of already visited switches
            
        Returns:
            Boolean numpy array (1 for valid actions, 0 for invalid)
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)
        
        switch = network.switches[current_switch]
        neighbors = switch.get_neighbors()
        
        for neighbor_idx, neighbor_id in enumerate(neighbors):
            # Check for routing loop
            if neighbor_id in visited_switches:
                continue  # Skip this neighbor (loop detection)
            
            port_id = switch.get_port_to_neighbor(neighbor_id)
            if port_id is None:
                continue
            
            # Check available slots
            for slot_idx in range(self.max_slots):
                if switch.is_slot_available(port_id, slot_idx):
                    action_idx = neighbor_idx * self.max_slots + slot_idx
                    if action_idx < self.action_dim:
                        mask[action_idx] = 1.0
        
        # If no valid actions, allow at least one (emergency fallback)
        if mask.sum() == 0 and self.action_dim > 0:
            mask[0] = 1.0
        
        return mask
    
    def get_value(self, state):
        """
        Get value estimate from critic network.
        
        Args:
            state: State observation
            
        Returns:
            Value estimate (float)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        with torch.no_grad():
            value = self.critic(state)
        
        return value.item()
    
    def __repr__(self):
        return f"MAIRSAgent(id={self.agent_id}, neighbors={self.num_neighbors})"
