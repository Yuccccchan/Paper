"""
Attention-based DDPG for WiFi Scheduling
Implements the Deep Deterministic Policy Gradient with self-attention mechanism
for dynamic WiFi band selection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from collections import deque
import random


class SelfAttentionLayer(nn.Module):
    """
    Self-Attention mechanism for capturing dependencies among flows.
    
    The attention mechanism helps the model understand relationships between
    different flows competing for network resources, similar to how tokens
    relate in NLP sequences.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize self-attention layer.
        
        Args:
            input_dim: Dimension of input features per flow
            hidden_dim: Hidden dimension for Q, K, V
        """
        super(SelfAttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Query, Key, Value projections
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Scaling factor for dot product attention
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x):
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, num_flows, input_dim)
            
        Returns:
            Attention output of shape (batch_size, num_flows, hidden_dim)
        """
        # Generate Q, K, V
        Q = self.query(x)  # (batch, num_flows, hidden_dim)
        K = self.key(x)    # (batch, num_flows, hidden_dim)
        V = self.value(x)  # (batch, num_flows, hidden_dim)
        
        # Compute attention scores
        # (batch, num_flows, num_flows)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        # (batch, num_flows, hidden_dim)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output


class AttentionActor(nn.Module):
    """
    Actor network with self-attention for band selection.
    
    Architecture:
    1. MLP layers for initial feature processing
    2. Self-attention layer to capture flow dependencies
    3. Residual connection
    4. Output layer with softmax for band selection probabilities
    """
    
    def __init__(self, state_dim: int, num_flows: int, hidden_dim: int = 64):
        """
        Initialize attention-based actor network.
        
        Args:
            state_dim: Dimension of state features (per flow)
            num_flows: Number of flows
            hidden_dim: Hidden dimension for MLP and attention
        """
        super(AttentionActor, self).__init__()
        
        self.state_dim = state_dim
        self.num_flows = num_flows
        self.hidden_dim = hidden_dim
        
        # MLP for initial feature processing
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Self-attention layer
        self.attention = SelfAttentionLayer(hidden_dim, hidden_dim)
        
        # Output layer (3 bands per flow)
        self.output = nn.Linear(hidden_dim, 3)
        
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch_size, num_flows, state_dim)
                   or (num_flows, state_dim) for single sample
            
        Returns:
            Band selection probabilities of shape (batch_size, num_flows, 3)
        """
        # Handle single sample (add batch dimension)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        
        # MLP layers with ReLU activation
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Self-attention with residual connection
        attention_out = self.attention(x)
        x = x + attention_out  # Residual connection
        
        # Output layer with softmax for band selection probabilities
        # Shape: (batch, num_flows, 3)
        logits = self.output(x)
        probs = F.softmax(logits, dim=-1)
        
        return probs


class AttentionCritic(nn.Module):
    """
    Critic network with self-attention for Q-value estimation.
    
    Architecture similar to Actor but outputs a single Q-value.
    """
    
    def __init__(self, state_dim: int, num_flows: int, hidden_dim: int = 64):
        """
        Initialize attention-based critic network.
        
        Args:
            state_dim: Dimension of state features (per flow)
            num_flows: Number of flows
            hidden_dim: Hidden dimension
        """
        super(AttentionCritic, self).__init__()
        
        self.state_dim = state_dim
        self.num_flows = num_flows
        self.hidden_dim = hidden_dim
        
        # MLP for state processing
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Self-attention layer
        self.attention = SelfAttentionLayer(hidden_dim, hidden_dim)
        
        # Action processing (band selections)
        self.action_fc = nn.Linear(3, hidden_dim)
        
        # Q-value output
        self.q1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch_size, num_flows, state_dim)
            action: Action tensor of shape (batch_size, num_flows, 3)
            
        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        # Handle single sample
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        if len(action.shape) == 2:
            action = action.unsqueeze(0)
        
        # Process state through MLP and attention
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        attention_out = self.attention(x)
        x = x + attention_out  # Residual connection
        
        # Pool over flows (mean pooling)
        x = torch.mean(x, dim=1)  # (batch, hidden_dim)
        
        # Process action
        a = F.relu(self.action_fc(action))
        a = torch.mean(a, dim=1)  # (batch, hidden_dim)
        
        # Concatenate state and action features
        xa = torch.cat([x, a], dim=1)  # (batch, hidden_dim * 2)
        
        # Q-value output
        q = F.relu(self.q1(xa))
        q = self.q2(q)  # (batch, 1)
        
        return q


class ReplayBuffer:
    """Experience replay buffer for DDPG."""
    
    def __init__(self, capacity: int = 8000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class AttentionDDPG:
    """
    Attention-based Deep Deterministic Policy Gradient for WiFi scheduling.
    
    This implements the DRL algorithm from the paper that uses self-attention
    to learn dependencies among flows for optimal band selection.
    """
    
    def __init__(self, state_dim: int, num_flows: int, 
                 hidden_dim: int = 64, lr: float = 0.0004,
                 gamma: float = 0.9, tau: float = 0.0004,
                 buffer_capacity: int = 8000, batch_size: int = 50,
                 device: str = 'cpu'):
        """
        Initialize Attention-based DDPG.
        
        Args:
            state_dim: Dimension of state features per flow
            num_flows: Number of flows
            hidden_dim: Hidden dimension for networks
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update parameter
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            device: Device to use ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.num_flows = num_flows
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # Actor networks
        self.actor = AttentionActor(state_dim, num_flows, hidden_dim).to(device)
        self.actor_target = AttentionActor(state_dim, num_flows, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic networks
        self.critic = AttentionCritic(state_dim, num_flows, hidden_dim).to(device)
        self.critic_target = AttentionCritic(state_dim, num_flows, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action (band for each flow) using current policy.
        
        Args:
            state: State array of shape (num_flows, state_dim)
            explore: Whether to add exploration noise
            
        Returns:
            Action array of shape (num_flows, 3) with band probabilities
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor).cpu().numpy()
        
        # Remove batch dimension if added
        if action_probs.shape[0] == 1:
            action_probs = action_probs[0]
        
        return action_probs
    
    def sample_action(self, action_probs: np.ndarray) -> np.ndarray:
        """
        Sample discrete actions from probability distribution.
        
        Args:
            action_probs: Probability distribution (num_flows, 3)
            
        Returns:
            Sampled band IDs (num_flows,)
        """
        num_flows = action_probs.shape[0]
        sampled_bands = np.array([
            np.random.choice(3, p=action_probs[i])
            for i in range(num_flows)
        ])
        return sampled_bands
    
    def update(self):
        """Update actor and critic networks using a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_action = self.actor(state)
        actor_loss = -self.critic(state, actor_action).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
    
    def _soft_update(self, target, source):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
