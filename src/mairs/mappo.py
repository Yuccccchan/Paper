"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Training framework for collaborative MAIRS agents

Based on the actor-critic architecture with PPO clipping mechanism
as described in Cao et al. 2025 paper.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class MAPPO:
    """
    Multi-Agent PPO trainer for MAIRS framework.
    
    Implements collaborative training with:
    - Shared critic information for better global understanding
    - PPO clipping mechanism for stable policy updates
    - Action masking for invalid actions
    """
    
    def __init__(self, agents, clip_epsilon=0.2, gamma=0.99, gae_lambda=0.95,
                 value_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 alpha=0.25):
        """
        Initialize MAPPO trainer.
        
        Args:
            agents: Dictionary of {agent_id: MAIRSAgent}
            clip_epsilon: PPO clipping parameter (default: 0.2)
            gamma: Discount factor (default: 0.99)
            gae_lambda: GAE lambda for advantage estimation (default: 0.95)
            value_loss_coef: Coefficient for value loss (default: 0.5)
            entropy_coef: Coefficient for entropy bonus (default: 0.01)
            max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
            alpha: Scaling factor for reward function (default: 0.25)
        """
        self.agents = agents
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def compute_reward(self, network, flow, transmission_success, max_utilization):
        """
        Compute reward for an episode.
        
        Reward = transmission_success_indicator - alpha * max_slot_utilization
        
        Args:
            network: TSNNetwork instance
            flow: TTFlow instance  
            transmission_success: Boolean indicating if flow was successfully scheduled
            max_utilization: Maximum slot utilization across all links
            
        Returns:
            Reward value (float)
        """
        success_indicator = 1.0 if transmission_success else 0.0
        reward = success_indicator - self.alpha * max_utilization
        return reward
    
    def rollout_flow(self, network, flow):
        """
        Perform a rollout for a single flow using current policies.
        
        Args:
            network: TSNNetwork instance
            flow: TTFlow instance
            
        Returns:
            Tuple of (trajectory, total_reward, success)
            where trajectory is list of (agent_id, state, action, log_prob, reward, mask)
        """
        trajectory = []
        current_switch = flow.source
        visited_switches = set([current_switch])
        flow.route = [current_switch]
        flow.schedule = {}
        
        max_hops = 10  # Prevent infinite loops
        hops = 0
        success = False
        
        while current_switch != flow.destination and hops < max_hops:
            # Get agent for current switch
            if current_switch not in self.agents:
                break
            
            agent = self.agents[current_switch]
            
            # Get local observation
            state = agent.get_local_observation(network, current_switch, flow, visited_switches)
            
            # Create action mask
            action_mask = agent.create_action_mask(network, current_switch, flow, visited_switches)
            
            # Select action
            action, log_prob = agent.select_action(state, action_mask, deterministic=False)
            
            # Decode action
            neighbor_idx, slot_idx = agent.decode_action(action)
            
            # Get actual neighbor
            switch = network.switches[current_switch]
            neighbors = switch.get_neighbors()
            
            if neighbor_idx >= len(neighbors):
                break  # Invalid action
            
            next_switch = neighbors[neighbor_idx]
            
            # Check for loop
            if next_switch in visited_switches:
                break  # Loop detected, terminate
            
            # Try to allocate slot
            port_id = switch.get_port_to_neighbor(next_switch)
            if port_id is None:
                break
            
            allocated = switch.allocate_slot(port_id, slot_idx, flow.flow_id)
            if not allocated:
                break  # Slot allocation failed
            
            # Record in flow schedule
            link = (current_switch, next_switch)
            flow.schedule[link] = slot_idx
            
            # Store transition
            trajectory.append({
                'agent_id': current_switch,
                'state': state,
                'action': action,
                'log_prob': log_prob,
                'action_mask': action_mask,
                'reward': 0.0  # Will be computed at episode end
            })
            
            # Move to next switch
            visited_switches.add(next_switch)
            flow.route.append(next_switch)
            current_switch = next_switch
            hops += 1
        
        # Check if successfully reached destination
        success = (current_switch == flow.destination)
        flow.successfully_scheduled = success
        
        # Compute maximum utilization across all links in route
        max_utilization = 0.0
        if len(flow.route) > 1:
            for i in range(len(flow.route) - 1):
                util = network.get_link_utilization(flow.route[i], flow.route[i+1])
                max_utilization = max(max_utilization, util)
        
        # Compute reward
        total_reward = self.compute_reward(network, flow, success, max_utilization)
        
        # Assign reward to all transitions in trajectory
        for transition in trajectory:
            transition['reward'] = total_reward
        
        return trajectory, total_reward, success
    
    def update_policy(self, trajectories, batch_size=32, epochs=4):
        """
        Update agent policies using collected trajectories.
        
        Args:
            trajectories: List of trajectories from rollouts
            batch_size: Batch size for training (default: 32)
            epochs: Number of epochs to train on each batch (default: 4)
            
        Returns:
            Dictionary of training statistics
        """
        if len(trajectories) == 0:
            return {}
        
        # Organize transitions by agent
        agent_transitions = defaultdict(list)
        for trajectory in trajectories:
            for transition in trajectory:
                agent_id = transition['agent_id']
                agent_transitions[agent_id].append(transition)
        
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'num_updates': 0
        }
        
        # Update each agent
        for agent_id, transitions in agent_transitions.items():
            if agent_id not in self.agents or len(transitions) == 0:
                continue
            
            agent = self.agents[agent_id]
            
            # Convert transitions to tensors
            states = torch.FloatTensor([t['state'] for t in transitions])
            actions = torch.LongTensor([t['action'] for t in transitions])
            old_log_probs = torch.FloatTensor([t['log_prob'] for t in transitions])
            rewards = torch.FloatTensor([t['reward'] for t in transitions])
            action_masks = torch.FloatTensor([t['action_mask'] for t in transitions])
            
            # Compute advantages (simplified - using reward directly)
            with torch.no_grad():
                values = agent.critic(states).squeeze()
            advantages = rewards - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update epochs
            for _ in range(epochs):
                # Forward pass through actor
                action_probs = agent.actor(states)
                
                # Apply action masks
                action_probs = action_probs * action_masks
                action_probs = action_probs / (action_probs.sum(dim=1, keepdim=True) + 1e-10)
                
                # Get log probabilities for taken actions
                log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-10)
                
                # Compute ratio and clipped ratio
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                # Policy loss with PPO clipping
                policy_loss = -torch.min(
                    ratio * advantages.detach(),
                    clipped_ratio * advantages.detach()
                ).mean()
                
                # Entropy bonus for exploration
                entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()
                
                # Update actor
                actor_loss = policy_loss - self.entropy_coef * entropy
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), self.max_grad_norm)
                agent.actor_optimizer.step()
                
                # Value loss (computed separately for critic)
                values_pred = agent.critic(states).squeeze()
                value_loss = nn.MSELoss()(values_pred, rewards)
                
                # Update critic
                agent.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), self.max_grad_norm)
                agent.critic_optimizer.step()
                
                # Record statistics
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.item()
                stats['num_updates'] += 1
        
        # Average statistics
        if stats['num_updates'] > 0:
            stats['policy_loss'] /= stats['num_updates']
            stats['value_loss'] /= stats['num_updates']
            stats['entropy'] /= stats['num_updates']
        
        return stats
    
    def train(self, network, flows, num_iterations=8000, batch_size=32):
        """
        Train agents using MAPPO.
        
        Args:
            network: TSNNetwork instance
            flows: List of TTFlow instances to schedule
            num_iterations: Number of training iterations (default: 8000)
            batch_size: Batch size for policy updates (default: 32)
            
        Returns:
            Training history
        """
        history = {
            'success_rates': [],
            'avg_rewards': [],
            'policy_losses': [],
            'value_losses': []
        }
        
        for iteration in range(num_iterations):
            trajectories = []
            rewards = []
            successes = []
            
            # Collect trajectories
            for flow in flows:
                # Reset network schedules
                network.reset_schedules()
                
                # Rollout flow
                trajectory, reward, success = self.rollout_flow(network, flow)
                
                trajectories.append(trajectory)
                rewards.append(reward)
                successes.append(success)
            
            # Update policies
            if len(trajectories) >= batch_size:
                stats = self.update_policy(trajectories, batch_size=batch_size)
                
                # Record history
                history['success_rates'].append(np.mean(successes))
                history['avg_rewards'].append(np.mean(rewards))
                history['policy_losses'].append(stats.get('policy_loss', 0.0))
                history['value_losses'].append(stats.get('value_loss', 0.0))
                
                # Print progress
                if (iteration + 1) % 100 == 0:
                    print(f"Iteration {iteration + 1}/{num_iterations}, "
                          f"Success Rate: {np.mean(successes):.3f}, "
                          f"Avg Reward: {np.mean(rewards):.3f}")
        
        return history
