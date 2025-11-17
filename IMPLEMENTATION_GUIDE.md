# Implementation Guide: TSN-WiFi Cross-Domain Scheduling with Attention-based DRL

This guide provides detailed instructions on how to implement the system described in "Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks" (ICCPS 2025).

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Implementation Steps](#implementation-steps)
4. [Detailed Component Design](#detailed-component-design)
5. [Algorithm Implementation](#algorithm-implementation)
6. [Training and Evaluation](#training-and-evaluation)

---

## System Overview

### Problem Statement
The system addresses cross-domain flow scheduling in TSN-WiFi hybrid networks that integrates:
- **TSN Domain**: Time-Sensitive Networking with deterministic wired communication
- **WiFi Domain**: IEEE 802.11be MLO (Multi-Link Operation) for wireless communication
- **Objective**: Achieve reliable, low-latency flow transmission by coordinating spatial-temporal-frequential resources

### Key Innovation
Uses **self-attention mechanism** from NLP to learn dependencies among flows in the network, analogous to how transformers learn token dependencies in sequences.

### Main Challenges Addressed
1. **Resource dimension mismatch**: TSN uses space-time, WiFi uses frequency-airtime
2. **Continuous decision space**: WiFi airtime allocation is continuous
3. **Scalability**: DRL model must handle variable number of flows
4. **Effectiveness**: Must learn inter-flow dependencies for optimal scheduling

---

## Architecture Components

### Overall System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TSN-WiFi Hybrid Network                   │
├──────────────────────────────┬──────────────────────────────┤
│        TSN Domain            │       WiFi Domain            │
│                              │                              │
│  ┌────────────────────┐     │     ┌──────────────────┐    │
│  │ Central Network    │     │     │  Access Point    │    │
│  │ Configuration      │     │     │  (AP)            │    │
│  │ (CNC)              │     │     │                  │    │
│  │                    │     │     │  ┌────────────┐  │    │
│  │ ┌──────────────┐   │     │     │  │ Attention- │  │    │
│  │ │Load-Aware    │   │     │     │  │ based DRL  │  │    │
│  │ │TSN Scheduler │   │◄────┼─────┼──┤ Scheduler  │  │    │
│  │ └──────────────┘   │     │     │  └────────────┘  │    │
│  │                    │     │     │                  │    │
│  │  GCL Configuration │     │     │  WiFi MLO        │    │
│  └────────────────────┘     │     │  Management      │    │
│           │                 │     └──────────────────┘    │
│           ▼                 │              │               │
│  ┌───────────────┐          │              ▼               │
│  │ TSN Switches  │──────────┼────────► MLD (2.4/5/6 GHz)  │
│  └───────────────┘          │                              │
│                              │                              │
└──────────────────────────────┴──────────────────────────────┘
```

### Component Breakdown

#### 1. TSN Domain Components
- **Central Network Configuration (CNC)**: Centralized controller
- **TSN Switches**: Time-Aware Shaper (TAS) with Gate Control Lists (GCL)
- **End Stations**: Source devices synchronized with IEEE 802.1AS

#### 2. WiFi Domain Components
- **Access Point (AP)**: Bridge between TSN and WiFi
- **Multi-Link Device (MLD)**: Supports 2.4 GHz, 5 GHz, 6 GHz bands
- **U-MAC Layer**: Traffic management and flow distribution
- **L-MAC Layer**: Per-link channel access with EDCA queues

#### 3. Scheduling Modules
- **Load-Aware TSN Scheduler**: Minimizes congestion in wired domain
- **Attention-based DDPG Scheduler**: Adaptive wireless scheduling with DRL

---

## Implementation Steps

### Phase 1: Environment Setup

#### 1.1 Dependencies
```python
# Core Dependencies
- Python 3.8+
- PyTorch 1.10+ (for DRL models)
- NumPy 1.20+
- NetworkX 2.6+ (for graph topology)
- Gym 0.21+ (for RL environment)

# Additional Libraries
- matplotlib (visualization)
- pandas (data analysis)
- scipy (optimization)
```

#### 1.2 Project Structure
```
tsn-wifi-scheduler/
├── config/
│   ├── network_topology.yaml
│   └── scheduling_params.yaml
├── tsn_domain/
│   ├── __init__.py
│   ├── topology.py
│   ├── flow.py
│   ├── switch.py
│   └── load_aware_scheduler.py
├── wifi_domain/
│   ├── __init__.py
│   ├── mld.py
│   ├── channel.py
│   └── airtime_manager.py
├── drl_agent/
│   ├── __init__.py
│   ├── attention_network.py
│   ├── ddpg_agent.py
│   └── replay_buffer.py
├── environment/
│   ├── __init__.py
│   ├── tsn_wifi_env.py
│   └── state_representation.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
├── main.py
├── train.py
└── evaluate.py
```

### Phase 2: TSN Domain Implementation

#### 2.1 Network Topology Model
```python
# tsn_domain/topology.py
import networkx as nx

class TSNTopology:
    """
    TSN network as directed graph G = (V, E)
    V = vertices (switches + end stations)
    E = edges (links with bandwidth and latency)
    """
    def __init__(self, topology_config):
        self.graph = nx.DiGraph()
        self.switches = []
        self.end_stations = []
        self.links = []
        self._build_topology(topology_config)
    
    def _build_topology(self, config):
        # Add switches and end stations as nodes
        for switch in config['switches']:
            self.graph.add_node(switch['id'], 
                              type='switch',
                              port_num=switch['ports'])
            self.switches.append(switch['id'])
        
        for station in config['end_stations']:
            self.graph.add_node(station['id'], type='end_station')
            self.end_stations.append(station['id'])
        
        # Add links with bandwidth and latency
        for link in config['links']:
            self.graph.add_edge(link['src'], link['dst'],
                              bandwidth=link['bandwidth'],  # Mbps
                              latency=link['latency'])      # microseconds
            self.links.append((link['src'], link['dst']))
    
    def get_all_paths(self, src, dst):
        """Get all simple paths between source and destination"""
        return list(nx.all_simple_paths(self.graph, src, dst))
    
    def get_link_load(self, link):
        """Calculate current load on a link"""
        # Return current bandwidth utilization
        return self.graph.edges[link].get('current_load', 0)
```

#### 2.2 Flow Model
```python
# tsn_domain/flow.py
class Flow:
    """
    Flow model with attributes:
    - src: source end station
    - dst: destination end station
    - size: packet size (bytes)
    - start_offset: initial offset in cycle (microseconds)
    - deadline: maximum end-to-end delay constraint
    - period: transmission period (for periodic flows)
    """
    def __init__(self, flow_id, src, dst, size, start_offset, 
                 deadline, period=None):
        self.flow_id = flow_id
        self.src = src
        self.dst = dst
        self.size = size  # bytes (typically 1500 MTU)
        self.start_offset = start_offset  # μs
        self.deadline = deadline  # μs
        self.period = period  # μs
        
        # Scheduling results
        self.path = None  # Selected path in TSN
        self.gcl_entries = []  # Gate Control List entries
        self.wifi_band = None  # Selected WiFi band
        self.wifi_airtime = 0  # Allocated airtime in WiFi
        self.reliability = 0  # Achieved reliability
    
    def get_transmission_time(self, bandwidth_mbps):
        """Calculate transmission time for this flow"""
        # size in bytes, bandwidth in Mbps
        return (self.size * 8) / (bandwidth_mbps * 1e6) * 1e6  # μs
```

#### 2.3 Load-Aware TSN Scheduling Algorithm
```python
# tsn_domain/load_aware_scheduler.py
import numpy as np

class LoadAwareTSNScheduler:
    """
    Algorithm 1 from paper: Load-aware TSN flow scheduling
    Minimizes maximum link load to avoid bottlenecks
    """
    def __init__(self, topology):
        self.topology = topology
        self.time_slot_duration = 10  # μs per time slot
        self.cycle_time = 1000  # μs (1ms cycle)
    
    def schedule_flows(self, flows):
        """
        Main scheduling algorithm
        Returns: routing and timing for all flows
        """
        scheduled_flows = []
        link_loads = {link: [] for link in self.topology.links}
        
        # Sort flows by start offset
        sorted_flows = sorted(flows, key=lambda f: f.start_offset)
        
        for flow in sorted_flows:
            # Step 1: Path selection using load-aware routing
            path = self._select_path_load_aware(flow, link_loads)
            
            # Step 2: Time slot allocation
            gcl_entries = self._allocate_time_slots(flow, path, link_loads)
            
            # Step 3: Update flow and link states
            flow.path = path
            flow.gcl_entries = gcl_entries
            scheduled_flows.append(flow)
            
            # Update link loads
            self._update_link_loads(flow, path, gcl_entries, link_loads)
        
        return scheduled_flows
    
    def _select_path_load_aware(self, flow, link_loads):
        """
        Select path that minimizes maximum link load
        Avoids bottleneck links
        """
        all_paths = self.topology.get_all_paths(flow.src, flow.dst)
        
        best_path = None
        min_max_load = float('inf')
        
        for path in all_paths:
            # Calculate maximum load on this path
            max_load = 0
            for i in range(len(path) - 1):
                link = (path[i], path[i+1])
                current_load = len(link_loads.get(link, []))
                max_load = max(max_load, current_load)
            
            # Select path with minimum maximum load
            if max_load < min_max_load:
                min_max_load = max_load
                best_path = path
        
        return best_path
    
    def _allocate_time_slots(self, flow, path, link_loads):
        """
        Allocate time slots for flow on each link in path
        Returns GCL entries for each switch
        """
        gcl_entries = {}
        current_offset = flow.start_offset
        
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            
            # Find available time slot
            slot = self._find_available_slot(link, current_offset, 
                                            flow, link_loads)
            
            # Create GCL entry
            gcl_entries[link] = {
                'offset': slot,
                'duration': flow.get_transmission_time(
                    self.topology.graph.edges[link]['bandwidth']
                ),
                'queue': self._get_priority_queue(flow)
            }
            
            # Update offset for next link (propagation delay)
            current_offset = slot + gcl_entries[link]['duration'] + \
                           self.topology.graph.edges[link]['latency']
        
        return gcl_entries
    
    def _find_available_slot(self, link, preferred_offset, flow, link_loads):
        """Find earliest available time slot on link"""
        occupied_slots = link_loads.get(link, [])
        
        # Round to time slot boundary
        slot = (preferred_offset // self.time_slot_duration) * \
               self.time_slot_duration
        
        # Check for conflicts
        while self._has_conflict(slot, flow, occupied_slots):
            slot += self.time_slot_duration
        
        return slot
    
    def _has_conflict(self, slot, flow, occupied_slots):
        """Check if slot conflicts with existing reservations"""
        trans_time = flow.get_transmission_time(100)  # approximate
        
        for existing in occupied_slots:
            if not (slot + trans_time <= existing['start'] or 
                   slot >= existing['end']):
                return True
        return False
    
    def _update_link_loads(self, flow, path, gcl_entries, link_loads):
        """Update link load tracking"""
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            entry = gcl_entries[link]
            link_loads[link].append({
                'flow': flow.flow_id,
                'start': entry['offset'],
                'end': entry['offset'] + entry['duration']
            })
    
    def _get_priority_queue(self, flow):
        """Map flow to priority queue (0-7)"""
        # Higher priority flows get higher queue numbers
        return 7  # Highest priority for time-sensitive flows
```

### Phase 3: WiFi Domain Implementation

#### 3.1 Multi-Link Device (MLD) Model
```python
# wifi_domain/mld.py
class WiFiMLO:
    """
    WiFi Multi-Link Operation model
    Supports 2.4 GHz, 5 GHz, 6 GHz bands
    """
    def __init__(self):
        self.bands = {
            '2.4GHz': {'frequency': 2.4, 'bandwidth': 20},  # MHz
            '5GHz': {'frequency': 5.0, 'bandwidth': 80},
            '6GHz': {'frequency': 6.0, 'bandwidth': 160}
        }
        
        # Channel state per band
        self.channel_states = {
            '2.4GHz': ChannelState(),
            '5GHz': ChannelState(),
            '6GHz': ChannelState()
        }
        
        # Airtime tracking
        self.airtime_used = {band: 0 for band in self.bands}
        self.max_airtime = 1000  # μs per cycle
    
    def get_available_airtime(self, band):
        """Return available airtime on band"""
        return self.max_airtime - self.airtime_used[band]
    
    def allocate_airtime(self, band, duration):
        """Allocate airtime for a flow"""
        if self.airtime_used[band] + duration <= self.max_airtime:
            self.airtime_used[band] += duration
            return True
        return False
    
    def reset_cycle(self):
        """Reset airtime counters for new cycle"""
        self.airtime_used = {band: 0 for band in self.bands}

class ChannelState:
    """Channel state information for a WiFi band"""
    def __init__(self):
        self.snr = 0  # Signal-to-Noise Ratio (dB)
        self.interference = 0
        self.packet_loss_rate = 0
        self.busy_ratio = 0
    
    def update_snr(self, snr_value):
        """Update SNR measurement"""
        self.snr = snr_value
    
    def get_data_rate(self):
        """Calculate data rate based on SNR using Shannon capacity"""
        # Simplified: actual WiFi uses MCS (Modulation and Coding Scheme)
        bandwidth_hz = 20e6  # 20 MHz
        capacity_bps = bandwidth_hz * np.log2(1 + 10**(self.snr/10))
        return capacity_bps / 1e6  # Mbps
    
    def calculate_reliability(self, packet_size_bytes):
        """
        Calculate packet delivery reliability
        Based on SNR and channel conditions
        """
        # Simplified model: reliability decreases with poor SNR
        ber = 0.5 * np.exp(-0.5 * 10**(self.snr/10))  # Bit Error Rate
        per = 1 - (1 - ber)**(packet_size_bytes * 8)  # Packet Error Rate
        return 1 - per  # Reliability
```

#### 3.2 WiFi Scheduling with DRL

### Phase 4: Attention-based DRL Implementation

#### 4.1 Self-Attention Network Architecture
```python
# drl_agent/attention_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    Learns dependencies among flows
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        x: (batch_size, num_flows, embed_dim)
        Returns: (batch_size, num_flows, embed_dim)
        """
        batch_size, num_flows, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, num_flows, self.num_heads, self.head_dim)
        k = k.view(batch_size, num_flows, self.num_heads, self.head_dim)
        v = v.view(batch_size, num_flows, self.num_heads, self.head_dim)
        
        # Transpose: (batch, heads, flows, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_flows, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output

class AttentionActorNetwork(nn.Module):
    """
    Actor network with attention mechanism
    Outputs scheduling decisions for all flows
    """
    def __init__(self, flow_feature_dim, embed_dim=128, num_heads=8, 
                 num_layers=3, num_bands=3):
        super().__init__()
        
        self.flow_feature_dim = flow_feature_dim
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        
        # Input embedding layer
        self.input_embedding = nn.Linear(flow_feature_dim, embed_dim)
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.band_selection_head = nn.Linear(embed_dim, num_bands)
        self.airtime_allocation_head = nn.Linear(embed_dim, 1)
    
    def forward(self, flow_features):
        """
        flow_features: (batch_size, num_flows, flow_feature_dim)
        Returns: 
            band_logits: (batch_size, num_flows, num_bands)
            airtime: (batch_size, num_flows, 1)
        """
        # Embed input features
        x = self.input_embedding(flow_features)
        
        # Apply attention layers with residual connections
        for i in range(len(self.attention_layers)):
            # Self-attention
            attn_out = self.attention_layers[i](x)
            x = self.layer_norms[i](x + attn_out)
            
            # Feed-forward
            ffn_out = self.ffn_layers[i](x)
            x = x + ffn_out
        
        # Generate outputs
        band_logits = self.band_selection_head(x)
        airtime = self.airtime_allocation_head(x)
        
        # Apply constraints
        airtime = torch.sigmoid(airtime)  # Normalize to [0, 1]
        
        return band_logits, airtime

class AttentionCriticNetwork(nn.Module):
    """
    Critic network with attention mechanism
    Estimates Q-value for state-action pairs
    """
    def __init__(self, flow_feature_dim, action_dim, embed_dim=128, 
                 num_heads=8, num_layers=3):
        super().__init__()
        
        # Input embedding
        self.state_embedding = nn.Linear(flow_feature_dim, embed_dim)
        self.action_embedding = nn.Linear(action_dim, embed_dim)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])
        
        # Q-value output
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, flow_features, actions):
        """
        flow_features: (batch_size, num_flows, flow_feature_dim)
        actions: (batch_size, num_flows, action_dim)
        Returns: Q-value (batch_size, 1)
        """
        # Embed state and action
        state_embed = self.state_embedding(flow_features)
        action_embed = self.action_embedding(actions)
        
        # Apply attention to state
        x = state_embed
        for i in range(len(self.attention_layers)):
            attn_out = self.attention_layers[i](x)
            x = self.layer_norms[i](x + attn_out)
        
        # Aggregate state and action
        state_agg = torch.mean(x, dim=1)  # (batch, embed_dim)
        action_agg = torch.mean(action_embed, dim=1)  # (batch, embed_dim)
        
        # Concatenate and predict Q-value
        combined = torch.cat([state_agg, action_agg], dim=-1)
        q_value = self.q_head(combined)
        
        return q_value
```

#### 4.2 DDPG Agent Implementation
```python
# drl_agent/ddpg_agent.py
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """Experience replay buffer for DDPG"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class AttentionDDPGAgent:
    """
    DDPG Agent with Attention-based Networks
    Algorithm 2 from paper
    """
    def __init__(self, flow_feature_dim, action_dim, num_bands=3,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001):
        
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Soft update parameter
        
        # Actor networks (policy)
        self.actor = AttentionActorNetwork(flow_feature_dim, num_bands=num_bands)
        self.actor_target = AttentionActorNetwork(flow_feature_dim, num_bands=num_bands)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks (Q-function)
        self.critic = AttentionCriticNetwork(flow_feature_dim, action_dim)
        self.critic_target = AttentionCriticNetwork(flow_feature_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
    
    def select_action(self, state, noise_scale=0.1):
        """
        Select action using actor network with exploration noise
        state: (num_flows, flow_feature_dim)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            band_logits, airtime = self.actor(state_tensor)
        
        # Band selection (categorical)
        band_probs = torch.softmax(band_logits, dim=-1)
        band_dist = torch.distributions.Categorical(band_probs)
        band_actions = band_dist.sample()
        
        # Airtime allocation (continuous) with noise
        airtime = airtime.squeeze(0)
        noise = torch.randn_like(airtime) * noise_scale
        airtime = torch.clamp(airtime + noise, 0, 1)
        
        # Convert to numpy
        band_actions = band_actions.squeeze(0).cpu().numpy()
        airtime = airtime.cpu().numpy()
        
        return band_actions, airtime
    
    def train_step(self):
        """
        Perform one training step
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ===== Update Critic =====
        with torch.no_grad():
            # Get next actions from target actor
            next_band_logits, next_airtime = self.actor_target(next_states)
            next_bands = torch.argmax(next_band_logits, dim=-1)
            
            # Combine into action representation
            next_actions = torch.cat([
                F.one_hot(next_bands, num_classes=3).float(),
                next_airtime
            ], dim=-1)
            
            # Compute target Q-value
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-value
        current_q = self.critic(states, actions)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== Update Actor =====
        band_logits, airtime = self.actor(states)
        bands = torch.argmax(band_logits, dim=-1)
        
        # Combine actions
        actions_pred = torch.cat([
            F.one_hot(bands, num_classes=3).float(),
            airtime
        ], dim=-1)
        
        # Actor loss (negative Q-value)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ===== Soft Update Target Networks =====
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
    
    def _soft_update(self, source, target):
        """Soft update: target = tau * source + (1-tau) * target"""
        for target_param, source_param in zip(target.parameters(), 
                                               source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """Save model checkpoints"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load model checkpoints"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
```

### Phase 5: Environment Implementation

#### 5.1 State Representation
```python
# environment/state_representation.py
import numpy as np

class StateEncoder:
    """
    Encodes environment state into feature vectors
    Features include: flow info, channel state, resource availability
    """
    def __init__(self, num_bands=3):
        self.num_bands = num_bands
    
    def encode_flow_features(self, flows, tsn_schedules, channel_states):
        """
        Encode features for all flows
        Returns: (num_flows, feature_dim) array
        """
        flow_features = []
        
        for flow in flows:
            features = []
            
            # 1. Flow characteristics
            features.append(flow.size / 1500.0)  # Normalized size
            features.append(flow.start_offset / 1000.0)  # Normalized offset
            features.append(flow.deadline / 1000.0)  # Normalized deadline
            
            # 2. TSN domain features
            if flow.path:
                # Path length
                features.append(len(flow.path) / 10.0)  # Normalized
                
                # TSN end time
                tsn_end_time = self._get_tsn_end_time(flow, tsn_schedules)
                features.append(tsn_end_time / 1000.0)
                
                # Remaining budget for WiFi
                remaining_budget = flow.deadline - tsn_end_time
                features.append(remaining_budget / 1000.0)
            else:
                features.extend([0, 0, 0])
            
            # 3. Channel state features (for each band)
            for band_name in ['2.4GHz', '5GHz', '6GHz']:
                channel = channel_states[band_name]
                features.append(channel.snr / 40.0)  # Normalized SNR
                features.append(channel.busy_ratio)
                features.append(1 - channel.packet_loss_rate)  # Success rate
            
            flow_features.append(features)
        
        return np.array(flow_features, dtype=np.float32)
    
    def _get_tsn_end_time(self, flow, tsn_schedules):
        """Calculate when flow completes TSN transmission"""
        if not flow.gcl_entries:
            return flow.start_offset
        
        # Last link end time
        last_link = list(flow.gcl_entries.keys())[-1]
        entry = flow.gcl_entries[last_link]
        return entry['offset'] + entry['duration']

class RewardCalculator:
    """
    Calculate reward for scheduling decisions
    """
    def __init__(self, target_reliability=0.99):
        self.target_reliability = target_reliability
    
    def calculate_reward(self, flows, scheduled_flows):
        """
        Reward based on:
        1. Average reliability
        2. Deadline satisfaction
        3. Resource efficiency
        """
        reliabilities = []
        deadline_violations = 0
        
        for flow, scheduled in zip(flows, scheduled_flows):
            # Reliability reward
            reliability = scheduled.reliability
            reliabilities.append(reliability)
            
            # Deadline check
            end_to_end_delay = self._calculate_delay(scheduled)
            if end_to_end_delay > flow.deadline:
                deadline_violations += 1
        
        # Average reliability
        avg_reliability = np.mean(reliabilities)
        
        # Reward components
        reliability_reward = avg_reliability * 100
        deadline_penalty = -deadline_violations * 50
        
        # Bonus for meeting target
        if avg_reliability >= self.target_reliability:
            bonus = 50
        else:
            bonus = 0
        
        total_reward = reliability_reward + deadline_penalty + bonus
        
        return total_reward, {
            'avg_reliability': avg_reliability,
            'deadline_violations': deadline_violations,
            'num_flows': len(flows)
        }
    
    def _calculate_delay(self, flow):
        """Calculate end-to-end delay"""
        if not flow.gcl_entries:
            return 0
        
        # TSN delay
        last_link = list(flow.gcl_entries.keys())[-1]
        tsn_delay = flow.gcl_entries[last_link]['offset'] + \
                   flow.gcl_entries[last_link]['duration']
        
        # WiFi delay
        wifi_delay = flow.wifi_airtime
        
        return tsn_delay + wifi_delay
```

#### 5.2 Complete Environment
```python
# environment/tsn_wifi_env.py
import gym
from gym import spaces
import numpy as np

class TSNWiFiEnv(gym.Env):
    """
    TSN-WiFi Scheduling Environment
    Integrates TSN and WiFi domains
    """
    def __init__(self, topology_config, num_flows=50, cycle_time=1000):
        super().__init__()
        
        # Initialize domains
        self.topology = TSNTopology(topology_config)
        self.tsn_scheduler = LoadAwareTSNScheduler(self.topology)
        self.wifi_mlo = WiFiMLO()
        
        # Environment parameters
        self.num_flows = num_flows
        self.cycle_time = cycle_time  # μs
        self.current_cycle = 0
        
        # State encoder and reward calculator
        self.state_encoder = StateEncoder()
        self.reward_calculator = RewardCalculator()
        
        # Observation and action spaces
        flow_feature_dim = 15  # Features per flow
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(num_flows, flow_feature_dim),
            dtype=np.float32
        )
        
        # Action: band selection (3) + airtime (1) per flow
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(num_flows, 4),
            dtype=np.float32
        )
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_cycle = 0
        
        # Generate new flows
        self.flows = self._generate_flows()
        
        # TSN domain scheduling
        self.tsn_scheduled_flows = self.tsn_scheduler.schedule_flows(self.flows)
        
        # Reset WiFi
        self.wifi_mlo.reset_cycle()
        self._update_channel_states()
        
        # Get initial state
        state = self.state_encoder.encode_flow_features(
            self.tsn_scheduled_flows,
            self.tsn_scheduler,
            self.wifi_mlo.channel_states
        )
        
        return state
    
    def step(self, action):
        """
        Execute scheduling action
        action: (num_flows, 4) - [band_one_hot(3), airtime(1)]
        """
        # Parse actions
        band_actions = np.argmax(action[:, :3], axis=1)
        airtime_actions = action[:, 3]
        
        # Apply WiFi scheduling
        scheduled_flows = self._apply_wifi_scheduling(
            self.tsn_scheduled_flows,
            band_actions,
            airtime_actions
        )
        
        # Calculate reward
        reward, info = self.reward_calculator.calculate_reward(
            self.flows, scheduled_flows
        )
        
        # Update cycle
        self.current_cycle += 1
        done = self.current_cycle >= 1  # One cycle per episode
        
        # Next state (for multi-cycle, would generate new flows)
        next_state = self.state_encoder.encode_flow_features(
            scheduled_flows,
            self.tsn_scheduler,
            self.wifi_mlo.channel_states
        )
        
        return next_state, reward, done, info
    
    def _generate_flows(self):
        """Generate random flows for current cycle"""
        flows = []
        
        for i in range(self.num_flows):
            # Random source and destination
            src = np.random.choice(self.topology.end_stations)
            dst_candidates = [s for s in self.topology.end_stations if s != src]
            dst = np.random.choice(dst_candidates)
            
            # Flow parameters
            size = 1500  # MTU
            start_offset = np.random.uniform(0, 400)  # μs
            deadline = np.random.uniform(600, 1000)  # μs
            
            flow = Flow(i, src, dst, size, start_offset, deadline)
            flows.append(flow)
        
        return flows
    
    def _apply_wifi_scheduling(self, flows, band_actions, airtime_actions):
        """Apply WiFi scheduling decisions"""
        band_names = ['2.4GHz', '5GHz', '6GHz']
        
        for flow, band_idx, airtime_norm in zip(flows, band_actions, airtime_actions):
            band_name = band_names[band_idx]
            
            # Calculate actual airtime needed
            channel_state = self.wifi_mlo.channel_states[band_name]
            data_rate = channel_state.get_data_rate()
            min_airtime = (flow.size * 8) / data_rate  # μs
            
            # Scale airtime
            max_airtime = self.wifi_mlo.get_available_airtime(band_name)
            allocated_airtime = min_airtime + airtime_norm * (max_airtime - min_airtime)
            
            # Allocate if possible
            if self.wifi_mlo.allocate_airtime(band_name, allocated_airtime):
                flow.wifi_band = band_name
                flow.wifi_airtime = allocated_airtime
                
                # Calculate reliability
                flow.reliability = channel_state.calculate_reliability(flow.size)
            else:
                # Fallback: use minimum airtime
                flow.wifi_band = band_name
                flow.wifi_airtime = min_airtime
                flow.reliability = 0.5  # Low reliability
        
        return flows
    
    def _update_channel_states(self):
        """Simulate dynamic channel conditions"""
        for band_name in self.wifi_mlo.channel_states:
            channel = self.wifi_mlo.channel_states[band_name]
            
            # Simulate SNR variations
            if band_name == '6GHz':
                snr = np.random.normal(35, 3)  # Best quality
            elif band_name == '5GHz':
                snr = np.random.normal(30, 4)
            else:  # 2.4GHz
                snr = np.random.normal(25, 5)  # Most interference
            
            channel.update_snr(np.clip(snr, 10, 40))
            channel.busy_ratio = np.random.uniform(0.1, 0.4)
            channel.packet_loss_rate = np.random.uniform(0.001, 0.01)
```

### Phase 6: Training Pipeline

#### 6.1 Training Script
```python
# train.py
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_ddpg_agent(env, agent, num_episodes=1000, max_steps=100):
    """
    Train DDPG agent on TSN-WiFi environment
    """
    episode_rewards = []
    episode_reliabilities = []
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action with exploration noise
            noise_scale = max(0.1, 1.0 - episode / num_episodes)
            band_actions, airtime_actions = agent.select_action(
                state, noise_scale=noise_scale
            )
            
            # Combine actions
            action = np.concatenate([
                np.eye(3)[band_actions],  # One-hot encode bands
                airtime_actions
            ], axis=-1)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            if len(agent.replay_buffer) > agent.batch_size:
                agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_reliabilities.append(info.get('avg_reliability', 0))
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_reliability = np.mean(episode_reliabilities[-10:])
            print(f"Episode {episode+1}: Reward={avg_reward:.2f}, "
                  f"Reliability={avg_reliability:.4f}")
        
        # Save checkpoint
        if (episode + 1) % 100 == 0:
            agent.save(f"checkpoints/agent_episode_{episode+1}.pt")
    
    return episode_rewards, episode_reliabilities

def plot_training_results(rewards, reliabilities):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Rewards
    ax1.plot(rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True)
    
    # Reliabilities
    ax2.plot(reliabilities)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reliability')
    ax2.set_title('Flow Reliability')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == '__main__':
    # Load topology
    topology_config = {
        'switches': [{'id': f's{i}', 'ports': 8} for i in range(10)],
        'end_stations': [f'es{i}' for i in range(20)],
        'links': [
            # Define network topology links
            {'src': 'es0', 'dst': 's0', 'bandwidth': 1000, 'latency': 10},
            # ... more links
        ]
    }
    
    # Create environment
    env = TSNWiFiEnv(topology_config, num_flows=50)
    
    # Create agent
    flow_feature_dim = 15
    action_dim = 4  # 3 for band + 1 for airtime
    agent = AttentionDDPGAgent(flow_feature_dim, action_dim)
    
    # Train
    rewards, reliabilities = train_ddpg_agent(env, agent, num_episodes=1000)
    
    # Plot results
    plot_training_results(rewards, reliabilities)
```

### Phase 7: Evaluation and Testing

#### 7.1 Evaluation Metrics
```python
# utils/metrics.py
import numpy as np

class PerformanceMetrics:
    """Calculate performance metrics for evaluation"""
    
    @staticmethod
    def calculate_reliability(flows):
        """Average reliability across all flows"""
        reliabilities = [f.reliability for f in flows]
        return np.mean(reliabilities)
    
    @staticmethod
    def calculate_deadline_satisfaction_rate(flows):
        """Percentage of flows meeting deadline"""
        satisfied = 0
        for flow in flows:
            end_to_end_delay = PerformanceMetrics._get_delay(flow)
            if end_to_end_delay <= flow.deadline:
                satisfied += 1
        return satisfied / len(flows)
    
    @staticmethod
    def calculate_resource_utilization(env):
        """Resource utilization in TSN and WiFi"""
        # TSN link utilization
        tsn_util = {}
        for link in env.topology.links:
            load = env.topology.get_link_load(link)
            bandwidth = env.topology.graph.edges[link]['bandwidth']
            tsn_util[link] = load / bandwidth
        
        # WiFi airtime utilization
        wifi_util = {}
        for band, used in env.wifi_mlo.airtime_used.items():
            wifi_util[band] = used / env.wifi_mlo.max_airtime
        
        return {
            'tsn': np.mean(list(tsn_util.values())),
            'wifi': np.mean(list(wifi_util.values()))
        }
    
    @staticmethod
    def _get_delay(flow):
        """Calculate end-to-end delay"""
        if not flow.gcl_entries:
            return float('inf')
        
        last_link = list(flow.gcl_entries.keys())[-1]
        tsn_end = flow.gcl_entries[last_link]['offset'] + \
                 flow.gcl_entries[last_link]['duration']
        
        return tsn_end + flow.wifi_airtime

def evaluate_agent(env, agent, num_episodes=100):
    """Evaluate trained agent"""
    metrics = {
        'reliability': [],
        'deadline_satisfaction': [],
        'resource_utilization': []
    }
    
    for _ in range(num_episodes):
        state = env.reset()
        
        # Get action without noise
        band_actions, airtime_actions = agent.select_action(state, noise_scale=0)
        
        action = np.concatenate([
            np.eye(3)[band_actions],
            airtime_actions
        ], axis=-1)
        
        _, _, _, info = env.step(action)
        
        # Collect metrics
        metrics['reliability'].append(info['avg_reliability'])
        metrics['deadline_satisfaction'].append(
            PerformanceMetrics.calculate_deadline_satisfaction_rate(env.flows)
        )
        util = PerformanceMetrics.calculate_resource_utilization(env)
        metrics['resource_utilization'].append(util)
    
    # Print results
    print(f"Average Reliability: {np.mean(metrics['reliability']):.4f}")
    print(f"Deadline Satisfaction: {np.mean(metrics['deadline_satisfaction']):.4f}")
    print(f"TSN Utilization: {np.mean([u['tsn'] for u in metrics['resource_utilization']]):.4f}")
    print(f"WiFi Utilization: {np.mean([u['wifi'] for u in metrics['resource_utilization']]):.4f}")
    
    return metrics
```

---

## Key Implementation Details

### 1. Feature Engineering for DRL

**Flow Features (Input to Attention Network)**:
- Flow size (normalized)
- Start offset (normalized)
- Deadline (normalized)
- Path length in TSN
- TSN completion time
- Remaining time budget for WiFi
- Per-band SNR values
- Per-band channel busy ratio
- Per-band success rate

### 2. Action Space Design

**Discrete Component**: Band selection (2.4 GHz, 5 GHz, 6 GHz)
**Continuous Component**: Airtime allocation (normalized 0-1)

### 3. Reward Function

```
Reward = α * average_reliability 
         - β * deadline_violations
         + γ * bonus_for_target_achievement
         
Where:
  α = 100 (reliability weight)
  β = 50 (penalty weight)
  γ = 50 (bonus for ≥99% reliability)
```

### 4. Training Hyperparameters

```python
HYPERPARAMETERS = {
    'learning_rate_actor': 1e-4,
    'learning_rate_critic': 1e-3,
    'gamma': 0.99,  # Discount factor
    'tau': 0.001,  # Soft update rate
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'num_attention_heads': 8,
    'embedding_dim': 128,
    'num_attention_layers': 3,
    'exploration_noise': 0.1 (decaying),
    'num_episodes': 1000
}
```

---

## Expected Results

Based on the paper, you should achieve:

1. **Reliability**: ~78.63% for 50 flows
2. **Improvement over baselines**:
   - +11.23% vs Policy Gradient
   - +10.53% vs Flow-wise DRL
   - +18.23% vs MLP-based DRL
   - +12.09% vs SLCI (Single Link Less Congested Interface)

3. **Convergence**: Stable reward curve after ~500-700 episodes

4. **TSN Scheduling**: Lower variance in link loads compared to shortest path

---

## Next Steps for Full Implementation

1. **Complete network topology configuration**: Define realistic industrial network topology
2. **Implement channel model**: Add more sophisticated wireless channel simulation
3. **Add baselines**: Implement comparison methods (Policy Gradient, MLP-based DRL, SLCI)
4. **Visualization tools**: Add real-time monitoring dashboard
5. **Deployment**: Create interface for actual TSN-WiFi hardware integration
6. **Testing**: Validate on different topologies and flow patterns

---

## References

- Paper: "Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks" (ICCPS 2025)
- IEEE 802.1 TSN Standards
- IEEE 802.11be EHT (WiFi 7) Standard
- DDPG: "Continuous control with deep reinforcement learning" (Lillicrap et al., 2015)
- Attention Mechanism: "Attention is All You Need" (Vaswani et al., 2017)

