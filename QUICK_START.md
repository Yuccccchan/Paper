# Quick Start Example

This document provides a minimal working example to get started with the TSN-WiFi scheduling implementation.

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy networkx matplotlib pandas scipy
```

## Minimal Working Example

### Step 1: Create a Simple Topology

```python
# example_topology.py
topology_config = {
    'switches': [
        {'id': 's0', 'ports': 8},
        {'id': 's1', 'ports': 8},
        {'id': 's2', 'ports': 8},
    ],
    'end_stations': ['es0', 'es1', 'es2', 'es3'],
    'links': [
        # End stations to switches
        {'src': 'es0', 'dst': 's0', 'bandwidth': 1000, 'latency': 5},
        {'src': 'es1', 'dst': 's0', 'bandwidth': 1000, 'latency': 5},
        {'src': 'es2', 'dst': 's1', 'bandwidth': 1000, 'latency': 5},
        {'src': 'es3', 'dst': 's1', 'bandwidth': 1000, 'latency': 5},
        
        # Switch to switch
        {'src': 's0', 'dst': 's1', 'bandwidth': 1000, 'latency': 10},
        {'src': 's1', 'dst': 's0', 'bandwidth': 1000, 'latency': 10},
        
        # Switches to AP (represented as s2)
        {'src': 's0', 'dst': 's2', 'bandwidth': 1000, 'latency': 10},
        {'src': 's1', 'dst': 's2', 'bandwidth': 1000, 'latency': 10},
    ]
}
```

### Step 2: Test TSN Scheduling

```python
# test_tsn_scheduling.py
from tsn_domain.topology import TSNTopology
from tsn_domain.flow import Flow
from tsn_domain.load_aware_scheduler import LoadAwareTSNScheduler

# Create topology
topology = TSNTopology(topology_config)

# Create some test flows
flows = [
    Flow(0, 'es0', 's2', size=1500, start_offset=0, deadline=500),
    Flow(1, 'es1', 's2', size=1500, start_offset=50, deadline=500),
    Flow(2, 'es2', 's2', size=1500, start_offset=100, deadline=500),
]

# Schedule flows
scheduler = LoadAwareTSNScheduler(topology)
scheduled_flows = scheduler.schedule_flows(flows)

# Print results
for flow in scheduled_flows:
    print(f"Flow {flow.flow_id}:")
    print(f"  Path: {' -> '.join(flow.path)}")
    print(f"  GCL Entries:")
    for link, entry in flow.gcl_entries.items():
        print(f"    {link}: offset={entry['offset']}μs, duration={entry['duration']}μs")
    print()
```

### Step 3: Test WiFi Channel Model

```python
# test_wifi_channel.py
from wifi_domain.mld import WiFiMLO, ChannelState
import numpy as np

# Create WiFi MLO
wifi = WiFiMLO()

# Simulate channel conditions
wifi.channel_states['6GHz'].update_snr(35)  # Good SNR
wifi.channel_states['5GHz'].update_snr(28)
wifi.channel_states['2.4GHz'].update_snr(22)  # Poor SNR

# Test data rates
for band_name, channel in wifi.channel_states.items():
    data_rate = channel.get_data_rate()
    reliability = channel.calculate_reliability(1500)
    print(f"{band_name}:")
    print(f"  SNR: {channel.snr} dB")
    print(f"  Data Rate: {data_rate:.2f} Mbps")
    print(f"  Reliability: {reliability:.4f}")
    print()

# Test airtime allocation
packet_size = 1500  # bytes
for band_name in ['6GHz', '5GHz', '2.4GHz']:
    channel = wifi.channel_states[band_name]
    data_rate_mbps = channel.get_data_rate()
    airtime_us = (packet_size * 8) / data_rate_mbps
    
    if wifi.allocate_airtime(band_name, airtime_us):
        print(f"Allocated {airtime_us:.2f}μs on {band_name}")
    else:
        print(f"Failed to allocate on {band_name}")
```

### Step 4: Train Simple DRL Agent

```python
# train_simple.py
import torch
from environment.tsn_wifi_env import TSNWiFiEnv
from drl_agent.ddpg_agent import AttentionDDPGAgent
from example_topology import topology_config

# Create environment
env = TSNWiFiEnv(topology_config, num_flows=10)

# Create agent
flow_feature_dim = 15
action_dim = 4
agent = AttentionDDPGAgent(flow_feature_dim, action_dim)

# Training loop
num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    # One episode = one scheduling cycle
    band_actions, airtime_actions = agent.select_action(state, noise_scale=0.1)
    
    # Combine actions
    action = np.concatenate([
        np.eye(3)[band_actions],
        airtime_actions
    ], axis=-1)
    
    # Execute
    next_state, reward, done, info = env.step(action)
    
    # Store and train
    agent.replay_buffer.push(state, action, reward, next_state, done)
    if len(agent.replay_buffer) > agent.batch_size:
        agent.train_step()
    
    total_reward += reward
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, "
              f"Reliability={info['avg_reliability']:.4f}")

# Save trained model
agent.save("trained_model.pt")
print("Training complete!")
```

### Step 5: Evaluate Trained Agent

```python
# evaluate.py
import numpy as np
from environment.tsn_wifi_env import TSNWiFiEnv
from drl_agent.ddpg_agent import AttentionDDPGAgent
from example_topology import topology_config

# Create environment and agent
env = TSNWiFiEnv(topology_config, num_flows=10)
agent = AttentionDDPGAgent(flow_feature_dim=15, action_dim=4)

# Load trained model
agent.load("trained_model.pt")

# Evaluate
num_test_episodes = 50
reliabilities = []
deadline_satisfactions = []

for _ in range(num_test_episodes):
    state = env.reset()
    
    # Get action (no exploration noise)
    band_actions, airtime_actions = agent.select_action(state, noise_scale=0)
    
    action = np.concatenate([
        np.eye(3)[band_actions],
        airtime_actions
    ], axis=-1)
    
    _, _, _, info = env.step(action)
    
    reliabilities.append(info['avg_reliability'])
    deadline_satisfactions.append(
        sum(1 for f in env.flows if f.reliability > 0.95) / len(env.flows)
    )

# Print results
print("Evaluation Results:")
print(f"Average Reliability: {np.mean(reliabilities):.4f} ± {np.std(reliabilities):.4f}")
print(f"Deadline Satisfaction: {np.mean(deadline_satisfactions):.4f}")
```

## Visualization Example

```python
# visualize_scheduling.py
import matplotlib.pyplot as plt
import numpy as np

def visualize_tsn_schedule(scheduled_flows):
    """Visualize TSN scheduling Gantt chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect all links
    all_links = set()
    for flow in scheduled_flows:
        for link in flow.gcl_entries.keys():
            all_links.add(link)
    
    link_list = sorted(list(all_links))
    link_to_y = {link: i for i, link in enumerate(link_list)}
    
    # Plot each flow
    colors = plt.cm.tab10(np.linspace(0, 1, len(scheduled_flows)))
    
    for flow, color in zip(scheduled_flows, colors):
        for link, entry in flow.gcl_entries.items():
            y = link_to_y[link]
            start = entry['offset']
            duration = entry['duration']
            
            ax.barh(y, duration, left=start, height=0.8, 
                   color=color, alpha=0.7, label=f"Flow {flow.flow_id}")
    
    # Formatting
    ax.set_yticks(range(len(link_list)))
    ax.set_yticklabels([f"{l[0]}→{l[1]}" for l in link_list])
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Link')
    ax.set_title('TSN Schedule Gantt Chart')
    ax.grid(True, alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.savefig('tsn_schedule.png', dpi=150)
    plt.show()

def visualize_wifi_allocation(scheduled_flows):
    """Visualize WiFi band allocation"""
    bands = {'2.4GHz': [], '5GHz': [], '6GHz': []}
    
    for flow in scheduled_flows:
        if flow.wifi_band:
            bands[flow.wifi_band].append({
                'flow_id': flow.flow_id,
                'airtime': flow.wifi_airtime,
                'reliability': flow.reliability
            })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Airtime allocation per band
    band_names = list(bands.keys())
    airtimes = [sum(f['airtime'] for f in bands[b]) for b in band_names]
    
    ax1.bar(band_names, airtimes, color=['blue', 'green', 'red'], alpha=0.7)
    ax1.set_ylabel('Total Airtime (μs)')
    ax1.set_title('Airtime Allocation per Band')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reliability per band
    reliabilities = [np.mean([f['reliability'] for f in bands[b]]) 
                    if bands[b] else 0 for b in band_names]
    
    ax2.bar(band_names, reliabilities, color=['blue', 'green', 'red'], alpha=0.7)
    ax2.set_ylabel('Average Reliability')
    ax2.set_ylim([0, 1])
    ax2.set_title('Reliability per Band')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wifi_allocation.png', dpi=150)
    plt.show()

# Example usage
if __name__ == '__main__':
    # Assuming you have scheduled_flows from previous examples
    visualize_tsn_schedule(scheduled_flows)
    visualize_wifi_allocation(scheduled_flows)
```

## Running the Complete Pipeline

Create a main script that ties everything together:

```python
# main.py
import numpy as np
from example_topology import topology_config
from environment.tsn_wifi_env import TSNWiFiEnv
from drl_agent.ddpg_agent import AttentionDDPGAgent
from utils.metrics import PerformanceMetrics, evaluate_agent
import matplotlib.pyplot as plt

def main():
    print("=== TSN-WiFi Scheduling Demo ===\n")
    
    # Configuration
    NUM_FLOWS = 20
    NUM_TRAIN_EPISODES = 500
    NUM_EVAL_EPISODES = 50
    
    print(f"Configuration:")
    print(f"  Number of flows: {NUM_FLOWS}")
    print(f"  Training episodes: {NUM_TRAIN_EPISODES}")
    print(f"  Evaluation episodes: {NUM_EVAL_EPISODES}\n")
    
    # Create environment
    print("Creating environment...")
    env = TSNWiFiEnv(topology_config, num_flows=NUM_FLOWS)
    
    # Create agent
    print("Creating DRL agent...")
    agent = AttentionDDPGAgent(flow_feature_dim=15, action_dim=4)
    
    # Training
    print("\nTraining agent...")
    episode_rewards = []
    episode_reliabilities = []
    
    for episode in range(NUM_TRAIN_EPISODES):
        state = env.reset()
        
        # Exploration noise (decaying)
        noise_scale = max(0.01, 0.1 * (1 - episode / NUM_TRAIN_EPISODES))
        
        # Get action
        band_actions, airtime_actions = agent.select_action(state, noise_scale)
        action = np.concatenate([np.eye(3)[band_actions], airtime_actions], axis=-1)
        
        # Execute
        next_state, reward, done, info = env.step(action)
        
        # Train
        agent.replay_buffer.push(state, action, reward, next_state, done)
        if len(agent.replay_buffer) > agent.batch_size:
            agent.train_step()
        
        episode_rewards.append(reward)
        episode_reliabilities.append(info['avg_reliability'])
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_rel = np.mean(episode_reliabilities[-50:])
            print(f"  Episode {episode+1}: Reward={avg_reward:.2f}, Reliability={avg_rel:.4f}")
    
    # Save model
    print("\nSaving trained model...")
    agent.save("trained_agent.pt")
    
    # Plot training curves
    print("Generating training plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(episode_rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True)
    
    ax2.plot(episode_reliabilities)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reliability')
    ax2.set_title('Training Reliability')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("  Saved: training_curves.png")
    
    # Evaluation
    print("\nEvaluating agent...")
    metrics = evaluate_agent(env, agent, num_episodes=NUM_EVAL_EPISODES)
    
    print("\n=== Final Results ===")
    print(f"Average Reliability: {np.mean(metrics['reliability']):.4f}")
    print(f"Deadline Satisfaction: {np.mean(metrics['deadline_satisfaction']):.4f}")
    print(f"TSN Utilization: {np.mean([u['tsn'] for u in metrics['resource_utilization']]):.4f}")
    print(f"WiFi Utilization: {np.mean([u['wifi'] for u in metrics['resource_utilization']]):.4f}")
    
    print("\nDemo complete!")

if __name__ == '__main__':
    main()
```

## Expected Output

When running the complete pipeline, you should see:

```
=== TSN-WiFi Scheduling Demo ===

Configuration:
  Number of flows: 20
  Training episodes: 500
  Evaluation episodes: 50

Creating environment...
Creating DRL agent...

Training agent...
  Episode 50: Reward=450.23, Reliability=0.6234
  Episode 100: Reward=520.45, Reliability=0.6891
  Episode 150: Reward=580.12, Reliability=0.7234
  ...
  Episode 500: Reward=710.34, Reliability=0.7863

Saving trained model...
Generating training plots...
  Saved: training_curves.png

Evaluating agent...

=== Final Results ===
Average Reliability: 0.7863
Deadline Satisfaction: 0.9200
TSN Utilization: 0.6543
WiFi Utilization: 0.5821

Demo complete!
```

## Next Steps

1. **Tune Hyperparameters**: Experiment with learning rates, network architecture
2. **Add More Flows**: Scale up to 50-100 flows
3. **Complex Topologies**: Try larger network topologies
4. **Baseline Comparison**: Implement comparison methods (MLP-based, Policy Gradient)
5. **Real Channel Data**: Integrate real WiFi channel measurements
6. **Hardware Integration**: Connect to actual TSN switches and WiFi APs

## Troubleshooting

### Issue: Training doesn't converge
**Solution**: 
- Reduce learning rates
- Increase exploration noise initially
- Check state normalization
- Verify reward function

### Issue: Low reliability
**Solution**:
- Increase airtime allocation
- Improve channel SNR simulation
- Check band selection logic
- Tune reward weights

### Issue: Memory error
**Solution**:
- Reduce replay buffer size
- Reduce batch size
- Use CPU instead of GPU for small models

### Issue: Slow training
**Solution**:
- Use GPU if available
- Reduce number of attention layers
- Simplify network topology
- Reduce number of flows initially

## Resources

- Full implementation guide: `IMPLEMENTATION_GUIDE.md`
- Algorithm details: `ALGORITHMS.md`
- Paper: `3716550.3722018.pdf`
- GitHub repository: https://github.com/Yuccccchan/Paper

