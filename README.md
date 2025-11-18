# TSN-WiFi Network Scheduler

Implementation and reproduction of the paper:

**"Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks"**

*Miao Guo et al., ICCPS 2025*

## Overview

This repository contains a Python implementation that reproduces the key algorithms and methodologies from the paper. The system implements cross-domain scheduling for Time-Sensitive Networking (TSN) integrated with WiFi Multi-Link Operation (MLO).

### Key Features

1. **TSN Network Model**: Complete implementation of Time-Sensitive Network with Time-Aware Shaper (TAS) switches
2. **WiFi MLO Model**: Multi-band WiFi (2.4GHz, 5GHz, 6GHz) with dynamic channel quality
3. **Load-Aware TSN Scheduling**: Implements the bucket effect principle for balanced load distribution
4. **Attention-based DDPG**: Deep Reinforcement Learning with self-attention mechanism for WiFi band selection
5. **Integrated Scheduler**: Cross-domain scheduling with constraint enforcement

## Paper Summary

The paper addresses the challenge of scheduling flows across TSN (wired) and WiFi (wireless) domains in industrial automation scenarios. The key innovation is using self-attention mechanisms (borrowed from NLP) to capture dependencies among flows competing for network resources.

### Main Contributions

1. **Cross-domain scheduling**: First comprehensive solution for spatial-temporal-frequential resource allocation in TSN-WiFi networks
2. **Load-aware TSN scheduling**: Balances network load using the "bucket effect" principle
3. **Attention-based DRL**: Uses self-attention to learn flow dependencies for optimal band selection
4. **Practical validation**: Tested on realistic topologies (SAE, AFDX, Ladder) with different flow volumes

### Technical Details

- **TSN Domain**: Spatial (routing) and temporal (timing) scheduling with load balancing
- **WiFi Domain**: Frequential (band selection) scheduling using DDPG with self-attention
- **Optimization Objective**: Maximize overall reliability while satisfying timing constraints
- **DRL Architecture**: Actor-Critic with self-attention layers and residual connections

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy

### Install Dependencies

```bash
pip install torch numpy
```

### Install Package

```bash
# From repository root
pip install -e .
```

Or use directly without installation:

```bash
python example.py
```

## Usage

### Basic Example

```python
from tsn_wifi_scheduler import TSNNetwork, WiFiMLO, Flow
from tsn_wifi_scheduler.integrated_scheduler import IntegratedScheduler

# Create TSN network
network = TSNNetwork(cycle_time=1000.0)  # 1000 μs cycle
network.add_switch(0)
network.add_switch(1)
network.add_link(0, 0, 1)

# Create WiFi MLO
wifi = WiFiMLO(cycle_time=1000.0)

# Create flows
flow = Flow(flow_id=0, src=0, dst=1, size=1500, start_offset=0)
network.add_flow(flow)

# Run integrated scheduling
scheduler = IntegratedScheduler(network, wifi, use_drl=False)
results = scheduler.schedule([flow])

print(f"Scheduled: {results['total_scheduled']}/{results['total_flows']}")
print(f"Average reliability: {results['avg_reliability']:.3f}")
```

### Running the Example

The repository includes a comprehensive example demonstrating all features:

```bash
python example.py
```

This will:
1. Create a ladder topology network
2. Generate 50 random flows
3. Run TSN load-aware scheduling
4. Run integrated TSN-WiFi scheduling
5. Demonstrate the attention-based DDPG architecture

### Expected Output

```
======================================================================
TSN-WiFi Network Scheduler - Example
Reproduction of: Pay Attention to Network
======================================================================

1. Creating network topology (ladder)...
   Created network with 8 switches and 10 links

2. Initializing WiFi MLO...
   WiFi bands: 3
   - Band 0 (2.4 GHz): Rate=72 Mbps, Reliability=0.85
   - Band 1 (5.0 GHz): Rate=433 Mbps, Reliability=0.92
   - Band 2 (6.0 GHz): Rate=600 Mbps, Reliability=0.96

3. Generating flows...
   Generated 50 flows

4. Running TSN load-aware scheduling...
   TSN Results:
   - Scheduled flows: 50/50 (100.0%)
   - Max link load: 25.32%
   - Avg link load: 15.67%
   - Load variance: 12.45

5. Running integrated TSN-WiFi scheduling (greedy)...
   Integrated Results:
   - Total scheduled: 50/50 (100.0%)
   - TSN scheduled: 50
   - WiFi scheduled: 50
   - Average reliability: 0.943
   - Max TSN load: 25.32%
   
   WiFi Band Utilization:
   - Band 0 (2.4 GHz): 15.23% utilized, reliability=0.850
   - Band 1 (5.0 GHz): 12.87% utilized, reliability=0.920
   - Band 2 (6.0 GHz): 71.90% utilized, reliability=0.960

6. Initializing Attention-based DDPG (no training)...
   DDPG initialized with:
   - State dimension: 12
   - Number of flows: 50
   - Hidden dimension: 64
   - Actor network: Self-Attention + MLP + Residual connection
   - Critic network: Self-Attention + MLP + Q-value estimation
```

## Architecture

### TSN Network (`tsn_network.py`)

- **TSNSwitch**: Time-Aware Shaper with Gate Control Lists (GCL)
- **TSNLink**: Network links with bandwidth scheduling
- **Flow**: Traffic flow with timing and routing information
- **TSNNetwork**: Complete network topology management

### WiFi MLO (`wifi_network.py`)

- **WiFiBand**: Individual frequency band (2.4/5/6 GHz) with dynamic reliability
- **WiFiMLO**: Multi-Link Operation access point with U-MAC and L-MAC layers

### Load-Aware Scheduler (`load_aware_scheduler.py`)

Implements the TSN scheduling algorithm:
- **Bucket Effect**: Balances load across network links
- **Alternative Paths**: Finds multiple routing options
- **Load-Aware Selection**: Chooses paths that minimize maximum link utilization

### Attention-based DDPG (`attention_ddpg.py`)

Deep Reinforcement Learning for WiFi scheduling:
- **SelfAttentionLayer**: Captures dependencies among flows
- **AttentionActor**: Policy network with self-attention for band selection
- **AttentionCritic**: Value network for Q-value estimation
- **ReplayBuffer**: Experience replay for stable training

### Integrated Scheduler (`integrated_scheduler.py`)

Combines TSN and WiFi scheduling:
- Enforces cross-domain constraints
- Manages FIFO ordering at WiFi AP
- Optimizes overall system reliability

## Implementation Details

### State Representation

For each flow in WiFi scheduling:
- Arrival time at AP (normalized)
- Packet size (normalized)
- TSN delay metric
- Channel state: TSNR, bandwidth, reliability for each band (9 features)

### Action Space

- Discrete: Select one of 3 bands for each flow
- Actor outputs probability distribution over bands
- Action adaptation: Earliest available time on selected band

### Reward Function

- Weighted combination of:
  - Number of successfully scheduled flows
  - Average reliability of scheduled flows
  - Penalty for deadline violations

### Network Topologies

Three topologies from the paper:
1. **Simple**: Linear chain topology
2. **Ladder**: 2×4 grid topology (from paper experiments)
3. **AFDX**: Star topology with redundant paths (avionics networks)

## Key Algorithms

### 1. Load-Aware TSN Scheduling

```
For each flow:
  1. Find alternative paths between src and dst
  2. Calculate load for each path
  3. Select path with minimum maximum link utilization (bucket effect)
  4. Schedule flow on selected path with earliest available time
  5. Update link loads
```

### 2. Attention-based WiFi Scheduling

```
For each scheduling cycle:
  1. Extract state features (arrival times, channel quality)
  2. Pass through attention-based actor network
  3. Get band selection probabilities for all flows
  4. Sample band for each flow
  5. Schedule at earliest available time on selected band
  6. Calculate reward and update networks
```

### 3. Cross-Domain Constraint

```
WiFi start time >= TSN end time + propagation delay

Ensures flows complete TSN transmission before starting WiFi transmission
```

## Performance Metrics

The implementation tracks:
- **Scheduling Rate**: Percentage of successfully scheduled flows
- **Reliability**: Average reliability across all flows
- **Load Variance**: Measure of load balancing quality (lower is better)
- **Max Link Load**: Maximum utilization across all links
- **Band Utilization**: Per-band resource usage

## Paper References

1. **Main Paper**: Guo, M., Yang, Y., He, S., Pan, J., Gu, C., & Chen, J. (2025). Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks. *ICCPS 2025*.

2. **Related Papers** (also in repository):
   - CaaS: Enabling Control-as-a-Service for Time-Sensitive Networking
   - Integrated Routing and Scheduling for Time-Sensitive Transmission

## Differences from Paper

This implementation focuses on the core algorithms and provides:
- ✅ Complete TSN network model with TAS
- ✅ WiFi MLO with 3 bands
- ✅ Load-aware scheduling with bucket effect
- ✅ Attention-based DDPG architecture
- ✅ Cross-domain constraint enforcement
- ⚠️ Simplified channel model (paper uses more complex wireless simulation)
- ⚠️ No full DRL training loop (architecture is complete, training requires extensive computation)
- ⚠️ Simplified FIFO constraint (paper has more detailed queue management)

## Future Extensions

To fully replicate paper results:
1. Implement complete DRL training loop with 1000+ episodes
2. Add realistic wireless channel simulation (Rayleigh fading, interference)
3. Implement additional baseline algorithms (MLP-based DRL, flow-wise scheduling)
4. Add visualization tools for network topology and scheduling Gantt charts
5. Benchmark on all three topologies (SAE, AFDX, Ladder) with varying flow volumes

## Contributing

This is a reproduction for educational purposes. Contributions welcome:
- Bug fixes
- Performance improvements
- Additional features from the paper
- Visualization tools
- Training scripts

## License

MIT License - See LICENSE file

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{guo2025attention,
  title={Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks},
  author={Guo, Miao and Yang, Yichuan and He, Shibo and Pan, Jianping and Gu, Chaojie and Chen, Jiming},
  booktitle={ICCPS},
  year={2025}
}
```

## Contact

For questions about the implementation, please open an issue in the repository.