# MAIRS: Multi-Agent Integrated Routing and Scheduling for TSN

Implementation of the algorithm from the paper:

**"How Can the Integrated Routing and Scheduling Enhance Optimality Bounds of Time-Sensitive Transmission"**  
*Zhenrui Cao, Tie Qiu, Qingyong Deng, Haolin Liu, and Xiaobo Zhou (2025)*

## Overview

This repository contains a Python implementation of MAIRS (Multi-Agent Deep Reinforcement Learning-based Integrated Routing and Scheduling), a novel approach for optimizing Time-Sensitive Networking (TSN) in Industrial Internet of Things (IIoT) applications.

### Key Features

- **Integrated Routing and Scheduling**: Synchronous decision-making at slot granularity to eliminate conflicts
- **Multi-Agent Deep Reinforcement Learning (MADRL)**: Collaborative agents using MAPPO (Multi-Agent PPO)
- **Time-Aware Shaper (TAS)**: IEEE 802.1Qbv compliant traffic shaping
- **Action Masking**: Prevents invalid actions (routing loops, slot conflicts)
- **Optimality Bounds Enhancement**: Maximizes transmission success rate of TT flows

## Paper Summary

### Problem Statement

Traditional TSN methods separate routing and scheduling decisions, leading to:
- Bandwidth allocation conflicts
- Suboptimal resource utilization  
- Failed deterministic transmission guarantees

### MAIRS Solution

MAIRS integrates routing and scheduling through:

1. **Synchronized Decision-Making**: Both routing and scheduling at slot-level granularity
2. **Collaborative MADRL**: Multiple agents (one per switch) working together
3. **Reward Function**: `R = transmission_success - α × max_slot_utilization`
4. **MAPPO Training**: Actor-critic architecture with PPO clipping for stable updates

### Key Results (from paper)

- Significantly outperforms existing algorithms (DeepSch, JTRS)
- Higher transmission success rates across varying flow counts
- Better adaptation to different slot durations (0.15ms, 0.2ms, 0.3ms)
- Effective load balancing across network links

## Repository Structure

```
.
├── src/
│   ├── environment/          # TSN network simulation
│   │   ├── tsn_network.py   # Network topology and management
│   │   ├── switch.py        # TSN switch with TAS support
│   │   └── tt_flow.py       # Time-Triggered flow representation
│   │
│   └── mairs/               # MAIRS algorithm implementation
│       ├── mairs_agent.py   # Individual switch agent
│       └── mappo.py         # Multi-Agent PPO trainer
│
├── examples/
│   └── mairs_demo.py        # Demonstration script
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yuccccchan/Paper.git
cd Paper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the demonstration example:

```bash
python examples/mairs_demo.py
```

This will:
1. Create an Orion CEV network topology (benchmark from paper)
2. Generate random TT flows with varying parameters
3. Train MAIRS agents using MAPPO
4. Display training progress and results

### Custom Network

```python
from src.environment import TSNNetwork, TTFlow
from src.mairs import MAIRSAgent, MAPPO

# Create network
network = TSNNetwork(slot_duration=0.2, link_bandwidth=100)

# Add switches
for i in range(5):
    network.add_switch(i, num_ports=8)

# Add links
network.add_link(0, 1)
network.add_link(1, 2)
# ... add more links

# Create TT flows
flow = TTFlow(
    flow_id=0,
    source=0,
    destination=4,
    data_size=1000,  # bytes
    period=4.8,      # ms
    deadline=4.8
)
network.add_flow(flow)

# Create agents for each switch
agents = {}
for switch_id in network.switches:
    agent = MAIRSAgent(
        agent_id=switch_id,
        num_neighbors=len(network.switches[switch_id].get_neighbors()),
        max_slots=network.max_slots,
        state_dim=20,  # Adjust based on network
        hidden_dim=128
    )
    agents[switch_id] = agent

# Train with MAPPO
trainer = MAPPO(agents, alpha=0.25)
history = trainer.train(network, [flow], num_iterations=1000)
```

## Algorithm Details

### State Space

Each agent observes local state including:
- Flow features (data size, period, deadline)
- Position features (is source, is destination)
- Neighbor features (distance to destination, link utilization, visited flag)

### Action Space

For each hop, agent decides:
- **Routing**: Which neighboring switch to forward to
- **Scheduling**: Which time slot to use for transmission

Total actions = `num_neighbors × max_slots`

### Reward Function

```
R = I_success - α × U_max
```

Where:
- `I_success`: Binary indicator (1 if flow reaches destination, 0 otherwise)
- `U_max`: Maximum slot utilization across route links
- `α`: Scaling factor (default: 0.25)

This design:
- Encourages successful transmission
- Promotes load balancing
- Prevents bandwidth bottlenecks

### Training Parameters (from paper)

- Iterations: 8000
- Learning rate: 0.003
- Batch size: 32
- PPO clip ε: 0.2
- Scaling factor α: 0.25

## Network Configuration (from paper)

- **Topology**: Orion CEV network (star, tree, ring structures)
- **Link bandwidth**: 100 Mb/s
- **Data sizes**: 60-1500 bytes
- **Periods**: {3.6, 4.8, 6.0} ms
- **Slot durations**: {0.15, 0.2, 0.3} ms

## Key Components

### TSNNetwork
- Manages network topology using NetworkX
- Tracks switch connectivity and link utilization
- Calculates hyperperiod (LCM of flow periods)
- Provides path finding utilities

### TSNSwitch  
- Implements Time-Aware Shaper (TAS)
- Manages egress ports with priority queues
- Maintains Gate Control Lists (GCL)
- Tracks slot allocations per port

### TTFlow
- Represents Time-Triggered flows
- Stores routing and scheduling decisions
- Calculates transmission time and slot requirements

### MAIRSAgent
- Neural network-based agent (actor-critic)
- Generates local observations
- Selects routing and scheduling actions
- Creates action masks for invalid actions

### MAPPO
- Coordinates multi-agent training
- Implements PPO with clipping
- Computes rewards and advantages
- Updates agent policies collaboratively

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{cao2025mairs,
  title={How Can the Integrated Routing and Scheduling Enhance Optimality Bounds of Time-Sensitive Transmission},
  author={Cao, Zhenrui and Qiu, Tie and Deng, Qingyong and Liu, Haolin and Zhou, Xiaobo},
  journal={IEEE Communications Magazine},
  year={2025},
  doi={10.1109/MCOM.002.2400635}
}
```

## References

The implementation is based on:

1. **Cao et al. (2025)** - Main MAIRS algorithm
2. **IEEE 802.1Qbv** - Time-Aware Shaper (TAS) standard
3. **IEEE 802.1Qcc** - Centralized network configuration for TSN
4. **Yu et al. (2022)** - MAPPO algorithm for cooperative multi-agent games

## License

This implementation is provided for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions about the implementation, please open an issue on GitHub.