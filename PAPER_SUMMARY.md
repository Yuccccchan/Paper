# Paper Summary

## Title
**Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks**

## Authors
- Miao Guo (Zhejiang University)
- Yichuan Yang (Zhejiang University)
- Shibo He (Zhejiang University)
- Jianping Pan (University of Victoria)
- Chaojie Gu (Zhejiang University)
- Jiming Chen (Zhejiang University)

## Publication
ICCPS 2025 (ACM/IEEE International Conference on Cyber-Physical Systems)

## Problem Context

### Background
- **TSN (Time-Sensitive Networking)**: Provides deterministic communication in wired networks
- **WiFi MLO (Multi-Link Operation)**: IEEE 802.11be feature enabling simultaneous transmission across multiple frequency bands
- **Industrial Automation**: Smart factories require reliable, real-time communication for control systems
- **Hybrid Networks**: Integration of TSN (wired) and WiFi (wireless) domains

### Challenge
Existing approaches for TSN-WiFi scheduling have limitations:
1. **Static assumptions**: Don't adapt to dynamic wireless channel conditions
2. **MLP-based DRL**: Treats flows independently, causing severe underfitting
3. **Flow-wise scheduling**: Inefficient, requires separate model per flow
4. **Lack of cross-domain coordination**: TSN and WiFi scheduled separately

## Key Innovation

### Main Insight
The mutual dependencies among flows competing for network resources are **analogous to contextual dependencies of tokens in language sequences**. This insight leads to using self-attention mechanisms from NLP for network scheduling.

### Novel Contributions

1. **First Cross-Domain Solution**: Comprehensive spatial-temporal-frequential resource allocation for TSN-WiFi networks
   - Spatial: Routing in TSN
   - Temporal: Timing in TSN
   - Frequential: Band selection in WiFi

2. **Global Scheduling Abstraction**: Mathematical framework describing all constraints:
   - TSN constraints: Port, queue, bandwidth, GCL
   - WiFi constraints: Band, occupancy
   - Cross-domain constraints: Temporal handoff, FIFO ordering

3. **Load-Aware TSN Algorithm**: Implements "bucket effect" principle
   - Balances load across network links
   - Selects paths that minimize maximum link utilization
   - Enables more flows to be scheduled

4. **Attention-Based DDPG**: Deep Reinforcement Learning with self-attention
   - Captures flow dependencies
   - Adapts to dynamic channel conditions
   - Outperforms MLP-based approaches

## Technical Approach

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Central Network Configuration (CNC)      │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │        Load-Aware TSN Scheduling                │   │
│  │  • Alternative path finding                     │   │
│  │  • Load-based path selection (bucket effect)    │   │
│  │  • GCL generation                               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           ↓
                   Flow arrival at AP
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    WiFi Access Point (AP)                │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │    Attention-Based DDPG WiFi Scheduling         │   │
│  │  • State: [arrival_time, channel_quality]       │   │
│  │  • Self-attention: Learn flow dependencies      │   │
│  │  • Action: Band selection (2.4/5/6 GHz)         │   │
│  │  • Reward: Reliability + scheduling success     │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### TSN Domain: Load-Aware Scheduling

**Objective**: 
```
Minimize max_link∈Links(utilization(link))
```

**Bucket Effect Principle**:
- Network capacity limited by most loaded link
- Like water in connected buckets - level determined by lowest bucket
- Balance load to increase overall capacity

**Algorithm**:
1. Compute current load for all links
2. Find alternative paths for each flow
3. Select path with lowest maximum link load
4. Schedule at earliest available time
5. Update link loads

### WiFi Domain: Attention-Based DDPG

**Network Architecture**:
```
Input (State) → MLP (Feature Processing)
                    ↓
              Self-Attention Layer
                    ↓
              Residual Connection
                    ↓
              Output (Band Probabilities)
```

**Self-Attention Mechanism**:
```python
Q = W_q × X  # Query: What does this flow need?
K = W_k × X  # Key: What are other flows requesting?
V = W_v × X  # Value: Flow characteristics

Attention_weights = softmax(Q × K^T / √d_k)
Output = Attention_weights × V
```

**State Features** (per flow):
- Arrival time at AP
- Packet size
- TSN domain delay
- Channel state: [TSNR, bandwidth, reliability] for each band

**Action Space**:
- Discrete: Select one of 3 bands
- Action adaptation: Compute earliest available time on selected band

**Reward Function**:
```
R = Σ(reliability × scheduled) / total_flows
```

### Cross-Domain Constraints

1. **Temporal Constraint**:
   ```
   WiFi_start ≥ TSN_end + propagation_delay
   ```

2. **FIFO Constraint**:
   ```
   arrival(f1) < arrival(f2) ∧ band(f1) = band(f2)
   → start(f1) < start(f2)
   ```

3. **Cycle Constraint**:
   ```
   start + transmission_time ≤ cycle_time
   ```

## Experimental Setup

### Network Topologies
1. **SAE**: Simple Automotive Ethernet topology
2. **AFDX**: Avionics Full-Duplex Switched Ethernet
3. **Ladder**: 2×n grid topology

### Parameters
- Cycle time: 1000 μs
- Time slot: 12 μs (MTU transmission time at 1 Gbps)
- Packet size: 1500 bytes (MTU)
- Flow sets: 50 and 100 flows
- WiFi bands: 2.4 GHz (72 Mbps), 5 GHz (433 Mbps), 6 GHz (600 Mbps)

### Baselines
1. **Shortest Path (SP)**: Standard routing without load awareness
2. **MLP-based DDPG**: DRL without attention mechanism
3. **Flow-wise scheduling**: Separate model per flow

### Metrics
- Scheduling success rate
- Average reliability
- Maximum TSN delay
- Link load variance
- Convergence speed

## Key Results

### TSN Scheduling (Ladder Topology, 50 flows)

| Algorithm | Max Delay | Load Variance | Success Rate |
|-----------|-----------|---------------|--------------|
| Shortest Path | ~65 μs | ~12.0 | ~85% |
| Load-Aware | ~40 μs | ~5.0 | ~95% |

**Improvement**: 38% reduction in delay, 58% reduction in variance

### WiFi Scheduling

| Algorithm | Reliability | Success Rate | Convergence |
|-----------|-------------|--------------|-------------|
| Flow-wise | ~85% | ~85% | Slow |
| MLP DDPG | ~88% | ~92% | Medium |
| Attention DDPG | ~95% | ~98% | Fast |

**Improvement**: 11% higher reliability, 15% better success rate

### Scalability

With 100 flows:
- Attention-based: 94% success rate, ~93% reliability
- MLP-based: 87% success rate, ~86% reliability
- Flow-wise: 80% success rate, ~82% reliability

## Insights and Analysis

### Why Self-Attention Works

1. **Flow Dependencies**: Captures competition for shared resources
   - Flows needing same band at similar times have high attention scores
   - Network learns to coordinate conflicting flows

2. **Variable Input Size**: Handles different number of flows naturally
   - No fixed architecture per flow count
   - Scales better than MLP

3. **Long-Range Dependencies**: Relates flows across entire network
   - Not limited by local receptive field
   - Global optimization perspective

### Bucket Effect Impact

Example with 3 links:
- Shortest Path: Loads [10%, 80%, 15%] → bottleneck at 80%
- Load-Aware: Loads [35%, 40%, 35%] → balanced at 40%

Result: Can schedule more flows without exceeding capacity

### Band Selection Patterns

- **Low congestion**: Prefer 6 GHz (highest reliability)
- **Medium congestion**: Mix of 5 GHz and 6 GHz
- **High congestion**: All bands utilized, DRL learns optimal distribution

### Cross-Domain Coordination

TSN delay impacts WiFi:
- Long TSN delay → late arrival at AP → fewer band options
- Load-balanced TSN → earlier arrivals → more WiFi flexibility
- Integrated approach better than separate optimization

## Limitations and Future Work

### Current Limitations
1. Simplified channel model (static reliability)
2. Single-hop WiFi (AP to station)
3. No mobility considerations
4. Fixed cycle time

### Future Directions
1. **Dynamic channel modeling**: Rayleigh fading, interference
2. **Multi-hop WiFi**: Mesh networks
3. **Mobility support**: Handover mechanisms
4. **Adaptive cycle time**: Based on traffic load
5. **Real-world deployment**: Hardware implementation and testing

## Implementation in This Repository

### What's Included
✅ Complete TSN network model with TAS  
✅ WiFi MLO with 3 bands  
✅ Load-aware scheduling algorithm  
✅ Attention-based DDPG architecture  
✅ Cross-domain constraint enforcement  
✅ Working examples and documentation  

### What's Simplified
⚠️ Static channel reliability (vs. dynamic simulation)  
⚠️ No full DRL training loop (architecture complete)  
⚠️ Simplified queue management  
⚠️ Basic visualization (vs. paper's extensive plots)  

### How to Extend
1. Implement realistic wireless channel simulator
2. Add full training loop with convergence monitoring
3. Implement baseline algorithms (MLP-based, flow-wise)
4. Add visualization tools for Gantt charts
5. Benchmark on all topologies with varying loads

## Related Work References

### TSN Standards
- IEEE 802.1Qbv: Time-Aware Shaper (TAS)
- IEEE 802.1Qbu: Frame Preemption
- IEEE 802.1AS: Timing and Synchronization

### WiFi Standards
- IEEE 802.11be: WiFi 7 with Multi-Link Operation
- IEEE 802.11ax: WiFi 6
- IEEE 802.11e: QoS enhancements (EDCA)

### Machine Learning
- Vaswani et al.: "Attention is All You Need" (Transformers)
- Lillicrap et al.: "Continuous Control with Deep RL" (DDPG)
- He et al.: "Deep Residual Learning" (ResNet)

## Conclusion

This paper makes significant contributions to TSN-WiFi scheduling:

1. **Novel approach**: First to use self-attention for network scheduling
2. **Comprehensive solution**: Addresses spatial, temporal, and frequential dimensions
3. **Strong results**: Outperforms baselines by 10-15% across metrics
4. **Practical relevance**: Applicable to smart factories, industrial automation

**Key Takeaway**: Borrowing concepts from NLP (self-attention) can significantly improve network scheduling by capturing flow dependencies that traditional approaches miss.

## Citation

```bibtex
@inproceedings{guo2025attention,
  title={Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential 
         Scheduling for TSN-WiFi Networks},
  author={Guo, Miao and Yang, Yichuan and He, Shibo and Pan, Jianping and 
          Gu, Chaojie and Chen, Jiming},
  booktitle={Proceedings of the ACM/IEEE 16th International Conference on 
             Cyber-Physical Systems (ICCPS)},
  year={2025},
  pages={TBD},
  doi={3716550.3722018}
}
```
