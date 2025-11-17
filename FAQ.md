# Frequently Asked Questions (FAQ)

Common questions and answers about implementing the TSN-WiFi scheduling system.

## General Questions

### Q1: What is the main contribution of this paper?
**A**: The paper introduces an attention-based Deep Reinforcement Learning approach for cross-domain scheduling in TSN-WiFi networks. The key innovation is using self-attention (from NLP) to learn dependencies among network flows, achieving 12% better reliability than previous methods.

### Q2: Why is this problem important?
**A**: Modern industrial applications (smart factories, AGVs, remote offices) need time-critical, reliable communication that spans both wired (TSN) and wireless (WiFi) domains. This system enables seamless integration of both domains with deterministic guarantees.

### Q3: What are the main challenges?
**A**: 
1. **Resource dimension mismatch**: TSN uses space-time, WiFi uses frequency-airtime
2. **Continuous action space**: Airtime allocation is continuous
3. **Dynamic wireless channels**: WiFi quality varies unpredictably
4. **Scalability**: Must handle variable number of flows
5. **Inter-flow dependencies**: Scheduling one flow affects others

### Q4: How long will implementation take?
**A**: Following the roadmap:
- **Minimal prototype**: 6-8 weeks
- **Full implementation**: 12-16 weeks
- **Research-quality results**: 16-20 weeks

Time varies based on your experience and resources.

---

## Technical Questions

### Q5: Why use attention mechanism instead of traditional MLP?
**A**: Attention learns relationships between flows:
```
Flow 1 uses Link A → affects → Flow 2 that needs Link A
```
Traditional MLP processes each flow independently and misses these dependencies. Attention achieves ~18% better performance than MLP-based DRL.

### Q6: Why DDPG instead of other DRL algorithms (PPO, SAC)?
**A**: DDPG is chosen because:
- **Continuous actions**: Airtime is continuous
- **Deterministic policy**: Faster convergence
- **Actor-critic**: Efficient training
- **Off-policy**: Better sample efficiency

Alternative: You could try TD3 or SAC for more stability.

### Q7: What is the difference between flow-wise and network-wise scheduling?

**Flow-wise scheduling**:
```python
for flow in flows:
    action = agent.select_action(flow)  # One at a time
```
- Slow: N forward passes for N flows
- Doesn't learn flow interactions

**Network-wise scheduling** (this paper):
```python
actions = agent.select_action(all_flows)  # All at once
```
- Fast: 1 forward pass for all flows
- Learns flow dependencies via attention

### Q8: How does load-aware TSN scheduling work?
**A**: Instead of shortest path routing, it:
1. Calculates load on each link
2. Selects path that minimizes maximum link load
3. Avoids creating bottlenecks

Example:
```
Path 1: [Link A (load=5), Link B (load=2)] → max_load = 5
Path 2: [Link C (load=3), Link D (load=3)] → max_load = 3 ✓ Choose this
```

### Q9: What are the key hyperparameters to tune?
**A**: Priority order:
1. **Learning rates** (α_actor=1e-4, α_critic=1e-3)
2. **Batch size** (64)
3. **Exploration noise** (0.1 → 0.01)
4. **Network architecture** (embed_dim=128, num_heads=8)
5. **Reward weights** (reliability=100, violation=-50)

Start with defaults, then tune learning rates first.

---

## Implementation Questions

### Q10: What dependencies do I need?
**A**: Core dependencies:
```bash
pip install torch numpy networkx gym matplotlib pandas scipy
```

Optional:
```bash
pip install tensorboard wandb  # Logging
pip install pytest  # Testing
pip install black flake8  # Code quality
```

Versions:
- Python 3.8+
- PyTorch 1.10+
- CUDA 11+ (for GPU)

### Q11: Can I run this without GPU?
**A**: Yes, but slower. For small experiments (10-20 flows):
- **CPU**: ~5-10 minutes per episode
- **GPU**: ~30 seconds per episode

For large experiments (100 flows), GPU highly recommended.

### Q12: How much memory is required?
**A**: Approximate memory usage:

| Component | Memory |
|-----------|--------|
| Model weights | ~10 MB |
| Replay buffer | ~1-5 GB |
| Training batch | ~100 MB |
| **Total** | **~2-6 GB** |

Reduce replay buffer size if memory limited.

### Q13: How do I validate my implementation?
**A**: Progressive validation:

1. **Unit tests**: Each component works
```python
def test_tsn_scheduler():
    scheduler = LoadAwareTSNScheduler(topology)
    flows = create_test_flows()
    scheduled = scheduler.schedule_flows(flows)
    assert all(f.path is not None for f in scheduled)
```

2. **Integration tests**: Components work together
```python
def test_environment():
    env = TSNWiFiEnv(config, num_flows=10)
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    assert 'avg_reliability' in info
```

3. **End-to-end test**: Training converges
```python
# Should see increasing reward
rewards = train_agent(env, agent, num_episodes=100)
assert rewards[-10:].mean() > rewards[:10].mean()
```

### Q14: My training doesn't converge. What should I do?
**A**: Debug checklist:

1. **Check state normalization**:
```python
state = env.reset()
print(f"State min: {state.min()}, max: {state.max()}")
# Should be roughly in [0, 1]
```

2. **Check reward scale**:
```python
# Rewards should be in reasonable range (-1000 to 1000)
print(f"Reward: {reward}")
```

3. **Monitor gradients**:
```python
for name, param in agent.actor.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: {grad_norm}")
        # Should be ~0.01-1.0, not 0 or >>1
```

4. **Reduce complexity**:
- Start with 5 flows
- Use simpler topology
- Increase learning rate

5. **Add logging**:
```python
import wandb
wandb.init(project="tsn-wifi")
wandb.log({"reward": reward, "reliability": reliability})
```

### Q15: How do I reproduce paper results?
**A**: Key requirements:

1. **Same hyperparameters**:
```python
lr_actor = 1e-4
lr_critic = 1e-3
gamma = 0.99
tau = 0.001
batch_size = 64
```

2. **Same network topology**: Use industrial network from paper
3. **Same flow distribution**: 50 flows, random start offsets
4. **Same channel model**: SNR distribution per band
5. **Same evaluation**: 100 test episodes

Expected results:
- Reliability: ~78% (±2%)
- Training time: 500-700 episodes
- Improvement over MLP: ~18%

### Q16: Can I modify the architecture?
**A**: Yes! Try these variations:

**Attention heads**:
```python
# Original: 8 heads
actor = AttentionActorNetwork(..., num_heads=4)  # Faster
actor = AttentionActorNetwork(..., num_heads=16)  # More expressive
```

**Network depth**:
```python
# Original: 3 layers
actor = AttentionActorNetwork(..., num_layers=2)  # Simpler
actor = AttentionActorNetwork(..., num_layers=4)  # Deeper
```

**Embedding dimension**:
```python
# Original: 128
actor = AttentionActorNetwork(..., embed_dim=64)   # Smaller
actor = AttentionActorNetwork(..., embed_dim=256)  # Larger
```

Start with defaults, then experiment.

---

## Domain-Specific Questions

### Q17: What is TSN and why do we need it?
**A**: Time-Sensitive Networking (TSN) is IEEE 802.1 standards for deterministic Ethernet:
- **Deterministic**: Guaranteed latency bounds
- **Synchronized**: Nanosecond-level time sync
- **Scheduled**: Gate Control Lists (GCL) for precise timing
- **Reliable**: No packet loss for critical traffic

Used in: industrial automation, automotive, audio/video production.

### Q18: What is WiFi MLO?
**A**: Multi-Link Operation (IEEE 802.11be / WiFi 7):
- Simultaneous transmission on multiple bands (2.4/5/6 GHz)
- Increased throughput and reliability
- Reduced latency through link diversity
- Better resource utilization

Think of it as "bonding" multiple WiFi links.

### Q19: What is a Gate Control List (GCL)?
**A**: TSN switches use GCLs to schedule packets:

```
Time Slot | Queue 0 | Queue 1 | ... | Queue 7
----------|---------|---------|-----|--------
0-100 μs  |   Open  | Closed  | ... | Closed
100-200   |  Closed |  Open   | ... | Closed
200-300   |  Closed | Closed  | ... |  Open
```

Each time slot specifies which queues can transmit. Ensures no conflicts.

### Q20: How does SNR relate to reliability?
**A**: Signal-to-Noise Ratio (SNR) determines bit error rate:

```
High SNR (35 dB):
  BER ≈ 10^-9 → Reliability ≈ 99.9%

Medium SNR (25 dB):
  BER ≈ 10^-6 → Reliability ≈ 99%

Low SNR (15 dB):
  BER ≈ 10^-3 → Reliability ≈ 90%
```

Higher SNR → Lower errors → Higher reliability

---

## Performance Questions

### Q21: What reliability can I expect?
**A**: Depends on configuration:

| Flows | Reliability | Deadline Satisfaction |
|-------|-------------|----------------------|
| 10    | ~85%        | ~95%                 |
| 20    | ~82%        | ~93%                 |
| 50    | ~78%        | ~90%                 |
| 100   | ~72%        | ~85%                 |

More flows = more congestion = lower reliability

### Q22: How fast is inference?
**A**: Timing breakdown:

```
TSN Scheduling: ~10 ms (50 flows)
WiFi DRL Inference: ~1-2 ms
Total: ~12 ms per cycle
```

For 1ms cycle time, would need optimization:
- Use C++ implementation
- Batch multiple cycles
- Precompute TSN schedules

### Q23: Does it scale to 1000+ flows?
**A**: Theoretically yes, but:
- **Memory**: O(N²) for attention (N = flows)
- **Computation**: Linear in N with efficient implementation
- **Training**: Slower convergence

For 1000+ flows, consider:
- Hierarchical scheduling
- Flow aggregation
- Distributed agents

### Q24: How does it compare to optimal solution?
**A**: Optimal scheduling is NP-hard (proven in paper). This DRL approach is heuristic:
- **Small networks** (<20 flows): ~90% of optimal
- **Medium networks** (50 flows): ~80-85% of optimal
- **Large networks** (100+ flows): Unknown (optimal infeasible)

Trade-off: Speed vs Optimality

---

## Practical Questions

### Q25: Can I use this in production?
**A**: Consider these factors:

**Ready for production**:
- Well-tested code
- Stable training
- Validated results
- Clear documentation

**Not ready**:
- Hardware integration needed
- Real-time requirements validation
- Safety certification required
- Extensive field testing needed

Start with pilot deployment.

### Q26: How do I integrate with real TSN hardware?
**A**: Integration steps:

1. **API Development**:
```python
class TSNHardwareInterface:
    def configure_gcl(self, switch_id, gcl_entries):
        # Send GCL to actual TSN switch
        pass
    
    def get_port_status(self, switch_id, port):
        # Query hardware status
        pass
```

2. **Time Synchronization**: Use IEEE 802.1AS (gPTP)
3. **Configuration Protocol**: Use NETCONF/YANG models
4. **Testing**: Start with simulation, then testbed, then production

### Q27: What about network topology changes?
**A**: Current implementation assumes static topology. For dynamic:

**Option 1: Retrain**
```python
# Topology changed
new_topology = discover_topology()
env = TSNWiFiEnv(new_topology)
agent.train(env)  # Quick fine-tuning
```

**Option 2: Transfer Learning**
```python
# Load pre-trained model
agent.load("pretrained_model.pt")
# Fine-tune on new topology
agent.train(env, num_episodes=100)
```

**Option 3: Online Learning**
```python
# Continuously adapt
while True:
    schedule_flows()
    observe_performance()
    update_model()
```

### Q28: How do I handle emergency traffic?
**A**: Add priority mechanism:

```python
class PriorityFlow(Flow):
    def __init__(self, ..., priority):
        super().__init__(...)
        self.priority = priority  # 0=highest
        
# Schedule high-priority first
flows = sorted(flows, key=lambda f: f.priority)
scheduled = scheduler.schedule_flows(flows)
```

Or use preemption in TSN.

### Q29: What monitoring do I need?
**A**: Key metrics:

1. **Real-time**:
   - Flow reliability
   - End-to-end latency
   - Queue occupancy
   - Channel quality

2. **Aggregate**:
   - Average reliability
   - Deadline miss rate
   - Resource utilization
   - Scheduling time

3. **Alerts**:
   - Reliability < threshold
   - Repeated failures
   - Hardware issues

### Q30: How do I debug in production?
**A**: Debugging strategy:

1. **Logging**:
```python
import logging
logging.info(f"Scheduled flow {flow.flow_id}: "
            f"reliability={flow.reliability}, "
            f"path={flow.path}")
```

2. **Metrics Export**:
```python
from prometheus_client import Counter, Histogram

reliability_metric = Histogram('flow_reliability', 
                               'Flow reliability distribution')
reliability_metric.observe(flow.reliability)
```

3. **Visualization**:
```python
# Live Gantt chart
visualize_schedule_live(scheduled_flows)

# Dashboard
streamlit.run_dashboard(env, agent)
```

4. **Fallback**:
```python
try:
    scheduled = drl_agent.schedule(flows)
except Exception as e:
    logging.error(f"DRL failed: {e}")
    scheduled = fallback_scheduler.schedule(flows)
```

---

## Research Questions

### Q31: Can I use this for my research?
**A**: Yes! Potential research directions:

1. **Algorithm improvements**:
   - Try other attention mechanisms
   - Compare with transformer
   - Multi-agent coordination

2. **Application domains**:
   - Automotive TSN
   - Audio/video TSN
   - 5G integration

3. **Theoretical analysis**:
   - Convergence guarantees
   - Sample complexity
   - Approximation bounds

### Q32: How do I cite this work?
**A**: 
```bibtex
@inproceedings{guo2025attention,
  title={Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks},
  author={Guo, Miao and Yang, Yichuan and He, Shibo and Pan, Jianping and Gu, Chaojie and Chen, Jiming},
  booktitle={ACM/IEEE 16th International Conference on Cyber-Physical Systems (ICCPS)},
  year={2025},
  doi={10.1145/3716550.3722018}
}
```

### Q33: What are good follow-up research topics?
**A**: 

1. **Multi-objective optimization**: Balance reliability, latency, energy
2. **Distributed scheduling**: Multiple coordinators
3. **Learning from demonstrations**: Expert guidance
4. **Predictive scheduling**: Use traffic prediction
5. **Federated learning**: Privacy-preserving training
6. **Explainable AI**: Interpret scheduling decisions

---

## Miscellaneous

### Q34: Where can I get help?
**A**: Resources:
1. **Documentation**: Check IMPLEMENTATION_GUIDE.md
2. **GitHub Issues**: Open issue with details
3. **Community**: Join discussions
4. **Paper Authors**: Contact via email (see paper)

### Q35: How can I contribute?
**A**: Contributions welcome:
1. **Bug fixes**: Submit pull requests
2. **New features**: Discuss first, then implement
3. **Documentation**: Improve clarity
4. **Examples**: Add use cases
5. **Testing**: Expand test coverage

### Q36: Is there a paper code repository?
**A**: The original authors haven't released official code. This implementation guide is based on the paper's detailed descriptions and algorithms.

### Q37: What license should I use?
**A**: Suggested licenses:
- **MIT**: Permissive, allows commercial use
- **Apache 2.0**: Permissive with patent protection
- **GPL v3**: Copyleft, must share modifications

Check your institution's policy.

### Q38: How do I stay updated?
**A**: 
1. **Watch this repository**: Get notifications
2. **Follow paper authors**: Check their publications
3. **TSN standards**: Monitor IEEE 802.1 updates
4. **WiFi 7**: Follow IEEE 802.11be development

---

## Additional Resources

### Papers
- Original paper: `3716550.3722018.pdf`
- DDPG paper: Lillicrap et al., "Continuous control with deep reinforcement learning" (2015)
- Attention paper: Vaswani et al., "Attention is All You Need" (2017)

### Standards
- IEEE 802.1AS: Time synchronization
- IEEE 802.1Qbv: Time-Aware Shaper
- IEEE 802.11be: WiFi 7 / EHT

### Code
- PyTorch: https://pytorch.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- OpenAI Gym: https://www.gymlibrary.dev/

### Communities
- TSN/IP: https://www.tsn-ip.org/
- IEEE 802.1 Working Group
- WiFi Alliance

---

**Last Updated**: Based on ICCPS 2025 paper

**Questions not answered here?** Open an issue on GitHub!

