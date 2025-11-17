# Implementation Roadmap

This document provides a structured roadmap for implementing the TSN-WiFi scheduling system from the paper "Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks" (ICCPS 2025).

## 🎯 Project Goals

Implement a complete cross-domain scheduling system that:
- ✅ Schedules flows in TSN domain with load-aware routing
- ✅ Schedules flows in WiFi domain with adaptive DRL
- ✅ Uses attention mechanism to learn flow dependencies
- ✅ Achieves ~78% reliability for 50 flows
- ✅ Outperforms baseline methods by 12-18%

## 📚 Documentation Overview

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **README.md** | Overview and quick reference | Start here for understanding |
| **IMPLEMENTATION_GUIDE.md** | Complete implementation details | Reference during coding |
| **ALGORITHMS.md** | Algorithm pseudocode | Understand the logic |
| **QUICK_START.md** | Hands-on examples | Test concepts quickly |
| **CONFIGURATION.md** | Setup and config templates | Configure your deployment |
| **ROADMAP.md** (this file) | Implementation plan | Track your progress |

## 🗺️ Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Set up environment and implement basic components

#### Week 1: Environment Setup
- [ ] Install dependencies (PyTorch, NumPy, NetworkX)
- [ ] Set up project structure following the template
- [ ] Create virtual environment
- [ ] Test basic imports

**Deliverables**:
- Working Python environment
- Project folder structure
- Basic imports verified

**Validation**:
```bash
python -c "import torch; import numpy; import networkx; print('All imports successful')"
```

#### Week 2: TSN Domain Implementation
- [ ] Implement `TSNTopology` class
- [ ] Implement `Flow` class
- [ ] Implement `LoadAwareTSNScheduler`
- [ ] Test with simple 3-node topology

**Deliverables**:
- `tsn_domain/topology.py`
- `tsn_domain/flow.py`
- `tsn_domain/load_aware_scheduler.py`
- Unit tests for TSN components

**Validation**:
```python
# Test script
topology = TSNTopology(simple_config)
flows = [Flow(0, 'es0', 'es1', 1500, 0, 500)]
scheduler = LoadAwareTSNScheduler(topology)
scheduled = scheduler.schedule_flows(flows)
assert len(scheduled) == 1
assert scheduled[0].path is not None
```

**Estimated Time**: 2 weeks
**Difficulty**: Medium

---

### Phase 2: WiFi Domain (Week 3-4)

#### Week 3: WiFi MLD Implementation
- [ ] Implement `WiFiMLO` class
- [ ] Implement `ChannelState` class
- [ ] Add SNR-based reliability calculation
- [ ] Test channel model with different SNR values

**Deliverables**:
- `wifi_domain/mld.py`
- `wifi_domain/channel.py`
- Channel simulation tests

**Validation**:
```python
wifi = WiFiMLO()
wifi.channel_states['6GHz'].update_snr(35)
reliability = wifi.channel_states['6GHz'].calculate_reliability(1500)
assert 0.95 <= reliability <= 1.0
```

#### Week 4: Airtime Management
- [ ] Implement airtime allocation logic
- [ ] Add band selection heuristics
- [ ] Test with multiple flows
- [ ] Validate constraints

**Deliverables**:
- Airtime manager
- Band selection logic
- Integration tests

**Estimated Time**: 2 weeks
**Difficulty**: Medium

---

### Phase 3: DRL Model (Week 5-7)

#### Week 5: Attention Network
- [ ] Implement `MultiHeadSelfAttention` module
- [ ] Implement `AttentionActorNetwork`
- [ ] Implement `AttentionCriticNetwork`
- [ ] Test forward pass with dummy data

**Deliverables**:
- `drl_agent/attention_network.py`
- Network architecture tests

**Validation**:
```python
actor = AttentionActorNetwork(flow_feature_dim=15, num_bands=3)
dummy_input = torch.randn(1, 10, 15)  # batch=1, flows=10, features=15
band_logits, airtime = actor(dummy_input)
assert band_logits.shape == (1, 10, 3)
assert airtime.shape == (1, 10, 1)
```

#### Week 6: DDPG Agent
- [ ] Implement `AttentionDDPGAgent`
- [ ] Implement `ReplayBuffer`
- [ ] Add action selection with noise
- [ ] Test training step

**Deliverables**:
- `drl_agent/ddpg_agent.py`
- `drl_agent/replay_buffer.py`
- Agent tests

**Validation**:
```python
agent = AttentionDDPGAgent(flow_feature_dim=15, action_dim=4)
state = np.random.rand(10, 15)
band_actions, airtime_actions = agent.select_action(state)
assert len(band_actions) == 10
assert len(airtime_actions) == 10
```

#### Week 7: Integration & Testing
- [ ] Integrate actor and critic
- [ ] Test training loop
- [ ] Verify gradient flow
- [ ] Debug and fix issues

**Estimated Time**: 3 weeks
**Difficulty**: Hard

---

### Phase 4: Environment (Week 8-9)

#### Week 8: State Representation
- [ ] Implement `StateEncoder`
- [ ] Implement `RewardCalculator`
- [ ] Test feature encoding
- [ ] Validate reward function

**Deliverables**:
- `environment/state_representation.py`
- Feature encoding tests

**Validation**:
```python
encoder = StateEncoder()
features = encoder.encode_flow_features(flows, tsn_schedules, channels)
assert features.shape == (len(flows), 15)
assert np.all((features >= 0) & (features <= 1))  # Normalized
```

#### Week 9: Complete Environment
- [ ] Implement `TSNWiFiEnv` (Gym environment)
- [ ] Integrate TSN and WiFi domains
- [ ] Test reset and step functions
- [ ] Validate constraint checking

**Deliverables**:
- `environment/tsn_wifi_env.py`
- Environment tests

**Validation**:
```python
env = TSNWiFiEnv(config, num_flows=10)
state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)
assert 'avg_reliability' in info
```

**Estimated Time**: 2 weeks
**Difficulty**: Medium-Hard

---

### Phase 5: Training Pipeline (Week 10-11)

#### Week 10: Training Loop
- [ ] Implement main training loop
- [ ] Add logging and checkpointing
- [ ] Implement exploration decay
- [ ] Add early stopping

**Deliverables**:
- `train.py`
- Training utilities

**Validation**:
- Run for 100 episodes
- Verify reward increases
- Check model saving

#### Week 11: Visualization & Metrics
- [ ] Implement performance metrics
- [ ] Add training curve plotting
- [ ] Create Gantt chart visualization
- [ ] Add band allocation plots

**Deliverables**:
- `utils/metrics.py`
- `utils/visualization.py`
- Plotting scripts

**Estimated Time**: 2 weeks
**Difficulty**: Medium

---

### Phase 6: Evaluation & Baselines (Week 12-14)

#### Week 12: Baseline Implementations
- [ ] Implement Random baseline
- [ ] Implement SLCI baseline
- [ ] Implement MLP-based DRL
- [ ] Implement Policy Gradient

**Deliverables**:
- `baselines/random.py`
- `baselines/slci.py`
- `baselines/mlp_drl.py`
- `baselines/policy_gradient.py`

#### Week 13-14: Comprehensive Evaluation
- [ ] Run experiments on all scenarios
- [ ] Compare with all baselines
- [ ] Generate result tables
- [ ] Create comparison plots
- [ ] Write evaluation report

**Deliverables**:
- `evaluate.py`
- Result tables and plots
- Evaluation report

**Validation**:
- Achieve ~78% reliability for 50 flows
- Show 12%+ improvement over baselines
- Demonstrate scalability

**Estimated Time**: 3 weeks
**Difficulty**: Medium

---

### Phase 7: Optimization & Deployment (Week 15-16)

#### Week 15: Performance Optimization
- [ ] Profile code for bottlenecks
- [ ] Optimize attention computation
- [ ] Add batch processing
- [ ] Enable GPU acceleration

**Deliverables**:
- Optimized code
- Performance benchmarks

#### Week 16: Documentation & Deployment
- [ ] Complete API documentation
- [ ] Create deployment guide
- [ ] Add hardware integration notes
- [ ] Prepare demo

**Deliverables**:
- Complete documentation
- Deployment package
- Demo materials

**Estimated Time**: 2 weeks
**Difficulty**: Medium

---

## 📊 Progress Tracking

Use this checklist to track overall progress:

### Core Components
- [ ] TSN Domain (Topology, Flow, Scheduler)
- [ ] WiFi Domain (MLD, Channel, Airtime)
- [ ] DRL Agent (Attention Networks, DDPG)
- [ ] Environment (State, Reward, Gym Interface)
- [ ] Training Pipeline
- [ ] Evaluation Framework

### Features
- [ ] Load-aware TSN routing
- [ ] Multi-head self-attention
- [ ] Adaptive WiFi scheduling
- [ ] Experience replay
- [ ] Target network updates
- [ ] Exploration strategy
- [ ] Reward shaping

### Testing & Validation
- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Performance benchmarks
- [ ] Comparison with baselines

### Documentation
- [x] README
- [x] Implementation Guide
- [x] Algorithm Reference
- [x] Quick Start
- [x] Configuration Guide
- [x] Roadmap (this document)
- [ ] API Documentation
- [ ] Deployment Guide

## 🎓 Learning Path

### Prerequisites
1. **Python Programming**: Intermediate level
2. **Deep Learning**: Understand neural networks, backpropagation
3. **Reinforcement Learning**: Know MDP, Q-learning, policy gradients
4. **Networking**: Basic TSN and WiFi concepts

### Recommended Study Materials

**Reinforcement Learning**:
- Paper: "Continuous control with deep reinforcement learning" (DDPG)
- Book: "Reinforcement Learning: An Introduction" by Sutton & Barto

**Attention Mechanism**:
- Paper: "Attention is All You Need" (Vaswani et al., 2017)
- Tutorial: PyTorch Transformer tutorial

**TSN & WiFi**:
- IEEE 802.1 TSN standards
- IEEE 802.11be (WiFi 7) documentation
- Industrial automation use cases

## 🐛 Common Pitfalls & Solutions

### Pitfall 1: Slow Training Convergence
**Symptoms**: Reward doesn't increase after many episodes

**Solutions**:
- Reduce learning rates (try 1e-5 for actor)
- Increase exploration initially
- Check reward normalization
- Verify gradient flow
- Try simpler topology first

### Pitfall 2: Instability in DRL
**Symptoms**: Loss explodes, NaN values

**Solutions**:
- Add gradient clipping
- Use smaller batch sizes
- Normalize state features
- Reduce network size
- Check replay buffer

### Pitfall 3: Poor Generalization
**Symptoms**: Works on training but fails on test

**Solutions**:
- Increase diversity in training
- Add regularization (dropout)
- Use more training episodes
- Validate on multiple scenarios

### Pitfall 4: Memory Issues
**Symptoms**: Out of memory errors

**Solutions**:
- Reduce replay buffer size
- Use smaller batch sizes
- Clear unused variables
- Use gradient checkpointing

## 🔧 Debugging Tips

### For TSN Scheduler
```python
# Add debug prints
print(f"Flow {flow.flow_id}: Path = {flow.path}")
print(f"  GCL entries: {flow.gcl_entries}")
print(f"  Link loads: {link_loads}")

# Visualize schedule
visualize_tsn_schedule(scheduled_flows)
```

### For DRL Agent
```python
# Monitor gradients
for name, param in agent.actor.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")

# Check attention weights
attn_weights = agent.actor.attention_layers[0](state)
print(f"Attention weights shape: {attn_weights.shape}")
print(f"Attention weights: {attn_weights[0, :5, :5]}")  # First 5x5
```

### For Environment
```python
# Validate state space
state = env.reset()
print(f"State shape: {state.shape}")
print(f"State min: {state.min()}, max: {state.max()}")
assert np.all(np.isfinite(state)), "State contains inf or nan"

# Validate reward
for _ in range(10):
    action = env.action_space.sample()
    _, reward, _, info = env.step(action)
    print(f"Reward: {reward}, Info: {info}")
```

## 📈 Success Metrics

### Minimum Viable Product (MVP)
- [ ] Runs end-to-end without errors
- [ ] Achieves >60% reliability
- [ ] Completes training in <24 hours
- [ ] Handles 10-20 flows

### Production Ready
- [ ] Achieves ~78% reliability (50 flows)
- [ ] Outperforms baselines by >10%
- [ ] Handles 100+ flows
- [ ] <1ms inference time
- [ ] Stable training (converges in <500 episodes)

### Research Quality
- [ ] Matches paper results (±2%)
- [ ] Comprehensive evaluation
- [ ] Multiple topologies tested
- [ ] Ablation studies conducted
- [ ] Publication-ready plots

## 🚀 Deployment Checklist

### Before Deployment
- [ ] All tests passing
- [ ] Code reviewed
- [ ] Documentation complete
- [ ] Performance benchmarked
- [ ] Security audit completed

### Integration with Hardware
- [ ] TSN switch compatibility verified
- [ ] WiFi AP configuration tested
- [ ] Time synchronization working
- [ ] End-to-end latency measured
- [ ] Reliability validated

### Production Deployment
- [ ] Monitoring setup
- [ ] Logging configured
- [ ] Backup strategy
- [ ] Rollback plan
- [ ] Incident response plan

## 📞 Getting Help

If you encounter issues:

1. **Check Documentation**: Review relevant docs first
2. **Search Issues**: Look for similar problems
3. **Debug Systematically**: Use debugging tips above
4. **Ask for Help**: Open an issue with:
   - Detailed problem description
   - Code snippet to reproduce
   - Error messages
   - What you've tried

## 🎉 Milestones

Track major achievements:

- [ ] 🎯 Milestone 1: TSN scheduler working (Week 2)
- [ ] 🎯 Milestone 2: WiFi model working (Week 4)
- [ ] 🎯 Milestone 3: DRL agent training (Week 7)
- [ ] 🎯 Milestone 4: Environment integrated (Week 9)
- [ ] 🎯 Milestone 5: First successful training run (Week 11)
- [ ] 🎯 Milestone 6: Baseline comparison complete (Week 14)
- [ ] 🎯 Milestone 7: Production ready (Week 16)
- [ ] 🎉 PROJECT COMPLETE: All goals achieved!

## 📝 Notes

### Key Design Decisions
1. **Attention over MLP**: Better flow dependency learning
2. **DDPG over PPO**: Continuous action space, faster convergence
3. **Load-aware routing**: Prevents bottlenecks in TSN
4. **Multi-band WiFi**: Improves reliability through diversity

### Future Enhancements
- [ ] Add priority-based scheduling
- [ ] Implement multi-agent coordination
- [ ] Support dynamic topology changes
- [ ] Add predictive channel modeling
- [ ] Enable real-time adaptation

### Research Extensions
- [ ] Compare with other attention mechanisms (cross-attention, sparse)
- [ ] Try other DRL algorithms (SAC, TD3)
- [ ] Investigate transfer learning
- [ ] Study multi-objective optimization
- [ ] Explore federated learning

## 🏁 Conclusion

This roadmap provides a structured 16-week plan to fully implement the TSN-WiFi scheduling system. Follow the phases sequentially, validate each component, and track your progress using the checklists.

**Remember**: Implementation is iterative. Don't hesitate to revisit earlier phases if needed. Good luck! 🚀

