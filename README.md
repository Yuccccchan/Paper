# TSN-WiFi Cross-Domain Scheduling with Attention-based Deep Reinforcement Learning

This repository contains research papers and implementation guidance for **"Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks"** (ICCPS 2025).

## 📄 Paper Summary

The main paper (`3716550.3722018.pdf`) presents a novel cross-domain scheduling scheme for TSN-WiFi hybrid networks using attention-based Deep Reinforcement Learning (DRL).

### Key Contributions

1. **Attention-based DRL Model**: Uses self-attention mechanism from NLP to learn dependencies among network flows, analogous to how transformers learn token relationships in sequences.

2. **Cross-Domain Scheduling**: First comprehensive solution for multi-dimensional spatial-temporal-frequential resource allocation in TSN-WiFi networks.

3. **Load-Aware TSN Algorithm**: Minimizes congestion by considering bottleneck resources on links.

4. **DDPG WiFi Scheduler**: Adaptive scheduling that handles dynamic wireless channel conditions.

### Problem Addressed

**Challenge**: Coordinate resource allocation between:
- **TSN Domain**: Time-Sensitive Networking (wired) using space-time resources
- **WiFi Domain**: IEEE 802.11be MLO (wireless) using frequency-airtime resources

**Goal**: Achieve reliable, low-latency flow transmission with deterministic guarantees.

### Technical Innovation

Unlike traditional DRL approaches (Flow-wise or MLP-based), the paper uses **Multi-Head Self-Attention** to:
- Learn network-wide dependencies among flows
- Handle variable number of flows (scalability)
- Achieve better scheduling effectiveness

**Results**: 12.09% reliability improvement over state-of-the-art methods, achieving 78.63% average reliability for 50 flows.

## 🚀 Implementation Guide

See **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** for detailed instructions on how to realize this work, including:

- Complete system architecture
- Step-by-step implementation for all components
- Full Python code for:
  - TSN domain (topology, flows, load-aware scheduler)
  - WiFi domain (MLO, channel modeling)
  - Attention-based DRL networks
  - DDPG training algorithm
  - Environment and evaluation
- Training procedures and hyperparameters
- Expected results and validation approach

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Average Reliability (50 flows) | 78.63% |
| Improvement vs Policy Gradient | +11.23% |
| Improvement vs Flow-wise DRL | +10.53% |
| Improvement vs MLP-based DRL | +18.23% |
| Improvement vs SLCI | +12.09% |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────┐
│           TSN-WiFi Hybrid Network               │
├────────────────────┬────────────────────────────┤
│   TSN Domain       │      WiFi Domain           │
│                    │                            │
│   ┌──────────┐    │    ┌─────────────────┐    │
│   │   CNC    │    │    │      AP         │    │
│   │          │    │    │                 │    │
│   │ Load-    │◄───┼────┤  Attention-     │    │
│   │ Aware    │    │    │  based DDPG     │    │
│   │ TSN      │    │    │  Scheduler      │    │
│   │ Scheduler│    │    │                 │    │
│   └──────────┘    │    └─────────────────┘    │
│        │          │            │               │
│        ▼          │            ▼               │
│   TSN Switches────┼─────► WiFi MLO            │
│   with GCL        │     (2.4/5/6 GHz)         │
└────────────────────┴────────────────────────────┘
```

## 🔬 Technical Components

### 1. TSN Domain
- **Time-Aware Shaper (TAS)**: Precise timing control with Gate Control Lists (GCL)
- **Load-Aware Routing**: Path selection minimizing bottleneck congestion
- **Time Slot Allocation**: Microsecond-level scheduling

### 2. WiFi Domain
- **Multi-Link Operation (MLO)**: Concurrent transmission on 2.4/5/6 GHz bands
- **Dynamic Channel Adaptation**: SNR-based reliability estimation
- **Airtime Management**: Per-band resource allocation

### 3. DRL Agent
- **Actor Network**: Multi-head self-attention + policy output
- **Critic Network**: Q-value estimation with attention
- **Training**: DDPG with experience replay and target networks

## 📦 Quick Start

```bash
# Clone repository
git clone https://github.com/Yuccccchan/Paper.git
cd Paper

# Install dependencies (Python 3.8+)
pip install torch numpy networkx gym matplotlib pandas scipy

# Follow detailed implementation in IMPLEMENTATION_GUIDE.md
```

## 📚 Related Papers

- `CaaS_Enabling_Control-as-a-Service_for_Time-Sensitive_Networking.pdf` - Control-as-a-Service for TSN
- `Cao 等 - 2025 - How can the integrated routing and scheduling enhance optimality bounds of time-sensitive transmissi.pdf` - Routing and scheduling optimization

## 🔗 Standards Referenced

- IEEE 802.1 TSN Task Group standards
- IEEE 802.1AS: Time Synchronization
- IEEE 802.1Qbv: Time-Aware Shaper (TAS)
- IEEE 802.11be: Extremely High Throughput (EHT) / WiFi 7

## 📖 Citation

```bibtex
@inproceedings{guo2025attention,
  title={Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks},
  author={Guo, Miao and Yang, Yichuan and He, Shibo and Pan, Jianping and Gu, Chaojie and Chen, Jiming},
  booktitle={ACM/IEEE 16th International Conference on Cyber-Physical Systems (ICCPS)},
  year={2025},
  pages={10},
  doi={10.1145/3716550.3722018}
}
```

## 🎯 Key Takeaways

1. **Self-attention is effective** for learning flow dependencies in network scheduling
2. **Cross-domain coordination** requires careful constraint modeling
3. **Load-aware TSN scheduling** significantly reduces congestion
4. **DRL with attention** outperforms traditional MLP-based approaches by ~12-18%
5. **Practical deployment** requires integration with TSN switches and WiFi APs

## 📧 Contact

For questions about implementation or the paper, please open an issue in this repository.

---

**Note**: This repository provides implementation guidance based on the published paper. Actual deployment may require hardware-specific adaptations for TSN switches and WiFi access points.