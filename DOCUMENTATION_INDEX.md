# Documentation Index

Complete guide to all documentation for implementing "Pay Attention to Network: Reliability-Aware Spatial-Temporal-Frequential Scheduling for TSN-WiFi Networks" (ICCPS 2025).

## 📖 Documentation Overview

This repository contains **7 comprehensive documents** (~4,400 lines total) providing everything needed to understand and implement the TSN-WiFi scheduling system.

## 🗺️ Quick Navigation

| If you want to... | Read this | Time |
|-------------------|-----------|------|
| Get an overview | README.md | 5 min |
| Understand concepts | FAQ.md (Q1-Q10) | 15 min |
| Try it quickly | QUICK_START.md | 30 min |
| Implement fully | IMPLEMENTATION_GUIDE.md | 2-3 days |
| Understand algorithms | ALGORITHMS.md | 2-4 hours |
| Plan your project | ROADMAP.md | 1 hour |
| Customize setup | CONFIGURATION.md | 1 hour |

## 📄 Document Descriptions

### 1. README.md (152 lines, 6.6 KB)
**Entry point and quick reference**

Contains:
- Paper summary and key contributions
- System architecture diagram
- Performance results (78% reliability, 12% improvement)
- Quick start instructions
- Citation information

👉 **Start here if you're new**

### 2. IMPLEMENTATION_GUIDE.md (1,419 lines, 49 KB)
**Complete implementation with full code**

Contains:
- System architecture breakdown
- Full Python implementation:
  - TSN domain (topology, flows, load-aware scheduler)
  - WiFi domain (MLO, channels, airtime management)
  - DRL agent (attention networks, DDPG)
  - Environment (Gym interface, state encoding, rewards)
  - Training and evaluation pipelines
- Code examples for every component
- Expected results and validation

👉 **Use this for coding**

### 3. ALGORITHMS.md (557 lines, 16 KB)
**Detailed pseudocode and logic**

Contains:
- Algorithm 1: Load-Aware TSN Scheduling
- Algorithm 2: WiFi DDPG with Attention
- Algorithm 3: Ternary Band Selection
- Algorithm 4: Cross-Domain Constraint Validation
- Algorithm 5: State Encoding for DRL
- Neural network architectures
- Complexity analysis
- Performance metrics calculations

👉 **Reference for understanding logic**

### 4. QUICK_START.md (492 lines, 14 KB)
**Hands-on examples and tutorials**

Contains:
- Installation guide
- Minimal working examples
- Step-by-step tutorials (TSN test, WiFi test, training, evaluation)
- Visualization examples
- Complete pipeline script
- Troubleshooting tips

👉 **Follow this to get started quickly**

### 5. CONFIGURATION.md (611 lines, 14 KB)
**Setup templates and configuration**

Contains:
- Network topology configs (small/medium/large)
- Training hyperparameters
- Evaluation settings
- Docker deployment setup
- Configuration loader code
- Best practices

👉 **Use for customization**

### 6. ROADMAP.md (544 lines, 14 KB)
**16-week implementation plan**

Contains:
- 7 phases with weekly breakdown
- Progress tracking checklists
- Common pitfalls and solutions
- Debugging strategies
- Success metrics
- Deployment checklist

👉 **Follow for project planning**

### 7. FAQ.md (595 lines, 16 KB)
**38+ frequently asked questions**

Contains:
- General concepts (What, Why, How)
- Technical details (Attention, DDPG, Algorithms)
- Implementation help (Setup, Training, Debugging)
- Domain specifics (TSN, WiFi, MLO, SNR)
- Performance expectations
- Production deployment guidance
- Research directions

👉 **Check when you have questions**

## 🎯 Recommended Reading Paths

### Path 1: Quick Understanding (30 minutes)
Perfect for: Getting familiar with the project
```
1. README.md (5 min) - Overview
2. FAQ.md Q1-Q10 (15 min) - Key concepts  
3. QUICK_START.md (10 min) - Examples
```

### Path 2: Full Implementation (2-3 weeks)
Perfect for: Building the complete system
```
1. README.md (5 min) - Context
2. ROADMAP.md (1 hour) - Planning
3. IMPLEMENTATION_GUIDE.md (3-5 days) - Coding
4. ALGORITHMS.md (as needed) - Reference
5. QUICK_START.md (1 day) - Testing
6. CONFIGURATION.md (1 day) - Customization
```

### Path 3: Research Study (1 week)
Perfect for: Academic research or deep understanding
```
1. Read paper + README.md (2-3 hours)
2. ALGORITHMS.md (1 day) - Theory
3. IMPLEMENTATION_GUIDE.md (3 days) - Code
4. FAQ.md Q31-Q38 (1 hour) - Research ideas
5. Run experiments (2 days)
```

### Path 4: Production Deployment (1 month)
Perfect for: Real-world deployment
```
1. IMPLEMENTATION_GUIDE.md (1 week) - Build
2. CONFIGURATION.md (3 days) - Setup
3. ROADMAP.md Phase 7 (3 days) - Deployment prep
4. FAQ.md Q25-Q30 (1 day) - Production tips
5. Testing & integration (2-3 weeks)
```

## 🔍 Topic Guide

### TSN (Time-Sensitive Networking)
- **Basics**: FAQ.md Q17-Q19
- **Algorithm**: ALGORITHMS.md Algorithm 1
- **Code**: IMPLEMENTATION_GUIDE.md Phase 2

### WiFi MLO (Multi-Link Operation)
- **Basics**: FAQ.md Q18, Q20
- **Code**: IMPLEMENTATION_GUIDE.md Phase 3
- **Band Selection**: ALGORITHMS.md Algorithm 3

### Attention Mechanism
- **Why use it**: FAQ.md Q5
- **Architecture**: ALGORITHMS.md (Actor Network)
- **Implementation**: IMPLEMENTATION_GUIDE.md Phase 4

### DDPG (Deep Deterministic Policy Gradient)
- **Algorithm**: ALGORITHMS.md Algorithm 2
- **Implementation**: IMPLEMENTATION_GUIDE.md Phase 4
- **Training**: QUICK_START.md Step 4

### Training & Evaluation
- **Quick example**: QUICK_START.md Step 4-5
- **Full pipeline**: IMPLEMENTATION_GUIDE.md Phase 6-7
- **Config**: CONFIGURATION.md Training Params

### Deployment
- **Planning**: ROADMAP.md Phase 7
- **Production**: FAQ.md Q25-Q30
- **Docker**: CONFIGURATION.md Docker section

## 📊 Statistics

| Metric | Value |
|--------|-------|
| Total Documents | 7 |
| Total Lines | 4,370 |
| Total Size | ~130 KB |
| Code Examples | 50+ |
| Algorithms | 5 detailed |
| FAQ Questions | 38+ |
| Implementation Time | 12-16 weeks |

## 🚀 For Different Roles

**👨‍🎓 Student/Researcher**
→ README.md → ALGORITHMS.md → FAQ.md Q31-Q33

**👨‍💻 Software Engineer**  
→ QUICK_START.md → IMPLEMENTATION_GUIDE.md → CONFIGURATION.md

**🏗️ System Architect**
→ README.md → ROADMAP.md → FAQ.md Q25-Q30

**🤖 ML Engineer**
→ ALGORITHMS.md → IMPLEMENTATION_GUIDE.md Phase 4 → CONFIGURATION.md

## 📦 What's Included

✅ Complete system architecture  
✅ Full Python implementation (all components)  
✅ 5 detailed algorithms with pseudocode  
✅ Step-by-step tutorials and examples  
✅ Configuration templates for 3 scenarios  
✅ 16-week implementation roadmap  
✅ 38+ FAQ covering all aspects  
✅ Training, evaluation, and deployment guides  
✅ Troubleshooting and debugging tips  
✅ Research directions and extensions  

## 🎯 Expected Results

Following this documentation, you should achieve:
- ✅ **78% reliability** for 50 flows
- ✅ **12% improvement** over baselines
- ✅ **Network-wise scheduling** with attention
- ✅ **Sub-millisecond inference** time
- ✅ **Convergence** in 500-700 episodes

## 📬 Support

**Have questions?**
1. Check FAQ.md (38+ questions answered)
2. Search this index for relevant docs
3. Open GitHub issue
4. Contact repository maintainer

**Found an error?**
1. Open an issue describing it
2. Submit a pull request
3. Help improve documentation

## ✅ Next Steps

**Choose your path:**

🟢 **New to project?**  
→ Start with README.md

🔵 **Ready to code?**  
→ Follow QUICK_START.md

🟡 **Planning implementation?**  
→ Review ROADMAP.md

🟣 **Need specific info?**  
→ Check FAQ.md or search this index

---

**Status**: ✅ Complete Documentation Package  
**Based On**: ICCPS 2025 Paper  
**Last Updated**: November 2024

**All documentation is ready to use!** 🎉
