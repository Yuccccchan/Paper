# Simple Training Guide for MAIRS

## What is MAIRS?

MAIRS helps data packets find the best path through a network and schedule when they should be sent. Think of it like a smart traffic controller for computer networks.

## How to Train the Agent - Step by Step

### Step 1: Install Requirements

First, make sure you have Python installed, then install the needed packages:

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (for machine learning)
- NumPy (for math operations)
- NetworkX (for network graphs)
- Matplotlib (for visualizations)

### Step 2: Create a Simple Network

Create a Python file (e.g., `train_simple.py`) and start with a basic network:

```python
import sys
sys.path.insert(0, 'src')

from environment import TSNNetwork, TTFlow
from mairs import MAIRSAgent, MAPPO

# Step 1: Create a simple network with 3 switches
network = TSNNetwork(slot_duration=0.2, link_bandwidth=100)

# Add 3 switches
network.add_switch(0)  # Switch 0
network.add_switch(1)  # Switch 1
network.add_switch(2)  # Switch 2

# Connect them: 0 -> 1 -> 2
network.add_link(0, 1)
network.add_link(1, 2)

print(f"Network created: {network}")
```

### Step 3: Create Data Flows

Add flows (data that needs to travel through the network):

```python
# Create a flow from switch 0 to switch 2
flow = TTFlow(
    flow_id=0,
    source=0,           # Start at switch 0
    destination=2,      # End at switch 2
    data_size=1000,     # 1000 bytes of data
    period=4.8          # Repeats every 4.8 milliseconds
)

network.add_flow(flow)
print(f"Flow added: {flow}")
```

### Step 4: Create AI Agents (One for Each Switch)

Each switch gets its own AI agent to make decisions:

```python
agents = {}

# Create an agent for each switch
for switch_id in network.switches:
    # Count how many neighbors this switch has
    num_neighbors = len(network.switches[switch_id].get_neighbors())
    if num_neighbors == 0:
        num_neighbors = 1  # At least 1
    
    # Create the agent
    agents[switch_id] = MAIRSAgent(
        agent_id=switch_id,
        num_neighbors=num_neighbors,
        max_slots=network.max_slots,  # Time slots available
        state_dim=20,                  # Information the agent sees
        hidden_dim=64                  # Neural network size
    )

print(f"Created {len(agents)} agents")
```

### Step 5: Train the Agents

Now train the agents to learn the best routing and scheduling:

```python
# Create the trainer
trainer = MAPPO(agents, alpha=0.25)

# Train for 100 iterations (start small for testing)
print("Starting training...")
history = trainer.train(
    network=network,
    flows=[flow],
    num_iterations=100,  # Try 100 iterations first
    batch_size=10
)

print("Training complete!")
```

### Step 6: Check the Results

See how well the agents learned:

```python
# Show training progress
if history['success_rates']:
    print(f"\nSuccess rate: {history['success_rates'][-1]:.2%}")
    print(f"Average reward: {history['avg_rewards'][-1]:.3f}")
    
    # Show improvement
    initial = history['success_rates'][0]
    final = history['success_rates'][-1]
    improvement = ((final - initial) / (initial + 0.001)) * 100
    print(f"Improvement: {improvement:.1f}%")
```

## Complete Example

Here's everything together in one file:

```python
import sys
sys.path.insert(0, 'src')

from environment import TSNNetwork, TTFlow
from mairs import MAIRSAgent, MAPPO

# 1. Create network
network = TSNNetwork(slot_duration=0.2, link_bandwidth=100)
for i in range(3):
    network.add_switch(i)
network.add_link(0, 1)
network.add_link(1, 2)

# 2. Add flow
flow = TTFlow(0, source=0, destination=2, data_size=1000, period=4.8)
network.add_flow(flow)

# 3. Create agents
agents = {}
for switch_id in network.switches:
    num_neighbors = len(network.switches[switch_id].get_neighbors()) or 1
    agents[switch_id] = MAIRSAgent(
        agent_id=switch_id,
        num_neighbors=num_neighbors,
        max_slots=network.max_slots,
        state_dim=20,
        hidden_dim=64
    )

# 4. Train
trainer = MAPPO(agents, alpha=0.25)
history = trainer.train(network, [flow], num_iterations=100, batch_size=10)

# 5. Results
print(f"Success rate: {history['success_rates'][-1]:.2%}")
print(f"Reward: {history['avg_rewards'][-1]:.3f}")
```

## What Happens During Training?

1. **Initial State**: Agents start with random behavior
2. **Learning Loop**: For each iteration:
   - Agents try to route the flow through the network
   - They get a reward (good = 1, bad = 0)
   - They learn from mistakes and improve
3. **Result**: After many iterations, agents learn to route flows successfully

## Key Parameters Explained

- `num_iterations`: How many times to practice (more = better learning, but slower)
  - Start with 100 for testing
  - Use 1000-8000 for real training
  
- `batch_size`: How many examples to learn from at once
  - Bigger = faster but uses more memory
  - 10-32 is a good range
  
- `alpha`: How much to care about network balance (0.25 = balanced)
  - Lower = focus on success
  - Higher = focus on spreading traffic evenly

## Tips for Beginners

1. **Start Small**: Begin with 3 switches and 1 flow
2. **Short Training**: Try 100 iterations first to make sure it works
3. **Watch Progress**: The code prints updates every 100 iterations
4. **Increase Gradually**: Once it works, add more flows and switches
5. **Be Patient**: Real training (8000 iterations) can take 10-30 minutes

## Running the Demo

To run the full example that comes with the code:

```bash
python examples/mairs_demo.py
```

This runs a bigger network with 50 flows and 500 training iterations.

## Common Issues

**Problem**: "ImportError: No module named 'torch'"
- **Solution**: Run `pip install -r requirements.txt`

**Problem**: Training is very slow
- **Solution**: Reduce `num_iterations` or use fewer flows

**Problem**: Success rate stays at 0%
- **Solution**: Your network might not have a path. Check that switches are connected.

## Next Steps

Once you understand the basics:
1. Try different network topologies (more switches, different connections)
2. Add more flows (10, 50, 100)
3. Increase training iterations for better results
4. Experiment with different `alpha` values
5. Visualize the routes the agents choose
