"""
SIMPLE TRAINING EXAMPLE
A beginner-friendly script showing how to train MAIRS agents step-by-step.
"""

import sys
sys.path.insert(0, 'src')

from environment import TSNNetwork, TTFlow
from mairs import MAIRSAgent, MAPPO

print("=" * 60)
print("SIMPLE MAIRS TRAINING EXAMPLE")
print("=" * 60)
print()

# ============================================================
# STEP 1: Create the Network
# ============================================================
print("STEP 1: Creating a simple network")
print("-" * 60)

network = TSNNetwork(slot_duration=0.2, link_bandwidth=100)

# Add 3 switches
network.add_switch(0)
network.add_switch(1)
network.add_switch(2)

# Connect them in a line: 0 -> 1 -> 2
network.add_link(0, 1)
network.add_link(1, 2)

print(f"✓ Created network with {len(network.switches)} switches")
print(f"  Connections: Switch 0 -> Switch 1 -> Switch 2")
print()

# ============================================================
# STEP 2: Add a Data Flow
# ============================================================
print("STEP 2: Adding a data flow")
print("-" * 60)

# Create a flow that needs to go from switch 0 to switch 2
flow = TTFlow(
    flow_id=0,
    source=0,           # Starts at switch 0
    destination=2,      # Needs to reach switch 2
    data_size=1000,     # 1000 bytes
    period=4.8          # Every 4.8 milliseconds
)

network.add_flow(flow)

print(f"✓ Flow created: {flow}")
print(f"  Goal: Get data from Switch {flow.source} to Switch {flow.destination}")
print(f"  Hyperperiod: {network.hyperperiod} ms")
print(f"  Available time slots: {network.max_slots}")
print()

# ============================================================
# STEP 3: Create AI Agents
# ============================================================
print("STEP 3: Creating AI agents (one for each switch)")
print("-" * 60)

agents = {}

for switch_id in network.switches:
    # Count neighbors for this switch
    num_neighbors = len(network.switches[switch_id].get_neighbors())
    if num_neighbors == 0:
        num_neighbors = 1  # Minimum of 1
    
    # Create an agent
    agents[switch_id] = MAIRSAgent(
        agent_id=switch_id,
        num_neighbors=num_neighbors,
        max_slots=network.max_slots,
        state_dim=20,       # How much info the agent sees
        hidden_dim=64       # Size of the neural network
    )
    
    print(f"  ✓ Agent created for Switch {switch_id} ({num_neighbors} neighbor(s))")

print()

# ============================================================
# STEP 4: Train the Agents
# ============================================================
print("STEP 4: Training the agents")
print("-" * 60)
print("The agents will learn how to route data and schedule time slots.")
print("This may take a minute...")
print()

# Create the trainer
trainer = MAPPO(agents, alpha=0.25)

# Train for 100 iterations (quick demonstration)
history = trainer.train(
    network=network,
    flows=[flow],
    num_iterations=100,  # 100 iterations for quick demo
    batch_size=10
)

print()
print("✓ Training complete!")
print()

# ============================================================
# STEP 5: Show Results
# ============================================================
print("STEP 5: Results")
print("-" * 60)

if len(history['success_rates']) > 0:
    # Get final results
    final_success = history['success_rates'][-1]
    final_reward = history['avg_rewards'][-1]
    
    print(f"Final Success Rate: {final_success:.1%}")
    print(f"Final Average Reward: {final_reward:.3f}")
    print()
    
    # Show improvement
    if len(history['success_rates']) > 10:
        initial_success = sum(history['success_rates'][:10]) / 10
        final_success_avg = sum(history['success_rates'][-10:]) / 10
        
        improvement = ((final_success_avg - initial_success) / (initial_success + 0.001)) * 100
        
        print(f"Learning Progress:")
        print(f"  First 10 iterations: {initial_success:.1%} success")
        print(f"  Last 10 iterations:  {final_success_avg:.1%} success")
        print(f"  Improvement: {improvement:+.1f}%")
    print()

# ============================================================
# STEP 6: Test the Trained Agents
# ============================================================
print("STEP 6: Testing the trained agents")
print("-" * 60)

# Reset the network
network.reset_schedules()

# Let the trained agents route the flow
trajectory, reward, success = trainer.rollout_flow(network, flow)

print(f"Test Result: {'SUCCESS ✓' if success else 'FAILED ✗'}")
print(f"Route taken: {' -> '.join(f'Switch {s}' for s in flow.route)}")
print(f"Reward earned: {reward:.3f}")
print()

# Show what the agents learned
if success and len(flow.route) > 1:
    print("Scheduled time slots:")
    for i in range(len(flow.route) - 1):
        link = (flow.route[i], flow.route[i+1])
        if link in flow.schedule:
            slot = flow.schedule[link]
            print(f"  Switch {link[0]} -> Switch {link[1]}: Time slot {slot}")

print()
print("=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print()
print("What just happened?")
print("1. Created a simple network with 3 switches")
print("2. Defined a data flow from switch 0 to switch 2")
print("3. Created AI agents for each switch")
print("4. Trained the agents through trial and error (100 iterations)")
print("5. Agents learned to route data and schedule transmission times")
print()
print("Try modifying this example:")
print("- Add more switches")
print("- Add more flows")
print("- Increase num_iterations to 1000 for better learning")
print("- Change the network topology")
