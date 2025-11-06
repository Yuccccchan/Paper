"""
Example: Using MAIRS for integrated routing and scheduling in TSN

This example demonstrates the MAIRS algorithm from Cao et al. 2025 paper:
"How Can the Integrated Routing and Scheduling Enhance Optimality 
Bounds of Time-Sensitive Transmission"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from environment import TSNNetwork, TTFlow
from mairs import MAIRSAgent, MAPPO

def create_orion_cev_network():
    """
    Create Orion CEV (Crew Exploration Vehicle) network topology.
    
    This is the benchmark topology used in the paper, integrating
    star, tree, and ring structures.
    
    Returns:
        TSNNetwork instance
    """
    network = TSNNetwork(slot_duration=0.2, link_bandwidth=100)
    
    # Add switches (simplified version with 10 switches)
    for i in range(10):
        network.add_switch(i, num_ports=8)
    
    # Create topology with star, tree, and ring structures
    # Central hub (star center)
    network.add_link(0, 1)
    network.add_link(0, 2)
    network.add_link(0, 3)
    
    # Tree structure
    network.add_link(1, 4)
    network.add_link(1, 5)
    network.add_link(2, 6)
    network.add_link(2, 7)
    
    # Ring structure
    network.add_link(3, 8)
    network.add_link(8, 9)
    network.add_link(9, 3)
    
    # Additional connections for redundancy
    network.add_link(4, 6)
    network.add_link(5, 7)
    
    return network

def generate_tt_flows(network, num_flows=100):
    """
    Generate random TT flows for the network.
    
    Args:
        network: TSNNetwork instance
        num_flows: Number of flows to generate
        
    Returns:
        List of TTFlow instances
    """
    flows = []
    switch_ids = list(network.switches.keys())
    
    # Possible periods (in ms)
    periods = [3.6, 4.8, 6.0]
    
    for i in range(num_flows):
        # Random source and destination
        source = np.random.choice(switch_ids)
        destination = np.random.choice(switch_ids)
        
        # Ensure source != destination
        while destination == source:
            destination = np.random.choice(switch_ids)
        
        # Random data size (60-1500 bytes)
        data_size = np.random.randint(60, 1501)
        
        # Random period
        period = np.random.choice(periods)
        
        # Create flow
        flow = TTFlow(
            flow_id=i,
            source=source,
            destination=destination,
            data_size=data_size,
            period=period,
            deadline=period  # Deadline equals period
        )
        
        flows.append(flow)
        network.add_flow(flow)
    
    return flows

def create_mairs_agents(network):
    """
    Create MAIRS agents for each switch in the network.
    
    Args:
        network: TSNNetwork instance
        
    Returns:
        Dictionary of {switch_id: MAIRSAgent}
    """
    agents = {}
    
    # Determine state dimension based on maximum neighbors
    max_neighbors = max(len(switch.get_neighbors()) 
                       for switch in network.switches.values())
    
    # State features: 
    # - Flow info (3): data_size, period, deadline
    # - Position info (2): is_source, is_destination  
    # - Per neighbor (3): distance_to_dest, link_util, visited_flag
    state_dim = 5 + max_neighbors * 3
    
    for switch_id, switch in network.switches.items():
        num_neighbors = len(switch.get_neighbors())
        
        agent = MAIRSAgent(
            agent_id=switch_id,
            num_neighbors=num_neighbors if num_neighbors > 0 else 1,
            max_slots=network.max_slots if network.max_slots else 100,
            state_dim=state_dim,
            hidden_dim=128,
            learning_rate=0.003
        )
        
        agents[switch_id] = agent
    
    return agents

def main():
    """Main function to demonstrate MAIRS algorithm."""
    
    print("=" * 60)
    print("MAIRS: Multi-Agent Deep Reinforcement Learning")
    print("for Integrated Routing and Scheduling in TSN")
    print("Based on Cao et al. 2025")
    print("=" * 60)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create network
    print("Creating Orion CEV network topology...")
    network = create_orion_cev_network()
    print(f"Network created: {network}")
    print()
    
    # Generate TT flows
    print("Generating TT flows...")
    num_flows = 50  # Start with smaller number for demonstration
    flows = generate_tt_flows(network, num_flows)
    print(f"Generated {len(flows)} TT flows")
    print(f"Hyperperiod: {network.hyperperiod} ms")
    print(f"Max slots: {network.max_slots}")
    print()
    
    # Create MAIRS agents
    print("Creating MAIRS agents for each switch...")
    agents = create_mairs_agents(network)
    print(f"Created {len(agents)} agents")
    print()
    
    # Create MAPPO trainer
    print("Initializing MAPPO trainer...")
    trainer = MAPPO(
        agents=agents,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        alpha=0.25  # Scaling factor from paper
    )
    print()
    
    # Train agents
    print("Training MAIRS agents...")
    print("(This may take a while for full 8000 iterations)")
    print("Using reduced iterations (500) for demonstration")
    print()
    
    history = trainer.train(
        network=network,
        flows=flows,
        num_iterations=500,  # Reduced for demonstration
        batch_size=32
    )
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()
    
    # Display results
    if len(history['success_rates']) > 0:
        final_success_rate = history['success_rates'][-1]
        final_avg_reward = history['avg_rewards'][-1]
        
        print(f"Final Success Rate: {final_success_rate:.3f}")
        print(f"Final Average Reward: {final_avg_reward:.3f}")
        print()
        
        # Show improvement over training
        if len(history['success_rates']) > 10:
            initial_success = np.mean(history['success_rates'][:10])
            final_success = np.mean(history['success_rates'][-10:])
            improvement = (final_success - initial_success) / (initial_success + 1e-10) * 100
            
            print(f"Improvement: {improvement:.1f}%")
            print(f"  Initial (first 10 iters): {initial_success:.3f}")
            print(f"  Final (last 10 iters): {final_success:.3f}")
    
    print()
    print("Example flow scheduling results:")
    print("-" * 60)
    
    # Test with trained agents on a few flows
    network.reset_schedules()
    successful = 0
    
    for flow in flows[:5]:  # Show first 5 flows
        trajectory, reward, success = trainer.rollout_flow(network, flow)
        successful += int(success)
        
        print(f"Flow {flow.flow_id}: {flow.source} -> {flow.destination}")
        print(f"  Route: {' -> '.join(map(str, flow.route))}")
        print(f"  Success: {'✓' if success else '✗'}")
        print(f"  Reward: {reward:.3f}")
        print()
    
    print(f"Successfully scheduled: {successful}/5 flows")
    print()
    print("MAIRS demonstration complete!")

if __name__ == "__main__":
    main()
