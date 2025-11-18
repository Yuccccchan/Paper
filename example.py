#!/usr/bin/env python3
"""
Example usage of TSN-WiFi Scheduler
Demonstrates the integrated scheduling system with sample topologies.
"""

import numpy as np
from tsn_wifi_scheduler import (
    TSNNetwork, Flow, WiFiMLO, LoadAwareScheduler, AttentionDDPG
)
from tsn_wifi_scheduler.integrated_scheduler import IntegratedScheduler


def create_sample_topology(topology_type: str = "simple") -> TSNNetwork:
    """
    Create sample TSN network topologies.
    
    Args:
        topology_type: Type of topology ("simple", "ladder", "afdx")
        
    Returns:
        TSN network with configured topology
    """
    network = TSNNetwork(cycle_time=1000.0)  # 1000 microseconds
    
    if topology_type == "simple":
        # Simple linear topology: 0 -- 1 -- 2 -- 3
        for i in range(4):
            network.add_switch(i)
        for i in range(3):
            network.add_link(i, i, i+1)
            
    elif topology_type == "ladder":
        # Ladder topology (2x4 grid)
        # 0 -- 1 -- 2 -- 3
        # |    |    |    |
        # 4 -- 5 -- 6 -- 7
        for i in range(8):
            network.add_switch(i)
        
        # Horizontal links
        for i in range(3):
            network.add_link(i, i, i+1)
            network.add_link(i+4, i+4, i+5)
        
        # Vertical links
        for i in range(4):
            network.add_link(i+4, i, i+4)
            
    elif topology_type == "afdx":
        # AFDX-like topology (Avionics Full-Duplex Switched Ethernet)
        # Star topology with redundant paths
        for i in range(7):
            network.add_switch(i)
        
        # Central switches: 0, 1
        # Edge switches: 2, 3, 4, 5, 6
        link_id = 0
        for edge in [2, 3, 4, 5, 6]:
            network.add_link(link_id, 0, edge)
            link_id += 1
            network.add_link(link_id, 1, edge)
            link_id += 1
        # Connect central switches
        network.add_link(link_id, 0, 1)
    
    return network


def create_sample_flows(network: TSNNetwork, num_flows: int = 50) -> list:
    """
    Create sample flows for the network.
    
    Args:
        network: TSN network
        num_flows: Number of flows to generate
        
    Returns:
        List of Flow objects
    """
    flows = []
    switches = list(network.switches.keys())
    
    for i in range(num_flows):
        # Random source and destination
        src = np.random.choice(switches)
        dst = np.random.choice(switches)
        
        # Random start offset (0-4 time slots)
        start_offset = np.random.randint(0, 5)
        
        flow = Flow(
            flow_id=i,
            src=src,
            dst=dst,
            size=1500,  # MTU size
            start_offset=start_offset
        )
        flows.append(flow)
        network.add_flow(flow)
    
    return flows


def run_scheduling_example():
    """Run a complete scheduling example."""
    print("=" * 70)
    print("TSN-WiFi Network Scheduler - Example")
    print("Reproduction of: Pay Attention to Network")
    print("=" * 70)
    print()
    
    # Create network topology
    print("1. Creating network topology (ladder)...")
    network = create_sample_topology("ladder")
    print(f"   Created network with {len(network.switches)} switches and {len(network.links)} links")
    
    # Create WiFi MLO
    print("\n2. Initializing WiFi MLO...")
    wifi = WiFiMLO(cycle_time=1000.0)
    print(f"   WiFi bands: {len(wifi.bands)}")
    for band in wifi.bands:
        print(f"   - Band {band.band_id} ({band.frequency} GHz): "
              f"Rate={band.get_transmission_rate()} Mbps, "
              f"Reliability={band.reliability:.2f}")
    
    # Generate flows
    print("\n3. Generating flows...")
    num_flows = 50
    flows = create_sample_flows(network, num_flows)
    print(f"   Generated {len(flows)} flows")
    
    # Test TSN scheduling only
    print("\n4. Running TSN load-aware scheduling...")
    tsn_scheduler = LoadAwareScheduler(network)
    tsn_scheduled, max_load = tsn_scheduler.schedule_flows(flows)
    tsn_results = tsn_scheduler.get_scheduling_results()
    
    print(f"   TSN Results:")
    print(f"   - Scheduled flows: {tsn_scheduled}/{num_flows} ({tsn_results['scheduling_rate']*100:.1f}%)")
    print(f"   - Max link load: {max_load:.2f}%")
    print(f"   - Avg link load: {tsn_results['avg_link_load']:.2f}%")
    print(f"   - Load variance: {tsn_results['load_variance']:.2f}")
    
    # Reset for integrated scheduling
    network.reset_schedules()
    wifi.reset_schedules()
    
    # Test integrated scheduling (without DRL for simplicity)
    print("\n5. Running integrated TSN-WiFi scheduling (greedy)...")
    integrated_scheduler = IntegratedScheduler(network, wifi, use_drl=False)
    results = integrated_scheduler.schedule(flows)
    
    print(f"   Integrated Results:")
    print(f"   - Total scheduled: {results['total_scheduled']}/{results['total_flows']} "
          f"({results['scheduling_rate']*100:.1f}%)")
    print(f"   - TSN scheduled: {results['tsn_scheduled']}")
    print(f"   - WiFi scheduled: {results['wifi_scheduled']}")
    print(f"   - Average reliability: {results['avg_reliability']:.3f}")
    print(f"   - Max TSN load: {results['max_tsn_load']:.2f}%")
    
    # WiFi statistics
    wifi_stats = results['wifi_stats']
    print(f"\n   WiFi Band Utilization:")
    for i, util in enumerate(wifi_stats['band_utilizations']):
        reliability = wifi_stats['band_reliabilities'][i]
        print(f"   - Band {i} ({wifi.bands[i].frequency} GHz): "
              f"{util:.2f}% utilized, reliability={reliability:.3f}")
    
    # Demonstrate attention-based DDPG (architecture only, no training)
    print("\n6. Initializing Attention-based DDPG (no training)...")
    state_dim = 12  # 3 flow features + 9 channel features
    ddpg = AttentionDDPG(state_dim=state_dim, num_flows=50, hidden_dim=64)
    print(f"   DDPG initialized with:")
    print(f"   - State dimension: {state_dim}")
    print(f"   - Number of flows: {50}")
    print(f"   - Hidden dimension: {64}")
    print(f"   - Actor network: Self-Attention + MLP + Residual connection")
    print(f"   - Critic network: Self-Attention + MLP + Q-value estimation")
    
    # Test forward pass
    dummy_state = np.random.rand(50, state_dim).astype(np.float32)
    action_probs = ddpg.select_action(dummy_state, explore=False)
    print(f"   - Action shape: {action_probs.shape} (flows × bands)")
    print(f"   - Example action probabilities for flow 0: {action_probs[0]}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run example
    run_scheduling_example()
