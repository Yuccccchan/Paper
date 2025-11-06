"""
Basic tests for MAIRS implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from environment import TSNNetwork, TTFlow, TSNSwitch
from mairs import MAIRSAgent, MAPPO

def test_tsn_switch():
    """Test TSN switch functionality."""
    print("Testing TSN Switch...")
    
    switch = TSNSwitch(switch_id=0, num_ports=8)
    assert switch.switch_id == 0
    assert switch.num_ports == 8
    
    # Test port connection
    switch.connect_port(0, 1)
    assert switch.ports[0]['connected_to'] == 1
    
    # Test slot allocation
    assert switch.is_slot_available(0, 0) == True
    switch.allocate_slot(0, 0, flow_id=1)
    assert switch.is_slot_available(0, 0) == False
    
    print("✓ TSN Switch tests passed")

def test_tt_flow():
    """Test TT flow functionality."""
    print("Testing TT Flow...")
    
    flow = TTFlow(
        flow_id=0,
        source=0,
        destination=3,
        data_size=1000,
        period=4.8
    )
    
    assert flow.flow_id == 0
    assert flow.source == 0
    assert flow.destination == 3
    assert flow.period == 4.8
    
    # Test transmission time calculation
    tx_time = flow.get_transmission_time(link_bandwidth=100)
    assert tx_time > 0
    
    # Test slot calculation
    slots = flow.get_required_slots(slot_duration=0.2)
    assert slots > 0
    
    print("✓ TT Flow tests passed")

def test_tsn_network():
    """Test TSN network functionality."""
    print("Testing TSN Network...")
    
    network = TSNNetwork(slot_duration=0.2, link_bandwidth=100)
    
    # Add switches
    for i in range(4):
        network.add_switch(i, num_ports=8)
    
    assert len(network.switches) == 4
    
    # Add links
    network.add_link(0, 1)
    network.add_link(1, 2)
    network.add_link(2, 3)
    
    # Test path finding
    path = network.get_shortest_path(0, 3)
    assert path is not None
    assert path[0] == 0
    assert path[-1] == 3
    
    # Add flow
    flow = TTFlow(0, 0, 3, 1000, 4.8)
    network.add_flow(flow)
    
    assert network.hyperperiod is not None
    assert network.max_slots is not None
    
    print("✓ TSN Network tests passed")

def test_mairs_agent():
    """Test MAIRS agent functionality."""
    print("Testing MAIRS Agent...")
    
    agent = MAIRSAgent(
        agent_id=0,
        num_neighbors=3,
        max_slots=24,
        state_dim=20,
        hidden_dim=64
    )
    
    assert agent.agent_id == 0
    assert agent.num_neighbors == 3
    assert agent.max_slots == 24
    
    # Test state generation
    state = np.random.rand(20).astype(np.float32)
    
    # Test action selection
    action, log_prob = agent.select_action(state)
    assert isinstance(action, int)
    assert action >= 0
    assert action < agent.action_dim
    
    # Test action decoding
    neighbor_idx, slot_idx = agent.decode_action(action)
    assert neighbor_idx >= 0
    assert slot_idx >= 0
    assert slot_idx < agent.max_slots
    
    # Test value estimation
    value = agent.get_value(state)
    assert isinstance(value, float)
    
    print("✓ MAIRS Agent tests passed")

def test_mappo():
    """Test MAPPO trainer functionality."""
    print("Testing MAPPO...")
    
    # Create simple network
    network = TSNNetwork(slot_duration=0.2, link_bandwidth=100)
    for i in range(3):
        network.add_switch(i, num_ports=8)
    
    network.add_link(0, 1)
    network.add_link(1, 2)
    
    # Add flow
    flow = TTFlow(0, 0, 2, 1000, 4.8)
    network.add_flow(flow)
    
    # Create agents
    agents = {}
    for switch_id in network.switches:
        agents[switch_id] = MAIRSAgent(
            agent_id=switch_id,
            num_neighbors=len(network.switches[switch_id].get_neighbors()) or 1,
            max_slots=network.max_slots or 24,
            state_dim=20,
            hidden_dim=64
        )
    
    # Create trainer
    trainer = MAPPO(agents, alpha=0.25)
    
    # Test rollout
    trajectory, reward, success = trainer.rollout_flow(network, flow)
    assert isinstance(trajectory, list)
    assert isinstance(reward, float)
    assert isinstance(success, bool)
    
    # Test reward computation
    reward = trainer.compute_reward(network, flow, True, 0.5)
    assert isinstance(reward, float)
    
    print("✓ MAPPO tests passed")

def test_integration():
    """Test full integration."""
    print("Testing full integration...")
    
    # Create network
    network = TSNNetwork(slot_duration=0.2, link_bandwidth=100)
    for i in range(5):
        network.add_switch(i, num_ports=8)
    
    # Create topology
    network.add_link(0, 1)
    network.add_link(1, 2)
    network.add_link(2, 3)
    network.add_link(1, 4)
    network.add_link(4, 3)
    
    # Create flows
    flows = []
    for i in range(3):
        flow = TTFlow(
            flow_id=i,
            source=0,
            destination=3,
            data_size=np.random.randint(60, 1500),
            period=np.random.choice([3.6, 4.8, 6.0])
        )
        network.add_flow(flow)
        flows.append(flow)
    
    # Create agents
    agents = {}
    state_dim = 20
    for switch_id in network.switches:
        num_neighbors = len(network.switches[switch_id].get_neighbors())
        agents[switch_id] = MAIRSAgent(
            agent_id=switch_id,
            num_neighbors=num_neighbors if num_neighbors > 0 else 1,
            max_slots=network.max_slots,
            state_dim=state_dim,
            hidden_dim=64
        )
    
    # Train for a few iterations
    trainer = MAPPO(agents)
    history = trainer.train(network, flows, num_iterations=10, batch_size=3)
    
    assert 'success_rates' in history
    assert 'avg_rewards' in history
    
    print("✓ Integration tests passed")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Running MAIRS Implementation Tests")
    print("=" * 60)
    print()
    
    np.random.seed(42)
    
    test_tsn_switch()
    test_tt_flow()
    test_tsn_network()
    test_mairs_agent()
    test_mappo()
    test_integration()
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

if __name__ == "__main__":
    main()
