#!/usr/bin/env python3
"""
Comprehensive Test Suite
Validates all implemented features from the paper.
"""

import numpy as np
import sys


def test_tsn_network():
    """Test TSN network components."""
    print("Testing TSN Network Components...")
    
    from tsn_wifi_scheduler import TSNNetwork, TSNSwitch, TSNLink, Flow
    
    # Create network
    network = TSNNetwork(cycle_time=1000.0)
    
    # Add switches
    for i in range(4):
        switch = network.add_switch(i)
        assert isinstance(switch, TSNSwitch)
        assert switch.switch_id == i
    
    # Add links
    for i in range(3):
        link = network.add_link(i, i, i+1)
        assert isinstance(link, TSNLink)
        assert link.link_id == i
    
    # Create and add flows
    flow1 = Flow(flow_id=0, src=0, dst=3, size=1500, start_offset=0)
    network.add_flow(flow1)
    assert len(network.flows) == 1
    
    # Test path finding
    path = network.get_shortest_path(0, 3)
    assert len(path) == 3  # Three hops
    
    # Test network stats
    stats = network.get_network_stats()
    assert stats['num_switches'] == 4
    assert stats['num_links'] == 3
    assert stats['num_flows'] == 1
    
    print("✓ TSN Network tests passed")
    return True


def test_wifi_mlo():
    """Test WiFi MLO components."""
    print("Testing WiFi MLO Components...")
    
    from tsn_wifi_scheduler import WiFiMLO, WiFiBand
    
    # Create WiFi MLO
    wifi = WiFiMLO(cycle_time=1000.0)
    
    # Check bands
    assert len(wifi.bands) == 3
    assert wifi.bands[0].frequency == 2.4
    assert wifi.bands[1].frequency == 5.0
    assert wifi.bands[2].frequency == 6.0
    
    # Check transmission rates
    assert wifi.bands[0].get_transmission_rate() == 72   # 2.4 GHz
    assert wifi.bands[1].get_transmission_rate() == 433  # 5 GHz
    assert wifi.bands[2].get_transmission_rate() == 600  # 6 GHz
    
    # Test channel state
    channel_state = wifi.get_channel_state()
    assert len(channel_state) == 9  # 3 bands × 3 features
    
    # Test scheduling
    scheduled = wifi.schedule_flow(0, 2, 100.0, 1500)  # flow 0, band 2, start 100μs, 1500 bytes
    assert scheduled == True
    
    # Test statistics
    stats = wifi.get_statistics()
    assert stats['num_bands'] == 3
    assert stats['scheduled_flows'] == 1
    
    print("✓ WiFi MLO tests passed")
    return True


def test_load_aware_scheduler():
    """Test load-aware scheduling algorithm."""
    print("Testing Load-Aware Scheduler...")
    
    from tsn_wifi_scheduler import TSNNetwork, Flow, LoadAwareScheduler
    
    # Create simple network
    network = TSNNetwork(cycle_time=1000.0)
    for i in range(4):
        network.add_switch(i)
    network.add_link(0, 0, 1)
    network.add_link(1, 1, 2)
    network.add_link(2, 2, 3)
    
    # Create flows
    flows = []
    for i in range(5):
        flow = Flow(flow_id=i, src=0, dst=3, size=1500, start_offset=i)
        flows.append(flow)
        network.add_flow(flow)
    
    # Schedule
    scheduler = LoadAwareScheduler(network)
    num_scheduled, max_load = scheduler.schedule_flows(flows)
    
    assert num_scheduled > 0
    assert max_load > 0
    
    # Check results
    results = scheduler.get_scheduling_results()
    assert results['total_flows'] == 5
    assert results['scheduled_flows'] == num_scheduled
    assert results['scheduling_rate'] > 0
    
    print(f"✓ Load-Aware Scheduler tests passed ({num_scheduled}/5 flows scheduled)")
    return True


def test_attention_ddpg():
    """Test attention-based DDPG components."""
    print("Testing Attention-based DDPG...")
    
    from tsn_wifi_scheduler import AttentionDDPG
    import torch
    
    # Create DDPG agent
    state_dim = 12
    num_flows = 10
    ddpg = AttentionDDPG(state_dim=state_dim, num_flows=num_flows, hidden_dim=64)
    
    # Test forward pass
    dummy_state = np.random.rand(num_flows, state_dim).astype(np.float32)
    action_probs = ddpg.select_action(dummy_state, explore=False)
    
    assert action_probs.shape == (num_flows, 3)  # 3 bands
    
    # Check probabilities sum to 1
    for i in range(num_flows):
        prob_sum = np.sum(action_probs[i])
        assert abs(prob_sum - 1.0) < 0.01  # Allow small numerical error
    
    # Test action sampling
    sampled_bands = ddpg.sample_action(action_probs)
    assert len(sampled_bands) == num_flows
    assert all(0 <= b <= 2 for b in sampled_bands)
    
    # Test replay buffer
    ddpg.replay_buffer.push(dummy_state, action_probs, 1.0, dummy_state, False)
    assert len(ddpg.replay_buffer) == 1
    
    print("✓ Attention-based DDPG tests passed")
    return True


def test_integrated_scheduler():
    """Test integrated TSN-WiFi scheduler."""
    print("Testing Integrated Scheduler...")
    
    from tsn_wifi_scheduler import TSNNetwork, WiFiMLO, Flow
    from tsn_wifi_scheduler.integrated_scheduler import IntegratedScheduler
    
    # Create network
    network = TSNNetwork(cycle_time=1000.0)
    for i in range(4):
        network.add_switch(i)
    for i in range(3):
        network.add_link(i, i, i+1)
    
    # Create WiFi
    wifi = WiFiMLO(cycle_time=1000.0)
    
    # Create flows
    flows = []
    for i in range(10):
        flow = Flow(flow_id=i, src=np.random.randint(0, 4), 
                   dst=np.random.randint(0, 4), size=1500, start_offset=i % 5)
        flows.append(flow)
        network.add_flow(flow)
    
    # Test greedy scheduling
    scheduler = IntegratedScheduler(network, wifi, use_drl=False)
    results = scheduler.schedule(flows)
    
    assert 'total_scheduled' in results
    assert 'scheduling_rate' in results
    assert 'avg_reliability' in results
    assert results['total_scheduled'] >= 0
    assert 0 <= results['scheduling_rate'] <= 1
    
    print(f"✓ Integrated Scheduler tests passed ({results['total_scheduled']}/10 flows)")
    return True


def test_topologies():
    """Test different network topologies."""
    print("Testing Different Topologies...")
    
    from tsn_wifi_scheduler import TSNNetwork
    
    # Test ladder topology
    network = TSNNetwork(cycle_time=1000.0)
    for i in range(8):
        network.add_switch(i)
    
    link_id = 0
    # Horizontal
    for i in range(3):
        network.add_link(link_id, i, i+1)
        link_id += 1
    for i in range(4, 7):
        network.add_link(link_id, i, i+1)
        link_id += 1
    # Vertical
    for i in range(4):
        network.add_link(link_id, i, i+4)
        link_id += 1
    
    assert len(network.switches) == 8
    assert len(network.links) == 10
    
    # Test path finding in ladder
    path = network.get_shortest_path(0, 7)
    assert len(path) > 0
    
    print("✓ Topology tests passed")
    return True


def test_constraints():
    """Test constraint satisfaction."""
    print("Testing Constraint Satisfaction...")
    
    from tsn_wifi_scheduler import TSNNetwork, WiFiMLO, Flow
    from tsn_wifi_scheduler.integrated_scheduler import IntegratedScheduler
    
    network = TSNNetwork(cycle_time=500.0)  # Shorter cycle
    for i in range(3):
        network.add_switch(i)
    network.add_link(0, 0, 1)
    network.add_link(1, 1, 2)
    
    wifi = WiFiMLO(cycle_time=500.0)
    
    # Create flows that should fit
    flows = []
    for i in range(3):
        flow = Flow(flow_id=i, src=0, dst=2, size=1500, start_offset=0)
        flows.append(flow)
        network.add_flow(flow)
    
    scheduler = IntegratedScheduler(network, wifi, use_drl=False)
    results = scheduler.schedule(flows)
    
    # Verify cycle constraint is respected
    for flow in flows:
        if flow.band is not None:
            band = wifi.get_band(flow.band)
            for flow_id, start, end in band.scheduled_flows:
                assert end <= wifi.cycle_time, f"Flow {flow_id} violates cycle constraint"
    
    print("✓ Constraint tests passed")
    return True


def run_all_tests():
    """Run all test suites."""
    print("="*80)
    print("COMPREHENSIVE TEST SUITE")
    print("Paper: Pay Attention to Network (ICCPS 2025)")
    print("="*80)
    print()
    
    tests = [
        ("TSN Network", test_tsn_network),
        ("WiFi MLO", test_wifi_mlo),
        ("Load-Aware Scheduler", test_load_aware_scheduler),
        ("Attention DDPG", test_attention_ddpg),
        ("Integrated Scheduler", test_integrated_scheduler),
        ("Topologies", test_topologies),
        ("Constraints", test_constraints),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} tests FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} tests FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("="*80)
    print(f"RESULTS: {passed}/{len(tests)} test suites passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} test suite(s) failed")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
