#!/usr/bin/env python3
"""
Detailed Analysis Script
Compares different scheduling approaches and analyzes performance.
"""

import numpy as np
from tsn_wifi_scheduler import TSNNetwork, Flow, WiFiMLO
from tsn_wifi_scheduler.integrated_scheduler import IntegratedScheduler
from tsn_wifi_scheduler.load_aware_scheduler import LoadAwareScheduler


def create_ladder_topology() -> TSNNetwork:
    """Create a ladder topology (2x4 grid)."""
    network = TSNNetwork(cycle_time=1000.0)
    
    # 0 -- 1 -- 2 -- 3
    # |    |    |    |
    # 4 -- 5 -- 6 -- 7
    for i in range(8):
        network.add_switch(i)
    
    # Horizontal links
    link_id = 0
    for i in range(3):
        network.add_link(link_id, i, i+1)
        link_id += 1
    for i in range(4, 7):
        network.add_link(link_id, i, i+1)
        link_id += 1
    
    # Vertical links
    for i in range(4):
        network.add_link(link_id, i, i+4)
        link_id += 1
    
    return network


def generate_flows(network: TSNNetwork, num_flows: int, seed: int = 42) -> list:
    """Generate random flows."""
    np.random.seed(seed)
    flows = []
    switches = list(network.switches.keys())
    
    for i in range(num_flows):
        src = np.random.choice(switches)
        dst = np.random.choice([s for s in switches if s != src])
        start_offset = np.random.randint(0, 5)
        
        flow = Flow(
            flow_id=i,
            src=src,
            dst=dst,
            size=1500,
            start_offset=start_offset
        )
        flows.append(flow)
        network.add_flow(flow)
    
    return flows


def compare_scheduling_approaches():
    """Compare different scheduling approaches."""
    print("=" * 80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Test with different flow counts
    flow_counts = [25, 50, 75]
    
    results_table = []
    
    for num_flows in flow_counts:
        print(f"\n{'='*80}")
        print(f"Testing with {num_flows} flows")
        print('='*80)
        
        # Create network
        network = create_ladder_topology()
        wifi = WiFiMLO(cycle_time=1000.0)
        flows = generate_flows(network, num_flows)
        
        # Test 1: TSN Only - Load-Aware
        print(f"\n1. TSN Load-Aware Scheduling")
        tsn_scheduler = LoadAwareScheduler(network)
        tsn_scheduled, max_load = tsn_scheduler.schedule_flows(flows)
        tsn_stats = tsn_scheduler.get_scheduling_results()
        
        print(f"   Scheduled: {tsn_scheduled}/{num_flows} ({tsn_stats['scheduling_rate']*100:.1f}%)")
        print(f"   Max link load: {max_load:.2f}%")
        print(f"   Avg link load: {tsn_stats['avg_link_load']:.2f}%")
        print(f"   Load variance: {tsn_stats['load_variance']:.2f}")
        
        # Reset for next test
        network.reset_schedules()
        
        # Test 2: Integrated - Greedy
        print(f"\n2. Integrated Scheduling (Greedy WiFi)")
        wifi.reset_schedules()
        scheduler_greedy = IntegratedScheduler(network, wifi, use_drl=False)
        results_greedy = scheduler_greedy.schedule(flows)
        
        print(f"   Total scheduled: {results_greedy['total_scheduled']}/{num_flows} "
              f"({results_greedy['scheduling_rate']*100:.1f}%)")
        print(f"   Average reliability: {results_greedy['avg_reliability']:.3f}")
        
        wifi_stats = results_greedy['wifi_stats']
        print(f"   WiFi band utilization:")
        for i, band in enumerate(wifi.bands):
            util = wifi_stats['band_utilizations'][i]
            rel = wifi_stats['band_reliabilities'][i]
            print(f"     Band {i} ({band.frequency} GHz): {util:.1f}% util, {rel:.3f} reliability")
        
        # Store results
        results_table.append({
            'flows': num_flows,
            'tsn_rate': tsn_stats['scheduling_rate'],
            'greedy_rate': results_greedy['scheduling_rate'],
            'greedy_reliability': results_greedy['avg_reliability'],
            'load_variance': tsn_stats['load_variance'],
        })
        
        # Reset for next iteration
        network.reset_schedules()
        wifi.reset_schedules()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Flows':<10} {'TSN Rate':<12} {'Greedy Rate':<14} {'Reliability':<14} {'Load Var':<12}")
    print("-"*80)
    
    for result in results_table:
        print(f"{result['flows']:<10} "
              f"{result['tsn_rate']*100:>10.1f}% "
              f"{result['greedy_rate']*100:>12.1f}% "
              f"{result['greedy_reliability']:>12.3f} "
              f"{result['load_variance']:>12.2f}")
    
    print("="*80)
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM PAPER")
    print("="*80)
    print("""
1. Load-Aware vs Shortest Path:
   - Load-aware routing reduces maximum link utilization (bucket effect)
   - Lower load variance indicates better load balancing
   - Enables more flows to be scheduled

2. Attention-based DDPG vs MLP-based:
   - Self-attention captures flow dependencies
   - Better band selection under varying channel conditions
   - Higher reliability and scheduling success rate

3. WiFi Band Selection:
   - 6 GHz band: Highest reliability (0.96), preferred for critical flows
   - 5 GHz band: Medium reliability (0.92), balanced choice
   - 2.4 GHz band: Lowest reliability (0.85), used when others are congested

4. Cross-Domain Constraints:
   - TSN scheduling directly impacts WiFi scheduling
   - FIFO constraint ensures fairness at AP
   - Cycle time constraint limits maximum schedulable flows

5. Scalability:
   - Network-wise scheduling scales better than flow-wise
   - Attention mechanism handles variable number of flows
   - Load balancing becomes more critical with more flows
    """)


def demonstrate_attention_mechanism():
    """Demonstrate the attention mechanism."""
    print("\n" + "="*80)
    print("ATTENTION MECHANISM DEMONSTRATION")
    print("="*80)
    print("""
The self-attention mechanism captures dependencies among flows:

1. Query (Q): "What resources does this flow need?"
2. Key (K): "What resources are each flow competing for?"
3. Value (V): "What is the state/priority of each flow?"

Attention Score = softmax(Q · K^T / √d_k)
Output = Attention Score · V

Example: If Flow A and Flow B both need Band 2 at similar times,
their attention scores will be high, helping the network learn that
they are in conflict and need careful coordination.

This is analogous to how words in a sentence relate to each other
(e.g., "The cat sat on the mat" - "cat" and "mat" are related through "sat").

Without attention (MLP): Each flow is treated independently → suboptimal
With attention: Flow dependencies are learned → better scheduling
    """)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TSN-WiFi Scheduler - Detailed Analysis")
    print("Paper: Pay Attention to Network (ICCPS 2025)")
    print("="*80)
    
    # Run comparison
    compare_scheduling_approaches()
    
    # Demonstrate attention
    demonstrate_attention_mechanism()
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
