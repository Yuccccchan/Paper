"""
TSN Network topology and management
Based on Cao et al. 2025 paper implementation
"""

import networkx as nx
import numpy as np
from .switch import TSNSwitch
from .tt_flow import TTFlow

class TSNNetwork:
    """
    Time-Sensitive Network (TSN) representation with switches and links.
    
    Supports centralized management for routing and scheduling of TT flows.
    """
    
    def __init__(self, slot_duration=0.2, link_bandwidth=100):
        """
        Initialize TSN network.
        
        Args:
            slot_duration: Duration of time slots in ms (0.15, 0.2, or 0.3)
            link_bandwidth: Link bandwidth in Mb/s (default: 100)
        """
        self.slot_duration = slot_duration  # ms
        self.link_bandwidth = link_bandwidth  # Mb/s
        
        # Network topology as directed graph
        self.graph = nx.DiGraph()
        
        # TSN switches indexed by ID
        self.switches = {}
        
        # TT flows to be scheduled
        self.flows = {}
        
        # Hyperperiod calculation (LCM of all flow periods)
        self.hyperperiod = None
        self.max_slots = None
        
    def add_switch(self, switch_id, num_ports=8):
        """Add a TSN switch to the network."""
        switch = TSNSwitch(switch_id, num_ports)
        self.switches[switch_id] = switch
        self.graph.add_node(switch_id)
        
    def add_link(self, from_switch, to_switch, from_port=None, to_port=None):
        """
        Add a bidirectional link between two switches.
        
        Args:
            from_switch: Source switch ID
            to_switch: Destination switch ID
            from_port: Port on source switch (auto-assigned if None)
            to_port: Port on destination switch (auto-assigned if None)
        """
        # Add edge in graph
        self.graph.add_edge(from_switch, to_switch)
        self.graph.add_edge(to_switch, from_switch)
        
        # Configure ports on switches
        if from_switch in self.switches and to_switch in self.switches:
            # Find available port on from_switch
            if from_port is None:
                from_port = self._find_available_port(from_switch)
            # Find available port on to_switch  
            if to_port is None:
                to_port = self._find_available_port(to_switch)
                
            self.switches[from_switch].connect_port(from_port, to_switch)
            self.switches[to_switch].connect_port(to_port, from_switch)
    
    def _find_available_port(self, switch_id):
        """Find an available port on a switch."""
        switch = self.switches[switch_id]
        for port_id in range(switch.num_ports):
            if switch.ports[port_id]['connected_to'] is None:
                return port_id
        return None
    
    def add_flow(self, flow):
        """Add a TT flow to the network."""
        self.flows[flow.flow_id] = flow
        self._update_hyperperiod()
    
    def _update_hyperperiod(self):
        """Update hyperperiod based on all flow periods."""
        if not self.flows:
            self.hyperperiod = None
            self.max_slots = None
            return
            
        import math
        periods = [flow.period for flow in self.flows.values()]
        
        # Calculate LCM of all periods
        def lcm(a, b):
            return abs(a * b) // math.gcd(int(a * 1000), int(b * 1000)) * 1000 / 1000
        
        self.hyperperiod = periods[0]
        for period in periods[1:]:
            self.hyperperiod = lcm(self.hyperperiod, period)
        
        # Calculate max number of slots in hyperperiod
        self.max_slots = int(self.hyperperiod / self.slot_duration)
    
    def get_shortest_path(self, source, destination):
        """
        Get shortest path between two switches using Dijkstra's algorithm.
        
        Args:
            source: Source switch ID
            destination: Destination switch ID
            
        Returns:
            List of switch IDs forming the path, or None if no path exists
        """
        try:
            return nx.shortest_path(self.graph, source, destination)
        except nx.NetworkXNoPath:
            return None
    
    def get_all_paths(self, source, destination, cutoff=None):
        """
        Get all simple paths between two switches.
        
        Args:
            source: Source switch ID
            destination: Destination switch ID
            cutoff: Maximum path length (None for no limit)
            
        Returns:
            List of paths, where each path is a list of switch IDs
        """
        try:
            paths = nx.all_simple_paths(self.graph, source, destination, cutoff=cutoff)
            return list(paths)
        except nx.NetworkXNoPath:
            return []
    
    def get_link_utilization(self, from_switch, to_switch):
        """
        Calculate utilization of a link (averaged across all slots).
        
        Args:
            from_switch: Source switch ID
            to_switch: Destination switch ID
            
        Returns:
            Utilization rate (0.0 to 1.0)
        """
        if from_switch not in self.switches or self.max_slots is None:
            return 0.0
        
        switch = self.switches[from_switch]
        port_id = switch.get_port_to_neighbor(to_switch)
        
        if port_id is None:
            return 0.0
            
        return switch.get_slot_utilization(port_id, self.max_slots)
    
    def reset_schedules(self):
        """Reset all flow schedules and slot allocations."""
        for flow in self.flows.values():
            flow.route = []
            flow.schedule = {}
            flow.successfully_scheduled = False
        
        for switch in self.switches.values():
            for port_id in switch.ports:
                switch.ports[port_id]['slot_assignments'] = {}
                switch.ports[port_id]['gcl'] = {}
    
    def get_network_state(self):
        """
        Get current network state representation.
        
        Returns:
            Dictionary containing network topology and current slot allocations
        """
        state = {
            'num_switches': len(self.switches),
            'num_flows': len(self.flows),
            'hyperperiod': self.hyperperiod,
            'max_slots': self.max_slots,
            'topology': nx.to_dict_of_lists(self.graph),
        }
        return state
    
    def __repr__(self):
        return (f"TSNNetwork(switches={len(self.switches)}, "
                f"flows={len(self.flows)}, slot_duration={self.slot_duration}ms)")
