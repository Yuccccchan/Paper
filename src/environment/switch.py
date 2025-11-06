"""
TSN Switch with Time-Aware Shaper (TAS) support
Based on IEEE 802.1Qbv standard as described in Cao et al. 2025
"""

import numpy as np

class TSNSwitch:
    """
    Represents a TSN switch with Time-Aware Shaper (TAS) capability.
    
    The switch manages egress ports with multiple priority queues and 
    gate control lists (GCL) for deterministic traffic shaping.
    """
    
    def __init__(self, switch_id, num_ports=8):
        """
        Initialize a TSN switch.
        
        Args:
            switch_id: Unique identifier for the switch
            num_ports: Number of egress ports
        """
        self.switch_id = switch_id
        self.num_ports = num_ports
        
        # Each port has multiple queues (Q0 for TT, Q1-Q2 for AVB, Q7 for BE)
        # We focus on Q0 (TT queue) for this implementation
        self.ports = {}
        for port_id in range(num_ports):
            self.ports[port_id] = {
                'connected_to': None,  # Adjacent switch ID
                'gcl': {},  # Gate Control List: {slot_id: 'open'/'close'}
                'slot_assignments': {}  # {slot_id: flow_id or None}
            }
    
    def connect_port(self, port_id, adjacent_switch_id):
        """Connect a port to an adjacent switch."""
        if port_id < self.num_ports:
            self.ports[port_id]['connected_to'] = adjacent_switch_id
    
    def is_slot_available(self, port_id, slot_id):
        """
        Check if a time slot is available on a port.
        
        Args:
            port_id: Port identifier
            slot_id: Time slot identifier
            
        Returns:
            True if slot is available, False otherwise
        """
        if port_id not in self.ports:
            return False
        
        slot_assignments = self.ports[port_id]['slot_assignments']
        return slot_id not in slot_assignments or slot_assignments[slot_id] is None
    
    def allocate_slot(self, port_id, slot_id, flow_id):
        """
        Allocate a time slot to a flow on a specific port.
        
        Args:
            port_id: Port identifier
            slot_id: Time slot identifier
            flow_id: Flow identifier
            
        Returns:
            True if allocation successful, False otherwise
        """
        if self.is_slot_available(port_id, slot_id):
            self.ports[port_id]['slot_assignments'][slot_id] = flow_id
            self.ports[port_id]['gcl'][slot_id] = 'open'
            return True
        return False
    
    def release_slot(self, port_id, slot_id):
        """Release a time slot on a port."""
        if port_id in self.ports:
            self.ports[port_id]['slot_assignments'][slot_id] = None
            self.ports[port_id]['gcl'][slot_id] = 'close'
    
    def get_slot_utilization(self, port_id, max_slots):
        """
        Calculate slot utilization rate for a port.
        
        Args:
            port_id: Port identifier
            max_slots: Maximum number of slots in a hyperperiod
            
        Returns:
            Utilization rate (0.0 to 1.0)
        """
        if port_id not in self.ports:
            return 0.0
        
        occupied_slots = sum(1 for slot_id, flow in 
                            self.ports[port_id]['slot_assignments'].items()
                            if flow is not None)
        return occupied_slots / max_slots if max_slots > 0 else 0.0
    
    def get_neighbors(self):
        """Get list of neighboring switch IDs."""
        neighbors = []
        for port_id, port_info in self.ports.items():
            if port_info['connected_to'] is not None:
                neighbors.append(port_info['connected_to'])
        return neighbors
    
    def get_port_to_neighbor(self, neighbor_id):
        """Get port ID connected to a specific neighbor."""
        for port_id, port_info in self.ports.items():
            if port_info['connected_to'] == neighbor_id:
                return port_id
        return None
    
    def __repr__(self):
        return f"TSNSwitch(id={self.switch_id}, ports={self.num_ports})"
