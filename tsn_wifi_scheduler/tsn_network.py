"""
TSN Network Model
Implements Time-Sensitive Network components including switches, links, and flows.
Based on the TSN model from the paper.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Flow:
    """
    Represents a flow in the TSN-WiFi network.
    
    Attributes:
        flow_id: Unique identifier for the flow
        src: Source device in TSN domain
        dst: Destination device in Wi-Fi domain
        size: Frame length in bytes (typically 1500 bytes/MTU)
        start_offset: Start time offset in time slots
        band: Selected band in WiFi domain (0: 2.4GHz, 1: 5GHz, 2: 6GHz, None: not assigned)
        reliability: Reliability of the flow (0-1)
        path: List of TSN links forming the path
        offsets: Dict mapping link_id to offset time
    """
    flow_id: int
    src: int
    dst: int
    size: int = 1500  # bytes (MTU)
    start_offset: int = 0  # time slots
    band: Optional[int] = None
    reliability: float = 1.0
    path: List[int] = field(default_factory=list)
    offsets: Dict[int, int] = field(default_factory=dict)
    scheduled: bool = False
    
    def get_transmission_time(self, link_speed: int = 1000) -> float:
        """
        Calculate transmission time in microseconds.
        
        Args:
            link_speed: Link speed in Mbps (default 1000 for 1Gbps)
        
        Returns:
            Transmission time in microseconds
        """
        # Time = (size in bits) / (speed in Mbps) = size * 8 / speed microseconds
        return (self.size * 8) / link_speed


@dataclass
class TSNLink:
    """
    Represents a link in the TSN network.
    
    Attributes:
        link_id: Unique identifier
        src_switch: Source switch ID
        dst_switch: Destination switch ID
        speed: Link speed in Mbps
        scheduled_flows: List of (flow_id, start_time, end_time) tuples
    """
    link_id: int
    src_switch: int
    dst_switch: int
    speed: int = 1000  # Mbps
    scheduled_flows: List[Tuple[int, float, float]] = field(default_factory=list)
    
    def get_load(self, cycle_time: float) -> float:
        """Calculate link utilization as a percentage."""
        total_time = sum(end - start for _, start, end in self.scheduled_flows)
        return (total_time / cycle_time) * 100 if cycle_time > 0 else 0
    
    def is_available(self, start_time: float, duration: float) -> bool:
        """Check if the link is available during the specified time window."""
        end_time = start_time + duration
        for _, flow_start, flow_end in self.scheduled_flows:
            # Check for overlap
            if not (end_time <= flow_start or start_time >= flow_end):
                return False
        return True
    
    def schedule_flow(self, flow_id: int, start_time: float, duration: float) -> bool:
        """
        Schedule a flow on this link.
        
        Returns:
            True if successfully scheduled, False otherwise
        """
        if self.is_available(start_time, duration):
            end_time = start_time + duration
            self.scheduled_flows.append((flow_id, start_time, end_time))
            self.scheduled_flows.sort(key=lambda x: x[1])  # Sort by start time
            return True
        return False


@dataclass
class TSNSwitch:
    """
    Represents a TSN switch with Time-Aware Shaper (TAS).
    
    Attributes:
        switch_id: Unique identifier
        num_ports: Number of ports
        time_slot_duration: Duration of each time slot in microseconds
        gcl: Gate Control List - dict mapping port to list of (queue, start, end) tuples
    """
    switch_id: int
    num_ports: int = 8
    time_slot_duration: float = 12.0  # microseconds (MTU transmission time at 1Gbps)
    gcl: Dict[int, List[Tuple[int, float, float]]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize GCL for each port with 8 queues (Q0-Q7)
        for port in range(self.num_ports):
            self.gcl[port] = []


class TSNNetwork:
    """
    Represents the complete TSN network topology.
    
    Attributes:
        switches: Dict of switch_id -> TSNSwitch
        links: Dict of link_id -> TSNLink
        adjacency: Dict mapping switch_id to list of (neighbor_switch_id, link_id)
        cycle_time: Cycle time in microseconds
        time_slot_duration: Duration of each time slot in microseconds
    """
    
    def __init__(self, cycle_time: float = 1000.0):
        """
        Initialize TSN network.
        
        Args:
            cycle_time: Cycle time in microseconds (default 1000 μs)
        """
        self.switches: Dict[int, TSNSwitch] = {}
        self.links: Dict[int, TSNLink] = {}
        self.adjacency: Dict[int, List[Tuple[int, int]]] = {}
        self.cycle_time = cycle_time
        self.time_slot_duration = 12.0  # microseconds
        self.flows: Dict[int, Flow] = {}
        
    def add_switch(self, switch_id: int, num_ports: int = 8) -> TSNSwitch:
        """Add a switch to the network."""
        switch = TSNSwitch(switch_id=switch_id, num_ports=num_ports)
        self.switches[switch_id] = switch
        if switch_id not in self.adjacency:
            self.adjacency[switch_id] = []
        return switch
    
    def add_link(self, link_id: int, src_switch: int, dst_switch: int, 
                 speed: int = 1000) -> TSNLink:
        """Add a bidirectional link between two switches."""
        link = TSNLink(link_id=link_id, src_switch=src_switch, 
                      dst_switch=dst_switch, speed=speed)
        self.links[link_id] = link
        
        # Update adjacency list
        if src_switch not in self.adjacency:
            self.adjacency[src_switch] = []
        if dst_switch not in self.adjacency:
            self.adjacency[dst_switch] = []
            
        self.adjacency[src_switch].append((dst_switch, link_id))
        self.adjacency[dst_switch].append((src_switch, link_id))
        
        return link
    
    def add_flow(self, flow: Flow) -> None:
        """Add a flow to the network."""
        self.flows[flow.flow_id] = flow
    
    def get_shortest_path(self, src: int, dst: int) -> List[int]:
        """
        Find shortest path between two switches using BFS.
        
        Returns:
            List of link IDs forming the path
        """
        if src == dst:
            return []
        
        # BFS
        queue = [(src, [])]
        visited = {src}
        
        while queue:
            current, path = queue.pop(0)
            
            if current == dst:
                return path
            
            if current in self.adjacency:
                for neighbor, link_id in self.adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [link_id]))
        
        return []  # No path found
    
    def calculate_link_loads(self) -> Dict[int, float]:
        """Calculate load (utilization percentage) for each link."""
        loads = {}
        for link_id, link in self.links.items():
            loads[link_id] = link.get_load(self.cycle_time)
        return loads
    
    def reset_schedules(self) -> None:
        """Reset all schedules in the network."""
        for link in self.links.values():
            link.scheduled_flows = []
        for switch in self.switches.values():
            for port in switch.gcl:
                switch.gcl[port] = []
        for flow in self.flows.values():
            flow.scheduled = False
            flow.path = []
            flow.offsets = {}
    
    def get_network_stats(self) -> Dict:
        """Get statistics about the network."""
        loads = self.calculate_link_loads()
        return {
            'num_switches': len(self.switches),
            'num_links': len(self.links),
            'num_flows': len(self.flows),
            'avg_link_load': np.mean(list(loads.values())) if loads else 0,
            'max_link_load': max(loads.values()) if loads else 0,
            'scheduled_flows': sum(1 for f in self.flows.values() if f.scheduled),
        }
