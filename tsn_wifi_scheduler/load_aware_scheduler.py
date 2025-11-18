"""
Load-Aware TSN Scheduler
Implements the load-aware routing and scheduling algorithm for TSN domain.
Based on the bucket effect principle from the paper.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .tsn_network import TSNNetwork, Flow, TSNLink


class LoadAwareScheduler:
    """
    Load-aware scheduler for TSN domain.
    
    Key principles:
    1. Bucket Effect: Evenly distribute load across network links
    2. Load-aware routing: Select paths that balance network load
    3. Constraint satisfaction: Ensure timing and bandwidth constraints
    
    The scheduler aims to minimize the maximum link utilization (bucket effect)
    while satisfying all timing constraints.
    """
    
    def __init__(self, network: TSNNetwork):
        """
        Initialize the load-aware scheduler.
        
        Args:
            network: TSN network to schedule
        """
        self.network = network
        self.link_loads: Dict[int, float] = {}
        
    def _calculate_path_load(self, path: List[int]) -> float:
        """
        Calculate the total load of a path.
        
        Args:
            path: List of link IDs
            
        Returns:
            Sum of link loads in the path
        """
        return sum(self.link_loads.get(link_id, 0.0) for link_id in path)
    
    def _get_alternative_paths(self, src: int, dst: int, 
                               max_paths: int = 3) -> List[List[int]]:
        """
        Find alternative paths between source and destination.
        
        Args:
            src: Source switch ID
            dst: Destination switch ID
            max_paths: Maximum number of paths to find
            
        Returns:
            List of paths (each path is a list of link IDs)
        """
        if src == dst:
            return [[]]
        
        paths = []
        
        # BFS to find multiple paths
        queue = [(src, [], {src})]
        
        while queue and len(paths) < max_paths:
            current, path, visited = queue.pop(0)
            
            if current == dst:
                paths.append(path)
                continue
            
            if current in self.network.adjacency:
                for neighbor, link_id in self.network.adjacency[current]:
                    if neighbor not in visited:
                        new_visited = visited.copy()
                        new_visited.add(neighbor)
                        queue.append((neighbor, path + [link_id], new_visited))
        
        if not paths:
            # Fallback to shortest path
            shortest = self.network.get_shortest_path(src, dst)
            if shortest:
                paths.append(shortest)
        
        return paths
    
    def _select_best_path(self, paths: List[List[int]]) -> Optional[List[int]]:
        """
        Select the best path based on load balancing (bucket effect).
        
        The best path is the one that results in the lowest maximum link load,
        implementing the bucket effect principle.
        
        Args:
            paths: List of candidate paths
            
        Returns:
            Best path or None if no valid path exists
        """
        if not paths:
            return None
        
        best_path = None
        best_max_load = float('inf')
        
        for path in paths:
            # Calculate what the maximum link load would be if we use this path
            max_load = 0
            for link_id in path:
                current_load = self.link_loads.get(link_id, 0.0)
                max_load = max(max_load, current_load)
            
            # Choose path with lowest maximum load (bucket effect)
            if max_load < best_max_load:
                best_max_load = max_load
                best_path = path
        
        return best_path
    
    def _schedule_flow_on_path(self, flow: Flow, path: List[int]) -> bool:
        """
        Schedule a flow on the given path.
        
        Args:
            flow: Flow to schedule
            path: Path (list of link IDs) to schedule on
            
        Returns:
            True if successfully scheduled, False otherwise
        """
        if not path:
            return True  # Local flow (src == dst)
        
        # Calculate transmission time
        link_speed = 1000  # Mbps (1 Gbps)
        trans_time = flow.get_transmission_time(link_speed)
        
        # Try to schedule on each link in the path
        current_time = flow.start_offset * self.network.time_slot_duration
        offsets = {}
        
        for link_id in path:
            link = self.network.links[link_id]
            
            # Find earliest available time on this link
            scheduled = False
            for attempt_time in np.arange(current_time, self.network.cycle_time, 
                                         self.network.time_slot_duration):
                if link.is_available(attempt_time, trans_time):
                    if link.schedule_flow(flow.flow_id, attempt_time, trans_time):
                        offsets[link_id] = attempt_time
                        current_time = attempt_time + trans_time
                        scheduled = True
                        break
            
            if not scheduled:
                # Cannot schedule on this link, rollback
                for prev_link_id in offsets:
                    prev_link = self.network.links[prev_link_id]
                    prev_link.scheduled_flows = [
                        s for s in prev_link.scheduled_flows if s[0] != flow.flow_id
                    ]
                return False
        
        # Successfully scheduled on all links
        flow.path = path
        flow.offsets = offsets
        flow.scheduled = True
        
        # Update link loads
        for link_id in path:
            link = self.network.links[link_id]
            self.link_loads[link_id] = link.get_load(self.network.cycle_time)
        
        return True
    
    def schedule_flows(self, flows: List[Flow]) -> Tuple[int, float]:
        """
        Schedule all flows using load-aware algorithm.
        
        Algorithm:
        1. Initialize link loads
        2. For each flow:
           a. Find alternative paths
           b. Select best path based on load balancing
           c. Schedule flow on selected path
           d. Update link loads
        
        Args:
            flows: List of flows to schedule
            
        Returns:
            Tuple of (num_scheduled, max_link_load)
        """
        # Initialize link loads
        self.link_loads = {link_id: 0.0 for link_id in self.network.links}
        
        # Sort flows by start offset (earlier flows first)
        sorted_flows = sorted(flows, key=lambda f: f.start_offset)
        
        num_scheduled = 0
        
        for flow in sorted_flows:
            # Find alternative paths
            paths = self._get_alternative_paths(flow.src, flow.dst)
            
            # Select best path (lowest maximum load - bucket effect)
            best_path = self._select_best_path(paths)
            
            if best_path is None:
                continue
            
            # Try to schedule on the selected path
            if self._schedule_flow_on_path(flow, best_path):
                num_scheduled += 1
        
        # Calculate final maximum link load
        max_load = max(self.link_loads.values()) if self.link_loads else 0.0
        
        return num_scheduled, max_load
    
    def get_load_variance(self) -> float:
        """
        Calculate the variance of link loads.
        Lower variance indicates better load balancing (bucket effect).
        
        Returns:
            Variance of link loads
        """
        if not self.link_loads:
            return 0.0
        
        loads = list(self.link_loads.values())
        return np.var(loads)
    
    def get_scheduling_results(self) -> Dict:
        """
        Get detailed scheduling results.
        
        Returns:
            Dictionary with scheduling statistics
        """
        total_flows = len(self.network.flows)
        scheduled_flows = sum(1 for f in self.network.flows.values() if f.scheduled)
        
        return {
            'total_flows': total_flows,
            'scheduled_flows': scheduled_flows,
            'scheduling_rate': scheduled_flows / total_flows if total_flows > 0 else 0,
            'max_link_load': max(self.link_loads.values()) if self.link_loads else 0,
            'avg_link_load': np.mean(list(self.link_loads.values())) if self.link_loads else 0,
            'load_variance': self.get_load_variance(),
        }
