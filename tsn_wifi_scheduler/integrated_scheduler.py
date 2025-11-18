"""
Integrated TSN-WiFi Scheduler
Combines TSN load-aware scheduling with WiFi attention-based DDPG scheduling.
"""

import numpy as np
from typing import List, Dict, Tuple
from .tsn_network import TSNNetwork, Flow
from .wifi_network import WiFiMLO
from .load_aware_scheduler import LoadAwareScheduler
from .attention_ddpg import AttentionDDPG


class IntegratedScheduler:
    """
    Integrated scheduler for cross-domain TSN-WiFi networks.
    
    Architecture:
    1. TSN Domain: Load-aware scheduling for spatial-temporal resource allocation
    2. WiFi Domain: Attention-based DDPG for frequential (band) selection
    3. Cross-domain constraint enforcement
    """
    
    def __init__(self, tsn_network: TSNNetwork, wifi_mlo: WiFiMLO,
                 use_drl: bool = True):
        """
        Initialize integrated scheduler.
        
        Args:
            tsn_network: TSN network
            wifi_mlo: WiFi MLO access point
            use_drl: Whether to use DRL for WiFi scheduling (vs. greedy)
        """
        self.tsn_network = tsn_network
        self.wifi_mlo = wifi_mlo
        self.use_drl = use_drl
        
        # TSN scheduler
        self.tsn_scheduler = LoadAwareScheduler(tsn_network)
        
        # WiFi scheduler (DRL)
        if use_drl:
            # State: [arrival_time, packet_size, TSN_delay] + channel_state (9 features)
            state_dim = 3 + 9  # Per flow features + channel state (repeated)
            self.wifi_scheduler = AttentionDDPG(
                state_dim=state_dim,
                num_flows=50,  # Will be adjusted based on actual flows
                hidden_dim=64,
                lr=0.0004,
                gamma=0.9,
                tau=0.0004,
            )
        else:
            self.wifi_scheduler = None
    
    def schedule(self, flows: List[Flow]) -> Dict:
        """
        Schedule flows across TSN and WiFi domains.
        
        Process:
        1. Schedule flows in TSN domain (spatial-temporal)
        2. Calculate arrival times at WiFi AP
        3. Schedule flows in WiFi domain (frequential - band selection)
        4. Check cross-domain constraints
        
        Args:
            flows: List of flows to schedule
            
        Returns:
            Dictionary with scheduling results and statistics
        """
        # Step 1: TSN scheduling
        tsn_scheduled, max_tsn_load = self.tsn_scheduler.schedule_flows(flows)
        
        # Step 2: Prepare flows for WiFi scheduling
        wifi_flows = []
        for flow in flows:
            if flow.scheduled:  # Only schedule flows that succeeded in TSN
                wifi_flows.append(flow)
        
        # Step 3: WiFi scheduling
        if self.use_drl and self.wifi_scheduler is not None:
            wifi_scheduled, avg_reliability = self._schedule_wifi_drl(wifi_flows)
        else:
            wifi_scheduled, avg_reliability = self._schedule_wifi_greedy(wifi_flows)
        
        # Step 4: Collect results
        total_scheduled = sum(1 for f in flows if f.scheduled and f.band is not None)
        
        results = {
            'total_flows': len(flows),
            'tsn_scheduled': tsn_scheduled,
            'wifi_scheduled': wifi_scheduled,
            'total_scheduled': total_scheduled,
            'scheduling_rate': total_scheduled / len(flows) if flows else 0,
            'max_tsn_load': max_tsn_load,
            'avg_reliability': avg_reliability,
            'tsn_stats': self.tsn_scheduler.get_scheduling_results(),
            'wifi_stats': self.wifi_mlo.get_statistics(),
        }
        
        return results
    
    def _calculate_arrival_time(self, flow: Flow) -> float:
        """
        Calculate when a flow arrives at the WiFi AP after TSN transmission.
        
        Args:
            flow: Flow with TSN scheduling completed
            
        Returns:
            Arrival time at AP in microseconds
        """
        if not flow.path or not flow.offsets:
            return flow.start_offset * self.tsn_network.time_slot_duration
        
        # Get the last link's offset and transmission time
        last_link_id = flow.path[-1]
        last_offset = flow.offsets[last_link_id]
        trans_time = flow.get_transmission_time()
        
        return last_offset + trans_time
    
    def _schedule_wifi_drl(self, flows: List[Flow]) -> Tuple[int, float]:
        """
        Schedule flows in WiFi domain using attention-based DDPG.
        
        Args:
            flows: List of flows to schedule
            
        Returns:
            Tuple of (num_scheduled, avg_reliability)
        """
        if not flows:
            return 0, 0.0
        
        # Prepare state for each flow
        channel_state = self.wifi_mlo.get_channel_state()  # 9 features
        states = []
        
        for flow in flows:
            arrival_time = self._calculate_arrival_time(flow)
            # State: [arrival_time_normalized, packet_size_normalized, tsn_delay_normalized, ...channel_state]
            state = np.array([
                arrival_time / self.tsn_network.cycle_time,
                flow.size / 1500.0,
                0.5,  # Simplified TSN delay metric
            ])
            # Append channel state
            state = np.concatenate([state, channel_state])
            states.append(state)
        
        state_array = np.array(states)
        
        # Get action probabilities from DRL agent
        action_probs = self.wifi_scheduler.select_action(state_array, explore=False)
        
        # Sample band for each flow
        num_scheduled = 0
        total_reliability = 0.0
        
        for i, flow in enumerate(flows):
            # Sample band based on probabilities
            band_id = self.wifi_scheduler.sample_action(action_probs[i:i+1])[0]
            
            # Calculate start time (earliest available on band)
            arrival_time = self._calculate_arrival_time(flow)
            start_time = self.wifi_mlo.get_earliest_available_time(
                band_id, arrival_time, flow.size
            )
            
            # Check if within cycle time
            band = self.wifi_mlo.get_band(band_id)
            trans_time = band.get_transmission_time(flow.size)
            
            if start_time + trans_time <= self.wifi_mlo.cycle_time:
                # Schedule the flow
                if self.wifi_mlo.schedule_flow(flow.flow_id, band_id, start_time, flow.size):
                    flow.band = band_id
                    flow.reliability = band.reliability
                    num_scheduled += 1
                    total_reliability += band.reliability
        
        avg_reliability = total_reliability / num_scheduled if num_scheduled > 0 else 0.0
        return num_scheduled, avg_reliability
    
    def _schedule_wifi_greedy(self, flows: List[Flow]) -> Tuple[int, float]:
        """
        Schedule flows in WiFi domain using greedy algorithm (baseline).
        Always selects the band with highest reliability.
        
        Args:
            flows: List of flows to schedule
            
        Returns:
            Tuple of (num_scheduled, avg_reliability)
        """
        if not flows:
            return 0, 0.0
        
        num_scheduled = 0
        total_reliability = 0.0
        
        for flow in flows:
            arrival_time = self._calculate_arrival_time(flow)
            
            # Try bands in order of reliability
            bands_by_reliability = sorted(
                enumerate(self.wifi_mlo.bands),
                key=lambda x: x[1].reliability,
                reverse=True
            )
            
            for band_id, band in bands_by_reliability:
                start_time = self.wifi_mlo.get_earliest_available_time(
                    band_id, arrival_time, flow.size
                )
                trans_time = band.get_transmission_time(flow.size)
                
                if start_time + trans_time <= self.wifi_mlo.cycle_time:
                    if self.wifi_mlo.schedule_flow(flow.flow_id, band_id, start_time, flow.size):
                        flow.band = band_id
                        flow.reliability = band.reliability
                        num_scheduled += 1
                        total_reliability += band.reliability
                        break
        
        avg_reliability = total_reliability / num_scheduled if num_scheduled > 0 else 0.0
        return num_scheduled, avg_reliability
    
    def reset(self):
        """Reset all schedules."""
        self.tsn_network.reset_schedules()
        self.wifi_mlo.reset_schedules()
