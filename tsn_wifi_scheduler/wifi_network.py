"""
WiFi Multi-Link Operation (MLO) Model
Implements Wi-Fi MLO with multiple bands (2.4GHz, 5GHz, 6GHz).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class WiFiBand:
    """
    Represents a WiFi band in Multi-Link Operation.
    
    Attributes:
        band_id: Band identifier (0: 2.4GHz, 1: 5GHz, 2: 6GHz)
        frequency: Frequency in GHz
        bandwidth: Bandwidth in MHz
        reliability: Channel reliability (0-1), affected by noise/interference
        airtime: Available airtime in microseconds per cycle
        scheduled_flows: List of (flow_id, start_time, end_time) tuples
    """
    band_id: int
    frequency: float
    bandwidth: int
    reliability: float = 0.95
    airtime: float = 1000.0  # microseconds
    scheduled_flows: List[Tuple[int, float, float]] = field(default_factory=list)
    
    def get_transmission_rate(self) -> int:
        """
        Get transmission rate in Mbps.
        Different bands have different typical rates.
        """
        rates = {
            0: 72,   # 2.4 GHz - lower rate
            1: 433,  # 5 GHz - medium rate
            2: 600,  # 6 GHz - highest rate
        }
        return rates.get(self.band_id, 433)
    
    def get_transmission_time(self, packet_size: int) -> float:
        """
        Calculate transmission time for a packet in microseconds.
        
        Args:
            packet_size: Packet size in bytes
            
        Returns:
            Transmission time in microseconds
        """
        rate_mbps = self.get_transmission_rate()
        # Time = (size in bits) / (rate in Mbps)
        return (packet_size * 8) / rate_mbps
    
    def is_available(self, start_time: float, duration: float) -> bool:
        """Check if band is available during the specified time window."""
        end_time = start_time + duration
        for _, flow_start, flow_end in self.scheduled_flows:
            # Check for overlap
            if not (end_time <= flow_start or start_time >= flow_end):
                return False
        return True
    
    def schedule_flow(self, flow_id: int, start_time: float, duration: float) -> bool:
        """
        Schedule a flow on this band.
        
        Returns:
            True if successfully scheduled, False otherwise
        """
        if self.is_available(start_time, duration):
            end_time = start_time + duration
            self.scheduled_flows.append((flow_id, start_time, end_time))
            self.scheduled_flows.sort(key=lambda x: x[1])
            return True
        return False
    
    def get_utilization(self, cycle_time: float) -> float:
        """Calculate band utilization as a percentage."""
        total_time = sum(end - start for _, start, end in self.scheduled_flows)
        return (total_time / cycle_time) * 100 if cycle_time > 0 else 0


class WiFiMLO:
    """
    WiFi Multi-Link Operation (MLO) Access Point.
    Manages three bands: 2.4 GHz, 5 GHz, and 6 GHz.
    
    Architecture:
    - U-MAC (Upper MAC): Traffic management and band selection
    - L-MAC (Lower MAC): Per-band EDCA queue management
    """
    
    def __init__(self, cycle_time: float = 1000.0):
        """
        Initialize WiFi MLO.
        
        Args:
            cycle_time: Cycle time in microseconds (default 1000 μs)
        """
        self.cycle_time = cycle_time
        
        # Initialize three bands
        self.bands = [
            WiFiBand(band_id=0, frequency=2.4, bandwidth=20, reliability=0.85),
            WiFiBand(band_id=1, frequency=5.0, bandwidth=80, reliability=0.92),
            WiFiBand(band_id=2, frequency=6.0, bandwidth=160, reliability=0.96),
        ]
        
        # EDCA queues for each band (8 priority queues per band)
        self.edca_queues: Dict[int, List[int]] = {i: [] for i in range(3)}
        
        # Statistics
        self.scheduled_flows: List[int] = []
        
    def get_band(self, band_id: int) -> Optional[WiFiBand]:
        """Get band by ID."""
        if 0 <= band_id < len(self.bands):
            return self.bands[band_id]
        return None
    
    def update_channel_quality(self, band_id: int, reliability: float) -> None:
        """
        Update channel quality (reliability) for a band.
        Simulates dynamic wireless channel conditions.
        
        Args:
            band_id: Band identifier
            reliability: New reliability value (0-1)
        """
        if 0 <= band_id < len(self.bands):
            self.bands[band_id].reliability = np.clip(reliability, 0.0, 1.0)
    
    def get_channel_state(self) -> np.ndarray:
        """
        Get current channel state for all bands.
        
        Returns:
            Array of [TSNR, bandwidth, reliability] for each band
            TSNR is simulated as reliability-based metric
        """
        state = []
        for band in self.bands:
            # TSNR (Time-Sensitive Network Reliability) - normalized
            tsnr = band.reliability * 100  # 0-100 scale
            # Bandwidth normalized
            bw_norm = band.bandwidth / 160  # Normalize by max bandwidth
            state.extend([tsnr, bw_norm, band.reliability])
        return np.array(state, dtype=np.float32)
    
    def schedule_flow(self, flow_id: int, band_id: int, start_time: float, 
                     packet_size: int) -> bool:
        """
        Schedule a flow on a specific band.
        
        Args:
            flow_id: Flow identifier
            band_id: Band to use (0-2)
            start_time: Start time in microseconds
            packet_size: Packet size in bytes
            
        Returns:
            True if successfully scheduled, False otherwise
        """
        band = self.get_band(band_id)
        if band is None:
            return False
        
        duration = band.get_transmission_time(packet_size)
        
        if band.schedule_flow(flow_id, start_time, duration):
            if flow_id not in self.scheduled_flows:
                self.scheduled_flows.append(flow_id)
            return True
        return False
    
    def get_earliest_available_time(self, band_id: int, arrival_time: float,
                                   packet_size: int) -> float:
        """
        Get the earliest available time to schedule a flow on a band.
        
        Args:
            band_id: Band identifier
            arrival_time: Flow arrival time at AP
            packet_size: Packet size in bytes
            
        Returns:
            Earliest available start time
        """
        band = self.get_band(band_id)
        if band is None:
            return arrival_time
        
        duration = band.get_transmission_time(packet_size)
        
        # Start with arrival time
        candidate_time = arrival_time
        
        # Find a gap in the schedule
        for _, flow_start, flow_end in sorted(band.scheduled_flows, key=lambda x: x[1]):
            if candidate_time + duration <= flow_start:
                # Found a gap
                return candidate_time
            elif candidate_time < flow_end:
                # Overlap, try after this flow
                candidate_time = flow_end
        
        # No conflicts, can schedule at candidate_time
        return candidate_time
    
    def reset_schedules(self) -> None:
        """Reset all schedules."""
        for band in self.bands:
            band.scheduled_flows = []
        for queue in self.edca_queues.values():
            queue.clear()
        self.scheduled_flows = []
    
    def get_statistics(self) -> Dict:
        """Get WiFi MLO statistics."""
        return {
            'num_bands': len(self.bands),
            'scheduled_flows': len(self.scheduled_flows),
            'band_utilizations': [
                band.get_utilization(self.cycle_time) for band in self.bands
            ],
            'band_reliabilities': [band.reliability for band in self.bands],
            'avg_reliability': np.mean([band.reliability for band in self.bands]),
        }
