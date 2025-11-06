"""
Time-Triggered (TT) Flow representation for TSN
Based on the Cao et al. 2025 paper on integrated routing and scheduling
"""

class TTFlow:
    """
    Represents a Time-Triggered (TT) flow in TSN network.
    
    Attributes:
        flow_id: Unique identifier for the flow
        source: Source switch ID
        destination: Destination switch ID
        data_size: Size of data in bytes (60-1500 bytes)
        period: Transmission period in ms (3.6, 4.8, or 6.0 ms)
        deadline: End-to-end deadline constraint
        route: Selected route (list of switch IDs)
        schedule: Allocated time slots on each hop
    """
    
    def __init__(self, flow_id, source, destination, data_size, period, deadline=None):
        """
        Initialize a TT flow.
        
        Args:
            flow_id: Unique identifier
            source: Source switch ID
            destination: Destination switch ID  
            data_size: Data size in bytes (60-1500)
            period: Transmission period in ms (3.6, 4.8, or 6.0)
            deadline: End-to-end deadline (defaults to period if not specified)
        """
        self.flow_id = flow_id
        self.source = source
        self.destination = destination
        self.data_size = data_size  # bytes
        self.period = period  # ms
        self.deadline = deadline if deadline is not None else period
        
        # To be determined by routing and scheduling
        self.route = []
        self.schedule = {}  # {link: [time_slots]}
        self.successfully_scheduled = False
        
    def get_transmission_time(self, link_bandwidth=100):
        """
        Calculate transmission time for the flow.
        
        Args:
            link_bandwidth: Link bandwidth in Mb/s (default: 100 Mb/s)
            
        Returns:
            Transmission time in ms
        """
        # Convert bytes to bits, bandwidth Mb/s to bits/ms
        bits = self.data_size * 8
        bandwidth_bits_per_ms = link_bandwidth * 1000  # Mb/s to bits/ms
        return bits / bandwidth_bits_per_ms
    
    def get_required_slots(self, slot_duration):
        """
        Calculate number of time slots required for transmission.
        
        Args:
            slot_duration: Duration of each time slot in ms
            
        Returns:
            Number of slots required (rounded up)
        """
        import math
        transmission_time = self.get_transmission_time()
        return math.ceil(transmission_time / slot_duration)
    
    def __repr__(self):
        return (f"TTFlow(id={self.flow_id}, {self.source}->{self.destination}, "
                f"size={self.data_size}B, period={self.period}ms)")
