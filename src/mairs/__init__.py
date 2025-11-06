"""
MAIRS - Multi-Agent Deep Reinforcement Learning-based 
Integrated Routing and Scheduling for TSN

Based on Cao et al. 2025 paper:
"How Can the Integrated Routing and Scheduling Enhance Optimality 
Bounds of Time-Sensitive Transmission"
"""

from .mairs_agent import MAIRSAgent
from .mappo import MAPPO

__all__ = ['MAIRSAgent', 'MAPPO']
