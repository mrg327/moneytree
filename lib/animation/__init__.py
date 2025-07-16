"""
Animation module for MoneyTree.

Provides pygame-based animation capabilities for creating dynamic visual content.
"""

from .entity import RPSEntity, EntityType
from .rps_simulator import RPSSimulator, SimulationConfig
from .animation_engine import AnimationEngine, AnimationConfig

__all__ = [
    'RPSEntity',
    'EntityType', 
    'RPSSimulator',
    'SimulationConfig',
    'AnimationEngine',
    'AnimationConfig'
]