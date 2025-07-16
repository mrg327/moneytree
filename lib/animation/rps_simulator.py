"""
Core RPS Simulation Engine.

Manages the Rock Paper Scissors battle simulation with entity interactions,
collision detection, and win condition tracking.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

from lib.utils.logging_config import get_logger
from .entity import RPSEntity, EntityType

logger = get_logger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for RPS simulation."""
    
    # Screen dimensions
    screen_width: int = 800
    screen_height: int = 600
    
    # Entity settings
    total_entities: int = 150
    entity_distribution: Dict[EntityType, float] = None  # None = equal distribution
    
    # Physics settings
    collision_enabled: bool = True
    physics_enabled: bool = True
    
    # Speed settings
    min_entity_speed: float = 30.0
    max_entity_speed: float = 80.0
    
    # Simulation settings
    max_simulation_time: float = 300.0  # 5 minutes max
    target_fps: int = 60
    
    def __post_init__(self):
        """Initialize default entity distribution if not provided."""
        if self.entity_distribution is None:
            # Equal distribution by default
            self.entity_distribution = {
                EntityType.ROCK: 1.0 / 3.0,
                EntityType.PAPER: 1.0 / 3.0,
                EntityType.SCISSORS: 1.0 / 3.0
            }


class RPSSimulator:
    """
    Core simulation engine for Rock Paper Scissors battles.
    
    Manages entity lifecycle, collision detection, battles, and win conditions.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the RPS simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.entities: List[RPSEntity] = []
        self.simulation_time = 0.0
        self.frame_count = 0
        self.is_running = False
        self.winner: Optional[EntityType] = None
        
        # Statistics tracking
        self.stats = {
            'battles_fought': 0,
            'conversions': defaultdict(int),
            'entity_counts': defaultdict(int),
            'simulation_events': []
        }
        
        # Performance tracking
        self.collision_checks = 0
        self.battles_this_frame = 0
        
        logger.info(f"Initialized RPS simulator with {config.total_entities} entities")
    
    def initialize_entities(self) -> None:
        """Initialize entities based on configuration."""
        self.entities.clear()
        
        # Calculate entity counts based on distribution
        distribution = self.config.entity_distribution
        entity_counts = {}
        
        for entity_type, ratio in distribution.items():
            count = int(self.config.total_entities * ratio)
            entity_counts[entity_type] = count
        
        # Adjust for rounding errors
        total_assigned = sum(entity_counts.values())
        if total_assigned < self.config.total_entities:
            # Add remaining entities to first type
            first_type = list(entity_counts.keys())[0]
            entity_counts[first_type] += self.config.total_entities - total_assigned
        
        # Create entities
        for entity_type, count in entity_counts.items():
            for _ in range(count):
                entity = RPSEntity.create_random(
                    self.config.screen_width,
                    self.config.screen_height,
                    entity_type
                )
                
                # Set speed limits from config
                entity.max_speed = self.config.max_entity_speed
                entity.add_random_velocity(
                    self.config.min_entity_speed,
                    self.config.max_entity_speed
                )
                
                self.entities.append(entity)
        
        # Update initial stats
        self._update_entity_counts()
        
        logger.info(f"Created {len(self.entities)} entities: "
                   f"Rock={entity_counts.get(EntityType.ROCK, 0)}, "
                   f"Paper={entity_counts.get(EntityType.PAPER, 0)}, "
                   f"Scissors={entity_counts.get(EntityType.SCISSORS, 0)}")
    
    def start_simulation(self) -> None:
        """Start the simulation."""
        if not self.entities:
            self.initialize_entities()
        
        self.is_running = True
        self.simulation_time = 0.0
        self.frame_count = 0
        self.winner = None
        
        logger.info("RPS simulation started")
    
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update simulation by one time step.
        
        Args:
            dt: Time delta in seconds
            
        Returns:
            Dictionary with simulation state information
        """
        if not self.is_running:
            return self._get_simulation_state()
        
        self.simulation_time += dt
        self.frame_count += 1
        self.battles_this_frame = 0
        self.collision_checks = 0
        
        # Update all entities
        for entity in self.entities:
            entity.update(dt, self.config.screen_width, self.config.screen_height)
        
        # Handle collisions and battles
        if self.config.collision_enabled:
            self._handle_collisions()
        
        # Check win condition
        self._check_win_condition()
        
        # Check timeout
        if self.simulation_time >= self.config.max_simulation_time:
            self._handle_timeout()
        
        # Update statistics
        self._update_entity_counts()
        
        return self._get_simulation_state()
    
    def _handle_collisions(self) -> None:
        """Handle collisions between entities."""
        for i in range(len(self.entities)):
            for j in range(i + 1, len(self.entities)):
                entity1 = self.entities[i]
                entity2 = self.entities[j]
                
                self.collision_checks += 1
                
                if entity1.is_colliding_with(entity2):
                    # Apply physics if enabled
                    if self.config.physics_enabled:
                        entity1.apply_collision_physics(entity2)
                    
                    # Handle battle
                    self._handle_battle(entity1, entity2)
    
    def _handle_battle(self, entity1: RPSEntity, entity2: RPSEntity) -> None:
        """
        Handle battle between two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
        """
        if entity1.entity_type == entity2.entity_type:
            return  # Same type, no battle
        
        winner_type = entity1.battle(entity2)
        
        if winner_type is None:
            return  # No battle result
        
        # Convert loser to winner's type
        if winner_type == entity1.entity_type:
            old_type = entity2.entity_type
            entity2.convert_to_type(winner_type)
            loser = entity2
        else:
            old_type = entity1.entity_type
            entity1.convert_to_type(winner_type)
            loser = entity1
        
        # Update statistics
        self.stats['battles_fought'] += 1
        self.stats['conversions'][f"{old_type.value}_to_{winner_type.value}"] += 1
        self.battles_this_frame += 1
        
        # Add some randomness to converted entity's velocity
        loser.add_random_velocity(
            self.config.min_entity_speed * 0.8,
            self.config.max_entity_speed * 1.2
        )
        
        # Log significant battles
        if self.stats['battles_fought'] % 100 == 0:
            logger.debug(f"Battle #{self.stats['battles_fought']}: "
                        f"{old_type.value} converted to {winner_type.value}")
    
    def _check_win_condition(self) -> None:
        """Check if simulation has reached win condition."""
        if self.winner is not None:
            return  # Already won
        
        # Count entity types
        type_counts = defaultdict(int)
        for entity in self.entities:
            type_counts[entity.entity_type] += 1
        
        # Check if only one type remains
        remaining_types = [t for t, count in type_counts.items() if count > 0]
        
        if len(remaining_types) == 1:
            self.winner = remaining_types[0]
            self.is_running = False
            
            total_battles = self.stats['battles_fought']
            logger.info(f"🏆 {self.winner.value.upper()} WINS! "
                       f"After {total_battles} battles in {self.simulation_time:.1f}s")
            
            # Record win event
            self.stats['simulation_events'].append({
                'type': 'win',
                'winner': self.winner.value,
                'time': self.simulation_time,
                'total_battles': total_battles
            })
    
    def _handle_timeout(self) -> None:
        """Handle simulation timeout."""
        if self.winner is None:
            # Find the type with most entities
            type_counts = defaultdict(int)
            for entity in self.entities:
                type_counts[entity.entity_type] += 1
            
            if type_counts:
                self.winner = max(type_counts.items(), key=lambda x: x[1])[0]
            
            self.is_running = False
            logger.info(f"⏰ Simulation timeout after {self.simulation_time:.1f}s. "
                       f"Winner by count: {self.winner.value if self.winner else 'None'}")
    
    def _update_entity_counts(self) -> None:
        """Update entity count statistics."""
        counts = defaultdict(int)
        for entity in self.entities:
            counts[entity.entity_type] += 1
        
        self.stats['entity_counts'] = dict(counts)
    
    def _get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            'is_running': self.is_running,
            'simulation_time': self.simulation_time,
            'frame_count': self.frame_count,
            'winner': self.winner.value if self.winner else None,
            'entity_counts': dict(self.stats['entity_counts']),
            'total_entities': len(self.entities),
            'battles_fought': self.stats['battles_fought'],
            'battles_this_frame': self.battles_this_frame,
            'collision_checks': self.collision_checks,
            'entities': self.entities  # For rendering
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed simulation statistics."""
        return {
            'simulation_time': self.simulation_time,
            'frame_count': self.frame_count,
            'battles_fought': self.stats['battles_fought'],
            'conversions': dict(self.stats['conversions']),
            'entity_counts': dict(self.stats['entity_counts']),
            'winner': self.winner.value if self.winner else None,
            'simulation_events': self.stats['simulation_events'].copy(),
            'performance': {
                'avg_collision_checks_per_frame': self.collision_checks / max(1, self.frame_count),
                'avg_battles_per_second': self.stats['battles_fought'] / max(0.1, self.simulation_time)
            }
        }
    
    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self.entities.clear()
        self.simulation_time = 0.0
        self.frame_count = 0
        self.is_running = False
        self.winner = None
        
        # Reset statistics
        self.stats = {
            'battles_fought': 0,
            'conversions': defaultdict(int),
            'entity_counts': defaultdict(int),
            'simulation_events': []
        }
        
        self.collision_checks = 0
        self.battles_this_frame = 0
        
        logger.info("Simulation reset")
    
    def add_entity(self, entity: RPSEntity) -> None:
        """Add a new entity to the simulation."""
        self.entities.append(entity)
        self._update_entity_counts()
    
    def remove_entity(self, entity: RPSEntity) -> bool:
        """
        Remove an entity from the simulation.
        
        Args:
            entity: Entity to remove
            
        Returns:
            True if entity was removed, False if not found
        """
        try:
            self.entities.remove(entity)
            self._update_entity_counts()
            return True
        except ValueError:
            return False
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[RPSEntity]:
        """Get all entities of a specific type."""
        return [entity for entity in self.entities if entity.entity_type == entity_type]
    
    def get_entity_positions(self) -> List[Tuple[float, float, str]]:
        """Get positions and types of all entities for external rendering."""
        return [(entity.position[0], entity.position[1], entity.entity_type.value) 
                for entity in self.entities]