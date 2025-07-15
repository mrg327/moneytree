"""Entity management for rock-paper-scissors battle royale."""

import random
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional
import pygame
import pymunk
import numpy as np


class EntityType(Enum):
    """Entity types for rock-paper-scissors game."""
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


@dataclass
class EntityConfig:
    """Configuration for game entities."""
    size: float = 20.0
    speed: float = 50.0
    mass: float = 1.0
    initial_count: int = 100
    colors: Dict[EntityType, Tuple[int, int, int]] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                EntityType.ROCK: (120, 120, 120),      # Gray
                EntityType.PAPER: (255, 255, 200),     # Light yellow
                EntityType.SCISSORS: (200, 200, 255)   # Light blue
            }


class Entity:
    """Individual entity in the battle royale."""
    
    def __init__(self, entity_id: int, entity_type: EntityType, x: float, y: float, 
                 config: EntityConfig, space: pymunk.Space):
        """Initialize an entity.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of entity (rock, paper, scissors)
            x: Initial x position
            y: Initial y position
            config: Entity configuration
            space: Pymunk physics space
        """
        self.id = entity_id
        self.type = entity_type
        self.config = config
        self.alive = True
        self.last_interaction_time = 0
        
        # Create physics body
        self.body = pymunk.Body(config.mass, pymunk.moment_for_circle(config.mass, 0, config.size))
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, config.size)
        self.shape.collision_type = abs(hash(entity_type.value)) % 1000  # Ensure positive
        self.shape.entity = self  # Back-reference
        
        # Add to physics space
        space.add(self.body, self.shape)
        
        # Visual properties
        self.color = config.colors[entity_type]
        self.radius = config.size
        
        # AI behavior
        self.target_direction = random.uniform(0, 2 * math.pi)
        self.direction_change_time = 0
        
    def update(self, dt: float, current_time: float):
        """Update entity state.
        
        Args:
            dt: Delta time since last update
            current_time: Current simulation time
        """
        if not self.alive:
            return
            
        # Simple AI: change direction periodically
        if current_time - self.direction_change_time > 2.0:
            self.target_direction = random.uniform(0, 2 * math.pi)
            self.direction_change_time = current_time
            
        # Apply movement force
        force_x = math.cos(self.target_direction) * self.config.speed
        force_y = math.sin(self.target_direction) * self.config.speed
        self.body.force = (force_x, force_y)
        
    def can_defeat(self, other: 'Entity') -> bool:
        """Check if this entity can defeat another entity.
        
        Args:
            other: Other entity to check against
            
        Returns:
            True if this entity defeats the other
        """
        if self.type == EntityType.ROCK:
            return other.type == EntityType.SCISSORS
        elif self.type == EntityType.PAPER:
            return other.type == EntityType.ROCK
        elif self.type == EntityType.SCISSORS:
            return other.type == EntityType.PAPER
        return False
        
    def defeat(self, other: 'Entity', current_time: float):
        """Defeat another entity and assimilate it.
        
        Args:
            other: Entity to defeat
            current_time: Current simulation time
        """
        if not self.can_defeat(other) or not other.alive:
            return
            
        # Convert the other entity to this type
        other.type = self.type
        other.color = self.config.colors[self.type]
        other.last_interaction_time = current_time
        
    def remove_from_space(self, space: pymunk.Space):
        """Remove entity from physics space."""
        if self.body in space.bodies:
            space.remove(self.body, self.shape)
        self.alive = False
        
    def draw(self, screen: pygame.Surface):
        """Draw the entity on screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        if not self.alive:
            return
            
        x, y = int(self.body.position.x), int(self.body.position.y)
        radius = max(1, int(self.radius))  # Ensure positive radius
        
        # Ensure position is within reasonable bounds
        if x >= 0 and y >= 0:
            pygame.draw.circle(screen, self.color, (x, y), radius)
            
            # Add a border to make entities more visible
            pygame.draw.circle(screen, (0, 0, 0), (x, y), radius, 2)


class EntityManager:
    """Manages all entities in the battle royale."""
    
    def __init__(self, config: EntityConfig, space: pymunk.Space, 
                 world_width: int, world_height: int):
        """Initialize the entity manager.
        
        Args:
            config: Entity configuration
            space: Pymunk physics space
            world_width: Width of the game world
            world_height: Height of the game world
        """
        self.config = config
        self.space = space
        self.world_width = world_width
        self.world_height = world_height
        self.entities: List[Entity] = []
        self.next_id = 0
        
        # Setup collision handlers
        # self._setup_collision_handlers()  # Disabled for now
        
    def _setup_collision_handlers(self):
        """Setup collision handlers for entity interactions."""
        def handle_collision(arbiter, space, data):
            entity1 = arbiter.shapes[0].entity
            entity2 = arbiter.shapes[1].entity
            
            if not (entity1.alive and entity2.alive):
                return True
                
            current_time = pygame.time.get_ticks() / 1000.0
            
            # Prevent rapid successive interactions
            if (current_time - entity1.last_interaction_time < 0.5 or 
                current_time - entity2.last_interaction_time < 0.5):
                return True
                
            # Determine winner
            if entity1.can_defeat(entity2):
                entity1.defeat(entity2, current_time)
                # TODO: Play sound effect here
            elif entity2.can_defeat(entity1):
                entity2.defeat(entity1, current_time)
                # TODO: Play sound effect here
                
            return True
            
        # Add collision handlers for all entity type combinations
        for type1 in EntityType:
            for type2 in EntityType:
                if type1 != type2:
                    handler = self.space.add_collision_handler(
                        abs(hash(type1.value)) % 1000, abs(hash(type2.value)) % 1000
                    )
                    handler.begin = handle_collision
                    
    def spawn_entities(self):
        """Spawn initial entities in the world."""
        entities_per_type = self.config.initial_count // 3
        
        for entity_type in EntityType:
            for _ in range(entities_per_type):
                x = random.uniform(self.config.size, self.world_width - self.config.size)
                y = random.uniform(self.config.size, self.world_height - self.config.size)
                
                entity = Entity(self.next_id, entity_type, x, y, self.config, self.space)
                self.entities.append(entity)
                self.next_id += 1
                
    def update(self, dt: float):
        """Update all entities.
        
        Args:
            dt: Delta time since last update
        """
        current_time = pygame.time.get_ticks() / 1000.0
        
        # Update all alive entities
        for entity in self.entities:
            if entity.alive:
                entity.update(dt, current_time)
                
        # Handle collisions manually
        self._handle_collisions(current_time)
                
        # Remove entities that are out of bounds
        self._handle_boundaries()
        
    def _handle_collisions(self, current_time: float):
        """Handle entity collisions manually.
        
        Args:
            current_time: Current simulation time
        """
        # Simple n^2 collision detection for now
        for i, entity1 in enumerate(self.entities):
            if not entity1.alive:
                continue
                
            for j, entity2 in enumerate(self.entities[i+1:], i+1):
                if not entity2.alive:
                    continue
                    
                # Check distance
                x1, y1 = entity1.body.position
                x2, y2 = entity2.body.position
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                
                # Check if entities are touching
                if distance < (entity1.radius + entity2.radius):
                    # Prevent rapid successive interactions
                    if (current_time - entity1.last_interaction_time < 0.5 or 
                        current_time - entity2.last_interaction_time < 0.5):
                        continue
                        
                    # Determine winner
                    if entity1.can_defeat(entity2):
                        entity1.defeat(entity2, current_time)
                        # TODO: Play sound effect here
                    elif entity2.can_defeat(entity1):
                        entity2.defeat(entity1, current_time)
                        # TODO: Play sound effect here
        
    def _handle_boundaries(self):
        """Handle entities that move out of bounds."""
        for entity in self.entities:
            if not entity.alive:
                continue
                
            x, y = entity.body.position
            
            # Bounce off walls
            if x <= entity.radius or x >= self.world_width - entity.radius:
                entity.body.velocity = (-entity.body.velocity.x, entity.body.velocity.y)
            if y <= entity.radius or y >= self.world_height - entity.radius:
                entity.body.velocity = (entity.body.velocity.x, -entity.body.velocity.y)
                
            # Clamp position
            entity.body.position = (
                max(entity.radius, min(x, self.world_width - entity.radius)),
                max(entity.radius, min(y, self.world_height - entity.radius))
            )
            
    def get_entity_counts(self) -> Dict[EntityType, int]:
        """Get count of each entity type.
        
        Returns:
            Dictionary mapping entity type to count
        """
        counts = {entity_type: 0 for entity_type in EntityType}
        
        for entity in self.entities:
            if entity.alive:
                counts[entity.type] += 1
                
        return counts
        
    def get_total_count(self) -> int:
        """Get total number of alive entities.
        
        Returns:
            Total count of alive entities
        """
        return sum(1 for entity in self.entities if entity.alive)
        
    def draw_all(self, screen: pygame.Surface):
        """Draw all entities.
        
        Args:
            screen: Pygame surface to draw on
        """
        for entity in self.entities:
            entity.draw(screen)
            
    def cleanup(self):
        """Clean up all entities."""
        for entity in self.entities:
            entity.remove_from_space(self.space)
        self.entities.clear()