"""
RPS Entity class for Rock Paper Scissors simulation.

Defines the basic entity that participates in the RPS battle simulation.
"""

import math
import random
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class EntityType(Enum):
    """Types of RPS entities."""
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


@dataclass
class RPSEntity:
    """
    A Rock Paper Scissors entity that moves around the screen and battles other entities.
    
    Attributes:
        entity_type: The type of entity (rock, paper, or scissors)
        position: Current (x, y) position on screen
        velocity: Current (vx, vy) velocity vector
        radius: Entity radius for collision detection
        color: RGBA color tuple for rendering
        mass: Entity mass for collision physics
        max_speed: Maximum speed limit
    """
    
    entity_type: EntityType
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    radius: float = 20.0
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    mass: float = 1.0
    max_speed: float = 100.0
    
    def __post_init__(self):
        """Initialize entity with default colors based on type."""
        if self.color == (255, 255, 255, 255):  # Default white, set type-specific color
            self.color = self.get_default_color()
    
    def get_default_color(self) -> Tuple[int, int, int, int]:
        """Get default color based on entity type."""
        color_map = {
            EntityType.ROCK: (128, 128, 128, 255),     # Gray, fully opaque
            EntityType.PAPER: (255, 255, 255, 255),    # White, fully opaque
            EntityType.SCISSORS: (255, 0, 0, 255)      # Red, fully opaque
        }
        return color_map.get(self.entity_type, (255, 255, 255, 255))
    
    def update(self, dt: float, screen_width: int, screen_height: int) -> None:
        """
        Update entity position and handle boundary collisions.
        
        Args:
            dt: Time delta in seconds
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        # Update position
        x, y = self.position
        vx, vy = self.velocity
        
        x += vx * dt
        y += vy * dt
        
        # Boundary collisions with bounce
        if x - self.radius <= 0:
            x = self.radius
            vx = abs(vx)  # Bounce right
        elif x + self.radius >= screen_width:
            x = screen_width - self.radius
            vx = -abs(vx)  # Bounce left
            
        if y - self.radius <= 0:
            y = self.radius
            vy = abs(vy)  # Bounce down
        elif y + self.radius >= screen_height:
            y = screen_height - self.radius
            vy = -abs(vy)  # Bounce up
        
        # Apply speed limit
        speed = math.sqrt(vx**2 + vy**2)
        if speed > self.max_speed:
            vx = (vx / speed) * self.max_speed
            vy = (vy / speed) * self.max_speed
        
        self.position = (x, y)
        self.velocity = (vx, vy)
    
    def get_distance_to(self, other: 'RPSEntity') -> float:
        """Calculate distance to another entity."""
        x1, y1 = self.position
        x2, y2 = other.position
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def is_colliding_with(self, other: 'RPSEntity') -> bool:
        """Check if this entity is colliding with another entity."""
        distance = self.get_distance_to(other)
        return distance <= (self.radius + other.radius)
    
    def battle(self, other: 'RPSEntity') -> Optional['RPSEntity']:
        """
        Battle with another entity using RPS rules.
        
        Returns:
            The winning entity type, or None if same type (no battle)
        """
        if self.entity_type == other.entity_type:
            return None  # Same type, no battle
        
        # Rock beats scissors, scissors beats paper, paper beats rock
        winning_combinations = {
            (EntityType.ROCK, EntityType.SCISSORS): EntityType.ROCK,
            (EntityType.SCISSORS, EntityType.PAPER): EntityType.SCISSORS,
            (EntityType.PAPER, EntityType.ROCK): EntityType.PAPER
        }
        
        combination = (self.entity_type, other.entity_type)
        reverse_combination = (other.entity_type, self.entity_type)
        
        if combination in winning_combinations:
            return self.entity_type
        elif reverse_combination in winning_combinations:
            return other.entity_type
        
        return None  # Should not happen with valid enum values
    
    def convert_to_type(self, new_type: EntityType) -> None:
        """Convert this entity to a different type."""
        self.entity_type = new_type
        self.color = self.get_default_color()
    
    def apply_collision_physics(self, other: 'RPSEntity') -> None:
        """
        Apply elastic collision physics between two entities.
        
        Args:
            other: The other entity involved in collision
        """
        # Calculate collision vector
        x1, y1 = self.position
        x2, y2 = other.position
        
        # Vector from this entity to other
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return  # Avoid division by zero
        
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        
        # Relative velocity
        vx1, vy1 = self.velocity
        vx2, vy2 = other.velocity
        dvx = vx2 - vx1
        dvy = vy2 - vy1
        
        # Relative velocity in collision normal direction
        dvn = dvx * nx + dvy * ny
        
        # Do not resolve if velocities are separating
        if dvn >= 0:
            return
        
        # Collision impulse
        impulse = 2 * dvn / (self.mass + other.mass)
        
        # Apply impulse to velocities
        self.velocity = (
            vx1 + impulse * other.mass * nx,
            vy1 + impulse * other.mass * ny
        )
        
        other.velocity = (
            vx2 - impulse * self.mass * nx,
            vy2 - impulse * self.mass * ny
        )
        
        # Separate entities to avoid overlap
        overlap = (self.radius + other.radius) - distance
        if overlap > 0:
            separation = overlap / 2
            self.position = (
                x1 - separation * nx,
                y1 - separation * ny
            )
            other.position = (
                x2 + separation * nx,
                y2 + separation * ny
            )
    
    def add_random_velocity(self, min_speed: float = 30.0, max_speed: float = 80.0) -> None:
        """Add random velocity to the entity."""
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(min_speed, max_speed)
        
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        
        self.velocity = (vx, vy)
    
    def render(self, screen) -> None:
        """
        Render the entity on a pygame screen.
        
        Args:
            screen: Pygame screen surface
        """
        if not HAS_PYGAME:
            return
        
        x, y = self.position
        pygame.draw.circle(screen, self.color, (int(x), int(y)), int(self.radius))
        
        # Add border for better visibility
        pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), int(self.radius), 2)
    
    @classmethod
    def create_random(cls, screen_width: int, screen_height: int, 
                     entity_type: EntityType = None) -> 'RPSEntity':
        """
        Create a random entity within screen boundaries.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            entity_type: Specific entity type, or None for random
            
        Returns:
            A new RPSEntity with random position and velocity
        """
        if entity_type is None:
            entity_type = random.choice(list(EntityType))
        
        radius = 20.0
        # Ensure entity spawns within screen bounds
        x = random.uniform(radius, screen_width - radius)
        y = random.uniform(radius, screen_height - radius)
        
        entity = cls(
            entity_type=entity_type,
            position=(x, y),
            velocity=(0, 0),
            radius=radius
        )
        
        # Add random initial velocity
        entity.add_random_velocity()
        
        return entity