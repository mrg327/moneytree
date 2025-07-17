"""
RPS Entity class for Rock Paper Scissors simulation.

Defines the basic entity that participates in the RPS battle simulation.
"""

import math
import random
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import pygame
    from PIL import Image
    HAS_PYGAME = True
    HAS_PIL = True
except ImportError:
    HAS_PYGAME = False
    HAS_PIL = False


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
        color: RGBA color tuple for rendering (used for fallback)
        mass: Entity mass for collision physics
        max_speed: Maximum speed limit
        emoji_image_path: Path to emoji image file for rendering
        use_emoji: Whether to use emoji rendering instead of colored circles
    """
    
    entity_type: EntityType
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    radius: float = 20.0
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    mass: float = 1.0
    max_speed: float = 100.0
    emoji_image_path: Optional[str] = None
    use_emoji: bool = False
    _emoji_surface: Optional[pygame.Surface] = None
    
    def __post_init__(self):
        """Initialize entity with default colors and emoji based on type."""
        if self.color == (255, 255, 255, 255):  # Default white, set type-specific color
            self.color = self.get_default_color()
        
        # Load emoji surface if emoji rendering is enabled
        if self.use_emoji and self.emoji_image_path:
            self._load_emoji_surface()
    
    def get_default_color(self) -> Tuple[int, int, int, int]:
        """Get default color based on entity type."""
        color_map = {
            EntityType.ROCK: (128, 128, 128, 255),     # Gray, fully opaque
            EntityType.PAPER: (255, 255, 255, 255),    # White, fully opaque
            EntityType.SCISSORS: (255, 0, 0, 255)      # Red, fully opaque
        }
        return color_map.get(self.entity_type, (255, 255, 255, 255))
    
    def _load_emoji_surface(self) -> None:
        """Load emoji image as pygame surface."""
        if not (HAS_PYGAME and HAS_PIL and self.emoji_image_path):
            logger.debug(f"Prerequisites not met: HAS_PYGAME={HAS_PYGAME}, HAS_PIL={HAS_PIL}, emoji_image_path={self.emoji_image_path}")
            return
        
        # Check if pygame is initialized (remove overly strict display check)
        if not pygame.get_init():
            logger.debug("Pygame not initialized, skipping emoji load")
            return  # Will be loaded later when pygame is ready
        
        try:
            logger.debug(f"Loading emoji from {self.emoji_image_path}")
            
            # Load PNG image with PIL
            pil_image = Image.open(self.emoji_image_path)
            logger.debug(f"PIL image loaded: {pil_image.size}, mode: {pil_image.mode}")
            
            # Convert to RGBA if not already
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')
                logger.debug("Converted to RGBA")
            
            # Scale to fit entity radius (emoji should be slightly smaller than radius)
            emoji_size = int(self.radius * 1.8)  # Make emoji slightly larger than circle
            pil_image = pil_image.resize((emoji_size, emoji_size), Image.Resampling.LANCZOS)
            logger.debug(f"Resized to {emoji_size}x{emoji_size}")
            
            # Convert PIL image to pygame surface
            mode = pil_image.mode
            size = pil_image.size
            data = pil_image.tobytes()
            logger.debug(f"Got image data: {len(data)} bytes")
            
            # Create pygame surface with alpha
            self._emoji_surface = pygame.image.fromstring(data, size, mode)
            self._emoji_surface = self._emoji_surface.convert_alpha()
            logger.debug(f"Created pygame surface: {self._emoji_surface.get_size()}")
            
        except Exception as e:
            logger.warning(f"Failed to load emoji image {self.emoji_image_path}: {e}")
            import traceback
            traceback.print_exc()
            self._emoji_surface = None
    
    def set_emoji(self, emoji_image_path: str) -> None:
        """
        Set emoji image for this entity.
        
        Args:
            emoji_image_path: Path to emoji PNG image
        """
        self.emoji_image_path = emoji_image_path
        self.use_emoji = True
        self._load_emoji_surface()
    
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
        
        # Update emoji if emoji rendering is enabled
        if self.use_emoji:
            self._update_emoji_for_type(new_type)
    
    def _update_emoji_for_type(self, entity_type: EntityType) -> None:
        """Update emoji image when entity type changes."""
        # Clear the current surface
        self._emoji_surface = None
        
        # Update emoji path based on new type
        # This assumes the emoji paths follow the pattern from EmojiRenderer
        if hasattr(self, '_emoji_path_mapping'):
            new_path = self._emoji_path_mapping.get(entity_type.value)
            if new_path:
                self.emoji_image_path = new_path
                logger.debug(f"Updated emoji path for {entity_type.value}: {new_path}")
        else:
            logger.debug(f"No emoji path mapping available for type conversion to {entity_type.value}")
    
    def set_emoji_path_mapping(self, emoji_paths: dict) -> None:
        """Set the mapping of entity types to emoji paths."""
        self._emoji_path_mapping = emoji_paths
    
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
        
        # Use emoji rendering if enabled
        if self.use_emoji:
            self._render_emoji(screen, x, y)
        else:
            # Fallback to colored circle rendering
            self._render_circle(screen, x, y)
    
    def _render_emoji(self, screen, x: float, y: float) -> None:
        """Render emoji image centered at position."""
        # Try to load emoji if not yet loaded
        if not self._emoji_surface and self.emoji_image_path:
            logger.debug(f"Loading emoji surface for {self.entity_type}: {self.emoji_image_path}")
            self._load_emoji_surface()
        
        if not self._emoji_surface:
            # Fallback to circle rendering if emoji loading failed
            logger.debug(f"Emoji surface not available for {self.entity_type}, using circle fallback")
            self._render_circle(screen, x, y)
            return
        
        # Get emoji surface dimensions
        emoji_width = self._emoji_surface.get_width()
        emoji_height = self._emoji_surface.get_height()
        
        # Calculate position to center the emoji
        emoji_x = int(x - emoji_width // 2)
        emoji_y = int(y - emoji_height // 2)
        
        # Blit emoji with transparency
        screen.blit(self._emoji_surface, (emoji_x, emoji_y))
    
    def _render_circle(self, screen, x: float, y: float) -> None:
        """Render colored circle (fallback rendering)."""
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