"""Core game engine for rock-paper-scissors battle royale."""

import time
import math
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import pygame
import pymunk
import numpy as np

from .entity_manager import EntityManager, EntityConfig, EntityType


@dataclass
class GameConfig:
    """Configuration for the game engine."""
    width: int = 1920
    height: int = 1080
    fps: int = 60
    duration: float = 60.0  # seconds
    background_color: tuple = (50, 50, 50)  # Dark gray
    
    # Battle royale settings
    enable_shrinking_zone: bool = True
    zone_start_time: float = 10.0  # seconds
    zone_shrink_duration: float = 40.0  # seconds
    zone_damage_rate: float = 10.0  # damage per second outside zone
    
    # Physics settings
    physics_dt: float = 1.0/60.0  # 60 FPS physics
    gravity: tuple = (0, 0)  # No gravity for top-down view


class BattleRoyaleZone:
    """Manages the shrinking battle royale zone."""
    
    def __init__(self, config: GameConfig):
        """Initialize the battle royale zone.
        
        Args:
            config: Game configuration
        """
        self.config = config
        self.center_x = config.width // 2
        self.center_y = config.height // 2
        self.max_radius = min(config.width, config.height) // 2 - 50
        self.min_radius = 100
        self.current_radius = self.max_radius
        self.active = False
        self.start_time = 0
        
    def update(self, current_time: float):
        """Update the zone state.
        
        Args:
            current_time: Current simulation time
        """
        if not self.config.enable_shrinking_zone:
            return
            
        if current_time >= self.config.zone_start_time and not self.active:
            self.active = True
            self.start_time = current_time
            
        if self.active:
            elapsed = current_time - self.start_time
            progress = min(elapsed / self.config.zone_shrink_duration, 1.0)
            
            # Smooth shrinking using easing function
            ease_progress = 1 - (1 - progress) ** 2  # Ease out quadratic
            self.current_radius = self.max_radius - (self.max_radius - self.min_radius) * ease_progress
            
    def is_inside_zone(self, x: float, y: float) -> bool:
        """Check if a point is inside the safe zone.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is inside safe zone
        """
        if not self.active:
            return True
            
        distance = math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        return distance <= self.current_radius
        
    def draw(self, screen: pygame.Surface):
        """Draw the zone boundary.
        
        Args:
            screen: Pygame surface to draw on
        """
        if not self.active:
            return
            
        # Just draw the zone boundary for now
        pass
        
        # Draw zone boundary
        radius = max(1, int(self.current_radius))  # Ensure positive radius
        pygame.draw.circle(screen, (255, 255, 0), (self.center_x, self.center_y), 
                         radius, 3)


class GameEngine:
    """Main game engine for rock-paper-scissors battle royale."""
    
    def __init__(self, game_config: GameConfig = None, entity_config: EntityConfig = None):
        """Initialize the game engine.
        
        Args:
            game_config: Game configuration
            entity_config: Entity configuration
        """
        self.game_config = game_config or GameConfig()
        self.entity_config = entity_config or EntityConfig()
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.Surface((self.game_config.width, self.game_config.height))
        self.clock = pygame.time.Clock()
        
        # Initialize physics
        self.space = pymunk.Space()
        self.space.gravity = self.game_config.gravity
        
        # Initialize game components
        self.entity_manager = EntityManager(
            self.entity_config, self.space, 
            self.game_config.width, self.game_config.height
        )
        self.zone = BattleRoyaleZone(self.game_config)
        
        # Game state
        self.running = False
        self.paused = False
        self.current_time = 0.0
        self.frame_count = 0
        
        # Callbacks
        self.on_frame_callback: Optional[Callable[[pygame.Surface, float], None]] = None
        self.on_collision_callback: Optional[Callable[[EntityType, EntityType], None]] = None
        
    def set_frame_callback(self, callback: Callable[[pygame.Surface, float], None]):
        """Set callback for each frame.
        
        Args:
            callback: Function to call with (screen, current_time) each frame
        """
        self.on_frame_callback = callback
        
    def set_collision_callback(self, callback: Callable[[EntityType, EntityType], None]):
        """Set callback for entity collisions.
        
        Args:
            callback: Function to call with (winner_type, loser_type) on collision
        """
        self.on_collision_callback = callback
        
    def start(self):
        """Start the game simulation."""
        self.running = True
        self.current_time = 0.0
        self.frame_count = 0
        
        # Spawn initial entities
        self.entity_manager.spawn_entities()
        
        # Main game loop
        while self.running and self.current_time < self.game_config.duration:
            dt = self.clock.tick(self.game_config.fps) / 1000.0
            self.current_time += dt
            
            if not self.paused:
                self.update(dt)
                
            self.render()
            
            # Check win condition
            if self.check_win_condition():
                break
                
            self.frame_count += 1
            
    def update(self, dt: float):
        """Update game state.
        
        Args:
            dt: Delta time since last update
        """
        # Update physics
        self.space.step(self.game_config.physics_dt)
        
        # Update game components
        self.entity_manager.update(dt)
        self.zone.update(self.current_time)
        
        # Handle zone damage
        self._handle_zone_damage()
        
    def _handle_zone_damage(self):
        """Handle entities taking damage outside the safe zone."""
        if not self.zone.active:
            return
            
        for entity in self.entity_manager.entities:
            if not entity.alive:
                continue
                
            x, y = entity.body.position
            if not self.zone.is_inside_zone(x, y):
                # Apply zone damage (for now, just push entities toward center)
                center_x, center_y = self.zone.center_x, self.zone.center_y
                dx = center_x - x
                dy = center_y - y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    # Push toward center
                    push_force = 200.0  # Adjust as needed
                    entity.body.force = (
                        entity.body.force.x + (dx / distance) * push_force,
                        entity.body.force.y + (dy / distance) * push_force
                    )
                    
    def render(self):
        """Render the current frame."""
        # Clear screen
        self.screen.fill(self.game_config.background_color)
        
        # Draw zone
        self.zone.draw(self.screen)
        
        # Draw entities
        self.entity_manager.draw_all(self.screen)
        
        # Draw UI
        self._draw_ui()
        
        # Call frame callback if set
        if self.on_frame_callback:
            self.on_frame_callback(self.screen, self.current_time)
            
    def _draw_ui(self):
        """Draw user interface elements."""
        # Font for UI text
        font = pygame.font.Font(None, 36)
        
        # Entity counts
        counts = self.entity_manager.get_entity_counts()
        y_offset = 20
        
        for entity_type, count in counts.items():
            color = self.entity_config.colors[entity_type]
            text = font.render(f"{entity_type.value.title()}: {count}", True, color)
            self.screen.blit(text, (20, y_offset))
            y_offset += 40
            
        # Time remaining
        time_remaining = max(0, self.game_config.duration - self.current_time)
        time_text = font.render(f"Time: {time_remaining:.1f}s", True, (255, 255, 255))
        self.screen.blit(time_text, (20, self.game_config.height - 60))
        
        # Zone info
        if self.zone.active:
            zone_text = font.render(f"Zone: {self.zone.current_radius:.0f}px", True, (255, 255, 0))
            self.screen.blit(zone_text, (20, self.game_config.height - 100))
            
    def check_win_condition(self) -> bool:
        """Check if the game has ended.
        
        Returns:
            True if game should end
        """
        counts = self.entity_manager.get_entity_counts()
        non_zero_counts = [count for count in counts.values() if count > 0]
        
        # Game ends if only one type remains or no entities left
        return len(non_zero_counts) <= 1 or self.entity_manager.get_total_count() == 0
        
    def get_winner(self) -> Optional[EntityType]:
        """Get the winning entity type.
        
        Returns:
            Winning entity type or None if no winner
        """
        counts = self.entity_manager.get_entity_counts()
        max_count = max(counts.values())
        
        if max_count == 0:
            return None
            
        # Find the type with the most entities
        for entity_type, count in counts.items():
            if count == max_count:
                return entity_type
                
        return None
        
    def get_game_stats(self) -> Dict[str, Any]:
        """Get current game statistics.
        
        Returns:
            Dictionary with game statistics
        """
        return {
            'current_time': self.current_time,
            'frame_count': self.frame_count,
            'entity_counts': self.entity_manager.get_entity_counts(),
            'total_entities': self.entity_manager.get_total_count(),
            'zone_active': self.zone.active,
            'zone_radius': self.zone.current_radius if self.zone.active else self.zone.max_radius,
            'winner': self.get_winner()
        }
        
    def pause(self):
        """Pause the game."""
        self.paused = True
        
    def resume(self):
        """Resume the game."""
        self.paused = False
        
    def stop(self):
        """Stop the game."""
        self.running = False
        
    def cleanup(self):
        """Clean up resources."""
        self.entity_manager.cleanup()
        pygame.quit()