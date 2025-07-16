"""Core game engine for rock-paper-scissors simulation."""

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
    
    # Physics settings
    physics_dt: float = 1.0/60.0  # 60 FPS physics
    gravity: tuple = (0, 0)  # No gravity for top-down view


class GameEngine:
    """Main game engine for rock-paper-scissors simulation."""
    
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
        
    def render(self):
        """Render the current frame."""
        # Clear screen
        self.screen.fill(self.game_config.background_color)
        
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
            
    def get_game_stats(self) -> Dict[str, Any]:
        """Get current game statistics.
        
        Returns:
            Dictionary with game statistics
        """
        return {
            'current_time': self.current_time,
            'frame_count': self.frame_count,
            'entity_counts': self.entity_manager.get_entity_counts(),
            'total_entities': self.entity_manager.get_total_count()
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