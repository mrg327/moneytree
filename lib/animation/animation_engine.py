"""
Pygame Animation Engine for RPS Simulation.

Handles rendering, frame capture, and video export for the RPS simulation.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from lib.utils.logging_config import get_logger
from .rps_simulator import RPSSimulator, SimulationConfig
from .entity import EntityType

logger = get_logger(__name__)

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    logger.warning("pygame not available - animation features disabled")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not available - frame capture disabled")


@dataclass
class AnimationConfig:
    """Configuration for animation rendering and export."""
    
    # Display settings
    screen_width: int = 800
    screen_height: int = 600
    fps: int = 60
    background_color: Tuple[int, int, int] = (0, 0, 0)  # Black background
    
    # Export settings
    export_frames: bool = False
    frame_output_dir: str = "temp_media/rps_frames"
    frame_format: str = "png"
    
    # Visual settings
    show_stats: bool = True
    show_fps: bool = True
    font_size: int = 24
    
    # Performance settings
    headless: bool = False  # True for video generation without display
    max_frame_skip: int = 5  # Skip frames if falling behind
    
    # Video export
    video_output_path: str = "video_output/rps_simulation.mp4"
    video_duration_limit: float = 120.0  # 2 minutes max


class AnimationEngine:
    """
    Pygame-based animation engine for RPS simulation.
    
    Handles rendering, user input, frame capture, and video export.
    """
    
    def __init__(self, animation_config: AnimationConfig, simulation_config: SimulationConfig):
        """
        Initialize the animation engine.
        
        Args:
            animation_config: Animation configuration
            simulation_config: Simulation configuration
        """
        self.animation_config = animation_config
        self.simulation_config = simulation_config
        self.simulator = RPSSimulator(simulation_config)
        
        # Pygame components
        self.screen = None
        self.clock = None
        self.font = None
        self.running = False
        
        # Frame capture
        self.frame_count = 0
        self.captured_frames: List[str] = []
        
        # Performance tracking
        self.actual_fps = 0
        self.frame_times: List[float] = []
        
        # Initialize pygame if available
        if HAS_PYGAME and not animation_config.headless:
            self._initialize_pygame()
        
        logger.info(f"Animation engine initialized: "
                   f"{animation_config.screen_width}x{animation_config.screen_height} "
                   f"@ {animation_config.fps}fps")
    
    def _initialize_pygame(self) -> None:
        """Initialize pygame display and components."""
        pygame.init()
        
        # Set up display
        if self.animation_config.headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        self.screen = pygame.display.set_mode(
            (self.animation_config.screen_width, self.animation_config.screen_height)
        )
        pygame.display.set_caption("RPS Battle Simulation")
        
        # Initialize clock and font
        self.clock = pygame.time.Clock()
        
        try:
            self.font = pygame.font.Font(None, self.animation_config.font_size)
        except pygame.error:
            logger.warning("Could not load font, using default")
            self.font = pygame.font.Font(None, 36)
        
        logger.info("Pygame initialized successfully")
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the complete RPS simulation with animation.
        
        Returns:
            Simulation results and statistics
        """
        if not HAS_PYGAME:
            logger.error("pygame not available - cannot run animation")
            return {"error": "pygame not available"}
        
        # Prepare frame capture directory
        if self.animation_config.export_frames:
            self._prepare_frame_capture()
        
        # Initialize simulation
        self.simulator.start_simulation()
        self.running = True
        self.frame_count = 0
        start_time = time.time()
        
        logger.info("Starting RPS simulation animation")
        
        # Main animation loop
        while self.running and self.simulator.is_running:
            frame_start = time.time()
            
            # Handle events
            self._handle_events()
            
            # Update simulation
            dt = self.clock.tick(self.animation_config.fps) / 1000.0
            sim_state = self.simulator.update(dt)
            
            # Render frame
            self._render_frame(sim_state)
            
            # Capture frame if enabled
            if self.animation_config.export_frames:
                self._capture_frame()
            
            # Update display
            if not self.animation_config.headless:
                pygame.display.flip()
            
            # Track performance
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 60:  # Keep last 60 frames
                self.frame_times.pop(0)
            
            self.frame_count += 1
            
            # Check time limits
            if time.time() - start_time > self.animation_config.video_duration_limit:
                logger.info("Animation time limit reached")
                break
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        results = {
            'simulation_stats': self.simulator.get_statistics(),
            'animation_stats': {
                'frames_rendered': self.frame_count,
                'total_time': total_time,
                'average_fps': avg_fps,
                'frames_captured': len(self.captured_frames)
            },
            'winner': self.simulator.winner.value if self.simulator.winner else None,
            'success': True
        }
        
        logger.info(f"Simulation complete! Winner: {results['winner']}")
        logger.info(f"Rendered {self.frame_count} frames in {total_time:.1f}s "
                   f"(avg {avg_fps:.1f} fps)")
        
        return results
    
    def _handle_events(self) -> None:
        """Handle pygame events."""
        if not HAS_PYGAME:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                logger.info("User quit simulation")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset simulation
                    self.simulator.reset()
                    self.simulator.start_simulation()
                    logger.info("Simulation reset")
                elif event.key == pygame.K_SPACE:
                    # Pause/unpause
                    self.simulator.is_running = not self.simulator.is_running
                    logger.info(f"Simulation {'resumed' if self.simulator.is_running else 'paused'}")
    
    def _render_frame(self, sim_state: Dict[str, Any]) -> None:
        """
        Render a single frame of the simulation.
        
        Args:
            sim_state: Current simulation state
        """
        if not HAS_PYGAME or not self.screen:
            return
        
        # Clear screen
        self.screen.fill(self.animation_config.background_color)
        
        # Render entities
        entities = sim_state.get('entities', [])
        for entity in entities:
            entity.render(self.screen)
        
        # Render UI overlay
        if self.animation_config.show_stats:
            self._render_stats(sim_state)
        
        if self.animation_config.show_fps:
            self._render_fps()
    
    def _render_stats(self, sim_state: Dict[str, Any]) -> None:
        """Render simulation statistics overlay."""
        if not self.font:
            return
        
        y_offset = 10
        line_height = 30
        
        # Entity counts
        entity_counts = sim_state.get('entity_counts', {})
        for entity_type, count in entity_counts.items():
            # Convert EntityType to string if needed
            entity_type_str = entity_type.value if hasattr(entity_type, 'value') else str(entity_type)
            color = self._get_entity_color(entity_type_str)
            text = f"{entity_type_str.upper()}: {count}"
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (10, y_offset))
            y_offset += line_height
        
        # Battles and time
        battles = sim_state.get('battles_fought', 0)
        time_str = f"Time: {sim_state.get('simulation_time', 0):.1f}s"
        battles_str = f"Battles: {battles}"
        
        time_surface = self.font.render(time_str, True, (255, 255, 255))
        battles_surface = self.font.render(battles_str, True, (255, 255, 255))
        
        self.screen.blit(time_surface, (10, y_offset))
        self.screen.blit(battles_surface, (10, y_offset + line_height))
        
        # Winner announcement
        if sim_state.get('winner'):
            winner_text = f"🏆 {sim_state['winner'].upper()} WINS!"
            winner_surface = self.font.render(winner_text, True, (255, 255, 0))
            
            # Center the winner text
            text_rect = winner_surface.get_rect()
            text_rect.centerx = self.animation_config.screen_width // 2
            text_rect.y = self.animation_config.screen_height // 2 - 50
            
            # Add background
            pygame.draw.rect(self.screen, (0, 0, 0), 
                           text_rect.inflate(20, 10))
            pygame.draw.rect(self.screen, (255, 255, 0), 
                           text_rect.inflate(20, 10), 2)
            
            self.screen.blit(winner_surface, text_rect)
    
    def _render_fps(self) -> None:
        """Render FPS counter."""
        if not self.font or not self.frame_times:
            return
        
        # Calculate FPS from recent frame times
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        fps_text = f"FPS: {fps:.1f}"
        fps_surface = self.font.render(fps_text, True, (255, 255, 255))
        
        # Position in top-right corner
        text_rect = fps_surface.get_rect()
        text_rect.topright = (self.animation_config.screen_width - 10, 10)
        
        self.screen.blit(fps_surface, text_rect)
    
    def _get_entity_color(self, entity_type: str) -> Tuple[int, int, int]:
        """Get display color for entity type."""
        color_map = {
            'rock': (128, 128, 128),     # Gray
            'paper': (255, 255, 255),    # White
            'scissors': (255, 0, 0)      # Red
        }
        return color_map.get(entity_type.lower(), (255, 255, 255))
    
    def _prepare_frame_capture(self) -> None:
        """Prepare directory for frame capture."""
        frame_dir = Path(self.animation_config.frame_output_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing frames
        for existing_frame in frame_dir.glob(f"*.{self.animation_config.frame_format}"):
            existing_frame.unlink()
        
        self.captured_frames.clear()
        logger.info(f"Frame capture prepared: {frame_dir}")
    
    def _capture_frame(self) -> None:
        """Capture current frame as image file."""
        if not HAS_PYGAME or not HAS_PIL or not self.screen:
            return
        
        # Get surface data
        frame_data = pygame.surfarray.array3d(self.screen)
        frame_data = frame_data.swapaxes(0, 1)  # Correct orientation
        
        # Convert to PIL Image
        frame_image = Image.fromarray(frame_data)
        
        # Save frame
        frame_filename = f"frame_{self.frame_count:06d}.{self.animation_config.frame_format}"
        frame_path = Path(self.animation_config.frame_output_dir) / frame_filename
        
        frame_image.save(frame_path)
        self.captured_frames.append(str(frame_path))
        
        if self.frame_count % 60 == 0:  # Log every second
            logger.debug(f"Captured frame {self.frame_count}")
    
    def export_to_video(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export captured frames to video file.
        
        Args:
            output_path: Output video path, uses default if None
            
        Returns:
            Export results
        """
        if not self.captured_frames:
            return {"error": "No frames captured"}
        
        if output_path is None:
            output_path = self.animation_config.video_output_path
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Import FFmpeg utils from video module
            from lib.video.ffmpeg_utils import FFmpegUtils
            
            ffmpeg = FFmpegUtils()
            
            # Create video from frames
            result = ffmpeg.create_video_from_frames(
                frame_dir=self.animation_config.frame_output_dir,
                output_path=output_path,
                fps=self.animation_config.fps,
                frame_pattern=f"frame_%06d.{self.animation_config.frame_format}"
            )
            
            if result['success']:
                logger.info(f"Video exported successfully: {output_path}")
                return {
                    'success': True,
                    'output_path': output_path,
                    'frames_used': len(self.captured_frames),
                    'duration': len(self.captured_frames) / self.animation_config.fps
                }
            else:
                logger.error(f"Video export failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            logger.error(f"Video export error: {e}")
            return {"error": str(e)}
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if HAS_PYGAME:
            pygame.quit()
        
        # Clean up captured frames if desired
        if self.animation_config.export_frames and self.captured_frames:
            try:
                for frame_path in self.captured_frames:
                    Path(frame_path).unlink(missing_ok=True)
                logger.info("Cleaned up captured frames")
            except Exception as e:
                logger.warning(f"Error cleaning up frames: {e}")
    
    def run_headless_simulation(self) -> Dict[str, Any]:
        """
        Run simulation without display for video generation.
        
        Returns:
            Simulation results
        """
        # Set headless mode
        original_headless = self.animation_config.headless
        self.animation_config.headless = True
        self.animation_config.export_frames = True
        
        try:
            # Initialize headless pygame
            if HAS_PYGAME:
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.animation_config.screen_width, self.animation_config.screen_height)
                )
                # Initialize clock for headless mode
                self.clock = pygame.time.Clock()
                
                # Initialize font for headless mode
                try:
                    self.font = pygame.font.Font(None, self.animation_config.font_size)
                except pygame.error:
                    self.font = pygame.font.Font(None, 36)
            
            # Run simulation
            results = self.run_simulation()
            
            # Export video
            if results.get('success'):
                video_result = self.export_to_video()
                results['video_export'] = video_result
            
            return results
            
        finally:
            # Restore original settings
            self.animation_config.headless = original_headless
            self.cleanup()