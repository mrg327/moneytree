"""Video renderer for RPS animations."""

import os
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import pygame
import cv2
import numpy as np
from dataclasses import dataclass

from .game_engine import GameEngine, GameConfig
from .entity_manager import EntityConfig


@dataclass
class VideoConfig:
    """Configuration for video rendering."""
    output_path: str = "video_output/rps_simulation.mp4"
    fps: int = 60
    codec: str = "mp4v"  # Or "H264" if available
    bitrate: int = 8000000  # 8 Mbps
    
    # Audio settings
    add_audio: bool = True
    background_music_path: Optional[str] = None
    sound_effects: bool = True
    
    # Visual settings
    add_hud: bool = True
    show_stats: bool = True
    smooth_camera: bool = True


class VideoRenderer:
    """Renders RPS simulation to video file."""
    
    def __init__(self, game_config: GameConfig = None, entity_config: EntityConfig = None,
                 video_config: VideoConfig = None):
        """Initialize the video renderer.
        
        Args:
            game_config: Game configuration
            entity_config: Entity configuration  
            video_config: Video configuration
        """
        self.game_config = game_config or GameConfig()
        self.entity_config = entity_config or EntityConfig()
        self.video_config = video_config or VideoConfig()
        
        # Create output directory
        output_dir = Path(self.video_config.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        self.video_writer = None
        self.temp_frames: List[np.ndarray] = []
        
    def render_game_to_video(self, progress_callback: Optional[callable] = None) -> str:
        """Render a complete game simulation to video.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the generated video file
        """
        # Initialize game engine
        game_engine = GameEngine(self.game_config, self.entity_config)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*self.video_config.codec)
        self.video_writer = cv2.VideoWriter(
            self.video_config.output_path,
            fourcc,
            self.video_config.fps,
            (self.game_config.width, self.game_config.height)
        )
        
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.video_config.output_path}")
            
        frame_count = 0
        total_frames = int(self.game_config.duration * self.video_config.fps)
        
        def frame_callback(screen: pygame.Surface, current_time: float):
            """Callback for each game frame."""
            nonlocal frame_count
            
            # Convert pygame surface to numpy array
            frame_array = pygame.surfarray.array3d(screen)
            # Rotate and flip to correct orientation for cv2
            frame_array = np.rot90(frame_array)
            frame_array = np.flipud(frame_array)
            # Convert RGB to BGR for OpenCV
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            self.video_writer.write(frame_array)
            
            frame_count += 1
            
            # Call progress callback if provided
            if progress_callback:
                progress = min(frame_count / total_frames, 1.0)
                progress_callback(progress, current_time)
                
        # Set frame callback and start game
        game_engine.set_frame_callback(frame_callback)
        
        try:
            game_engine.start()
        finally:
            # Clean up
            if self.video_writer:
                self.video_writer.release()
            game_engine.cleanup()
            
        return self.video_config.output_path
        
    def render_with_audio(self, background_music_path: Optional[str] = None,
                         progress_callback: Optional[callable] = None) -> str:
        """Render game with audio track.
        
        Args:
            background_music_path: Path to background music file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the generated video file with audio
        """
        # First render video without audio
        video_path = self.render_game_to_video(progress_callback)
        
        if not self.video_config.add_audio:
            return video_path
            
        # Use background music if provided
        music_path = background_music_path or self.video_config.background_music_path
        if not music_path or not os.path.exists(music_path):
            print(f"Warning: Background music not found at {music_path}")
            return video_path
            
        # Create output path for video with audio
        output_with_audio = self.video_config.output_path.replace('.mp4', '_with_audio.mp4')
        
        try:
            # Use ffmpeg to combine video and audio
            import subprocess
            
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', video_path,  # Input video
                '-i', music_path,  # Input audio
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',  # Use AAC audio codec
                '-shortest',  # End when shortest stream ends
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input
                output_with_audio
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return output_with_audio
            else:
                print(f"FFmpeg error: {result.stderr}")
                return video_path
                
        except Exception as e:
            print(f"Error adding audio: {e}")
            return video_path
            
    def create_preview_gif(self, duration: float = 10.0, fps: int = 10) -> str:
        """Create a preview GIF of the game.
        
        Args:
            duration: Duration of the preview in seconds
            fps: Frames per second for the GIF
            
        Returns:
            Path to the generated GIF file
        """
        # Create a shorter config for preview
        preview_config = GameConfig(
            width=self.game_config.width // 2,  # Half resolution
            height=self.game_config.height // 2,
            fps=fps,
            duration=duration
        )
        
        # Initialize game engine
        game_engine = GameEngine(preview_config, self.entity_config)
        
        frames = []
        
        def frame_callback(screen: pygame.Surface, current_time: float):
            """Callback for each game frame."""
            # Convert pygame surface to numpy array
            frame_array = pygame.surfarray.array3d(screen)
            frame_array = np.rot90(frame_array)
            frame_array = np.flipud(frame_array)
            frames.append(frame_array)
            
        # Set frame callback and start game
        game_engine.set_frame_callback(frame_callback)
        
        try:
            game_engine.start()
        finally:
            game_engine.cleanup()
            
        # Create GIF
        gif_path = self.video_config.output_path.replace('.mp4', '_preview.gif')
        
        try:
            from PIL import Image
            
            # Convert frames to PIL Images
            pil_frames = []
            for frame in frames:
                pil_frame = Image.fromarray(frame)
                pil_frames.append(pil_frame)
                
            # Save as GIF
            if pil_frames:
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=int(1000 / fps),  # Duration in milliseconds
                    loop=0
                )
                
        except ImportError:
            print("Warning: PIL not available, cannot create GIF")
            return ""
        except Exception as e:
            print(f"Error creating GIF: {e}")
            return ""
            
        return gif_path
        
    def get_video_info(self) -> dict:
        """Get information about the video configuration.
        
        Returns:
            Dictionary with video information
        """
        return {
            'output_path': self.video_config.output_path,
            'resolution': (self.game_config.width, self.game_config.height),
            'fps': self.video_config.fps,
            'duration': self.game_config.duration,
            'codec': self.video_config.codec,
            'estimated_file_size_mb': self._estimate_file_size(),
            'total_frames': int(self.game_config.duration * self.video_config.fps)
        }
        
    def _estimate_file_size(self) -> float:
        """Estimate the output file size in MB.
        
        Returns:
            Estimated file size in megabytes
        """
        # Rough estimation based on bitrate and duration
        bits_per_second = self.video_config.bitrate
        total_bits = bits_per_second * self.game_config.duration
        total_bytes = total_bits / 8
        total_mb = total_bytes / (1024 * 1024)
        
        return round(total_mb, 2)