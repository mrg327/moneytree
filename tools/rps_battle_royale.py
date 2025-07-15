#!/usr/bin/env python3
"""
Rock-Paper-Scissors Battle Royale Animation Generator.

Generate engaging battle royale animations using rock-paper-scissors mechanics.
Creates MP4 videos with customizable parameters, entity counts, and visual effects.

Usage:
    uv run python -m tools.rps_battle_royale --output my_battle.mp4
    uv run python -m tools.rps_battle_royale --entities 200 --duration 90
    uv run python -m tools.rps_battle_royale --resolution 1920x1080 --fps 60
    uv run python -m tools.rps_battle_royale --music background.mp3 --preview
"""

import sys
import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.animation.game_engine import GameEngine, GameConfig
from lib.animation.entity_manager import EntityConfig, EntityType
from lib.animation.renderer import VideoRenderer, VideoConfig


def parse_resolution(resolution_str: str) -> Tuple[int, int]:
    """Parse resolution string like '1920x1080' into (width, height).
    
    Args:
        resolution_str: Resolution string in format 'WIDTHxHEIGHT'
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        ValueError: If resolution string is invalid
    """
    try:
        width, height = resolution_str.split('x')
        return int(width), int(height)
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid resolution format: {resolution_str}. Use format like '1920x1080'")


def create_progress_callback(total_duration: float):
    """Create a progress callback function.
    
    Args:
        total_duration: Total duration of the animation
        
    Returns:
        Progress callback function
    """
    start_time = time.time()
    
    def progress_callback(progress: float, current_time: float):
        """Progress callback for video rendering.
        
        Args:
            progress: Progress from 0.0 to 1.0
            current_time: Current simulation time
        """
        elapsed_real_time = time.time() - start_time
        eta = (elapsed_real_time / max(progress, 0.001)) * (1 - progress) if progress > 0 else 0
        
        # Create progress bar
        bar_width = 50
        filled_width = int(bar_width * progress)
        bar = '█' * filled_width + '░' * (bar_width - filled_width)
        
        print(f"\rRendering: [{bar}] {progress:.1%} | "
              f"Time: {current_time:.1f}/{total_duration:.1f}s | "
              f"ETA: {eta:.0f}s", end='', flush=True)
        
    return progress_callback


def main():
    """Run the rock-paper-scissors battle royale generator."""
    parser = argparse.ArgumentParser(
        description='MoneyTree: Rock-Paper-Scissors Battle Royale Animation Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic battle royale with default settings
  uv run python -m tools.rps_battle_royale
  
  # Custom entity count and duration
  uv run python -m tools.rps_battle_royale --entities 300 --duration 120
  
  # High resolution with background music
  uv run python -m tools.rps_battle_royale --resolution 2560x1440 --music background.mp3
  
  # Quick preview GIF
  uv run python -m tools.rps_battle_royale --preview --duration 30
  
  # Disable zone shrinking for pure RPS simulation
  uv run python -m tools.rps_battle_royale --no-zone --duration 180
        """
    )
    
    # Output options
    parser.add_argument('--output', '-o', default='video_output/rps_battle_royale.mp4',
                       help='Output video file path (default: video_output/rps_battle_royale.mp4)')
    parser.add_argument('--preview', action='store_true',
                       help='Create a preview GIF instead of full video')
    
    # Game configuration
    parser.add_argument('--entities', '-e', type=int, default=150,
                       help='Number of entities to spawn (default: 150)')
    parser.add_argument('--duration', '-d', type=float, default=60.0,
                       help='Animation duration in seconds (default: 60.0)')
    parser.add_argument('--resolution', '-r', default='1920x1080',
                       help='Video resolution (default: 1920x1080)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Video framerate (default: 60)')
    
    # Battle royale settings
    parser.add_argument('--no-zone', action='store_true',
                       help='Disable shrinking zone (pure RPS simulation)')
    parser.add_argument('--zone-start', type=float, default=10.0,
                       help='Time before zone starts shrinking (default: 10.0)')
    parser.add_argument('--zone-duration', type=float, default=40.0,
                       help='Duration of zone shrinking (default: 40.0)')
    
    # Entity settings
    parser.add_argument('--entity-size', type=float, default=20.0,
                       help='Size of entities in pixels (default: 20.0)')
    parser.add_argument('--entity-speed', type=float, default=50.0,
                       help='Movement speed of entities (default: 50.0)')
    
    # Visual settings
    parser.add_argument('--background-color', default='50,50,50',
                       help='Background color as R,G,B (default: 50,50,50)')
    parser.add_argument('--rock-color', default='120,120,120',
                       help='Rock color as R,G,B (default: 120,120,120)')
    parser.add_argument('--paper-color', default='255,255,200',
                       help='Paper color as R,G,B (default: 255,255,200)')
    parser.add_argument('--scissors-color', default='200,200,255',
                       help='Scissors color as R,G,B (default: 200,200,255)')
    
    # Audio settings
    parser.add_argument('--music', '-m', help='Background music file path')
    parser.add_argument('--no-audio', action='store_true',
                       help='Disable audio output')
    
    # Advanced settings
    parser.add_argument('--codec', default='mp4v',
                       help='Video codec (default: mp4v)')
    parser.add_argument('--bitrate', type=int, default=8000000,
                       help='Video bitrate in bits per second (default: 8000000)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = parse_resolution(args.resolution)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Parse colors
    def parse_color(color_str: str) -> Tuple[int, int, int]:
        try:
            r, g, b = color_str.split(',')
            return (int(r), int(g), int(b))
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid color format: {color_str}. Use format like '255,0,0'")
    
    try:
        background_color = parse_color(args.background_color)
        rock_color = parse_color(args.rock_color)
        paper_color = parse_color(args.paper_color)
        scissors_color = parse_color(args.scissors_color)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create configurations
    game_config = GameConfig(
        width=width,
        height=height,
        fps=args.fps,
        duration=args.duration,
        background_color=background_color,
        enable_shrinking_zone=not args.no_zone,
        zone_start_time=args.zone_start,
        zone_shrink_duration=args.zone_duration
    )
    
    entity_config = EntityConfig(
        size=args.entity_size,
        speed=args.entity_speed,
        initial_count=args.entities,
        colors={
            EntityType.ROCK: rock_color,
            EntityType.PAPER: paper_color,
            EntityType.SCISSORS: scissors_color
        }
    )
    
    video_config = VideoConfig(
        output_path=args.output,
        fps=args.fps,
        codec=args.codec,
        bitrate=args.bitrate,
        add_audio=not args.no_audio,
        background_music_path=args.music
    )
    
    # Validate inputs
    if args.entities < 3:
        print("Error: Need at least 3 entities for a battle royale")
        sys.exit(1)
    
    if args.duration <= 0:
        print("Error: Duration must be positive")
        sys.exit(1)
    
    if args.music and not os.path.exists(args.music):
        print(f"Warning: Background music file not found: {args.music}")
    
    # Create renderer
    renderer = VideoRenderer(game_config, entity_config, video_config)
    
    if not args.quiet:
        print("MoneyTree Rock-Paper-Scissors Battle Royale Generator")
        print("=" * 60)
        
        # Display configuration
        video_info = renderer.get_video_info()
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {args.duration}s")
        print(f"FPS: {args.fps}")
        print(f"Entities: {args.entities}")
        print(f"Zone enabled: {not args.no_zone}")
        print(f"Estimated file size: {video_info['estimated_file_size_mb']} MB")
        print(f"Total frames: {video_info['total_frames']}")
        print(f"Output: {args.output}")
        print()
    
    # Create progress callback
    progress_callback = None if args.quiet else create_progress_callback(args.duration)
    
    try:
        start_time = time.time()
        
        if args.preview:
            # Create preview GIF
            if not args.quiet:
                print("Creating preview GIF...")
            output_path = renderer.create_preview_gif(
                duration=min(args.duration, 10.0),
                fps=min(args.fps, 15)
            )
        else:
            # Create full video
            if not args.quiet:
                print("Starting video generation...")
            
            if args.music and not args.no_audio:
                output_path = renderer.render_with_audio(
                    args.music, progress_callback
                )
            else:
                output_path = renderer.render_game_to_video(progress_callback)
        
        end_time = time.time()
        
        if not args.quiet:
            print(f"\n\nGeneration completed in {end_time - start_time:.1f} seconds")
            print(f"Output saved to: {output_path}")
            
            # Display final file info
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"File size: {file_size:.2f} MB")
        
        print(output_path)  # Always print output path for scripting
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()