#!/usr/bin/env python3
"""
RPS Battle Royale Video Generator.

Creates engaging Rock Paper Scissors battle simulation videos with statistics,
background music, optional audio commentary, and emoji rendering support.
"""

from lib.utils.logging_config import setup_logging, get_logger, LoggedOperation
setup_logging(log_level="INFO", console_output=True, detailed_format=True)
logger = get_logger(__name__)

logger.info("Importing RPS modules")

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

logger.info("Importing: RPS Animation components")
from lib.animation.animation_engine import AnimationEngine, AnimationConfig
from lib.animation.rps_simulator import SimulationConfig
from lib.animation.entity import EntityType

logger.info("Importing: Video processing components")
from lib.video.clip import VideoClip, VideoConfig
from lib.video.ffmpeg_utils import FFmpegUtils

logger.info("Importing: Audio components")
from lib.audio.duration_estimator import AudioDurationEstimator


def create_rps_commentary(simulation_stats: Dict[str, Any]) -> str:
    """
    Generate commentary text for the RPS simulation.
    
    Args:
        simulation_stats: Simulation statistics
        
    Returns:
        Commentary text for TTS
    """
    winner = simulation_stats.get('winner', 'unknown')
    battles = simulation_stats.get('battles_fought', 0)
    duration = simulation_stats.get('simulation_time', 0)
    
    commentary_parts = [
        f"Welcome to the ultimate Rock Paper Scissors battle royale!",
        f"Watch as hundreds of entities battle for dominance.",
        f"After {battles} intense battles lasting {duration:.1f} seconds,",
        f"{winner} emerges victorious!",
        f"The rules are simple: Rock crushes Scissors, Scissors cuts Paper, Paper covers Rock.",
        f"But in this chaotic battlefield, only one type can survive.",
        f"Who will claim victory in this epic showdown?"
    ]
    
    return " ".join(commentary_parts)


def generate_rps_video(
    output_path: str,
    simulation_config: SimulationConfig,
    animation_config: AnimationConfig,
    music_path: Optional[str] = None,
    add_commentary: bool = False,
    use_emojis: bool = False
) -> Dict[str, Any]:
    """
    Generate complete RPS battle video.
    
    Args:
        output_path: Output video file path
        simulation_config: RPS simulation settings
        animation_config: Animation rendering settings
        music_path: Optional background music file
        add_commentary: Whether to add TTS commentary
        
    Returns:
        Generation results
    """
    logger.info("🎮 Starting RPS battle video generation")
    logger.info(f"Output: {output_path}")
    logger.info(f"Entities: {simulation_config.total_entities}")
    logger.info(f"Resolution: {animation_config.screen_width}x{animation_config.screen_height}")
    logger.info(f"Emoji rendering: {'enabled' if use_emojis else 'disabled'}")
    
    try:
        # Step 1: Setup emoji rendering if requested
        emoji_paths = None
        if use_emojis:
            with LoggedOperation(logger, "emoji setup"):
                try:
                    from tools.animation.emoji_renderer import EmojiRenderer
                    
                    renderer = EmojiRenderer(default_size=40)
                    emoji_paths = renderer.render_rps_emojis(size=40)
                    
                    if emoji_paths:
                        logger.info(f"Generated emojis: {list(emoji_paths.keys())}")
                    else:
                        logger.warning("Failed to generate emojis, falling back to circles")
                        use_emojis = False
                        
                except ImportError as e:
                    logger.warning(f"Emoji renderer not available: {e}")
                    use_emojis = False
                except Exception as e:
                    logger.warning(f"Emoji setup failed: {e}")
                    use_emojis = False
        
        # Step 2: Run RPS simulation and capture frames
        with LoggedOperation(logger, "RPS simulation and frame capture"):
            engine = AnimationEngine(animation_config, simulation_config)
            
            # Enable emoji rendering for entities if requested
            if use_emojis and emoji_paths:
                # Hook into the simulator to enable emoji rendering
                original_initialize = engine.simulator.initialize_entities
                
                def initialize_entities_with_emojis():
                    original_initialize()
                    
                    # Enable emoji rendering for all entities
                    for entity in engine.simulator.entities:
                        emoji_path = emoji_paths.get(entity.entity_type.value)
                        if emoji_path:
                            entity.emoji_image_path = emoji_path
                            entity.use_emoji = True
                            entity.set_emoji_path_mapping(emoji_paths)
                    
                    logger.info(f"Enabled emoji rendering for {len(engine.simulator.entities)} entities")
                
                engine.simulator.initialize_entities = initialize_entities_with_emojis
            
            # Enable frame capture for video generation
            animation_config.export_frames = True
            
            # Run headless simulation
            simulation_results = engine.run_headless_simulation()
            
            if not simulation_results.get('success'):
                logger.error("RPS simulation failed")
                return simulation_results
            
            logger.info(f"Simulation complete! Winner: {simulation_results.get('winner', 'Unknown')}")
            logger.info(f"Battles fought: {simulation_results['simulation_stats']['battles_fought']}")
            logger.info(f"Duration: {simulation_results['simulation_stats']['simulation_time']:.1f}s")
        
        # Step 3: Export simulation to video
        with LoggedOperation(logger, "video export from frames"):
            video_result = simulation_results.get('video_export', {})
            
            if not video_result.get('success'):
                logger.error("Video export failed")
                return video_result
            
            base_video_path = video_result['output_path']
            logger.info(f"Base video created: {Path(base_video_path).name}")
            logger.info(f"Video duration: {video_result['duration']:.1f}s")
        
        # Step 4: Generate commentary audio (optional)
        commentary_audio_path = None
        if add_commentary:
            with LoggedOperation(logger, "commentary generation"):
                try:
                    from lib.tts.coqui_speech_generator import CoquiSpeechGenerator, CoquiTTSConfig
                    
                    # Generate commentary text
                    commentary_text = create_rps_commentary(simulation_results['simulation_stats'])
                    logger.info(f"Generated commentary: {len(commentary_text)} characters")
                    
                    # Create simple monologue format
                    monologue = {
                        'generated_text': commentary_text,
                        'word_count': len(commentary_text.split()),
                        'estimated_duration': len(commentary_text) / 15  # Rough estimate
                    }
                    
                    # Generate speech
                    tts_config = CoquiTTSConfig(
                        model_name="tts_models/en/ljspeech/tacotron2-DDC"
                    )
                    speech_gen = CoquiSpeechGenerator(tts_config)
                    
                    if speech_gen.tts:
                        audio_result = speech_gen.generate_speech_from_monologue(monologue)
                        if audio_result['success']:
                            commentary_audio_path = audio_result['output_path']
                            logger.info(f"Commentary audio generated: {Path(commentary_audio_path).name}")
                        else:
                            logger.warning("Commentary generation failed, continuing without audio")
                    else:
                        logger.warning("TTS not available, skipping commentary")
                        
                except Exception as e:
                    logger.warning(f"Commentary generation error: {e}")
        
        # Step 5: Enhance video with audio and effects
        with LoggedOperation(logger, "video enhancement"):
            try:
                with VideoClip(base_video_path) as video_clip:
                    # Add commentary audio if available
                    if commentary_audio_path:
                        narration_result = video_clip.add_narration_audio(commentary_audio_path)
                        if narration_result['success']:
                            logger.info(f"Commentary audio added: {narration_result['narration_duration']:.1f}s")
                    
                    # Add background music if provided
                    if music_path and Path(music_path).exists():
                        music_result = video_clip.add_background_music(
                            music_path,
                            volume=0.3,  # Lower volume for background
                            fade_in=2.0,
                            fade_out=2.0
                        )
                        if music_result['success']:
                            logger.info(f"Background music added: {music_result['effects']}")
                    
                    # Configure final video settings
                    video_config = VideoConfig(
                        quality='high',
                        fps=animation_config.fps,
                        output_format='mp4',
                        vertical_format=False  # Horizontal format for RPS battles
                    )
                    
                    # Render final video
                    logger.info("Rendering final enhanced video")
                    render_result = video_clip.render_video(output_path=output_path, config=video_config)
                    
                    if render_result['success']:
                        logger.info("🎬 RPS battle video completed successfully!")
                        logger.info(f"Final video: {render_result['output_path']}")
                        logger.info(f"File size: {render_result['file_size']:,} bytes")
                        logger.info(f"Duration: {render_result['duration']:.1f} seconds")
                        
                        return {
                            'success': True,
                            'output_path': render_result['output_path'],
                            'simulation_stats': simulation_results['simulation_stats'],
                            'video_info': {
                                'duration': render_result['duration'],
                                'file_size': render_result['file_size'],
                                'resolution': render_result['resolution']
                            }
                        }
                    else:
                        logger.error(f"Final video rendering failed: {render_result.get('error', 'Unknown error')}")
                        return render_result
                        
            except Exception as e:
                logger.error(f"Video enhancement failed: {e}")
                # Return basic video if enhancement fails
                return {
                    'success': True,
                    'output_path': base_video_path,
                    'simulation_stats': simulation_results['simulation_stats'],
                    'warning': f"Enhancement failed: {e}"
                }
        
    except Exception as e:
        logger.error(f"RPS video generation failed: {e}")
        return {'success': False, 'error': str(e)}
    
    finally:
        # Cleanup
        try:
            if 'engine' in locals():
                engine.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


def main():
    """Generate RPS battle royale video."""
    parser = argparse.ArgumentParser(description='RPS Battle Royale Video Generator')
    
    # Basic settings
    parser.add_argument('--output', default='video_output/rps_battle_royale.mp4',
                       help='Output video file path')
    parser.add_argument('--entities', type=int, default=150,
                       help='Total number of entities in simulation')
    parser.add_argument('--duration', type=float, default=120.0,
                       help='Maximum simulation duration in seconds')
    
    # Visual settings
    parser.add_argument('--width', type=int, default=800,
                       help='Video width in pixels')
    parser.add_argument('--height', type=int, default=600,
                       help='Video height in pixels')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video frame rate')
    
    # Entity distribution
    parser.add_argument('--rock-ratio', type=float, default=0.33,
                       help='Ratio of rock entities (0.0-1.0)')
    parser.add_argument('--paper-ratio', type=float, default=0.33,
                       help='Ratio of paper entities (0.0-1.0)')
    parser.add_argument('--scissors-ratio', type=float, default=0.34,
                       help='Ratio of scissors entities (0.0-1.0)')
    
    # Physics settings
    parser.add_argument('--min-speed', type=float, default=30.0,
                       help='Minimum entity speed')
    parser.add_argument('--max-speed', type=float, default=80.0,
                       help='Maximum entity speed')
    parser.add_argument('--no-physics', action='store_true',
                       help='Disable collision physics')
    
    # Audio settings
    parser.add_argument('--music', help='Background music file path')
    parser.add_argument('--commentary', action='store_true',
                       help='Add TTS commentary to video')
    
    # Visual enhancement settings
    parser.add_argument('--emojis', action='store_true',
                       help='Use emoji rendering instead of colored circles')
    
    # Advanced settings
    parser.add_argument('--preset', choices=['fast', 'balanced', 'chaos', 'epic'],
                       default='balanced', help='Simulation preset')
    parser.add_argument('--export-stats', help='Export simulation statistics to JSON file')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.rock_ratio + args.paper_ratio + args.scissors_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.error(f"Entity ratios must sum to 1.0, got {total_ratio:.3f}")
        return
    
    # Apply presets
    preset_configs = {
        'fast': {'entities': 75, 'duration': 60.0, 'fps': 30},
        'balanced': {'entities': 150, 'duration': 120.0, 'fps': 30},
        'chaos': {'entities': 300, 'duration': 180.0, 'fps': 30},
        'epic': {'entities': 500, 'duration': 300.0, 'fps': 60}
    }
    
    if args.preset in preset_configs:
        preset = preset_configs[args.preset]
        if args.entities == 150:  # Only apply if not explicitly set
            args.entities = preset['entities']
        if args.duration == 120.0:
            args.duration = preset['duration']
        if args.fps == 30:
            args.fps = preset['fps']
    
    logger.info(f"🎮 RPS Battle Royale Generator")
    logger.info(f"Preset: {args.preset}")
    logger.info(f"Entities: {args.entities}")
    logger.info(f"Max duration: {args.duration}s")
    logger.info(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    logger.info(f"Emoji rendering: {'enabled' if args.emojis else 'disabled'}")
    
    # Create configurations
    simulation_config = SimulationConfig(
        screen_width=args.width,
        screen_height=args.height,
        total_entities=args.entities,
        entity_distribution={
            EntityType.ROCK: args.rock_ratio,
            EntityType.PAPER: args.paper_ratio,
            EntityType.SCISSORS: args.scissors_ratio
        },
        collision_enabled=True,
        physics_enabled=not args.no_physics,
        min_entity_speed=args.min_speed,
        max_entity_speed=args.max_speed,
        max_simulation_time=args.duration,
        target_fps=args.fps
    )
    
    animation_config = AnimationConfig(
        screen_width=args.width,
        screen_height=args.height,
        fps=args.fps,
        background_color=(245, 245, 220, 255),  # Cream background for better emoji visibility
        export_frames=True,
        headless=True,
        show_stats=True,
        video_duration_limit=args.duration
    )
    
    # Generate video
    result = generate_rps_video(
        output_path=args.output,
        simulation_config=simulation_config,
        animation_config=animation_config,
        music_path=args.music,
        add_commentary=args.commentary,
        use_emojis=args.emojis
    )
    
    if result['success']:
        logger.info("✅ RPS battle video generation completed successfully!")
        logger.info(f"📁 Output: {result['output_path']}")
        
        # Show simulation statistics
        stats = result['simulation_stats']
        logger.info(f"🏆 Winner: {stats['winner']}")
        logger.info(f"⚔️  Battles: {stats['battles_fought']}")
        logger.info(f"⏱️  Duration: {stats['simulation_time']:.1f}s")
        
        # Export statistics if requested
        if args.export_stats:
            try:
                with open(args.export_stats, 'w') as f:
                    json.dump(result['simulation_stats'], f, indent=2)
                logger.info(f"📊 Statistics exported: {args.export_stats}")
            except Exception as e:
                logger.warning(f"Failed to export stats: {e}")
        
        # Show Windows path for convenience
        windows_path = str(result['output_path']).replace('/mnt/c/', 'C:\\\\').replace('/', '\\\\')
        logger.info(f"🪟 Windows path: {windows_path}")
        
    else:
        logger.error("❌ RPS battle video generation failed!")
        if 'error' in result:
            logger.error(f"Error: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()