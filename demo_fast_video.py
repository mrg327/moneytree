#!/usr/bin/env python3
"""
Fast video generation demo that uses existing audio files to bypass TTS delays.

This creates videos quickly by reusing previously generated audio files,
focusing on the video composition and rendering performance.
"""

import sys
import argparse
import time
from pathlib import Path

from lib.utils.logging_config import setup_logging, get_logger, LoggedOperation
from lib.wiki.crawler import WikipediaCrawler
from lib.llm.discussion_generator import HumorousDiscussionGenerator, DiscussionFormat
from lib.video.clip import VideoClip, CaptionStyle, VideoConfig

# Setup logging
setup_logging(log_level="INFO", console_output=True)
logger = get_logger(__name__)


def find_existing_audio(topic: str, audio_dir: Path = Path("audio_output")) -> str:
    """Find existing audio file for the topic."""
    # Try different naming patterns
    patterns = [
        f"{topic}_chattts.wav",
        f"{topic}_coqui.wav",
        f"{topic.lower()}_chattts.wav", 
        f"{topic.lower()}_coqui.wav"
    ]
    
    for pattern in patterns:
        audio_path = audio_dir / pattern
        if audio_path.exists():
            return str(audio_path)
    
    return None


def main():
    """Run fast video generation using existing audio."""
    parser = argparse.ArgumentParser(description='MoneyTree: Fast Video Generation (Existing Audio)')
    parser.add_argument('topic', help='Wikipedia topic to process')
    parser.add_argument('--template', default='downloads/videos/minecraft_parkour.mp4',
                       help='Path to template video file')
    parser.add_argument('--audio', help='Specific audio file to use (overrides auto-detection)')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='low',
                       help='Video output quality (low=fastest)')
    parser.add_argument('--format', choices=['vertical', 'horizontal'], default='vertical',
                       help='Video format')
    parser.add_argument('--preset', choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium'], 
                       default='ultrafast', help='Rendering speed preset')
    
    args = parser.parse_args()
    
    logger.info("MoneyTree: Fast Video Generation")
    logger.info(f"Topic: {args.topic}")
    logger.info(f"Template: {Path(args.template).name}")
    logger.info(f"Quality: {args.quality} (preset: {args.preset})")
    
    # Check template exists
    if not Path(args.template).exists():
        logger.error(f"Template not found: {args.template}")
        return
    
    # Find or specify audio
    if args.audio:
        audio_path = args.audio
        if not Path(audio_path).exists():
            logger.error(f"Audio file not found: {audio_path}")
            return
    else:
        audio_path = find_existing_audio(args.topic)
        if not audio_path:
            logger.error(f"No existing audio found for '{args.topic}' in audio_output/")
            logger.info("Available audio files:")
            audio_dir = Path("audio_output")
            if audio_dir.exists():
                for f in audio_dir.glob("*.wav"):
                    logger.info(f"   - {f.name}")
            logger.info("Generate audio first with: python demo_natural_tts.py")
            return
    
    logger.info(f"Using audio: {Path(audio_path).name}")
    
    # Get Wikipedia content for captions
    with LoggedOperation(logger, "Wikipedia content fetching"):
        crawler = WikipediaCrawler()
        content = crawler.get_page_summary(args.topic)
        
        if not content:
            logger.error(f"Could not find Wikipedia page for '{args.topic}'")
            return
        
        logger.info(f"Found: {content.get('title', args.topic)}")
    
    # Generate content for captions
    with LoggedOperation(logger, "caption content generation"):
        generator = HumorousDiscussionGenerator()
        monologue = generator.generate_discussion(content, DiscussionFormat.MONOLOGUE)
    
    # Extract text for captions
    if 'script' in monologue:
        text_parts = []
        for turn in monologue['script']:
            if hasattr(turn, 'content'):
                text_parts.append(turn.content)
            elif isinstance(turn, dict):
                text_parts.append(turn.get('content', ''))
        caption_text = ' '.join(text_parts)
    else:
        caption_text = content.get('content', '')[:500]  # Fallback to Wikipedia content
    
        logger.info(f"Caption text ready: {len(caption_text)} characters")
    
    # Create video
    with LoggedOperation(logger, "video creation with fast settings"):
    
    try:
        with VideoClip(args.template) as video_clip:
            # Use fast configurations
            if args.quality == 'low':
                video_config = VideoConfig.for_fast_rendering()
                caption_style = CaptionStyle.for_fast_rendering()
            else:
                video_config = VideoConfig(
                    quality=args.quality,
                    fps=24,
                    preset=args.preset,
                    vertical_format=(args.format == 'vertical')
                )
                caption_style = CaptionStyle.for_vertical_video() if args.format == 'vertical' else CaptionStyle()
            
            logger.info(f"Target resolution: {video_config.get_target_resolution()}")
            logger.info(f"Render settings: {video_config.fps}fps, {video_config.preset} preset")
            
            # Add audio
            logger.info("Adding narration audio")
            narration_result = video_clip.add_narration_audio(audio_path)
            if not narration_result['success']:
                logger.error(f"Audio integration failed: {narration_result.get('error')}")
                return
            
            logger.info(f"Audio added: {narration_result['narration_duration']:.1f}s")
            
            # Add captions
            logger.info("Adding synchronized captions")
            caption_result = video_clip.add_synchronized_captions(
                caption_text, 
                audio_path, 
                caption_style
            )
            
            if caption_result['success']:
                logger.info(f"Captions added: {caption_result['caption_count']} segments")
            else:
                logger.warning(f"Caption issues (proceeding anyway): {caption_result}")
            
            # Render video
            logger.info("Rendering final video")
            import time
            start_time = time.time()
            
            render_result = video_clip.render_video(config=video_config)
            
            render_time = time.time() - start_time
            
            if render_result['success']:
                logger.info(f"Video rendered successfully in {render_time:.1f} seconds!")
                logger.info(f"Saved to: {render_result['output_path']}")
                logger.debug(f"File size: {render_result['file_size']:,} bytes")
                logger.debug(f"Duration: {render_result['duration']:.1f} seconds")
                logger.debug(f"Resolution: {render_result['resolution'][0]}x{render_result['resolution'][1]}")
                
                # Windows path
                windows_path = str(render_result['output_path']).replace('/mnt/c/', 'C:\\\\').replace('/', '\\\\')
                logger.info(f"Windows path: {windows_path}")
                
                # Performance summary
                logger.info("Performance Summary:")
                logger.info(f"   Render time: {render_time:.1f}s")
                logger.info(f"   Video duration: {render_result['duration']:.1f}s")
                logger.info(f"   Ratio: {render_time/render_result['duration']:.1f}x realtime")
                
            else:
                logger.error(f"Video rendering failed: {render_result.get('error')}")
    
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return


if __name__ == "__main__":
    main()