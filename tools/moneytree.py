#!/usr/bin/env python3
"""
MoneyTree: Complete Wikipedia → Video → YouTube Upload Pipeline

Takes a Wikipedia article name and creates + uploads an educational video with
AI-generated descriptions and tags.

This script combines the complete MoneyTree workflow:
1. Wikipedia content extraction
2. LLM content generation (REQUIRED - fails if Ollama unavailable)
3. Video creation with optimized pipeline
4. LLM-generated YouTube descriptions and tags  
5. Automatic YouTube upload

Examples:
    # Basic usage
    uv run python -m tools.moneytree "Quantum Physics" \
      --template downloads/videos/minecraft_parkour.mp4 \
      --client-secrets client_secrets.json

    # Full featured upload
    uv run python -m tools.moneytree "Ancient Rome" \
      --template downloads/videos/minecraft_parkour.mp4 \
      --music downloads/audio/background.mp3 \
      --client-secrets client_secrets.json \
      --privacy public \
      --quality high \
      --include-image \
      --use-whisper

    # Scheduled upload
    uv run python -m tools.moneytree "Climate Change" \
      --template downloads/videos/minecraft_parkour.mp4 \
      --client-secrets client_secrets.json \
      --privacy private \
      --publish-at "2024-12-25T10:00:00Z"
"""

from lib.utils.logging_config import setup_logging, get_logger, LoggedOperation
setup_logging(log_level="INFO", console_output=True, detailed_format=True)
logger = get_logger(__name__)

logger.info("Importing MoneyTree modules")

import sys
import argparse
import re
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

# Wikipedia and content generation
logger.info("Importing: WikipediaCrawler")
from lib.wiki.crawler import WikipediaCrawler
logger.info("Importing: LLMMonologueGenerator, LLMConfig")
from lib.llm.llm_generator import LLMMonologueGenerator, LLMConfig
logger.info("Importing: LLMDescriptionGenerator, DescriptionConfig")
from lib.llm.description_generator import LLMDescriptionGenerator, DescriptionConfig

# Video generation
logger.info("Importing: VideoClip, CaptionStyle, VideoConfig, create_sample_template")
from lib.video.clip import VideoClip, CaptionStyle, VideoConfig, create_sample_template
logger.info("Importing: MediaController, MediaConfig")
from lib.pipeline.media_controller import MediaController, MediaConfig

# YouTube upload
logger.info("Importing: YouTubeClient")
from lib.upload.youtube import YouTubeClient


def find_first_file_in_directory(directory: str, extensions: List[str]) -> Optional[str]:
    """
    Find the first file with specified extensions in a directory.
    
    Args:
        directory: Directory path to search
        extensions: List of file extensions to look for (e.g., ['.mp4', '.avi'])
        
    Returns:
        Path to first matching file, or None if not found
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    for ext in extensions:
        for file_path in dir_path.glob(f"*{ext}"):
            if file_path.is_file():
                return str(file_path)
    
    return None


def get_default_template() -> Optional[str]:
    """Get the first available template video file."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    return find_first_file_in_directory('downloads/videos', video_extensions)


def get_default_music() -> Optional[str]:
    """Get the first available music file."""
    audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']
    return find_first_file_in_directory('downloads/audio', audio_extensions)


def sanitize_topic_for_filename(topic: str) -> str:
    """
    Sanitize a topic name to be safe for use in filenames.
    
    Args:
        topic: The topic name to sanitize
        
    Returns:
        A filename-safe version of the topic
    """
    # Remove or replace problematic characters
    sanitized = re.sub(r'[^\w\s-]', '', topic)  # Remove special chars except word chars, spaces, hyphens
    sanitized = re.sub(r'\s+', '_', sanitized)  # Replace spaces with underscores
    sanitized = re.sub(r'_+', '_', sanitized)   # Replace multiple underscores with single
    sanitized = sanitized.strip('_')            # Remove leading/trailing underscores
    
    # Limit length and ensure it's not empty
    if len(sanitized) > 50:
        sanitized = sanitized[:50].rstrip('_')
    if not sanitized:
        sanitized = "unknown_topic"
    
    return sanitized


def validate_and_set_defaults(args) -> bool:
    """
    Validate requirements and set smart defaults for optional arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if all requirements are met, False otherwise
    """
    logger.info("🔍 Validating requirements and setting defaults...")
    
    # Auto-detect template video if not provided
    if not args.template:
        args.template = get_default_template()
        if args.template:
            logger.info(f"📁 Auto-detected template: {Path(args.template).name}")
        else:
            logger.error("❌ No template video specified and none found in downloads/videos/")
            logger.info("Place a video file in downloads/videos/ or use --template to specify path")
            return False
    
    # Validate template exists
    template_path = Path(args.template)
    if not template_path.exists():
        logger.error(f"❌ Template video not found: {template_path}")
        return False
    
    # Auto-detect music file if not provided
    if not args.music:
        args.music = get_default_music()
        if args.music:
            logger.info(f"🎵 Auto-detected music: {Path(args.music).name}")
        else:
            logger.info("ℹ️  No music file specified and none found in downloads/audio/")
            logger.info("Video will be created without background music")
    
    # Validate music file if provided
    if args.music:
        music_path = Path(args.music)
        if not music_path.exists():
            logger.error(f"❌ Music file not found: {music_path}")
            return False
    
    # Validate YouTube credentials
    secrets_path = Path(args.client_secrets)
    if not secrets_path.exists():
        logger.error(f"❌ Client secrets file not found: {secrets_path}")
        logger.info("Download client_secrets.json from Google Cloud Console or specify path with --client-secrets")
        return False
    
    # Check YouTube API key
    if not os.getenv('YOUTUBE_API_KEY'):
        logger.error("❌ YOUTUBE_API_KEY environment variable not set")
        logger.info("Set your YouTube API key: export YOUTUBE_API_KEY='your_key_here'")
        return False
    
    # Set new defaults based on inverted flags
    args.use_whisper = not args.no_whisper
    args.include_image = not args.no_image
    args.single_word_captions = not args.no_single_word_captions
    
    logger.info("✅ All requirements validated and defaults set")
    logger.info(f"🎯 Defaults: Image={args.include_image}, SingleWord={args.single_word_captions}, Whisper={args.use_whisper}")
    return True


def test_ollama_connection(config: LLMConfig) -> bool:
    """
    Test Ollama connection and fail fast if unavailable.
    
    Args:
        config: LLM configuration
        
    Returns:
        True if Ollama is available, False otherwise
    """
    logger.info("🔌 Testing Ollama connection...")
    
    try:
        # Create a test generator to check connection
        test_generator = LLMMonologueGenerator(config)
        
        # Try a simple connection test
        test_content = {
            "title": "Connection Test",
            "description": "Testing Ollama connectivity",
            "extract": "This is a test to verify Ollama is running and accessible."
        }
        
        # This will attempt to connect and fail fast if Ollama is unavailable
        result = test_generator.generate_monologue(test_content, target_length=50)
        
        if result.get('model_used') == 'fallback':
            logger.error("❌ Ollama connection failed - LLM model not available")
            logger.error("Please ensure Ollama is running and accessible")
            logger.info("WSL users: Make sure Ollama is running on Windows at 172.31.32.1:11434")
            logger.info("Local users: Make sure Ollama is running at localhost:11434")
            return False
        
        logger.info(f"✅ Ollama connection successful (model: {result.get('model_used')})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ollama connection test failed: {e}")
        logger.error("MoneyTree requires Ollama to be running for content generation")
        return False


def main():
    """Run the complete MoneyTree pipeline: Wikipedia → Video → YouTube Upload."""
    parser = argparse.ArgumentParser(
        description='MoneyTree: Complete Wikipedia → Video → YouTube Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses smart defaults)
  uv run python -m tools.moneytree "Quantum Physics"

  # Override specific settings
  uv run python -m tools.moneytree "Ancient Rome" \\
    --privacy public \\
    --quality high \\
    --no-single-word-captions

  # Custom files and scheduled upload
  uv run python -m tools.moneytree "Climate Change" \\
    --template downloads/videos/custom.mp4 \\
    --music downloads/audio/custom.mp3 \\
    --publish-at "2024-12-25T10:00:00Z"

  # Disable default features
  uv run python -m tools.moneytree "History Topic" \\
    --no-image \\
    --no-whisper \\
    --no-single-word-captions

Smart Defaults:
  - Template: First video file in downloads/videos/
  - Music: First audio file in downloads/audio/
  - Client secrets: client_secrets.json
  - Image inclusion: Enabled
  - Single-word captions: Enabled  
  - Whisper captions: Enabled
  - Privacy: Private
  - Made for kids: No (not made for kids)
        """
    )
    
    # Required argument
    parser.add_argument('wikipedia_article', help='Wikipedia article name to create video about')
    
    # Video generation options
    parser.add_argument('--template', help='Path to template video file (default: first file in downloads/videos/)')
    parser.add_argument('--engine', choices=['chattts', 'coqui'], default='coqui',
                       help='TTS engine to use (default: coqui)')
    parser.add_argument('--voice', choices=['natural', 'conversational', 'expressive', 'calm', 'consistent', 'high_quality'],
                       default='natural', help='ChatTTS voice style (default: natural)')
    parser.add_argument('--model', choices=['tacotron2', 'fast_pitch', 'vits', 'jenny', 'xtts_v2'],
                       default='jenny', help='Coqui TTS model (default: tacotron2)')
    parser.add_argument('--music', help='Path to background music file (default: first file in downloads/audio/)')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='high',
                       help='Video output quality (default: high)')
    parser.add_argument('--format', choices=['vertical', 'horizontal'], default='vertical',
                       help='Video format: vertical for TikTok/YT Shorts, horizontal for standard (default: vertical)')
    
    # Advanced video options (now with better defaults)
    parser.add_argument('--disable-optimization', action='store_true',
                       help='Disable pipeline optimizations (for debugging)')
    parser.add_argument('--buffer-factor', type=float, default=1.15,
                       help='Buffer factor for early media trimming (default: 1.15 = 15%% buffer)')
    parser.add_argument('--no-whisper', action='store_true',
                       help='Disable Whisper ASR for audio-synchronized captions (enabled by default)')
    parser.add_argument('--clone-voice', help='Path to reference audio for voice cloning (6+ seconds)')
    parser.add_argument('--language', default='en', help='Language for XTTS-v2 generation (default: en)')
    parser.add_argument('--no-image', action='store_true',
                       help='Disable Wikipedia page image in video (enabled by default)')
    parser.add_argument('--no-single-word-captions', action='store_true',
                       help='Disable single-word captions (enabled by default)')
    
    # YouTube upload options
    parser.add_argument('--client-secrets', default='client_secrets.json', help='Path to YouTube OAuth2 client secrets file (default: client_secrets.json)')
    parser.add_argument('--token-file', default='youtube_token.json', help='Path to YouTube token file (default: youtube_token.json)')
    parser.add_argument('--privacy', choices=['private', 'public', 'unlisted'], default='private',
                       help='YouTube video privacy setting (default: private)')
    parser.add_argument('--language-code', help='YouTube video language code (ISO 639-1, e.g., "en")')
    parser.add_argument('--made-for-kids', action='store_true', help='Mark video as made for kids (COPPA compliance)')
    parser.add_argument('--not-made-for-kids', action='store_true', help='Mark video as NOT made for kids (default behavior)')
    parser.add_argument('--publish-at', help='Schedule publish time (ISO 8601 format, e.g., "2024-12-25T10:00:00Z")')
    parser.add_argument('--no-embedding', action='store_true', help='Disable video embedding')
    parser.add_argument('--hide-stats', action='store_true', help='Hide view count and stats')
    parser.add_argument('--no-notifications', action='store_true', help="Don't notify subscribers")
    parser.add_argument('--thumbnail', help='Path to custom thumbnail image (JPEG/PNG, max 2MB)')
    parser.add_argument('--synthetic-media', action='store_true', help='Mark video as containing AI-generated content (YouTube disclosure)')
    
    # LLM configuration
    parser.add_argument('--llm-model', default='llama3.1:8b', help='LLM model to use (default: llama3.1:8b)')
    parser.add_argument('--llm-temperature', type=float, default=0.8, help='LLM temperature for content generation (default: 0.8)')
    parser.add_argument('--description-temperature', type=float, default=0.7, help='LLM temperature for description generation (default: 0.7)')
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.made_for_kids and args.not_made_for_kids:
        logger.error("❌ Cannot specify both --made-for-kids and --not-made-for-kids")
        return 1
    
    # Validate requirements and set smart defaults
    if not validate_and_set_defaults(args):
        return 1
    
    logger.info("🚀 MoneyTree: Complete Wikipedia → Video → YouTube Pipeline")
    logger.info(f"📖 Article: {args.wikipedia_article}")
    logger.info(f"🎬 Output: {args.format.title()} video ({'9:16' if args.format == 'vertical' else '16:9'})")
    logger.info(f"🎤 TTS Engine: {args.engine.upper()}")
    logger.info(f"📺 Upload Privacy: {args.privacy.title()}")
    
    # Initialize configurations
    llm_config = LLMConfig(
        model=args.llm_model,
        temperature=args.llm_temperature
    )
    
    description_config = DescriptionConfig(
        model=args.llm_model,
        temperature=args.description_temperature
    )
    
    media_config = MediaConfig(
        enable_early_trimming=not args.disable_optimization,
        buffer_factor=args.buffer_factor,
        enable_quality_validation=True,
        enable_background_music_optimization=True
    )
    
    # Test Ollama connection early (fail fast)
    if not test_ollama_connection(llm_config):
        logger.error("❌ MoneyTree requires Ollama for content generation. Exiting.")
        return 1
    
    # Initialize components
    media_controller = MediaController(media_config)
    description_generator = LLMDescriptionGenerator(description_config)
    
    if not args.disable_optimization:
        logger.info(f"🚀 Pipeline optimizations enabled (buffer factor: {args.buffer_factor:.1%})")
    
    try:
        # Step 1: Get Wikipedia content
        with LoggedOperation(logger, "Wikipedia content extraction"):
            crawler = WikipediaCrawler()
            wikipedia_content = crawler.get_page_summary(args.wikipedia_article)
            
            if not wikipedia_content:
                logger.error(f"❌ Could not find Wikipedia page for '{args.wikipedia_article}'")
                logger.info("Please check the article name and try again")
                return 1
            
            # Get categories for enhanced tag generation
            wikipedia_categories = crawler.get_page_categories(args.wikipedia_article)
            
            article_title = wikipedia_content.get('title', args.wikipedia_article)
            logger.info(f"✅ Found article: {article_title}")
            logger.debug(f"Description: {wikipedia_content.get('description', 'No description')}")
            if wikipedia_categories:
                logger.debug(f"Categories: {', '.join(wikipedia_categories[:5])}{'...' if len(wikipedia_categories) > 5 else ''}")
        
        # Step 2: Download Wikipedia image if requested
        wikipedia_image_path = None
        if args.include_image:
            with LoggedOperation(logger, "Wikipedia image download"):
                try:
                    from lib.wiki.image_downloader import download_wikipedia_image
                    
                    wikipedia_image_path = download_wikipedia_image(wikipedia_content)
                    if wikipedia_image_path:
                        logger.info(f"✅ Wikipedia image downloaded: {Path(wikipedia_image_path).name}")
                    else:
                        logger.warning("⚠️  No Wikipedia image available for this page")
                except Exception as e:
                    logger.warning(f"⚠️  Wikipedia image download failed: {e}")
                    wikipedia_image_path = None
        
        # Step 3: Generate content using LLM (required)
        with LoggedOperation(logger, "LLM content generation"):
            logger.info("🧠 Using LLM-based content generation (required)")
            llm_generator = LLMMonologueGenerator(llm_config)
            monologue = llm_generator.generate_monologue(wikipedia_content, target_length=180)
            
            # Check if LLM generation succeeded
            if monologue.get('model_used') == 'fallback':
                logger.error("❌ LLM content generation failed")
                logger.error("MoneyTree requires Ollama for content generation")
                return 1
            
            logger.info(f"✅ Content generated ({monologue['word_count']} words, ~{monologue['estimated_duration']:.1f}s)")
            
            # Optimize media processing pipeline
            logger.info("🎯 Optimizing media pipeline with early duration estimation")
            media_optimization = media_controller.process_content_optimized(
                monologue, args.engine, args.template, args.music
            )
            
            # Show optimization results
            audio_estimate = media_optimization['audio_estimate']
            logger.info(f"📊 Audio duration estimate: {audio_estimate.estimated_duration:.1f}s "
                       f"(confidence: {audio_estimate.confidence_level:.2f})")
            
            if audio_estimate.quality_warnings:
                for warning in audio_estimate.quality_warnings:
                    logger.warning(f"⚠️  Content analysis: {warning}")
        
        # Get text content for captions
        if 'generated_text' in monologue:
            caption_text = monologue['generated_text']
        elif 'script' in monologue:
            # Extract text from script format
            text_parts = []
            for turn in monologue['script']:
                if hasattr(turn, 'content'):
                    text_parts.append(turn.content)
                elif isinstance(turn, dict):
                    text_parts.append(turn.get('content', ''))
            caption_text = ' '.join(text_parts)
        else:
            caption_text = "No caption text available"
        
        # Step 4: Generate speech
        with LoggedOperation(logger, "enhanced speech generation"):
            if args.engine == 'chattts':
                logger.info("🎤 Using enhanced ChatTTS for natural speech")
                from lib.tts.chattts_speech_generator import ChatTTSSpeechGenerator, ChatTTSConfig, get_recommended_voice_settings
                
                voice_settings = next((v for v in get_recommended_voice_settings() if v['name'] == args.voice),
                                     get_recommended_voice_settings()[0])
                
                tts_config = ChatTTSConfig(
                    temperature=voice_settings['temperature'],
                    top_k=voice_settings['top_k'],
                    top_p=voice_settings['top_p']
                )
                
                speech_gen = ChatTTSSpeechGenerator(tts_config)
                
                if not speech_gen.chat:
                    logger.error("❌ ChatTTS not available")
                    return 1
                
                audio_result = speech_gen.generate_speech_from_monologue(monologue)
            else:
                logger.info("🎤 Using enhanced Coqui TTS for synthetic speech")
                from lib.tts.coqui_speech_generator import CoquiSpeechGenerator, CoquiTTSConfig
                
                # Handle XTTS-v2 voice cloning
                if args.model == 'xtts_v2' or args.clone_voice:
                    from lib.tts.coqui_speech_generator import create_voice_cloning_config
                    
                    if args.clone_voice:
                        logger.info(f"Using XTTS-v2 with voice cloning from {Path(args.clone_voice).name}")
                        tts_config = create_voice_cloning_config(
                            reference_audio=args.clone_voice,
                            language=args.language,
                            gpu=False  # CPU for stability in pipeline
                        )
                    else:
                        logger.info("Using XTTS-v2 with default voice")
                        tts_config = CoquiTTSConfig.for_xtts_v2(
                            speaker_wav=None,
                            language=args.language,
                            gpu=False
                        )
                else:
                    # Traditional models
                    model_map = {
                        'tacotron2': "tts_models/en/ljspeech/tacotron2-DDC",
                        'fast_pitch': "tts_models/en/ljspeech/fast_pitch",
                        'vits': "tts_models/en/vctk/vits",
                        'jenny': "tts_models/en/jenny/jenny"
                    }
                    
                    tts_config = CoquiTTSConfig(
                        model_name=model_map[args.model]
                    )
                
                speech_gen = CoquiSpeechGenerator(tts_config)
                
                if not speech_gen.tts:
                    logger.error("❌ Coqui TTS not available")
                    return 1
                
                audio_result = speech_gen.generate_speech_from_monologue(monologue)
            
            if not audio_result['success']:
                logger.error(f"❌ Speech generation failed: {audio_result.get('error', 'Unknown error')}")
                return 1
            
            audio_path = audio_result['output_path']
            actual_duration = audio_result['estimated_duration']
            logger.info(f"✅ Speech generated: {Path(audio_path).name} ({actual_duration:.1f}s)")
            
            # Validate audio quality
            should_proceed, validation_report = media_controller.validate_audio_before_media_processing(audio_path)
            if not should_proceed:
                logger.error("❌ Audio quality too low for video processing")
                logger.info("Use --disable-optimization to skip quality validation")
                return 1
            
            # Finalize media optimization
            final_media = media_controller.finalize_media_with_actual_duration(audio_result, media_optimization)
            if not final_media['success']:
                logger.error("❌ Media finalization failed")
                return 1
        
        # Step 5: Generate YouTube metadata using LLM
        with LoggedOperation(logger, "YouTube metadata generation"):
            logger.info("📝 Generating YouTube description and tags using LLM")
            metadata_result = description_generator.generate_youtube_metadata(
                wikipedia_content, 
                actual_duration,
                wikipedia_categories
            )
            
            if not metadata_result['success']:
                logger.error(f"❌ YouTube metadata generation failed: {metadata_result.get('error')}")
                logger.error("MoneyTree requires Ollama for description generation")
                return 1
            
            # Log the generated metadata
            logger.info(description_generator.format_metadata_summary(metadata_result))
            
            # Use the article title as the YouTube title (more descriptive than generated)
            youtube_title = article_title
            youtube_description = metadata_result['description']
            youtube_tags = metadata_result['tags']
        
        # Step 6: Create video
        video_output_path = None
        with LoggedOperation(logger, "optimized video creation"):
            try:
                # Use optimized media paths
                optimized_template = final_media.get('final_template_path', args.template)
                optimized_music = final_media.get('final_music_path', args.music)
                
                logger.info(f"🎬 Using optimized template: {Path(optimized_template).name}")
                if optimized_music and optimized_music != args.music:
                    logger.info(f"🎵 Using optimized music: {Path(optimized_music).name}")
                
                with VideoClip(optimized_template) as video_clip:
                    # Configure caption style
                    if args.single_word_captions:
                        caption_style = CaptionStyle.for_single_word(font_size=90)
                    elif args.format == 'vertical':
                        caption_style = CaptionStyle.for_vertical_video(font_size=110)
                    else:
                        caption_style = CaptionStyle.for_horizontal_video(font_size=80)
                    
                    # Add narration audio
                    narration_result = video_clip.add_narration_audio(audio_path)
                    if narration_result['success']:
                        logger.info(f"🎤 Narration added: {narration_result['narration_duration']:.1f}s")
                    
                    # Add Wikipedia image if available
                    if wikipedia_image_path:
                        image_result = video_clip.add_wikipedia_image(
                            image_path=wikipedia_image_path,
                            duration=5.0,
                            fade_duration=0.5
                        )
                        if image_result['success']:
                            logger.info("🖼️  Wikipedia image added to video")
                    
                    # Add synchronized captions
                    caption_result = video_clip.add_synchronized_captions(
                        caption_text, 
                        audio_path, 
                        caption_style,
                        use_whisper=args.use_whisper
                    )
                    if caption_result['success']:
                        logger.info(f"📝 Captions added: {caption_result['caption_count']} segments")
                    
                    # Add background music if provided
                    if optimized_music and Path(optimized_music).exists():
                        music_result = video_clip.add_background_music(
                            optimized_music,
                            volume=0.25,
                            fade_in=3.0,
                            fade_out=3.0
                        )
                        if music_result['success']:
                            logger.info(f"🎵 Background music added: {music_result['effects']}")
                
                # Configure video output
                video_config = VideoConfig(
                    quality=args.quality,
                    fps=30,
                    output_format='mp4',
                    vertical_format=(args.format == 'vertical')
                )
                
                # Generate output path
                sanitized_topic = sanitize_topic_for_filename(article_title)
                timestamp = int(time.time())
                video_output_path = f"video_output/{sanitized_topic}_{timestamp}.mp4"
                
                # Render video
                logger.info("🎬 Rendering final video...")
                render_result = video_clip.render_video(output_path=video_output_path, config=video_config)
                
                if not render_result['success']:
                    logger.error(f"❌ Video rendering failed: {render_result.get('error', 'Unknown error')}")
                    return 1
                
                video_output_path = render_result['output_path']
                logger.info(f"✅ Video rendered: {Path(video_output_path).name}")
                logger.info(f"📊 Size: {render_result['file_size']:,} bytes, Duration: {render_result['duration']:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Video creation failed: {e}")
                return 1
        
        # Step 7: Upload to YouTube
        with LoggedOperation(logger, "YouTube upload"):
            try:
                logger.info("📺 Uploading to YouTube...")
                
                # Initialize YouTube client
                youtube_client = YouTubeClient()
                
                # Authenticate
                if not youtube_client.authenticate(args.client_secrets, args.token_file):
                    logger.error("❌ YouTube authentication failed")
                    return 1
                
                # Handle COPPA compliance (default to NOT made for kids)
                made_for_kids = False  # Default to not made for kids
                if args.made_for_kids:
                    made_for_kids = True
                elif args.not_made_for_kids:
                    made_for_kids = False
                
                # Upload video
                upload_result = youtube_client.upload_video(
                    video_path=video_output_path,
                    title=youtube_title,
                    description=youtube_description,
                    tags=youtube_tags,
                    privacy_status=args.privacy,
                    default_language=args.language_code,
                    made_for_kids=made_for_kids,
                    contains_synthetic_media=args.synthetic_media,  # Optional disclosure for AI-generated content
                    publish_at=args.publish_at,
                    embeddable=not args.no_embedding,
                    public_stats_viewable=not args.hide_stats,
                    notify_subscribers=not args.no_notifications
                )
                
                if not upload_result:
                    logger.error("❌ YouTube upload failed")
                    return 1
                
                video_id = upload_result['id']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                logger.info("✅ Upload successful!")
                logger.info(f"📺 Video ID: {video_id}")
                logger.info(f"🔗 URL: {video_url}")
                
                # Get privacy status safely (it's in the 'status' section, not 'snippet')
                privacy_status = upload_result.get('status', {}).get('privacyStatus', args.privacy)
                logger.info(f"🔒 Privacy: {privacy_status}")
                
                # Upload custom thumbnail if provided
                if args.thumbnail:
                    logger.info(f"🖼️  Uploading custom thumbnail: {args.thumbnail}")
                    if youtube_client.set_thumbnail(video_id, args.thumbnail):
                        logger.info("✅ Thumbnail uploaded successfully")
                    else:
                        logger.warning("⚠️  Thumbnail upload failed")
                
                # Show upload status
                status = youtube_client.get_upload_status(video_id)
                if status:
                    upload_status = status.get('status', {})
                    logger.info(f"📊 Upload Status: {upload_status.get('uploadStatus', 'Unknown')}")
                    if args.publish_at:
                        logger.info(f"⏰ Scheduled for: {args.publish_at}")
                
            except Exception as e:
                logger.error(f"❌ YouTube upload failed: {e}")
                return 1
        
        # Final summary
        logger.info("🎉 MoneyTree Pipeline Complete!")
        logger.info(f"📖 Article: {article_title}")
        logger.info(f"🎬 Video: {Path(video_output_path).name}")
        logger.info(f"📺 YouTube: {video_url}")
        logger.info(f"🔒 Privacy: {args.privacy}")
        if args.publish_at:
            logger.info(f"⏰ Scheduled: {args.publish_at}")
        
        # Show optimization summary
        if not args.disable_optimization:
            savings = final_media['savings_summary']
            logger.info(f"🚀 Optimization: {savings['duration_accuracy_percent']:.1f}% duration accuracy")
            if savings['early_trimming_effective']:
                logger.info(f"💰 Resource savings: {savings['summary']}")
        
        # Cleanup
        try:
            media_controller.cleanup_temp_files()
        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup temp files: {e}")
        
        logger.info("✨ MoneyTree: Wikipedia → Video → YouTube complete!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
