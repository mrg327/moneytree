#!/usr/bin/env python3
"""
Demo showcasing the complete MoneyTree pipeline: Wikipedia → LLM → TTS → Video.

Creates educational videos with synchronized captions and background music.
"""

import sys
import argparse
from pathlib import Path

from lib.utils.logging_config import setup_logging, get_logger, LoggedOperation
from lib.wiki.crawler import WikipediaCrawler
from lib.llm.discussion_generator import HumorousDiscussionGenerator, DiscussionFormat
from lib.llm.llm_generator import LLMMonologueGenerator, LLMConfig
from lib.tts.chattts_speech_generator import ChatTTSSpeechGenerator, ChatTTSConfig
from lib.tts.coqui_speech_generator import CoquiSpeechGenerator, CoquiTTSConfig
from lib.video.clip import VideoClip, CaptionStyle, VideoConfig, create_sample_template
from lib.pipeline.media_controller import MediaController, MediaConfig
from lib.audio.duration_estimator import AudioDurationEstimator

# Setup logging
setup_logging(log_level="INFO", console_output=True)
logger = get_logger(__name__)


def main():
    """Run the complete video generation pipeline."""
    parser = argparse.ArgumentParser(description='MoneyTree: Complete Video Generation Pipeline')
    parser.add_argument('topic', help='Wikipedia topic to process')
    parser.add_argument('--template', help='Path to template video file')
    parser.add_argument('--engine', choices=['chattts', 'coqui'], default='chattts',
                       help='TTS engine to use')
    parser.add_argument('--voice', choices=['natural', 'conversational', 'expressive', 'calm', 'consistent', 'high_quality'],
                       default='natural', help='ChatTTS voice style')
    parser.add_argument('--model', choices=['tacotron2', 'fast_pitch', 'vits', 'jenny'],
                       default='tacotron2', help='Coqui TTS model')
    parser.add_argument('--music', help='Path to background music file')
    parser.add_argument('--use-rule-based', action='store_true',
                       help='Use rule-based generator instead of LLM')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='medium',
                       help='Video output quality')
    parser.add_argument('--format', choices=['vertical', 'horizontal'], default='vertical',
                       help='Video format: vertical for TikTok/YT Shorts, horizontal for standard')
    parser.add_argument('--create-template', help='Create a sample template video at this path')
    parser.add_argument('--disable-optimization', action='store_true',
                       help='Disable pipeline optimizations (for debugging)')
    parser.add_argument('--buffer-factor', type=float, default=1.15,
                       help='Buffer factor for early media trimming (default: 1.15 = 15% buffer)')
    parser.add_argument('--use-whisper', action='store_true',
                       help='Use Whisper ASR for audio-synchronized captions (fallback to speech analysis if fails)')
    
    args = parser.parse_args()
    
    # Create template if requested
    if args.create_template:
        logger.info("Creating sample template video")
        try:
            vertical_format = (args.format == 'vertical')
            template_path = create_sample_template(args.create_template, duration=90.0, vertical=vertical_format)
            logger.info(f"Template created: {template_path}")
            return
        except Exception as e:
            logger.error(f"Template creation failed: {e}")
            return
    
    # Validate template
    if not args.template:
        logger.error("Template video is required. Use --template or --create-template")
        logger.info("Example: python demo_video.py \"Cat\" --create-template template.mp4")
        return
    
    template_path = Path(args.template)
    if not template_path.exists():
        logger.error(f"Template video not found: {template_path}")
        return
    
    logger.info("MoneyTree: Optimized Video Generation Pipeline")
    logger.info(f"Topic: {args.topic}")
    logger.info(f"TTS Engine: {args.engine}")
    logger.info(f"Format: {args.format.title()} ({'9:16' if args.format == 'vertical' else '16:9'})")
    logger.info(f"Template: {template_path.name}")
    if args.music:
        logger.info(f"Background Music: {Path(args.music).name}")
    
    # Initialize optimized pipeline components
    media_config = MediaConfig(
        enable_early_trimming=not args.disable_optimization,
        buffer_factor=args.buffer_factor,
        enable_quality_validation=True,
        enable_background_music_optimization=True
    )
    media_controller = MediaController(media_config)
    
    if not args.disable_optimization:
        logger.info(f"🚀 Pipeline optimizations enabled (buffer factor: {args.buffer_factor:.1%})")
    
    # Step 1: Get Wikipedia content
    with LoggedOperation(logger, "Wikipedia content fetching"):
        crawler = WikipediaCrawler()
        content = crawler.get_page_summary(args.topic)
        
        if not content:
            logger.error(f"Could not find Wikipedia page for '{args.topic}'")
            return
        
        logger.info(f"Found: {content.get('title', args.topic)}")
        logger.debug(f"Description: {content.get('description', 'No description')}")
    
    # Step 2: Generate content and optimize media pipeline
    with LoggedOperation(logger, "content generation and pipeline optimization"):
        if args.use_rule_based:
            logger.info("Using rule-based generation")
            generator = HumorousDiscussionGenerator()
            monologue = generator.generate_discussion(content, DiscussionFormat.MONOLOGUE)
        else:
            logger.info("Using LLM-based generation")
            config = LLMConfig()
            generator = LLMMonologueGenerator(config)
            monologue = generator.generate_monologue(content, target_length=180)
            
            if monologue['model_used'] == 'fallback':
                logger.warning("LLM not available, falling back to rule-based generation")
                generator = HumorousDiscussionGenerator()
                monologue = generator.generate_discussion(content, DiscussionFormat.MONOLOGUE)
        
        logger.info(f"Generated content ({monologue['word_count']} words)")
        logger.debug(f"Estimated duration: {monologue['estimated_duration']:.1f} seconds")
        
        # Optimize media processing pipeline
        logger.info("🎯 Optimizing media pipeline with early duration estimation")
        media_optimization = media_controller.process_content_optimized(
            monologue, args.engine, str(template_path), args.music
        )
        
        # Show optimization results
        audio_estimate = media_optimization['audio_estimate']
        logger.info(f"📊 Audio duration estimate: {audio_estimate.estimated_duration:.1f}s "
                   f"(confidence: {audio_estimate.confidence_level:.2f})")
        
        if audio_estimate.quality_warnings:
            for warning in audio_estimate.quality_warnings:
                logger.warning(f"⚠️  Content analysis: {warning}")
        
        # Show recommended TTS config
        tts_recommendations = media_optimization['recommended_tts_config']
        if tts_recommendations['suggested_adjustments']:
            logger.info("💡 TTS recommendations:")
            for adjustment in tts_recommendations['suggested_adjustments']:
                logger.info(f"   • {adjustment}")
    
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
    
    # Step 3: Generate speech with enhanced quality
    with LoggedOperation(logger, "enhanced speech generation"):
        if args.engine == 'chattts':
            logger.info("Using enhanced ChatTTS for natural speech")
            from lib.tts.chattts_speech_generator import get_recommended_voice_settings
            
            voice_settings = next((v for v in get_recommended_voice_settings() if v['name'] == args.voice),
                                 get_recommended_voice_settings()[0])
            
            tts_config = ChatTTSConfig(
                temperature=voice_settings['temperature'],
                top_k=voice_settings['top_k'],
                top_p=voice_settings['top_p']
            )
            
            speech_gen = ChatTTSSpeechGenerator(tts_config)
            
            if speech_gen.chat:
                audio_result = speech_gen.generate_speech_from_monologue(monologue)
            else:
                logger.error("ChatTTS not available")
                return
        else:
            logger.info("Using enhanced Coqui TTS for synthetic speech")
            
            model_map = {
                'tacotron2': "tts_models/en/ljspeech/tacotron2-DDC",
                'fast_pitch': "tts_models/en/ljspeech/fast_pitch",
                'vits': "tts_models/en/vctk/vits",
                'jenny': "tts_models/en/jenny/jenny"
            }
            
            tts_config = CoquiTTSConfig(
                model_name=model_map[args.model],
                device='cpu'
            )
            
            speech_gen = CoquiSpeechGenerator(tts_config)
            
            if speech_gen.tts:
                audio_result = speech_gen.generate_speech_from_monologue(monologue)
            else:
                logger.error("Coqui TTS not available")
                return
        
        if not audio_result['success']:
            logger.error(f"Speech generation failed: {audio_result.get('error', 'Unknown error')}")
            return
        
        audio_path = audio_result['output_path']
        logger.info(f"Enhanced speech generated: {Path(audio_path).name}")
        logger.info(f"Duration: {audio_result['estimated_duration']:.1f} seconds")
        
        # Show quality metrics if available
        if 'quality_metrics' in audio_result:
            quality = audio_result['quality_metrics']
            logger.info(f"Quality score: {quality.get('quality_score', 'N/A'):.2f}")
            if quality.get('enhancements_applied'):
                logger.info(f"Enhancements applied: {', '.join(quality['enhancements_applied'])}")
        
        # Validate audio before expensive video processing
        should_proceed, validation_report = media_controller.validate_audio_before_media_processing(audio_path)
        if not should_proceed:
            logger.error("Audio quality too low for video processing. Consider regenerating audio.")
            logger.info("Use --disable-optimization to skip quality validation")
            return
        
        # Finalize media optimization with actual audio duration
        logger.info("🎬 Finalizing media optimization with actual audio duration")
        final_media = media_controller.finalize_media_with_actual_duration(audio_result, media_optimization)
        
        if not final_media['success']:
            logger.error("Media finalization failed")
            return
        
        # Show optimization savings
        savings = final_media['savings_summary']
        if savings['early_trimming_effective']:
            logger.info(f"💰 Resource savings: {savings['summary']}")
            logger.info(f"Duration accuracy: {savings['duration_accuracy_percent']:.1f}%")
    
    # Step 4: Create video with optimized media
    with LoggedOperation(logger, "optimized video creation with captions"):
        try:
            # Use optimized template path
            optimized_template = final_media.get('final_template_path', str(template_path))
            optimized_music = final_media.get('final_music_path', args.music)
            
            logger.info(f"Using optimized template: {Path(optimized_template).name}")
            if optimized_music and optimized_music != args.music:
                logger.info(f"Using optimized music: {Path(optimized_music).name}")
            
            with VideoClip(optimized_template) as video_clip:
                # Configure caption style based on format
                if args.format == 'vertical':
                    caption_style = CaptionStyle.for_vertical_video(font_size=36)
                    logger.debug(f"Using vertical caption style: {caption_style.words_per_caption} words per line")
                else:
                    caption_style = CaptionStyle.for_horizontal_video(font_size=56)
                    logger.debug(f"Using horizontal caption style: {caption_style.words_per_caption} words per line")
                
                # Add narration audio
                narration_result = video_clip.add_narration_audio(audio_path)
                if narration_result['success']:
                    logger.info(f"Narration added: {narration_result['narration_duration']:.1f}s")
            
                # Add synchronized captions
                caption_result = video_clip.add_synchronized_captions(
                    caption_text, 
                    audio_path, 
                    caption_style,
                    use_whisper=args.use_whisper
                )
                if caption_result['success']:
                    logger.info(f"Captions added: {caption_result['caption_count']} segments")
            
                # Add background music if provided (use optimized version)
                if optimized_music and Path(optimized_music).exists():
                    music_result = video_clip.add_background_music(
                        optimized_music,
                        volume=0.25,  # Lower volume to not overpower narration
                        fade_in=3.0,
                        fade_out=3.0
                    )
                    if music_result['success']:
                        logger.info(f"Background music added: {music_result['effects']}")
            
            # Configure video output
            video_config = VideoConfig(
                quality=args.quality,
                fps=30,
                output_format='mp4',
                vertical_format=(args.format == 'vertical')
            )
            
            # Render final video
            logger.info("Rendering final video")
            render_result = video_clip.render_video(config=video_config)
            
            if render_result['success']:
                logger.info("Video rendered successfully!")
                logger.info(f"Saved to: {render_result['output_path']}")
                logger.debug(f"File size: {render_result['file_size']:,} bytes")
                logger.debug(f"Duration: {render_result['duration']:.1f} seconds")
                logger.debug(f"Resolution: {render_result['resolution'][0]}x{render_result['resolution'][1]}")
                logger.debug(f"Components: {render_result['components']['captions']} captions, {render_result['components']['audio_tracks']} audio tracks")
                
                # Provide Windows path
                windows_path = str(render_result['output_path']).replace('/mnt/c/', 'C:\\\\').replace('/', '\\\\')
                logger.info(f"Windows path: {windows_path}")
            else:
                logger.error(f"Video rendering failed: {render_result.get('error', 'Unknown error')}")
    
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return
    
    # Final summary with optimization metrics
    logger.info("Optimized Pipeline Summary:")
    logger.info(f"Topic: {content.get('title', args.topic)}")
    logger.info(f"Content: {'LLM' if not args.use_rule_based and monologue.get('model_used') != 'fallback' else 'Rule-based'}")
    logger.info(f"TTS: {args.engine.upper()} (Enhanced)")
    logger.info(f"Format: {args.format.title()} ({'9:16' if args.format == 'vertical' else '16:9'})")
    logger.info(f"Video: {args.quality} quality MP4")
    logger.info(f"Features: Synchronized captions{'+ optimized background music' if optimized_music else ''}")
    
    # Show optimization results
    if not args.disable_optimization:
        logger.info("🚀 Optimization Results:")
        
        # Audio quality metrics
        if 'quality_metrics' in audio_result:
            quality = audio_result['quality_metrics']
            logger.info(f"   Audio quality score: {quality.get('quality_score', 'N/A'):.2f}/1.0")
            logger.info(f"   Speech content: {quality.get('speech_percentage', 'N/A'):.1f}%")
        
        # Resource savings
        savings = final_media['savings_summary']
        logger.info(f"   Duration estimation accuracy: {savings['duration_accuracy_percent']:.1f}%")
        if savings['early_trimming_effective']:
            logger.info(f"   Resources saved: {savings['summary']}")
        else:
            logger.info("   No media trimming was needed (optimal content length)")
        
        # Buffer efficiency
        buffer_efficiency = savings.get('buffer_efficiency', 1.0)
        if buffer_efficiency < 0.9:
            logger.info(f"   Buffer efficiency: {buffer_efficiency:.1%} (consider reducing buffer factor)")
        elif buffer_efficiency > 1.0:
            logger.warning(f"   Buffer exceeded: {buffer_efficiency:.1%} (audio longer than estimated)")
        else:
            logger.info(f"   Buffer efficiency: {buffer_efficiency:.1%} (optimal)")
    
    # Cleanup temporary files
    try:
        media_controller.cleanup_temp_files()
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files: {e}")


if __name__ == "__main__":
    main()
