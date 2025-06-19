"""
Media processing controller with audio-first optimization.

Orchestrates the optimized pipeline that prevents resource waste by estimating
audio duration early and trimming other media components accordingly.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from lib.utils.logging_config import get_logger
from lib.audio.duration_estimator import AudioDurationEstimator, AudioEstimate
from lib.audio.quality_validator import AudioQualityValidator

logger = get_logger(__name__)

try:
    from moviepy import VideoFileClip, AudioFileClip
    HAS_VIDEO = True
except ImportError:
    HAS_VIDEO = False
    logger.warning("moviepy not available, video optimization will be limited")


@dataclass 
class MediaConfig:
    """
    Configuration for optimized media processing.
    
    Attributes:
        enable_early_trimming: Whether to enable early media trimming
        buffer_factor: Safety buffer for pre-trimming (1.1 = 10% extra)
        enable_quality_validation: Whether to validate audio quality
        max_template_duration: Maximum template duration to process
        enable_background_music_optimization: Whether to optimize background music processing
        temp_dir: Directory for temporary files
    """
    enable_early_trimming: bool = True
    buffer_factor: float = 1.15  # 15% buffer for safety
    enable_quality_validation: bool = True
    max_template_duration: float = 300.0  # 5 minutes max
    enable_background_music_optimization: bool = True
    temp_dir: str = "temp_media"


class MediaController:
    """
    Controls media processing with audio-first optimization approach.
    
    Implements early duration estimation to prevent waste of processing
    resources on video and music that will be trimmed later.
    """
    
    def __init__(self, config: Optional[MediaConfig] = None):
        """
        Initialize the media controller.
        
        Args:
            config: Media processing configuration
        """
        self.config = config or MediaConfig()
        self.duration_estimator = AudioDurationEstimator()
        self.quality_validator = AudioQualityValidator()
        
        # Ensure temp directory exists
        self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"MediaController initialized with early trimming: {self.config.enable_early_trimming}")
    
    def process_content_optimized(self, content: Dict[str, Any], tts_engine: str,
                                template_path: str, music_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process content with audio-first optimization to prevent resource waste.
        
        Args:
            content: Content dictionary (monologue/script)
            tts_engine: TTS engine to use ('chattts' or 'coqui')
            template_path: Path to template video
            music_path: Optional path to background music
            
        Returns:
            Dictionary with processing results and optimization metrics
        """
        optimization_metrics = {
            'early_trimming_enabled': self.config.enable_early_trimming,
            'resources_saved': {},
            'processing_time_saved': 0.0,
            'audio_first_workflow': True
        }
        
        # Step 1: Estimate audio duration early
        logger.info("Step 1: Estimating audio duration for pipeline optimization")
        audio_estimate = self.duration_estimator.estimate_from_monologue(content, tts_engine)
        
        estimated_duration = audio_estimate.estimated_duration
        buffer_duration = estimated_duration * self.config.buffer_factor
        
        logger.info(f"Audio duration estimate: {estimated_duration:.1f}s "
                   f"(with {self.config.buffer_factor:.1%} buffer: {buffer_duration:.1f}s)")
        
        # Step 2: Pre-process template video if early trimming enabled
        processed_template_path = template_path
        if self.config.enable_early_trimming and HAS_VIDEO:
            processed_template_path = self._pre_trim_template_video(
                template_path, buffer_duration, optimization_metrics
            )
        
        # Step 3: Pre-process background music if provided
        processed_music_path = music_path
        if (music_path and self.config.enable_background_music_optimization and 
            self.config.enable_early_trimming):
            processed_music_path = self._pre_trim_background_music(
                music_path, buffer_duration, optimization_metrics
            )
        
        # Step 4: Return optimized media paths and metrics
        return {
            'audio_estimate': audio_estimate,
            'processed_template_path': processed_template_path,
            'processed_music_path': processed_music_path,
            'optimization_metrics': optimization_metrics,
            'buffer_duration': buffer_duration,
            'recommended_tts_config': self._get_recommended_tts_config(audio_estimate, tts_engine)
        }
    
    def finalize_media_with_actual_duration(self, audio_result: Dict[str, Any], 
                                          processed_media: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize media processing with actual audio duration.
        
        Args:
            audio_result: Result from TTS generation
            processed_media: Result from process_content_optimized
            
        Returns:
            Dictionary with final processing results
        """
        if not audio_result.get('success'):
            logger.error("Audio generation failed, cannot finalize media")
            return {'success': False, 'error': 'Audio generation failed'}
        
        actual_duration = audio_result.get('estimated_duration', 0.0)
        estimated_duration = processed_media['audio_estimate'].estimated_duration
        
        # Calculate accuracy of estimate
        accuracy = 0.0
        if estimated_duration > 0:
            accuracy = (1 - abs(actual_duration - estimated_duration) / estimated_duration) * 100
        
        logger.info(f"Audio duration accuracy: {accuracy:.1f}% "
                   f"(estimated: {estimated_duration:.1f}s, actual: {actual_duration:.1f}s)")
        
        # Final trimming if needed
        final_template_path = processed_media['processed_template_path']
        final_music_path = processed_media['processed_music_path']
        
        if self.config.enable_early_trimming and HAS_VIDEO:
            # Check if further trimming is needed
            buffer_duration = processed_media['buffer_duration']
            if actual_duration < buffer_duration * 0.9:  # If actual is much less than buffer
                logger.info("Applying final trim to match actual audio duration")
                final_template_path = self._final_trim_template(
                    processed_media['processed_template_path'], actual_duration
                )
        
        # Update optimization metrics
        optimization_metrics = processed_media['optimization_metrics']
        optimization_metrics['duration_accuracy'] = accuracy
        optimization_metrics['final_duration'] = actual_duration
        optimization_metrics['buffer_efficiency'] = actual_duration / processed_media['buffer_duration']
        
        return {
            'success': True,
            'final_template_path': final_template_path,
            'final_music_path': final_music_path,
            'actual_duration': actual_duration,
            'optimization_metrics': optimization_metrics,
            'quality_metrics': audio_result.get('quality_metrics', {}),
            'savings_summary': self._calculate_savings_summary(optimization_metrics)
        }
    
    def validate_audio_before_media_processing(self, audio_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate audio quality before proceeding with expensive media processing.
        
        Args:
            audio_path: Path to generated audio file
            
        Returns:
            Tuple of (should_proceed, quality_report_dict)
        """
        if not self.config.enable_quality_validation:
            return True, {'validation_skipped': True}
        
        logger.info("Validating audio quality before media processing")
        quality_report = self.quality_validator.analyze_audio(audio_path)
        
        # Determine if quality is acceptable for media processing
        quality_threshold = 0.6  # Minimum acceptable quality score
        should_proceed = quality_report.quality_score >= quality_threshold
        
        if not should_proceed:
            logger.warning(f"Audio quality below threshold ({quality_report.quality_score:.2f} < {quality_threshold})")
            logger.warning("Consider regenerating audio before expensive media processing")
        else:
            logger.info(f"Audio quality acceptable ({quality_report.quality_score:.2f}), proceeding with media processing")
        
        return should_proceed, {
            'quality_score': quality_report.quality_score,
            'speech_percentage': quality_report.speech_percentage,
            'silence_percentage': quality_report.silence_percentage,
            'dynamic_range_db': quality_report.dynamic_range_db,
            'should_proceed': should_proceed,
            'quality_threshold': quality_threshold
        }
    
    def _pre_trim_template_video(self, template_path: str, target_duration: float, 
                               metrics: Dict[str, Any]) -> str:
        """
        Pre-trim template video to avoid processing excess content.
        
        Args:
            template_path: Path to original template video
            target_duration: Target duration including buffer
            metrics: Optimization metrics to update
            
        Returns:
            Path to pre-trimmed template video
        """
        try:
            # Load template to get original duration
            template_clip = VideoFileClip(template_path)
            original_duration = template_clip.duration
            
            if original_duration <= target_duration:
                # No trimming needed
                template_clip.close()
                logger.info(f"Template duration ({original_duration:.1f}s) <= target ({target_duration:.1f}s), no pre-trimming needed")
                return template_path
            
            # Calculate savings
            duration_saved = original_duration - target_duration
            processing_time_saved = duration_saved * 2.5  # Rough estimate: 2.5x processing time
            
            # Create pre-trimmed version
            trimmed_path = self.temp_dir / f"pre_trimmed_{Path(template_path).name}"
            
            logger.info(f"Pre-trimming template: {original_duration:.1f}s -> {target_duration:.1f}s")
            logger.info(f"Estimated processing time saved: {processing_time_saved:.1f}s")
            
            # Trim and save
            trimmed_clip = template_clip.subclipped(0, target_duration)
            trimmed_clip.write_videofile(str(trimmed_path), logger=None)
            
            # Cleanup
            trimmed_clip.close()
            template_clip.close()
            
            # Update metrics
            metrics['resources_saved']['video_duration'] = duration_saved
            metrics['processing_time_saved'] += processing_time_saved
            
            return str(trimmed_path)
            
        except Exception as e:
            logger.error(f"Failed to pre-trim template video: {e}")
            return template_path  # Return original on error
    
    def _pre_trim_background_music(self, music_path: str, target_duration: float,
                                 metrics: Dict[str, Any]) -> str:
        """
        Pre-trim background music to avoid processing excess audio.
        
        Args:
            music_path: Path to original music file
            target_duration: Target duration including buffer
            metrics: Optimization metrics to update
            
        Returns:
            Path to pre-trimmed music file
        """
        try:
            # Load music to get original duration
            music_clip = AudioFileClip(music_path)
            original_duration = music_clip.duration
            
            if original_duration <= target_duration:
                # No trimming needed
                music_clip.close()
                logger.info(f"Music duration ({original_duration:.1f}s) <= target ({target_duration:.1f}s), no pre-trimming needed")
                return music_path
            
            # Calculate savings
            duration_saved = original_duration - target_duration
            
            # Create pre-trimmed version
            trimmed_path = self.temp_dir / f"pre_trimmed_{Path(music_path).name}"
            
            logger.info(f"Pre-trimming background music: {original_duration:.1f}s -> {target_duration:.1f}s")
            
            # Trim and save
            trimmed_clip = music_clip.subclipped(0, target_duration)
            trimmed_clip.write_audiofile(str(trimmed_path), logger=None)
            
            # Cleanup
            trimmed_clip.close()
            music_clip.close()
            
            # Update metrics
            metrics['resources_saved']['music_duration'] = duration_saved
            
            return str(trimmed_path)
            
        except Exception as e:
            logger.error(f"Failed to pre-trim background music: {e}")
            return music_path  # Return original on error
    
    def _final_trim_template(self, template_path: str, actual_duration: float) -> str:
        """
        Apply final trimming to template based on actual audio duration.
        
        Args:
            template_path: Path to template video (may be pre-trimmed)
            actual_duration: Actual audio duration
            
        Returns:
            Path to final trimmed template
        """
        try:
            template_clip = VideoFileClip(template_path)
            current_duration = template_clip.duration
            
            if current_duration <= actual_duration:
                template_clip.close()
                return template_path
            
            # Create final trimmed version
            final_path = self.temp_dir / f"final_trimmed_{Path(template_path).name}"
            
            logger.info(f"Final template trim: {current_duration:.1f}s -> {actual_duration:.1f}s")
            
            # Trim and save
            final_clip = template_clip.subclipped(0, actual_duration)
            final_clip.write_videofile(str(final_path), logger=None)
            
            # Cleanup
            final_clip.close()
            template_clip.close()
            
            return str(final_path)
            
        except Exception as e:
            logger.error(f"Failed to apply final trim: {e}")
            return template_path
    
    def _get_recommended_tts_config(self, audio_estimate: AudioEstimate, engine: str) -> Dict[str, Any]:
        """
        Get recommended TTS configuration based on audio estimate.
        
        Args:
            audio_estimate: Audio duration and quality estimate
            engine: TTS engine being used
            
        Returns:
            Dictionary with recommended configuration adjustments
        """
        recommendations = {
            'engine': engine,
            'confidence_level': audio_estimate.confidence_level,
            'suggested_adjustments': []
        }
        
        # Recommend adjustments based on content characteristics
        if audio_estimate.confidence_level < 0.7:
            recommendations['suggested_adjustments'].append('Use slower speaking rate for better quality')
        
        if 'High number content detected' in audio_estimate.quality_warnings:
            recommendations['suggested_adjustments'].append('Enable enhanced number pronunciation')
        
        if 'Technical content detected' in audio_estimate.quality_warnings:
            recommendations['suggested_adjustments'].append('Use higher quality model for technical terms')
        
        # Engine-specific recommendations
        if engine == 'chattts':
            if audio_estimate.estimated_duration > 60:  # Long content
                recommendations['suggested_adjustments'].append('Consider using consistent voice for long content')
            recommendations['suggested_adjustments'].append('Enable crossfade for smooth concatenation')
        elif engine == 'coqui':
            recommendations['suggested_adjustments'].append('Enable quality validation and enhancement')
        
        return recommendations
    
    def _calculate_savings_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary of resource savings achieved.
        
        Args:
            metrics: Optimization metrics
            
        Returns:
            Dictionary with savings summary
        """
        resources_saved = metrics.get('resources_saved', {})
        
        total_video_saved = resources_saved.get('video_duration', 0.0)
        total_music_saved = resources_saved.get('music_duration', 0.0)
        processing_time_saved = metrics.get('processing_time_saved', 0.0)
        
        return {
            'video_duration_saved_seconds': total_video_saved,
            'music_duration_saved_seconds': total_music_saved,
            'estimated_processing_time_saved_seconds': processing_time_saved,
            'buffer_efficiency': metrics.get('buffer_efficiency', 1.0),
            'duration_accuracy_percent': metrics.get('duration_accuracy', 0.0),
            'early_trimming_effective': total_video_saved > 0 or total_music_saved > 0,
            'summary': self._generate_savings_summary_text(total_video_saved, total_music_saved, processing_time_saved)
        }
    
    def _generate_savings_summary_text(self, video_saved: float, music_saved: float, time_saved: float) -> str:
        """Generate human-readable savings summary."""
        if video_saved == 0 and music_saved == 0:
            return "No media trimming was needed - content duration matched template length"
        
        parts = []
        if video_saved > 0:
            parts.append(f"{video_saved:.1f}s of video processing")
        if music_saved > 0:
            parts.append(f"{music_saved:.1f}s of music processing")
        
        saved_text = " and ".join(parts)
        
        if time_saved > 0:
            return f"Saved {saved_text}, estimated {time_saved:.1f}s processing time reduction"
        else:
            return f"Saved {saved_text}"
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
                logger.info("Cleaned up temporary media files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")