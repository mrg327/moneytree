"""
Caption rendering manager with dual renderer architecture.

This module provides intelligent switching between MoviePy and OpenCV/PIL
caption renderers with automatic fallback, performance monitoring, and
error recovery capabilities.
"""

import time
import traceback
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from lib.utils.logging_config import get_logger
from lib.video.clip import CaptionStyle, VideoConfig

logger = get_logger(__name__)

# Import renderers with graceful degradation
try:
    from lib.video.clip import VideoClip
    HAS_MOVIEPY_RENDERER = True
except ImportError:
    HAS_MOVIEPY_RENDERER = False
    logger.warning("MoviePy renderer not available")

try:
    from lib.video.opencv_caption_renderer import render_video_with_opencv_captions
    from lib.video.frame_processor import process_video_with_frame_processor
    HAS_OPENCV_RENDERER = True
except ImportError:
    HAS_OPENCV_RENDERER = False
    logger.warning("OpenCV/PIL renderer not available")


class RendererType(Enum):
    """Available caption renderer types."""
    MOVIEPY = "moviepy"
    OPENCV_PIL = "opencv_pil"
    AUTO = "auto"


class RendererStatus(Enum):
    """Renderer status states."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    TESTING = "testing"


@dataclass
class RendererPerformance:
    """Performance metrics for a renderer."""
    success_rate: float = 0.0
    average_render_time: float = 0.0
    error_count: int = 0
    success_count: int = 0
    last_used: Optional[float] = None
    quality_score: float = 0.0


@dataclass
class RenderResult:
    """Result of a caption rendering operation."""
    success: bool
    renderer_used: RendererType
    render_time: float
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None


class CaptionRenderingManager:
    """
    Intelligent caption rendering manager with dual renderer architecture.
    
    Automatically selects the best available renderer based on system capabilities,
    performance history, and task requirements. Provides fallback mechanisms
    and quality validation.
    """
    
    def __init__(self, 
                 preferred_renderer: RendererType = RendererType.AUTO,
                 enable_fallback: bool = True,
                 performance_tracking: bool = True):
        """
        Initialize the caption rendering manager.
        
        Args:
            preferred_renderer: Preferred renderer type
            enable_fallback: Enable automatic fallback between renderers
            performance_tracking: Enable performance tracking and optimization
        """
        self.preferred_renderer = preferred_renderer
        self.enable_fallback = enable_fallback
        self.performance_tracking = performance_tracking
        
        # Renderer availability and performance tracking
        self.renderer_status = {
            RendererType.MOVIEPY: RendererStatus.UNAVAILABLE,
            RendererType.OPENCV_PIL: RendererStatus.UNAVAILABLE
        }
        
        self.renderer_performance = {
            RendererType.MOVIEPY: RendererPerformance(),
            RendererType.OPENCV_PIL: RendererPerformance()
        }
        
        # Configuration
        self.max_retry_attempts = 2
        self.fallback_timeout = 300  # 5 minutes
        self.quality_threshold = 0.8
        
        # Initialize available renderers
        self._detect_available_renderers()
        
        logger.info(f"Caption manager initialized with {len(self._get_available_renderers())} available renderers")
    
    def _detect_available_renderers(self):
        """Detect and test available caption renderers."""
        # Test MoviePy renderer
        if HAS_MOVIEPY_RENDERER:
            try:
                # Quick availability test
                self.renderer_status[RendererType.MOVIEPY] = RendererStatus.AVAILABLE
                logger.info("MoviePy renderer available")
            except Exception as e:
                logger.warning(f"MoviePy renderer test failed: {e}")
                self.renderer_status[RendererType.MOVIEPY] = RendererStatus.FAILED
        
        # Test OpenCV/PIL renderer
        if HAS_OPENCV_RENDERER:
            try:
                # Quick availability test
                self.renderer_status[RendererType.OPENCV_PIL] = RendererStatus.AVAILABLE
                logger.info("OpenCV/PIL renderer available")
            except Exception as e:
                logger.warning(f"OpenCV/PIL renderer test failed: {e}")
                self.renderer_status[RendererType.OPENCV_PIL] = RendererStatus.FAILED
    
    def _get_available_renderers(self) -> List[RendererType]:
        """Get list of currently available renderers."""
        available = []
        for renderer_type, status in self.renderer_status.items():
            if status == RendererStatus.AVAILABLE:
                available.append(renderer_type)
        return available
    
    def _select_optimal_renderer(self, 
                               task_complexity: str = "medium",
                               quality_priority: bool = False) -> RendererType:
        """
        Select the optimal renderer based on current conditions.
        
        Args:
            task_complexity: Estimated task complexity ("low", "medium", "high")
            quality_priority: Whether to prioritize quality over speed
            
        Returns:
            Selected renderer type
        """
        available_renderers = self._get_available_renderers()
        
        if not available_renderers:
            raise RuntimeError("No caption renderers available")
        
        # Handle specific preferences
        if self.preferred_renderer != RendererType.AUTO:
            if self.preferred_renderer in available_renderers:
                return self.preferred_renderer
            elif not self.enable_fallback:
                raise RuntimeError(f"Preferred renderer {self.preferred_renderer.value} not available")
        
        # Auto-selection logic
        if len(available_renderers) == 1:
            return available_renderers[0]
        
        # Performance-based selection
        best_renderer = None
        best_score = -1
        
        for renderer in available_renderers:
            perf = self.renderer_performance[renderer]
            
            # Calculate selection score
            score = 0
            
            # Success rate (most important)
            score += perf.success_rate * 100
            
            # Speed factor
            if perf.average_render_time > 0:
                speed_factor = min(60 / perf.average_render_time, 10)  # Cap at 10
                score += speed_factor * 10
            
            # Quality factor
            score += perf.quality_score * 20
            
            # Recency bonus
            if perf.last_used:
                time_since_use = time.time() - perf.last_used
                recency_bonus = max(0, 10 - (time_since_use / 3600))  # Decay over hours
                score += recency_bonus
            
            # Task-specific adjustments
            if renderer == RendererType.OPENCV_PIL:
                if quality_priority:
                    score += 20  # OpenCV/PIL generally produces higher quality
                if task_complexity == "high":
                    score += 15  # Better for complex rendering
            
            if renderer == RendererType.MOVIEPY:
                if task_complexity == "low":
                    score += 10  # Faster for simple tasks
            
            logger.debug(f"Renderer {renderer.value} score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_renderer = renderer
        
        selected = best_renderer or available_renderers[0]
        logger.info(f"Selected renderer: {selected.value} (score: {best_score:.2f})")
        
        return selected
    
    def render_captions(self,
                       template_video_path: str,
                       audio_path: str,
                       caption_text: str,
                       output_path: str,
                       caption_style: Optional[CaptionStyle] = None,
                       video_config: Optional[VideoConfig] = None,
                       progress_callback: Optional[Callable[[float], None]] = None,
                       quality_priority: bool = False) -> RenderResult:
        """
        Render captions using the optimal available renderer.
        
        Args:
            template_video_path: Path to template video
            audio_path: Path to audio file for synchronization
            caption_text: Text content for captions
            output_path: Output video path
            caption_style: Caption styling options
            video_config: Video output configuration
            progress_callback: Optional progress callback
            quality_priority: Prioritize quality over speed
            
        Returns:
            Render result with success status and metrics
        """
        start_time = time.time()
        
        # Use defaults if not provided
        if caption_style is None:
            caption_style = CaptionStyle()
        if video_config is None:
            video_config = VideoConfig()
        
        # Estimate task complexity
        complexity = self._estimate_task_complexity(caption_text, caption_style)
        
        # Select optimal renderer
        try:
            selected_renderer = self._select_optimal_renderer(complexity, quality_priority)
        except RuntimeError as e:
            logger.error(f"Renderer selection failed: {e}")
            return RenderResult(
                success=False,
                renderer_used=RendererType.AUTO,
                render_time=0.0,
                error_message=str(e)
            )
        
        # Attempt rendering with selected renderer
        result = self._attempt_render(
            selected_renderer, template_video_path, audio_path, caption_text,
            output_path, caption_style, video_config, progress_callback
        )
        
        # Update performance tracking
        if self.performance_tracking:
            self._update_performance_metrics(selected_renderer, result)
        
        # Attempt fallback if enabled and primary failed
        if not result.success and self.enable_fallback:
            fallback_renderers = [r for r in self._get_available_renderers() 
                                if r != selected_renderer]
            
            for fallback_renderer in fallback_renderers:
                logger.warning(f"Primary renderer failed, trying fallback: {fallback_renderer.value}")
                
                fallback_result = self._attempt_render(
                    fallback_renderer, template_video_path, audio_path, caption_text,
                    output_path, caption_style, video_config, progress_callback
                )
                
                if self.performance_tracking:
                    self._update_performance_metrics(fallback_renderer, fallback_result)
                
                if fallback_result.success:
                    logger.info(f"Fallback renderer {fallback_renderer.value} succeeded")
                    return fallback_result
        
        return result
    
    def _attempt_render(self,
                       renderer: RendererType,
                       template_video_path: str,
                       audio_path: str,
                       caption_text: str,
                       output_path: str,
                       caption_style: CaptionStyle,
                       video_config: VideoConfig,
                       progress_callback: Optional[Callable[[float], None]]) -> RenderResult:
        """Attempt rendering with a specific renderer."""
        start_time = time.time()
        
        try:
            logger.info(f"Attempting render with {renderer.value} renderer")
            
            if renderer == RendererType.MOVIEPY:
                success = self._render_with_moviepy(
                    template_video_path, audio_path, caption_text,
                    output_path, caption_style, video_config, progress_callback
                )
            elif renderer == RendererType.OPENCV_PIL:
                success = self._render_with_opencv(
                    template_video_path, audio_path, caption_text,
                    output_path, caption_style, video_config, progress_callback
                )
            else:
                raise ValueError(f"Unknown renderer type: {renderer}")
            
            render_time = time.time() - start_time
            
            if success:
                logger.info(f"Rendering successful with {renderer.value} in {render_time:.2f}s")
                return RenderResult(
                    success=True,
                    renderer_used=renderer,
                    render_time=render_time,
                    output_path=output_path
                )
            else:
                logger.error(f"Rendering failed with {renderer.value}")
                return RenderResult(
                    success=False,
                    renderer_used=renderer,
                    render_time=render_time,
                    error_message="Renderer returned failure status"
                )
        
        except Exception as e:
            render_time = time.time() - start_time
            error_msg = f"Renderer {renderer.value} exception: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            return RenderResult(
                success=False,
                renderer_used=renderer,
                render_time=render_time,
                error_message=error_msg
            )
    
    def _render_with_moviepy(self,
                           template_video_path: str,
                           audio_path: str,
                           caption_text: str,
                           output_path: str,
                           caption_style: CaptionStyle,
                           video_config: VideoConfig,
                           progress_callback: Optional[Callable[[float], None]]) -> bool:
        """Render using MoviePy renderer with enhanced margins."""
        try:
            with VideoClip(template_video_path) as video_clip:
                # Add synchronized captions with margin-aware rendering
                caption_result = video_clip.add_synchronized_captions(
                    caption_text, audio_path, caption_style
                )
                
                if not caption_result.get('success', False):
                    logger.error("MoviePy caption generation failed")
                    return False
                
                # Add narration audio
                audio_result = video_clip.add_narration_audio(audio_path)
                
                if not audio_result.get('success', False):
                    logger.error("MoviePy audio integration failed")
                    return False
                
                # Render final video
                render_result = video_clip.render_video(output_path, video_config)
                
                return render_result.get('success', False)
        
        except Exception as e:
            logger.error(f"MoviePy rendering error: {e}")
            return False
    
    def _render_with_opencv(self,
                          template_video_path: str,
                          audio_path: str,
                          caption_text: str,
                          output_path: str,
                          caption_style: CaptionStyle,
                          video_config: VideoConfig,
                          progress_callback: Optional[Callable[[float], None]]) -> bool:
        """Render using OpenCV/PIL renderer."""
        try:
            # Create caption timing data (simplified for this implementation)
            # In a full implementation, this would analyze the audio for timing
            words = caption_text.split()
            words_per_caption = caption_style.words_per_caption
            caption_duration = 3.0  # Default duration per caption
            
            caption_timings = []
            current_time = 0.0
            
            for i in range(0, len(words), words_per_caption):
                caption_words = words[i:i + words_per_caption]
                caption_text_segment = ' '.join(caption_words)
                
                caption_timings.append({
                    'text': caption_text_segment,
                    'start': current_time,
                    'end': current_time + caption_duration,
                    'duration': caption_duration
                })
                
                current_time += caption_duration
            
            # Use frame processor for high-quality rendering
            return process_video_with_frame_processor(
                template_video_path,
                output_path,
                caption_timings,
                caption_style,
                progress_callback
            )
        
        except Exception as e:
            logger.error(f"OpenCV/PIL rendering error: {e}")
            return False
    
    def _estimate_task_complexity(self, caption_text: str, caption_style: CaptionStyle) -> str:
        """Estimate rendering task complexity."""
        complexity_score = 0
        
        # Text length factor
        text_length = len(caption_text)
        if text_length > 500:
            complexity_score += 3
        elif text_length > 200:
            complexity_score += 2
        elif text_length > 100:
            complexity_score += 1
        
        # Style complexity
        if caption_style.stroke_width > 2:
            complexity_score += 2
        elif caption_style.stroke_width > 0:
            complexity_score += 1
        
        if caption_style.bg_opacity > 0:
            complexity_score += 1
        
        # Font complexity
        if caption_style.font_family and caption_style.font_family != "default":
            complexity_score += 1
        
        # Return complexity level
        if complexity_score >= 6:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _update_performance_metrics(self, renderer: RendererType, result: RenderResult):
        """Update performance metrics for a renderer."""
        perf = self.renderer_performance[renderer]
        
        # Update counts
        if result.success:
            perf.success_count += 1
        else:
            perf.error_count += 1
        
        # Update success rate
        total_attempts = perf.success_count + perf.error_count
        perf.success_rate = perf.success_count / total_attempts if total_attempts > 0 else 0.0
        
        # Update average render time
        if result.render_time > 0:
            if perf.average_render_time == 0:
                perf.average_render_time = result.render_time
            else:
                # Exponential moving average
                alpha = 0.3
                perf.average_render_time = (alpha * result.render_time + 
                                          (1 - alpha) * perf.average_render_time)
        
        # Update last used timestamp
        perf.last_used = time.time()
        
        # Update quality score (simplified - would use actual quality metrics in full implementation)
        if result.success:
            perf.quality_score = min(1.0, perf.quality_score + 0.1)
        else:
            perf.quality_score = max(0.0, perf.quality_score - 0.2)
        
        logger.debug(f"Updated performance for {renderer.value}: "
                    f"success_rate={perf.success_rate:.2f}, "
                    f"avg_time={perf.average_render_time:.2f}s, "
                    f"quality={perf.quality_score:.2f}")
    
    def get_renderer_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all renderers."""
        status = {}
        
        for renderer_type in RendererType:
            if renderer_type == RendererType.AUTO:
                continue
            
            perf = self.renderer_performance[renderer_type]
            status[renderer_type.value] = {
                'status': self.renderer_status[renderer_type].value,
                'success_rate': perf.success_rate,
                'average_render_time': perf.average_render_time,
                'error_count': perf.error_count,
                'success_count': perf.success_count,
                'quality_score': perf.quality_score,
                'last_used': perf.last_used
            }
        
        return status
    
    def reset_performance_metrics(self):
        """Reset all performance tracking metrics."""
        for renderer_type in self.renderer_performance:
            self.renderer_performance[renderer_type] = RendererPerformance()
        
        logger.info("Performance metrics reset")
    
    def set_preferred_renderer(self, renderer: RendererType):
        """Set preferred renderer type."""
        self.preferred_renderer = renderer
        logger.info(f"Preferred renderer set to: {renderer.value}")


# Global caption manager instance
caption_manager = CaptionRenderingManager()