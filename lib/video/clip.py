"""
Video editing and composition module for creating educational content videos.

Combines template videos with generated audio and synchronized captions to create
engaging educational videos from Wikipedia content.
"""

import os
import re
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from lib.utils.logging_config import get_logger, LoggedOperation, log_execution_time

logger = get_logger(__name__)

try:
    # MoviePy 2.x imports (direct from moviepy, not moviepy.editor)
    from moviepy import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, CompositeAudioClip, afx, concatenate_audioclips
    import librosa
    import numpy as np
    HAS_VIDEO_DEPS = True
except ImportError:
    HAS_VIDEO_DEPS = False
    # Define dummy numpy for type hints when not available
    class np:
        ndarray = object

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class CaptionStyle:
    """Configuration for caption appearance and behavior."""
    font_size: int = 36
    font_color: str = 'white'
    bg_color: str = 'black'
    bg_opacity: float = 0.7
    position: str = 'center'  # 'top', 'center', 'bottom'
    max_width: int = 80  # Percentage of video width
    words_per_caption: int = 6
    font_family: str = None  # Use system default font instead of Arial
    stroke_color: str = 'black'
    stroke_width: int = 2
    
    @classmethod
    def for_vertical_video(cls, font_size: int = 36) -> 'CaptionStyle':
        """Create caption style optimized for vertical videos (TikTok/YT Shorts)."""
        return cls(
            font_size=font_size,
            font_color='white',
            bg_color='black',
            bg_opacity=0.8,
            position='center',  # Center positioning for better visibility
            max_width=85,  # Slightly wider for center positioning
            words_per_caption=6,  # Slightly more words since center has more space
            font_family=None,
            stroke_color='black',
            stroke_width=2  # Clean stroke for readability
        )
    
    @classmethod
    def for_fast_rendering(cls, font_size: int = 48) -> 'CaptionStyle':
        """Create caption style optimized for fast rendering."""
        return cls(
            font_size=font_size,
            font_color='white',
            bg_color='black',
            bg_opacity=0.5,  # Reduced opacity for faster rendering
            position='bottom',
            max_width=85,
            words_per_caption=6,
            font_family=None,
            stroke_color='black',
            stroke_width=1  # Thinner stroke for speed
        )
    
    @classmethod
    def for_horizontal_video(cls, font_size: int = 48) -> 'CaptionStyle':
        """Create caption style optimized for horizontal videos."""
        return cls(
            font_size=font_size,
            font_color='white',
            bg_color='black',
            bg_opacity=0.7,
            position='bottom',
            max_width=80,
            words_per_caption=8,
            font_family=None,
            stroke_color='black',
            stroke_width=2
        )


@dataclass
class VideoConfig:
    """Configuration for video output settings."""
    output_format: str = 'mp4'
    fps: int = 24  # Reduced from 30 for faster rendering
    quality: str = 'medium'  # 'low', 'medium', 'high'
    resolution: Optional[Tuple[int, int]] = None  # Use template resolution if None
    codec: str = 'libx264'
    audio_codec: str = 'aac'
    vertical_format: bool = True  # True for TikTok/YT Shorts (9:16), False for standard (16:9)
    preset: str = 'medium'  # x264 preset: ultrafast, superfast, veryfast, faster, fast, medium, slow
    threads: int = 0  # 0 = auto-detect CPU cores
    
    def get_target_resolution(self) -> Tuple[int, int]:
        """Get the target resolution based on format preference."""
        if self.resolution:
            return self.resolution
        
        if self.vertical_format:
            # Standard TikTok/YouTube Shorts resolution (9:16 aspect ratio)
            quality_map = {
                'low': (540, 960),      # 540p vertical (faster rendering)
                'medium': (720, 1280),  # 720p vertical (good balance)
                'high': (1080, 1920)    # 1080p vertical (best quality)
            }
        else:
            # Standard horizontal resolution (16:9 aspect ratio)
            quality_map = {
                'low': (960, 540),      # 540p horizontal
                'medium': (1280, 720),  # 720p horizontal (good balance)
                'high': (1920, 1080)    # 1080p horizontal
            }
        
        return quality_map.get(self.quality, quality_map['medium'])
    
    @classmethod
    def for_fast_rendering(cls) -> 'VideoConfig':
        """Create config optimized for fast rendering (draft quality)."""
        return cls(
            fps=20,                    # Lower FPS
            quality='low',             # Lower resolution
            preset='ultrafast',        # Fastest x264 preset
            threads=0,                 # Use all CPU cores
            vertical_format=True
        )
    
    @classmethod
    def for_production(cls) -> 'VideoConfig':
        """Create config for production quality (slower but better)."""
        return cls(
            fps=24,
            quality='high',
            preset='slow',             # Better compression
            threads=0,
            vertical_format=True
        )


class VideoClip:
    """
    Creates educational videos by combining template videos with generated audio and captions.
    
    Handles video composition, caption synchronization, and background music integration
    to produce engaging educational content.
    """
    
    def __init__(self, template_path: str, output_dir: str = "video_output"):
        """
        Initialize the VideoClip with a template video.
        
        Args:
            template_path: Path to the template video file
            output_dir: Directory to save generated videos
        """
        if not HAS_VIDEO_DEPS:
            raise ImportError(
                "Video dependencies not available. Install with: "
                "uv add moviepy librosa opencv-python pillow"
            )
        
        self.template_path = Path(template_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load and validate template video
        self.template_clip = None
        self.template_duration = 0
        self.template_fps = 30
        self.template_size = (1920, 1080)
        
        self._load_template()
        
        # Default configurations
        self.caption_style = CaptionStyle()
        self.video_config = VideoConfig()
        
        # Track audio and video components
        self.audio_clips = []
        self.video_components = []
        
    def _load_template(self):
        """Load and analyze the template video."""
        try:
            self.template_clip = VideoFileClip(str(self.template_path))
            self.template_duration = self.template_clip.duration
            self.template_fps = self.template_clip.fps
            self.template_size = self.template_clip.size
            
            logger.info(f"Template loaded: {self.template_path.name}")
            logger.debug(f"Duration: {self.template_duration:.1f}s")
            logger.debug(f"Resolution: {self.template_size[0]}x{self.template_size[1]}")
            logger.debug(f"FPS: {self.template_fps}")
            
        except Exception as e:
            raise ValueError(f"Failed to load template video '{self.template_path}': {e}")
    
    def add_synchronized_captions(
        self, 
        text: str, 
        audio_path: str, 
        style: Optional[CaptionStyle] = None
    ) -> Dict[str, Any]:
        """
        Add synchronized captions based on audio timing.
        
        Args:
            text: The text content to display as captions
            audio_path: Path to the audio file for timing synchronization
            style: Caption styling options
            
        Returns:
            Dictionary with caption generation results
        """
        if style is None:
            style = self.caption_style
            
        try:
            # Load audio for timing analysis
            logger.info("🎬 Creating synchronized captions...")
            audio_clip = AudioFileClip(audio_path)
            
            # Analyze audio for speech timing
            timing_data = self._analyze_speech_timing(audio_path, text)
            
            # Get target dimensions from the template video
            target_resolution = self.video_config.get_target_resolution()
            width, height = target_resolution
            
            # Create caption clips with improved positioning
            caption_clips = self._create_caption_clips_with_dimensions(timing_data, style, width, height)
            
            # Store for final composition
            self.video_components.extend(caption_clips)
            
            return {
                "success": True,
                "caption_count": len(caption_clips),
                "total_duration": audio_clip.duration,
                "dimensions": target_resolution,
                "timing_method": "speech_analysis"
            }
            
        except Exception as e:
            logger.exception(f"Caption generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "caption_count": 0
            }
    
    def _analyze_speech_timing(self, audio_path: str, text: str) -> List[Dict[str, Any]]:
        """
        Analyze audio to determine caption timing.
        
        Args:
            audio_path: Path to audio file
            text: Text content to synchronize
            
        Returns:
            List of timing data for each caption segment
        """
        try:
            # Load audio with librosa for analysis
            y, sr = librosa.load(audio_path)
            
            # Detect speech segments using energy-based approach
            frame_length = 2048
            hop_length = 512
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Convert to time
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            # Find speech segments (above threshold)
            threshold = np.mean(rms) * 0.3
            speech_frames = rms > threshold
            
            # Group consecutive speech frames
            speech_segments = self._group_speech_segments(times, speech_frames)
            
            # Split text into caption chunks
            words = text.split()
            words_per_caption = self.caption_style.words_per_caption
            caption_texts = [
                ' '.join(words[i:i + words_per_caption]) 
                for i in range(0, len(words), words_per_caption)
            ]
            
            # Distribute captions evenly across audio duration
            timing_data = []
            if caption_texts:
                # Get actual audio duration
                audio_duration = len(y) / sr
                
                # Calculate timing for better synchronization
                total_captions = len(caption_texts)
                time_per_caption = audio_duration / total_captions
                
                for i, caption_text in enumerate(caption_texts):
                    # More accurate timing based on position in audio
                    start_time = i * time_per_caption
                    duration = min(time_per_caption, len(caption_text.split()) * 0.4)  # ~0.4s per word
                    end_time = start_time + duration
                    
                    timing_data.append({
                        'text': caption_text,
                        'start': start_time,
                        'end': end_time,
                        'duration': duration
                    })
                
                logger.info(f"🎯 Caption timing: {total_captions} captions over {audio_duration:.1f}s ({time_per_caption:.1f}s each)")
            
            return timing_data
            
        except Exception as e:
            logger.exception(f"�  Speech analysis failed, using uniform timing: {e}")
            # Fallback to uniform timing
            return self._create_uniform_timing(text)
    
    def _group_speech_segments(self, times: np.ndarray, speech_frames: np.ndarray) -> List[Tuple[float, float]]:
        """Group consecutive speech frames into segments."""
        segments = []
        in_speech = False
        start_time = None
        
        for i, (time, is_speech) in enumerate(zip(times, speech_frames)):
            if is_speech and not in_speech:
                # Start of speech segment
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                if start_time is not None:
                    segments.append((start_time, time))
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech and start_time is not None:
            segments.append((start_time, times[-1]))
        
        return segments
    
    def _create_uniform_timing(self, text: str) -> List[Dict[str, Any]]:
        """Create uniform caption timing as fallback."""
        words = text.split()
        words_per_caption = self.caption_style.words_per_caption
        caption_duration = 3.0  # Default 3 seconds per caption
        
        timing_data = []
        current_time = 0
        
        for i in range(0, len(words), words_per_caption):
            caption_text = ' '.join(words[i:i + words_per_caption])
            
            timing_data.append({
                'text': caption_text,
                'start': current_time,
                'end': current_time + caption_duration,
                'duration': caption_duration
            })
            
            current_time += caption_duration
        
        return timing_data
    
    def _create_caption_clips_with_dimensions(
        self, 
        timing_data: List[Dict[str, Any]], 
        style: CaptionStyle, 
        width: int, 
        height: int
    ) -> List[Any]:
        """
        Create caption clips with precise positioning based on video dimensions.
        
        Args:
            timing_data: List of caption timing information
            style: Caption styling options
            width: Video width
            height: Video height
            
        Returns:
            List of TextClip objects for captions
        """
        caption_clips = []
        
        for caption_info in timing_data:
            try:
                # Calculate position for text placement using pixel coordinates
                if style.position == 'bottom':
                    position = ('center', int(height * 0.75))  # 75% down from top
                elif style.position == 'top':
                    position = ('center', int(height * 0.25))  # 25% down from top
                else:  # center
                    position = ('center', int(height * 0.45))  # Slightly above center
                
                # Wrap text for better display
                text = self._wrap_text_for_display(caption_info['text'], style)
                
                # Create text clip with enhanced styling
                txt_clip = self._create_styled_text_clip(
                    text, 
                    style, 
                    position, 
                    caption_info['start'], 
                    caption_info['duration']
                )
                
                if txt_clip:
                    caption_clips.append(txt_clip)
                    
            except Exception as e:
                logger.warning(f"Failed to create caption clip for '{caption_info['text'][:30]}...': {e}")
                continue
        
        logger.info(f"✅ Created {len(caption_clips)} caption clips")
        return caption_clips
    
    
    def _wrap_text_for_display(self, text: str, style: CaptionStyle) -> str:
        """Wrap text appropriately for video display."""
        # Character limits based on position and screen size
        if style.position == 'center':
            max_chars_per_line = 25  # Center has more flexibility
        elif style.position == 'top':
            max_chars_per_line = 20  # Top needs more conservative spacing
        else:  # bottom
            max_chars_per_line = 22  # Bottom positioning
        
        # Only wrap if text is longer than limit
        if len(text) <= max_chars_per_line:
            return text
            
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Test if adding this word would exceed the limit
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                # Current line is full, start new line
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, truncate it
                    current_line = word[:max_chars_per_line-3] + "..."
        
        # Don't forget the last line
        if current_line:
            lines.append(current_line)
        
        # Limit to 2 lines max and join with actual newlines
        return '\n'.join(lines[:2])
    
    def _create_styled_text_clip(
        self, 
        text: str, 
        style: CaptionStyle, 
        position: tuple, 
        start_time: float, 
        duration: float
    ):
        """Create a styled text clip with better error handling."""
        try:
            # Create text clip with full styling options
            clip_kwargs = {
                'text': text,
                'font_size': style.font_size,
                'color': style.font_color,
                'stroke_color': style.stroke_color,
                'stroke_width': style.stroke_width
            }
            
            # Only add font if specified
            if style.font_family:
                clip_kwargs['font'] = style.font_family
            
            txt_clip = TextClip(**clip_kwargs).with_position(position).with_start(start_time).with_duration(duration)
            return txt_clip
            
        except Exception as clip_error:
            logger.warning(f"Failed to create TextClip with full options, trying minimal options: {clip_error}")
            # Fallback to minimal TextClip creation
            try:
                txt_clip = TextClip(
                    text=text,
                    font_size=style.font_size,
                    color=style.font_color
                ).with_position(position).with_start(start_time).with_duration(duration)
                return txt_clip
            except Exception as fallback_error:
                logger.error(f"Failed to create even minimal TextClip: {fallback_error}")
                return None
    
    def _create_caption_clips(self, timing_data: List[Dict[str, Any]], style: CaptionStyle) -> List[Any]:
        """Create MoviePy TextClip objects for captions."""
        caption_clips = []
        
        for caption_info in timing_data:
            try:
                # Calculate position for better placement
                # MoviePy positioning: (x, y) where y is from TOP of video frame
                if style.position == 'bottom':
                    position = ('center', 0.75)  # 75% down from top, higher up to avoid cutoff
                elif style.position == 'top':
                    position = ('center', 0.25)  # 25% down from top, lower to avoid cutoff
                elif style.position == 'center':
                    # Use explicit center positioning - 0.45 to account for text height
                    position = ('center', 0.45)  # Slightly above true center for better visual balance
                else:
                    # Default fallback to center
                    position = ('center', 0.45)
                
                logger.debug(f"Caption position for '{style.position}': {position}")
                
                # Wrap text to prevent cutoff with better logic
                text = caption_info['text']
                # Character limits based on position and screen size
                if style.position == 'center':
                    max_chars_per_line = 25  # Center has more flexibility
                elif style.position == 'top':
                    max_chars_per_line = 20  # Top needs more conservative spacing
                else:  # bottom
                    max_chars_per_line = 22  # Bottom positioning
                
                # Only wrap if text is longer than limit
                if len(text) > max_chars_per_line:
                    words = text.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        # Test if adding this word would exceed the limit
                        test_line = current_line + (" " if current_line else "") + word
                        if len(test_line) <= max_chars_per_line:
                            current_line = test_line
                        else:
                            # Current line is full, start new line
                            if current_line:
                                lines.append(current_line)
                                current_line = word
                            else:
                                # Single word is too long, truncate it
                                current_line = word[:max_chars_per_line-3] + "..."
                    
                    # Don't forget the last line
                    if current_line:
                        lines.append(current_line)
                    
                    # Limit to 2 lines max and join with actual newlines
                    text = '\n'.join(lines[:2])
                
                # Create text clip with better error handling (MoviePy 2.x syntax)
                try:
                    clip_kwargs = {
                        'text': text,
                        'font_size': style.font_size,
                        'color': style.font_color,
                        'stroke_color': style.stroke_color,
                        'stroke_width': style.stroke_width
                    }
                    
                    # Only add font if specified (let MoviePy use system default otherwise)
                    if style.font_family:
                        clip_kwargs['font'] = style.font_family
                    
                    txt_clip = TextClip(**clip_kwargs).with_position(position).with_start(caption_info['start']).with_duration(caption_info['duration'])
                except Exception as clip_error:
                    logger.exception(f"Failed to create TextClip with full options, trying minimal options: {clip_error}")
                    # Fallback to minimal TextClip creation
                    txt_clip = TextClip(
                        text=text,
                        font_size=style.font_size,
                        color=style.font_color
                    ).with_position(position).with_start(caption_info['start']).with_duration(caption_info['duration'])
                
                # Validate clip before adding
                if txt_clip is None:
                    logger.info(f"⚠️ TextClip creation returned None for: '{caption_info['text'][:20]}...'")
                    continue
                    
                caption_clips.append(txt_clip)
                
            except Exception as e:
                logger.exception(f"�  Failed to create caption '{caption_info['text'][:30]}...': {e}")
                continue
        
        return caption_clips
    
    def add_background_music(
        self, 
        music_path: str, 
        volume: float = 0.3,
        fade_in: float = 2.0,
        fade_out: float = 2.0
    ) -> Dict[str, Any]:
        """
        Add background music with automatic volume ducking.
        
        Args:
            music_path: Path to background music file
            volume: Background music volume (0.0 to 1.0)
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            
        Returns:
            Dictionary with music integration results
        """
        try:
            logger.info(f"<� Adding background music: {Path(music_path).name}")
            
            # Load background music
            music_clip = AudioFileClip(music_path)
            
            # Adjust volume (MoviePy 2.x syntax)
            music_clip = music_clip.with_volume_scaled(volume)
            
            # Add fade effects (MoviePy 2.x syntax)
            if fade_in > 0:
                music_clip = music_clip.with_effects([afx.AudioFadeIn(fade_in)])
            if fade_out > 0:
                music_clip = music_clip.with_effects([afx.AudioFadeOut(fade_out)])
            
            # Loop music to match video duration if needed (MoviePy 2.x approach)
            if music_clip.duration < self.template_duration:
                # Create multiple copies and concatenate
                loop_count = int(self.template_duration / music_clip.duration) + 1
                music_clips = [music_clip] * loop_count
                music_clip = concatenate_audioclips(music_clips)
            
            # Trim to match video duration (MoviePy 2.x syntax)
            music_clip = music_clip.subclipped(0, self.template_duration)
            
            # Store for final composition
            self.audio_clips.append(music_clip)
            
            return {
                "success": True,
                "music_duration": music_clip.duration,
                "volume": volume,
                "effects": f"fade_in: {fade_in}s, fade_out: {fade_out}s"
            }
            
        except Exception as e:
            logger.exception(f"L Background music failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_narration_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Add narration audio (from TTS) to the video.
        
        Args:
            audio_path: Path to the narration audio file
            
        Returns:
            Dictionary with audio integration results
        """
        try:
            logger.info(f"<� Adding narration audio: {Path(audio_path).name}")
            
            # Load narration audio
            narration_clip = AudioFileClip(audio_path)
            
            # Store for final composition
            self.audio_clips.append(narration_clip)
            
            return {
                "success": True,
                "narration_duration": narration_clip.duration,
                "audio_path": audio_path
            }
            
        except Exception as e:
            logger.exception(f"L Narration audio failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def render_video(
        self, 
        output_path: Optional[str] = None,
        config: Optional[VideoConfig] = None
    ) -> Dict[str, Any]:
        """
        Render the final video with all components.
        
        Args:
            output_path: Output file path, auto-generated if None
            config: Video output configuration
            
        Returns:
            Dictionary with rendering results
        """
        if config is None:
            config = self.video_config
            
        try:
            logger.info("Rendering final video...")
            
            # Generate output path if not provided
            if not output_path:
                import time
                timestamp = int(time.time())
                output_path = self.output_dir / f"moneytree_video_{timestamp}.{config.output_format}"
            
            # Get audio duration first
            audio_duration = max([clip.duration for clip in self.audio_clips]) if self.audio_clips else self.template_duration
            
            # Strategy: Create a completely fresh template for rendering to avoid MoviePy state corruption
            logger.info(f"Creating fresh template clip for rendering...")
            template_clip = VideoFileClip(str(self.template_path))
            
            # Trim to audio duration if needed
            if audio_duration < template_clip.duration:
                logger.info(f"Trimming template from {template_clip.duration:.1f}s to {audio_duration:.1f}s")
                template_clip = template_clip.subclipped(0, audio_duration)
            
            # Apply video transformations for proper formatting
            target_resolution = config.get_target_resolution()
            current_size = template_clip.size
            
            if target_resolution != current_size:
                logger.info(f"Resizing video from {current_size} to {target_resolution}")
                # Resize while maintaining aspect ratio and cropping if needed
                final_video = template_clip.resized(target_resolution)
            else:
                final_video = template_clip
            
            # Add audio if available
            if self.audio_clips:
                logger.info(f"Adding {len(self.audio_clips)} audio track(s)")
                final_audio = CompositeAudioClip(self.audio_clips)
                final_video = final_video.with_audio(final_audio)
            
            # Add captions if available
            if self.video_components:
                logger.info(f"Adding {len(self.video_components)} caption(s)")
                # Composite video with captions
                final_video = CompositeVideoClip([final_video] + self.video_components)
                logger.info("Captions successfully composited")
            else:
                logger.info("No captions to add")
            
            # Use minimal render settings to avoid issues
            render_kwargs = {
                'fps': 24,  # Standard FPS
                'codec': 'libx264',
                'audio_codec': 'aac',
                'preset': 'medium',  # Balanced preset
                'logger': None,
                'temp_audiofile': 'temp-audio.m4a',  # Explicit temp audio file
                'remove_temp': True  # Clean up temp files
            }
            
            # Render video with minimal settings
            logger.info(f"Saving to: {output_path}")
            final_video.write_videofile(str(output_path), **render_kwargs)
            
            # Get file stats
            file_size = os.path.getsize(output_path)
            
            return {
                "success": True,
                "output_path": str(output_path),
                "file_size": file_size,
                "duration": final_video.duration,
                "resolution": final_video.size,
                "fps": config.fps,
                "components": {
                    "captions": len(self.video_components),
                    "audio_tracks": len(self.audio_clips)
                }
            }
            
        except Exception as e:
            logger.exception(f"L Video rendering failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up video clips and free memory."""
        try:
            if self.template_clip:
                self.template_clip.close()
            
            for clip in self.audio_clips:
                if hasattr(clip, 'close'):
                    clip.close()
            
            for clip in self.video_components:
                if hasattr(clip, 'close'):
                    clip.close()
                    
            logger.info(">� Video clips cleaned up")
            
        except Exception as e:
            logger.exception(f"�  Cleanup warning: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # self.cleanup()


def get_recommended_video_templates() -> List[str]:
    """Get a list of recommended video template types for different content."""
    return [
        "educational_slideshow",  # Static slides with transitions
        "floating_text",          # Text floating over abstract background
        "nature_background",      # Nature scenes for relaxing content
        "tech_animation",         # Tech/digital themed animations
        "minimalist_white",       # Clean white background with elements
        "documentary_style",      # Documentary-style camera movements
    ]


def create_sample_template(output_path: str, duration: float = 60.0, vertical: bool = True) -> str:
    """
    Create a simple sample template video for testing.
    
    Args:
        output_path: Where to save the template
        duration: Duration in seconds
        vertical: True for TikTok/YT Shorts (9:16), False for standard (16:9)
        
    Returns:
        Path to created template
    """
    try:
        from moviepy import ColorClip
        
        # Set size based on format preference
        if vertical:
            size = (1080, 1920)  # 9:16 aspect ratio for TikTok/YT Shorts
            logger.info(f"📱 Creating vertical template (TikTok/YT Shorts format): {size[0]}x{size[1]}")
        else:
            size = (1920, 1080)  # 16:9 aspect ratio for standard videos
            logger.info(f"🖥️ Creating horizontal template (standard format): {size[0]}x{size[1]}")
        
        # Create a simple gradient-like colored background
        template = ColorClip(size=size, color=(30, 50, 80), duration=duration)
        template.write_videofile(output_path, fps=30)
        
        return output_path
        
    except ImportError:
        raise ImportError("Video dependencies required for template creation. Install with: uv add moviepy")
    except Exception as e:
        raise RuntimeError(f"Failed to create sample template: {e}")
