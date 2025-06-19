"""
OpenCV-based caption renderer for high-quality text overlay on videos.

This module provides an alternative to MoviePy's TextClip system using OpenCV
for video processing and PIL for high-quality text rendering. It's designed
to work reliably in WSL2 environments and provide superior text quality.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import tempfile
import os

from lib.utils.logging_config import get_logger
from lib.video.clip import CaptionStyle

logger = get_logger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.error("PIL not available - OpenCV caption renderer will not work")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.error("OpenCV not available - OpenCV caption renderer will not work")


@dataclass
class CaptionTiming:
    """Timing information for a caption segment."""
    text: str
    start_time: float
    end_time: float
    duration: float


class OpenCVCaptionRenderer:
    """
    High-quality caption renderer using OpenCV and PIL.
    
    Provides frame-by-frame video processing with PIL-based text rendering
    for superior quality and reliability compared to MoviePy TextClip.
    """
    
    def __init__(self, input_video_path: str, output_video_path: str):
        """
        Initialize the OpenCV caption renderer.
        
        Args:
            input_video_path: Path to input video file
            output_video_path: Path to output video file
        """
        if not HAS_OPENCV or not HAS_PIL:
            raise ImportError("OpenCV and PIL are required for this renderer")
        
        self.input_video_path = Path(input_video_path)
        self.output_video_path = Path(output_video_path)
        
        # Video properties
        self.video_cap = None
        self.video_writer = None
        self.fps = 24
        self.frame_width = 1080
        self.frame_height = 1920
        self.total_frames = 0
        self.duration = 0.0
        
        # Caption data
        self.caption_timings: List[CaptionTiming] = []
        self.current_frame = 0
        self.current_time = 0.0
        
        # Font cache for performance
        self._font_cache = {}
        
        self._load_video_properties()
    
    def _load_video_properties(self):
        """Load video properties from input file."""
        try:
            self.video_cap = cv2.VideoCapture(str(self.input_video_path))
            
            if not self.video_cap.isOpened():
                raise ValueError(f"Could not open video file: {self.input_video_path}")
            
            # Get video properties
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            logger.info(f"Video properties: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            logger.info(f"Duration: {self.duration:.2f}s, Total frames: {self.total_frames}")
            
        except Exception as e:
            logger.error(f"Failed to load video properties: {e}")
            raise
    
    def add_caption_timings(self, timing_data: List[Dict[str, Any]]):
        """
        Add caption timing data.
        
        Args:
            timing_data: List of dictionaries with caption timing information
        """
        self.caption_timings = []
        
        for timing in timing_data:
            caption = CaptionTiming(
                text=timing['text'],
                start_time=timing['start'],
                end_time=timing['end'],
                duration=timing['duration']
            )
            self.caption_timings.append(caption)
        
        logger.info(f"Added {len(self.caption_timings)} caption timings")
    
    def _get_cached_font(self, font_path: str, font_size: int) -> Optional[ImageFont.FreeTypeFont]:
        """Get cached font or create new one."""
        cache_key = f"{font_path}_{font_size}"
        
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]
        
        try:
            font = ImageFont.truetype(font_path, font_size)
            self._font_cache[cache_key] = font
            return font
        except Exception as e:
            logger.debug(f"Failed to load font {font_path}: {e}")
            return None
    
    def _get_best_font(self, style: CaptionStyle) -> ImageFont.FreeTypeFont:
        """Get the best available font for the given style."""
        # WSL2-optimized font paths
        font_paths = [
            # WSL2 specific paths - Linux system fonts
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            
            # WSL2 access to Windows fonts
            "/mnt/c/Windows/Fonts/arial.ttf",
            "/mnt/c/Windows/Fonts/arialbd.ttf",
            "/mnt/c/Windows/Fonts/calibri.ttf",
            "/mnt/c/Windows/Fonts/calibrib.ttf",
        ]
        
        # Try specified font family first
        if style.font_family:
            font = self._get_cached_font(style.font_family, style.font_size)
            if font:
                return font
        
        # Try font paths
        for font_path in font_paths:
            font = self._get_cached_font(font_path, style.font_size)
            if font:
                logger.debug(f"Using font: {font_path}")
                return font
        
        # Final fallback
        try:
            font = ImageFont.load_default()
            logger.warning("Using PIL default font")
            return font
        except:
            logger.error("No usable font found")
            return None
    
    def _render_text_on_frame(self, frame: np.ndarray, text: str, style: CaptionStyle) -> np.ndarray:
        """
        Render text on a single frame using PIL.
        
        Args:
            frame: OpenCV frame (BGR format)
            text: Text to render
            style: Caption styling options
            
        Returns:
            Frame with text rendered
        """
        try:
            # Convert BGR to RGB for PIL
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_frame)
            
            # Get font
            font = self._get_best_font(style)
            if not font:
                logger.warning("No font available, skipping text rendering")
                return frame
            
            # Calculate text dimensions with proper descender handling
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position based on style
            x_pos, y_pos = self._calculate_text_position(
                text_width, text_height, style, self.frame_width, self.frame_height
            )
            
            # Add background with proper margins
            if style.bg_opacity > 0:
                bg_margin = 10
                bg_bbox = (
                    x_pos - bg_margin,
                    y_pos - bg_margin,
                    x_pos + text_width + bg_margin,
                    y_pos + text_height + bg_margin
                )
                
                # Convert color and apply opacity
                bg_color = self._convert_color_with_opacity(style.bg_color, style.bg_opacity)
                draw.rectangle(bg_bbox, fill=bg_color)
            
            # Draw text with stroke if specified
            if style.stroke_width > 0:
                # Draw stroke by rendering text multiple times with slight offsets
                stroke_color = self._convert_color(style.stroke_color)
                for dx in range(-style.stroke_width, style.stroke_width + 1):
                    for dy in range(-style.stroke_width, style.stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text(
                                (x_pos + dx, y_pos + dy), 
                                text, 
                                font=font, 
                                fill=stroke_color
                            )
            
            # Draw main text
            text_color = self._convert_color(style.font_color)
            draw.text((x_pos, y_pos), text, font=font, fill=text_color)
            
            # Convert back to BGR for OpenCV
            frame_with_text = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            
            return frame_with_text
            
        except Exception as e:
            logger.error(f"Failed to render text on frame: {e}")
            return frame  # Return original frame on error
    
    def _calculate_text_position(self, text_width: int, text_height: int, 
                               style: CaptionStyle, frame_width: int, frame_height: int) -> Tuple[int, int]:
        """Calculate optimal text position."""
        # Safety margins
        margin_x = int(frame_width * 0.1)
        margin_y = int(frame_height * 0.1)
        
        # Calculate X position (center horizontally)
        x_pos = (frame_width - text_width) // 2
        x_pos = max(margin_x, min(x_pos, frame_width - text_width - margin_x))
        
        # Calculate Y position based on style
        if style.position == 'top':
            y_pos = margin_y
        elif style.position == 'bottom':
            y_pos = frame_height - text_height - margin_y
        else:  # center
            y_pos = (frame_height - text_height) // 2
        
        # Ensure within bounds
        y_pos = max(margin_y, min(y_pos, frame_height - text_height - margin_y))
        
        return (x_pos, y_pos)
    
    def _convert_color(self, color_str: str) -> Tuple[int, int, int]:
        """Convert color string to RGB tuple."""
        color_map = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255)
        }
        return color_map.get(color_str.lower(), (255, 255, 255))  # Default to white
    
    def _convert_color_with_opacity(self, color_str: str, opacity: float) -> Tuple[int, int, int, int]:
        """Convert color string to RGBA tuple with opacity."""
        rgb = self._convert_color(color_str)
        alpha = int(opacity * 255)
        return rgb + (alpha,)
    
    def _get_active_captions(self, current_time: float) -> List[CaptionTiming]:
        """Get captions that should be displayed at the current time."""
        active_captions = []
        
        for caption in self.caption_timings:
            if caption.start_time <= current_time <= caption.end_time:
                active_captions.append(caption)
        
        return active_captions
    
    def render_video_with_captions(self, caption_style: CaptionStyle, 
                                 progress_callback: Optional[callable] = None) -> bool:
        """
        Render the complete video with captions.
        
        Args:
            caption_style: Styling options for captions
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting video rendering with OpenCV/PIL")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.output_video_path),
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            
            if not self.video_writer.isOpened():
                raise ValueError("Could not open video writer")
            
            # Reset video capture to beginning
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            frame_count = 0
            
            while True:
                ret, frame = self.video_cap.read()
                
                if not ret:
                    break
                
                # Calculate current time
                current_time = frame_count / self.fps
                
                # Get active captions for this time
                active_captions = self._get_active_captions(current_time)
                
                # Render captions on frame
                for caption in active_captions:
                    frame = self._render_text_on_frame(frame, caption.text, caption_style)
                
                # Write frame
                self.video_writer.write(frame)
                
                frame_count += 1
                
                # Progress callback
                if progress_callback and frame_count % 30 == 0:  # Every second
                    progress = frame_count / self.total_frames
                    progress_callback(progress)
                
                # Log progress periodically
                if frame_count % 300 == 0:  # Every 10 seconds
                    progress = frame_count / self.total_frames * 100
                    logger.info(f"Rendering progress: {progress:.1f}% ({frame_count}/{self.total_frames})")
            
            logger.info(f"Rendering complete: {frame_count} frames processed")
            return True
            
        except Exception as e:
            logger.error(f"Video rendering failed: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.video_cap:
                self.video_cap.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            logger.debug("OpenCV resources cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


def render_video_with_opencv_captions(
    input_video: str,
    output_video: str,
    caption_data: List[Dict[str, Any]],
    caption_style: CaptionStyle,
    progress_callback: Optional[callable] = None
) -> bool:
    """
    Convenience function to render video with captions using OpenCV/PIL.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video  
        caption_data: List of caption timing data
        caption_style: Caption styling options
        progress_callback: Optional progress callback
        
    Returns:
        True if successful, False otherwise
    """
    try:
        renderer = OpenCVCaptionRenderer(input_video, output_video)
        renderer.add_caption_timings(caption_data)
        return renderer.render_video_with_captions(caption_style, progress_callback)
    except Exception as e:
        logger.error(f"OpenCV caption rendering failed: {e}")
        return False