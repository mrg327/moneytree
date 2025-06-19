"""
PIL-based text rendering engine for high-quality caption generation.

This module provides precise text measurement, advanced font handling,
and high-quality anti-aliased text rendering using PIL/Pillow.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import re

from lib.utils.logging_config import get_logger
from lib.video.clip import CaptionStyle

logger = get_logger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.error("PIL not available - PIL text engine will not work")


@dataclass
class TextMetrics:
    """Detailed text measurement results."""
    width: int
    height: int
    ascent: int
    descent: int
    line_count: int
    char_count: int
    actual_bbox: Tuple[int, int, int, int]  # left, top, right, bottom


@dataclass 
class WrappedText:
    """Text wrapping results."""
    lines: List[str]
    total_width: int
    total_height: int
    line_heights: List[int]
    overflow: bool


class PILTextEngine:
    """
    High-quality text rendering engine using PIL.
    
    Provides precise text measurement, advanced font handling, and
    anti-aliased text rendering with comprehensive layout controls.
    """
    
    def __init__(self):
        """Initialize the PIL text engine."""
        if not HAS_PIL:
            raise ImportError("PIL is required for the PIL text engine")
        
        self.font_cache = {}
        self.measurement_cache = {}
        
    def get_font(self, style: CaptionStyle) -> Optional[ImageFont.FreeTypeFont]:
        """
        Get font with intelligent fallback and caching.
        
        Args:
            style: Caption style configuration
            
        Returns:
            Font object or None if no font available
        """
        cache_key = f"{style.font_family or 'default'}_{style.font_size}"
        
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        font = self._load_best_font(style)
        self.font_cache[cache_key] = font
        return font
    
    def _load_best_font(self, style: CaptionStyle) -> Optional[ImageFont.FreeTypeFont]:
        """Load the best available font for the style."""
        # Font search paths optimized for WSL2
        font_candidates = []
        
        # Add user-specified font if provided
        if style.font_family:
            font_candidates.append(style.font_family)
        
        # WSL2-optimized font paths
        font_candidates.extend([
            # High-quality system fonts with full paths
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
            
            # Regular weight fallbacks
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-Regular.ttf",
            
            # WSL2 Windows font access
            "/mnt/c/Windows/Fonts/arial.ttf",
            "/mnt/c/Windows/Fonts/arialbd.ttf",  # Arial Bold
            "/mnt/c/Windows/Fonts/calibri.ttf",
            "/mnt/c/Windows/Fonts/calibrib.ttf",  # Calibri Bold
            
            # Font name fallbacks
            "DejaVuSans-Bold.ttf",
            "LiberationSans-Bold.ttf",
            "Arial-Bold.ttf",
            "DejaVuSans.ttf",
            "Arial.ttf"
        ])
        
        # Score and select best font
        best_font = None
        best_score = 0
        
        for font_path in font_candidates:
            font, score = self._validate_font(font_path, style.font_size)
            if font and score > best_score:
                best_font = font
                best_score = score
                logger.debug(f"Better font found: {font_path} (score: {score})")
        
        if best_font:
            logger.info(f"Selected font with quality score: {best_score}")
            return best_font
        
        # Final fallback to PIL default
        try:
            default_font = ImageFont.load_default()
            logger.warning("Using PIL default font - text quality may be poor")
            return default_font
        except Exception:
            logger.error("No fonts available - text rendering will fail")
            return None
    
    def _validate_font(self, font_path: str, font_size: int) -> Tuple[Optional[ImageFont.FreeTypeFont], int]:
        """
        Validate font and calculate quality score.
        
        Args:
            font_path: Path to font file
            font_size: Font size to test
            
        Returns:
            Tuple of (font_object, quality_score)
        """
        try:
            font = ImageFont.truetype(font_path, font_size)
            
            # Test font with sample text
            test_img = Image.new('RGB', (200, 100), 'white')
            draw = ImageDraw.Draw(test_img)
            test_text = "Test gjpqy AW"  # Mix of letters with descenders, ascenders
            
            # Attempt text measurement
            bbox = draw.textbbox((0, 0), test_text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Calculate quality score
            score = 50  # Base score
            
            # Prefer full system paths (more reliable)
            if font_path.startswith('/'):
                score += 25
            
            # Prefer bold fonts
            if 'bold' in font_path.lower():
                score += 20
            
            # Prefer well-known quality fonts
            quality_fonts = ['dejavu', 'liberation', 'ubuntu', 'noto', 'arial', 'calibri']
            if any(name in font_path.lower() for name in quality_fonts):
                score += 15
            
            # Ensure reasonable text dimensions
            if width > 30 and height > 10:
                score += 10
            
            # Test font metrics if available
            try:
                if hasattr(font, 'getmetrics'):
                    ascent, descent = font.getmetrics()
                    if ascent > 0 and descent > 0:
                        score += 10
            except:
                pass
            
            return font, score
            
        except Exception as e:
            logger.debug(f"Font validation failed for {font_path}: {e}")
            return None, 0
    
    def measure_text(self, text: str, style: CaptionStyle) -> TextMetrics:
        """
        Precisely measure text dimensions and properties.
        
        Args:
            text: Text to measure
            style: Caption style configuration
            
        Returns:
            Detailed text metrics
        """
        cache_key = f"{text}_{style.font_family or 'default'}_{style.font_size}"
        
        if cache_key in self.measurement_cache:
            return self.measurement_cache[cache_key]
        
        font = self.get_font(style)
        if not font:
            # Fallback measurements
            lines = text.split('\\n')
            char_count = len(text)
            line_count = len(lines)
            # Rough estimates
            width = int(char_count * style.font_size * 0.6)
            height = int(line_count * style.font_size * 1.2)
            metrics = TextMetrics(
                width=width, height=height, ascent=int(style.font_size * 0.8),
                descent=int(style.font_size * 0.2), line_count=line_count,
                char_count=char_count, actual_bbox=(0, 0, width, height)
            )
            self.measurement_cache[cache_key] = metrics
            return metrics
        
        # Create measurement surface
        measure_img = Image.new('RGB', (2000, 1000), 'white')
        draw = ImageDraw.Draw(measure_img)
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Get font metrics
        ascent = descent = 0
        try:
            if hasattr(font, 'getmetrics'):
                ascent, descent = font.getmetrics()
            else:
                # Estimate from font size
                ascent = int(style.font_size * 0.8)
                descent = int(style.font_size * 0.2)
        except:
            ascent = int(style.font_size * 0.8)
            descent = int(style.font_size * 0.2)
        
        # Count lines and characters
        lines = text.split('\\n')
        line_count = len(lines)
        char_count = len(text)
        
        metrics = TextMetrics(
            width=width,
            height=height,
            ascent=ascent,
            descent=descent,
            line_count=line_count,
            char_count=char_count,
            actual_bbox=bbox
        )
        
        self.measurement_cache[cache_key] = metrics
        logger.debug(f"Text measurement: '{text[:20]}...' -> {width}x{height}")
        
        return metrics
    
    def wrap_text(self, text: str, max_width: int, style: CaptionStyle) -> WrappedText:
        """
        Intelligently wrap text to fit within specified width.
        
        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            style: Caption style configuration
            
        Returns:
            Wrapped text result
        """
        font = self.get_font(style)
        if not font:
            # Simple character-based fallback
            char_width = int(style.font_size * 0.6)
            chars_per_line = max(1, max_width // char_width)
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if len(test_line) <= chars_per_line:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            return WrappedText(
                lines=lines,
                total_width=max_width,
                total_height=len(lines) * int(style.font_size * 1.2),
                line_heights=[int(style.font_size * 1.2)] * len(lines),
                overflow=False
            )
        
        # PIL-based precise text wrapping
        measure_img = Image.new('RGB', (max_width * 2, 1000), 'white')
        draw = ImageDraw.Draw(measure_img)
        
        words = text.split()
        lines = []
        line_heights = []
        current_line = ""
        total_width = 0
        overflow = False
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line = test_line
            else:
                # Line would be too wide
                if current_line:
                    # Add current line
                    line_bbox = draw.textbbox((0, 0), current_line, font=font)
                    lines.append(current_line)
                    line_heights.append(line_bbox[3] - line_bbox[1])
                    total_width = max(total_width, line_bbox[2] - line_bbox[0])
                    current_line = word
                else:
                    # Single word too long - truncate with ellipsis
                    while len(word) > 3:
                        test_word = word[:-1] + "..."
                        bbox = draw.textbbox((0, 0), test_word, font=font)
                        if bbox[2] - bbox[0] <= max_width:
                            current_line = test_word
                            overflow = True
                            break
                        word = word[:-1]
                    else:
                        current_line = word  # Give up and use as-is
                        overflow = True
        
        # Add final line
        if current_line:
            line_bbox = draw.textbbox((0, 0), current_line, font=font)
            lines.append(current_line)
            line_heights.append(line_bbox[3] - line_bbox[1])
            total_width = max(total_width, line_bbox[2] - line_bbox[0])
        
        # Limit to reasonable number of lines (2-3 max for captions)
        if len(lines) > 3:
            lines = lines[:3]
            line_heights = line_heights[:3]
            overflow = True
        
        total_height = sum(line_heights)
        
        return WrappedText(
            lines=lines,
            total_width=total_width,
            total_height=total_height,
            line_heights=line_heights,
            overflow=overflow
        )
    
    def render_text(self, image: Image.Image, text: str, position: Tuple[int, int], 
                   style: CaptionStyle, 
                   wrapped_text: Optional[WrappedText] = None) -> Image.Image:
        """
        Render high-quality anti-aliased text on image.
        
        Args:
            image: PIL Image to render text on
            text: Text to render (or use wrapped_text if provided)
            position: (x, y) position for text
            style: Caption style configuration
            wrapped_text: Pre-wrapped text (optional)
            
        Returns:
            Image with text rendered
        """
        font = self.get_font(style)
        if not font:
            logger.warning("No font available for text rendering")
            return image
        
        draw = ImageDraw.Draw(image)
        
        # Use wrapped text if provided
        if wrapped_text:
            lines = wrapped_text.lines
            line_heights = wrapped_text.line_heights
        else:
            lines = [text]
            line_heights = [style.font_size]
        
        x, y = position
        current_y = y
        
        for line, line_height in zip(lines, line_heights):
            if not line.strip():
                current_y += line_height
                continue
            
            # Calculate centered X position for this line
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = x - (line_width // 2)  # Center the line
            
            # Draw background if specified
            if style.bg_opacity > 0:
                bg_margin = 8
                bg_color = self._convert_color_with_opacity(style.bg_color, style.bg_opacity)
                bg_bbox = (
                    line_x - bg_margin,
                    current_y - bg_margin,
                    line_x + line_width + bg_margin,
                    current_y + line_height + bg_margin
                )
                draw.rectangle(bg_bbox, fill=bg_color)
            
            # Draw stroke if specified
            if style.stroke_width > 0:
                stroke_color = self._convert_color(style.stroke_color)
                # Multiple passes for smooth stroke
                for dx in range(-style.stroke_width, style.stroke_width + 1):
                    for dy in range(-style.stroke_width, style.stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text(
                                (line_x + dx, current_y + dy),
                                line,
                                font=font,
                                fill=stroke_color
                            )
            
            # Draw main text
            text_color = self._convert_color(style.font_color)
            draw.text((line_x, current_y), line, font=font, fill=text_color)
            
            current_y += line_height
        
        return image
    
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
            'magenta': (255, 0, 255),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128)
        }
        
        # Handle hex colors
        if color_str.startswith('#'):
            try:
                hex_color = color_str[1:]
                if len(hex_color) == 6:
                    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
        
        return color_map.get(color_str.lower(), (255, 255, 255))
    
    def _convert_color_with_opacity(self, color_str: str, opacity: float) -> Tuple[int, int, int, int]:
        """Convert color string to RGBA tuple with opacity."""
        rgb = self._convert_color(color_str)
        alpha = int(opacity * 255)
        return rgb + (alpha,)
    
    def optimize_text_for_readability(self, text: str, style: CaptionStyle, 
                                    max_width: int, max_height: int) -> Tuple[CaptionStyle, WrappedText]:
        """
        Optimize text and style for maximum readability within constraints.
        
        Args:
            text: Text to optimize
            style: Initial caption style
            max_width: Maximum width constraint
            max_height: Maximum height constraint
            
        Returns:
            Tuple of (optimized_style, wrapped_text)
        """
        # Try different font sizes to find optimal fit
        original_size = style.font_size
        best_style = style
        best_wrapped = None
        
        for size_multiplier in [1.0, 0.9, 0.8, 0.7, 0.6]:
            test_style = CaptionStyle(
                font_size=int(original_size * size_multiplier),
                font_color=style.font_color,
                bg_color=style.bg_color,
                bg_opacity=style.bg_opacity,
                position=style.position,
                max_width=style.max_width,
                words_per_caption=style.words_per_caption,
                font_family=style.font_family,
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width
            )
            
            wrapped = self.wrap_text(text, max_width, test_style)
            
            if wrapped.total_height <= max_height and not wrapped.overflow:
                best_style = test_style
                best_wrapped = wrapped
                break
        
        if best_wrapped is None:
            # Fallback to heavily reduced text
            best_wrapped = self.wrap_text(text, max_width, best_style)
        
        logger.debug(f"Text optimization: {original_size} -> {best_style.font_size} font size")
        
        return best_style, best_wrapped
    
    def clear_cache(self):
        """Clear font and measurement caches."""
        self.font_cache.clear()
        self.measurement_cache.clear()
        logger.debug("PIL text engine caches cleared")


# Singleton instance for global use
pil_text_engine = PILTextEngine()