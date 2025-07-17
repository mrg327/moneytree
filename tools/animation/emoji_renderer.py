"""
Emoji renderer that converts Unicode emojis to PNG images with transparent backgrounds.

This module provides functionality to render emojis as high-quality PNG images
with transparent backgrounds, suitable for use in animations and overlays.
"""

import os
from typing import Optional, Tuple
from pathlib import Path

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.error("PIL not available - emoji renderer will not work")


class EmojiRenderer:
    """
    Renders Unicode emojis as PNG images with transparent backgrounds.
    
    Uses system fonts that support emoji rendering, with intelligent fallback
    for different operating systems and environments.
    """
    
    def __init__(self, default_size: int = 64, cache_dir: str = "temp_media/emoji_cache"):
        """
        Initialize the emoji renderer.
        
        Args:
            default_size: Default emoji size in pixels
            cache_dir: Directory to cache rendered emoji images
        """
        if not HAS_PIL:
            raise ImportError("PIL is required for emoji rendering")
        
        self.default_size = default_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.font_cache = {}
        self._load_emoji_fonts()
    
    def _load_emoji_fonts(self) -> None:
        """Load emoji-capable fonts with intelligent fallback."""
        # Emoji font candidates optimized for different environments
        emoji_font_candidates = [
            # Windows emoji fonts (accessible via WSL2)
            "/mnt/c/Windows/Fonts/seguiemj.ttf",  # Windows emoji font
            "/mnt/c/Windows/Fonts/seguisym.ttf",  # Windows symbol font
            "/mnt/c/Windows/Fonts/segmdl2.ttf",   # Windows UI symbol font
            
            # Linux emoji fonts
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
            "/usr/share/fonts/truetype/color-emoji/NotoColorEmoji.ttf",
            "/usr/share/fonts/truetype/ancient-scripts/NotoEmoji-Regular.ttf",
            
            # Fallback fonts with limited emoji support
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-Regular.ttf",
            
            # System default fonts
            "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
            "/usr/share/fonts/truetype/fonts-japanese/VLGothic-Regular.ttf",  # Japanese with some emoji
        ]
        
        self.emoji_fonts = []
        
        for font_path in emoji_font_candidates:
            if os.path.exists(font_path):
                try:
                    # Test different sizes to find a working font
                    test_font = ImageFont.truetype(font_path, self.default_size)
                    
                    # Test if font can render a simple emoji
                    if self._test_emoji_support(test_font):
                        self.emoji_fonts.append(font_path)
                        logger.info(f"Loaded emoji font: {font_path}")
                    else:
                        logger.debug(f"Font {font_path} exists but doesn't support emoji")
                        
                except Exception as e:
                    logger.debug(f"Failed to load font {font_path}: {e}")
        
        if not self.emoji_fonts:
            logger.warning("No emoji-capable fonts found. Emoji rendering may not work properly.")
        else:
            logger.info(f"Found {len(self.emoji_fonts)} emoji-capable fonts")
    
    def _test_emoji_support(self, font: ImageFont.FreeTypeFont) -> bool:
        """Test if a font can render emoji characters."""
        try:
            # Test with a simple emoji that should be widely supported
            test_emoji = "😊"  # Smiling face
            
            # Create a small test image
            test_img = Image.new('RGBA', (32, 32), (255, 255, 255, 0))
            test_draw = ImageDraw.Draw(test_img)
            
            # Try to get text size - this will fail if font doesn't support the character
            bbox = test_draw.textbbox((0, 0), test_emoji, font=font)
            
            # If bbox is too small, font probably doesn't support emoji
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            return width > 5 and height > 5
            
        except Exception:
            return False
    
    def get_font(self, size: int) -> Optional[ImageFont.FreeTypeFont]:
        """
        Get the best emoji font for the given size.
        
        Args:
            size: Font size in pixels
            
        Returns:
            Font object or None if no font available
        """
        cache_key = f"emoji_{size}"
        
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        font = None
        
        # Try to load the first available emoji font
        for font_path in self.emoji_fonts:
            try:
                font = ImageFont.truetype(font_path, size)
                break
            except Exception as e:
                logger.debug(f"Failed to load font {font_path} at size {size}: {e}")
                continue
        
        # Fallback to default font if no emoji font works
        if font is None:
            try:
                font = ImageFont.load_default()
                logger.warning(f"Using default font for size {size} - emoji may not render properly")
            except Exception as e:
                logger.error(f"Failed to load any font for size {size}: {e}")
        
        self.font_cache[cache_key] = font
        return font
    
    def render_emoji(self, emoji: str, size: int = None, output_path: str = None) -> Optional[str]:
        """
        Render an emoji to a PNG image with transparent background.
        
        Args:
            emoji: Unicode emoji character(s)
            size: Size in pixels (uses default_size if None)
            output_path: Path to save the image (auto-generated if None)
            
        Returns:
            Path to the saved PNG image, or None if rendering failed
        """
        if size is None:
            size = self.default_size
        
        # Generate cache filename
        if output_path is None:
            # Create a safe filename from emoji
            emoji_name = emoji.encode('unicode_escape').decode('ascii')
            output_path = self.cache_dir / f"emoji_{emoji_name}_{size}.png"
        else:
            output_path = Path(output_path)
        
        # Check if already cached
        if output_path.exists():
            logger.debug(f"Using cached emoji: {output_path}")
            return str(output_path)
        
        # Get font
        font = self.get_font(size)
        if font is None:
            logger.error(f"No font available for emoji rendering")
            return None
        
        try:
            # Create image with transparent background
            # Use a larger canvas to ensure emoji fits properly
            canvas_size = int(size * 1.2)
            img = Image.new('RGBA', (canvas_size, canvas_size), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # Get text dimensions
            bbox = draw.textbbox((0, 0), emoji, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the emoji on the canvas
            x = (canvas_size - text_width) // 2
            y = (canvas_size - text_height) // 2
            
            # Draw the emoji
            draw.text((x, y), emoji, font=font, fill=(0, 0, 0, 255))
            
            # Crop to actual content to remove excess transparent space
            img = self._crop_to_content(img)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as PNG with transparency
            img.save(output_path, "PNG", optimize=True)
            
            logger.info(f"Rendered emoji '{emoji}' to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to render emoji '{emoji}': {e}")
            return None
    
    def _crop_to_content(self, img: Image.Image) -> Image.Image:
        """Crop image to remove excess transparent space around content."""
        try:
            # Get bounding box of non-transparent content
            bbox = img.getbbox()
            if bbox:
                return img.crop(bbox)
            else:
                # If no content found, return original
                return img
        except Exception as e:
            logger.debug(f"Failed to crop image: {e}")
            return img
    
    def render_rps_emojis(self, size: int = None) -> dict:
        """
        Render Rock Paper Scissors emojis.
        
        Args:
            size: Size in pixels (uses default_size if None)
            
        Returns:
            Dictionary mapping entity types to image paths
        """
        if size is None:
            size = self.default_size
        
        rps_emojis = {
            'rock': '🪨',      # Rock emoji
            'paper': '📄',     # Paper emoji  
            'scissors': '✂️'   # Scissors emoji
        }
        
        results = {}
        
        for entity_type, emoji in rps_emojis.items():
            image_path = self.render_emoji(emoji, size)
            if image_path:
                results[entity_type] = image_path
                logger.info(f"Rendered {entity_type} emoji: {image_path}")
            else:
                logger.error(f"Failed to render {entity_type} emoji")
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the emoji cache directory."""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Emoji cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear emoji cache: {e}")


def main():
    """Test the emoji renderer with RPS emojis."""
    print("Testing Emoji Renderer...")
    
    renderer = EmojiRenderer(default_size=64)
    
    # Test individual emojis
    test_emojis = ['🪨', '📄', '✂️', '😊', '🎮']
    
    print("\nTesting individual emojis:")
    for emoji in test_emojis:
        result = renderer.render_emoji(emoji)
        if result:
            print(f"✅ {emoji} -> {result}")
        else:
            print(f"❌ {emoji} -> Failed")
    
    # Test RPS emoji set
    print("\nTesting RPS emoji set:")
    rps_results = renderer.render_rps_emojis()
    for entity_type, path in rps_results.items():
        print(f"✅ {entity_type}: {path}")
    
    print(f"\nEmoji cache directory: {renderer.cache_dir}")


if __name__ == "__main__":
    main()