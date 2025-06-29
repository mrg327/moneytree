"""
Image processing for Wikipedia images in video generation.

Handles resizing, positioning, and effect application for Wikipedia images
to be overlaid on videos with proper aspect ratio and positioning.
"""

import os
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from PIL import Image, ImageFilter, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.error("PIL not available - image processing will not work")

try:
    from moviepy import ImageClip
    HAS_MOVIEPY_IMAGE = True
except ImportError:
    HAS_MOVIEPY_IMAGE = False
    logger.error("MoviePy ImageClip not available")


class WikipediaImageProcessor:
    """
    Processes Wikipedia images for video overlay.
    
    Handles resizing, cropping, and effect application to prepare
    Wikipedia images for integration into educational videos.
    """
    
    def __init__(self):
        """Initialize the image processor."""
        if not HAS_PIL or not HAS_MOVIEPY_IMAGE:
            raise ImportError("PIL and MoviePy ImageClip are required for image processing")
        
        logger.debug("Image processor initialized")
    
    def create_video_image_clip(self, 
                               image_path: str,
                               video_width: int,
                               video_height: int,
                               duration: float = 5.0,
                               position: str = "top_third",
                               fade_duration: float = 0.5) -> Optional[Any]:
        """
        Create a MoviePy ImageClip for video overlay.
        
        Args:
            image_path: Path to the image file
            video_width: Target video width
            video_height: Target video height  
            duration: How long the image should be displayed
            position: Where to position the image ("top_third", "center", etc.)
            fade_duration: Duration of fade in/out effects
            
        Returns:
            MoviePy ImageClip ready for composition, or None if failed
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            logger.info(f"Processing image for video overlay: {Path(image_path).name}")
            
            # Calculate target dimensions based on position
            if position == "top_third":
                # Image takes up top third of video with some margin
                target_height = int(video_height * 0.3)  # 30% of video height
                margin = int(video_width * 0.05)  # 5% margin on sides
                target_width = video_width - (2 * margin)
            else:
                # Default to smaller centered image
                target_width = int(video_width * 0.6)
                target_height = int(video_height * 0.4)
            
            # Process the image
            processed_image_path = self._process_image(
                image_path, target_width, target_height
            )
            
            if not processed_image_path:
                logger.error("Image processing failed")
                return None
            
            # Create MoviePy ImageClip
            image_clip = ImageClip(processed_image_path, duration=duration)
            
            # Calculate position
            if position == "top_third":
                # Center horizontally, position in top third with margin
                x_pos = 'center'
                y_pos = int(video_height * 0.05)  # 5% from top
            else:
                # Center both horizontally and vertically
                x_pos = 'center'
                y_pos = 'center'
            
            # Apply position
            image_clip = image_clip.with_position((x_pos, y_pos))
            
            # Apply fade effects
            if fade_duration > 0:
                from moviepy.video.fx import FadeIn, FadeOut
                # Fade in at start and fade out at end
                image_clip = image_clip.with_effects([
                    # FadeIn(fade_duration),
                    FadeOut(fade_duration)
                ])
            
            logger.info(f"Image clip created: {target_width}x{target_height}, duration={duration}s")
            return image_clip
            
        except Exception as e:
            logger.error(f"Failed to create image clip: {e}")
            return None
    
    def _process_image(self, 
                      image_path: str, 
                      target_width: int, 
                      target_height: int) -> Optional[str]:
        """
        Process image to fit target dimensions with effects.
        
        Args:
            image_path: Path to source image
            target_width: Target width in pixels
            target_height: Target height in pixels
            
        Returns:
            Path to processed image file, or None if failed
        """
        try:
            # Open image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                original_width, original_height = img.size
                logger.debug(f"Original image size: {original_width}x{original_height}")
                
                # Calculate scaling to fit within target dimensions while maintaining aspect ratio
                scale_w = target_width / original_width
                scale_h = target_height / original_height
                scale = min(scale_w, scale_h)  # Use smaller scale to fit within bounds
                
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                
                # Resize image
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create target-sized canvas with subtle background
                canvas = Image.new('RGB', (target_width, target_height), (245, 245, 245))  # Light gray
                
                # Calculate position to center the image on canvas
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                
                # Paste resized image onto canvas
                canvas.paste(img_resized, (paste_x, paste_y))
                
                # Apply subtle enhancements
                canvas = self._apply_image_effects(canvas)
                
                # Generate output path
                output_path = self._generate_processed_path(image_path, target_width, target_height)
                
                # Save processed image
                canvas.save(output_path, 'JPEG', quality=90, optimize=True)
                
                logger.debug(f"Processed image saved: {output_path}")
                logger.debug(f"Final size: {target_width}x{target_height}")
                
                return output_path
                
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return None
    
    def _apply_image_effects(self, img: Image.Image) -> Image.Image:
        """
        Apply subtle visual effects to enhance the image.
        
        Args:
            img: PIL Image to enhance
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Slightly enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)  # 10% more contrast
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.05)  # 5% more sharpness
            
            # Very subtle gaussian blur to smooth any artifacts
            img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
            
            return img
            
        except Exception as e:
            logger.warning(f"Failed to apply image effects: {e}")
            return img
    
    def _generate_processed_path(self, 
                                original_path: str, 
                                width: int, 
                                height: int) -> str:
        """
        Generate path for processed image file.
        
        Args:
            original_path: Path to original image
            width: Target width
            height: Target height
            
        Returns:
            Path for processed image
        """
        original_path = Path(original_path)
        output_dir = original_path.parent / "processed"
        output_dir.mkdir(exist_ok=True)
        
        stem = original_path.stem
        processed_name = f"{stem}_processed_{width}x{height}.jpg"
        
        return str(output_dir / processed_name)
    
    def get_recommended_position_for_captions(self, 
                                            video_height: int,
                                            image_duration: float,
                                            current_time: float) -> str:
        """
        Recommend caption position to avoid overlap with image.
        
        Args:
            video_height: Video height in pixels
            image_duration: How long image is displayed
            current_time: Current time in video
            
        Returns:
            Recommended caption position ("bottom", "center", "top")
        """
        # If image is active (first 5 seconds typically), use bottom positioning
        if current_time <= image_duration:
            return "bottom"
        else:
            return "center"  # Default position when no image


def create_wikipedia_image_clip(image_path: str,
                              video_width: int, 
                              video_height: int,
                              duration: float = 5.0,
                              fade_duration: float = 0.5) -> Optional[Any]:
    """
    Convenience function to create a Wikipedia image clip.
    
    Args:
        image_path: Path to Wikipedia image
        video_width: Target video width
        video_height: Target video height
        duration: Display duration in seconds
        fade_duration: Fade in/out duration
        
    Returns:
        MoviePy ImageClip or None if failed
    """
    processor = WikipediaImageProcessor()
    return processor.create_video_image_clip(
        image_path, video_width, video_height, 
        duration=duration, fade_duration=fade_duration
    )
