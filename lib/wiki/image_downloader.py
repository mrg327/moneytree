"""
Wikipedia image downloader for fetching and caching page images.

Handles downloading Wikipedia thumbnail images with validation, caching,
and error handling for use in video generation.
"""

import os
import hashlib
import requests
from typing import Optional, Dict, Any
from pathlib import Path
from urllib.parse import urlparse

from lib.utils.logging_config import get_logger, LoggedOperation

logger = get_logger(__name__)


class WikipediaImageDownloader:
    """
    Downloads and caches Wikipedia images for video generation.
    
    Provides robust image downloading with validation, caching, and 
    proper error handling for integration with the video pipeline.
    """
    
    def __init__(self, cache_dir: str = "image_cache"):
        """
        Initialize the image downloader.
        
        Args:
            cache_dir: Directory to cache downloaded images
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        
        # Setup session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MoneyTree/1.0 (https://github.com/user/moneytree)",
            "Accept": "image/*,*/*;q=0.8"
        })
        
        logger.debug(f"Image downloader initialized with cache dir: {self.cache_dir}")
    
    def download_wikipedia_image(self, image_info: Dict[str, Any], 
                                page_title: str = "") -> Optional[str]:
        """
        Download a Wikipedia image from thumbnail info.
        
        Args:
            image_info: Wikipedia thumbnail/image info dict
            page_title: Page title for naming cached files
            
        Returns:
            Path to downloaded image file, or None if failed
        """
        if not image_info or not isinstance(image_info, dict):
            logger.debug("No image info provided")
            return None
        
        # Extract image URL
        image_url = image_info.get('source', '') or image_info.get('url', '')
        if not image_url:
            logger.debug("No image URL found in image info")
            return None
        
        with LoggedOperation(logger, f"downloading Wikipedia image for '{page_title}'"):
            try:
                # Generate cache filename
                cache_filename = self._generate_cache_filename(image_url, page_title)
                cache_path = self.cache_dir / cache_filename
                
                # Check if already cached
                if cache_path.exists():
                    logger.info(f"Using cached image: {cache_path.name}")
                    return str(cache_path)
                
                # Download image
                logger.info(f"Downloading image from: {image_url}")
                response = self.session.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Validate image content
                if not self._validate_image_content(response.content, image_url):
                    logger.warning("Downloaded content is not a valid image")
                    return None
                
                # Save to cache
                with open(cache_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Image downloaded and cached: {cache_path.name}")
                logger.debug(f"Image size: {len(response.content):,} bytes")
                
                return str(cache_path)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download image from {image_url}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error downloading image: {e}")
                return None
    
    def get_best_image_url(self, content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract the best available image from Wikipedia content.
        
        Args:
            content: Wikipedia page content from crawler
            
        Returns:
            Best image info dict, or None if no suitable image found
        """
        # Try thumbnail first (usually best quality for our use case)
        thumbnail = content.get('thumbnail', {})
        if thumbnail and thumbnail.get('source'):
            # Prefer larger thumbnails if available
            width = thumbnail.get('width', 0)
            height = thumbnail.get('height', 0)
            
            # Check if thumbnail is reasonable size (at least 200px on one side)
            if width >= 200 or height >= 200:
                logger.debug(f"Using thumbnail image: {width}x{height}")
                return thumbnail
        
        # Try original image as fallback
        original_image = content.get('original_image', {})
        if original_image and original_image.get('source'):
            logger.debug("Using original image")
            return original_image
        
        logger.debug("No suitable image found in Wikipedia content")
        return None
    
    def _generate_cache_filename(self, image_url: str, page_title: str) -> str:
        """
        Generate a cache filename based on URL and page title.
        
        Args:
            image_url: URL of the image
            page_title: Wikipedia page title
            
        Returns:
            Cache filename
        """
        # Create hash of URL for uniqueness
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
        
        # Clean page title for filename
        clean_title = "".join(c for c in page_title if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_title = clean_title.replace(' ', '_')[:30]  # Limit length
        
        # Extract file extension
        parsed_url = urlparse(image_url)
        path = parsed_url.path.lower()
        extension = None
        
        for ext in self.supported_formats:
            if path.endswith(ext):
                extension = ext
                break
        
        if not extension:
            # Default to .jpg if can't determine
            extension = '.jpg'
        
        # Combine components
        if clean_title:
            filename = f"{clean_title}_{url_hash}{extension}"
        else:
            filename = f"wikipedia_image_{url_hash}{extension}"
        
        return filename
    
    def _validate_image_content(self, content: bytes, url: str) -> bool:
        """
        Validate that downloaded content is a valid image.
        
        Args:
            content: Downloaded file content
            url: Original URL for logging
            
        Returns:
            True if valid image, False otherwise
        """
        if len(content) < 100:  # Too small to be a real image
            logger.warning(f"Downloaded content too small: {len(content)} bytes")
            return False
        
        # Check for common image file signatures
        image_signatures = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89\x50\x4e\x47',  # PNG
            b'\x47\x49\x46\x38',  # GIF
            b'\x52\x49\x46\x46',  # WebP (RIFF header)
        ]
        
        for sig in image_signatures:
            if content.startswith(sig):
                return True
        
        logger.warning(f"Downloaded content doesn't match known image signatures from {url}")
        return False
    
    def cleanup_cache(self, max_age_days: int = 7):
        """
        Clean up old cached images.
        
        Args:
            max_age_days: Remove cached files older than this many days
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            removed_count = 0
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old cached images")
            
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")


def download_wikipedia_image(content: Dict[str, Any], 
                           cache_dir: str = "image_cache") -> Optional[str]:
    """
    Convenience function to download Wikipedia image from content.
    
    Args:
        content: Wikipedia page content from crawler
        cache_dir: Directory to cache images
        
    Returns:
        Path to downloaded image, or None if failed
    """
    downloader = WikipediaImageDownloader(cache_dir)
    
    # Get best image info
    image_info = downloader.get_best_image_url(content)
    if not image_info:
        return None
    
    # Download image
    page_title = content.get('title', 'unknown')
    return downloader.download_wikipedia_image(image_info, page_title)