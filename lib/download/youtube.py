"""
YouTube video and audio downloader for MoneyTree.

Downloads YouTube content for use as template videos or background music
in educational content creation.
"""

import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, parse_qs

try:
    import yt_dlp
    HAS_YT_DLP = True
except ImportError:
    HAS_YT_DLP = False


@dataclass
class DownloadConfig:
    """Configuration for YouTube downloads."""
    output_dir: str = "downloads"
    video_quality: str = "720p"  # "144p", "240p", "360p", "480p", "720p", "1080p", "1440p", "2160p", "4320p", "high", "highest", "best"
    audio_quality: str = "128"   # "64", "128", "192", "256", "320", "best"
    output_format: str = "mp4"   # For video: "mp4", "webm", "mkv", "any" (auto-select best)
    audio_format: str = "mp3"    # For audio: "mp3", "m4a", "wav", "flac"
    include_subtitles: bool = True
    subtitle_language: str = "en"
    max_filesize: str = "100M"   # "50M", "100M", "500M", "1G", "2G", "5G"
    codec_preference: str = "any"  # "h264", "vp9", "av1", "any" (prefer better codecs)
    prefer_60fps: bool = False   # Prefer 60fps when available
    show_available_formats: bool = False  # Debug: show all available formats


@dataclass
class VideoInfo:
    """Information about a YouTube video."""
    id: str
    title: str
    description: str
    duration: float
    view_count: int
    uploader: str
    upload_date: str
    thumbnail_url: str
    url: str
    available_formats: List[str]


class YouTubeDownloader:
    """
    Downloads YouTube videos and audio for educational content creation.
    
    Provides high-quality downloads with configurable formats and quality settings
    for use as template videos or background music in MoneyTree.
    """
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        """
        Initialize the YouTube downloader.
        
        Args:
            config: Download configuration, uses defaults if None
        """
        if not HAS_YT_DLP:
            raise ImportError(
                "YouTube download dependencies not available. Install with: "
                "uv add yt-dlp"
            )
        
        self.config = config or DownloadConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create organized subdirectories
        self.video_dir = self.output_dir / "videos"
        self.audio_dir = self.output_dir / "audio"
        self.video_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        
        # Configure yt-dlp options
        self._setup_downloader()
    
    def _setup_downloader(self):
        """Configure yt-dlp with optimal settings."""
        self.ydl_opts_base = {
            'restrictfilenames': True,  # Use ASCII filenames
            'ignoreerrors': True,
            'no_warnings': False,
            'extractflat': False,
            'writethumbnail': False,
            'writeinfojson': False,
        }
    
    def _get_video_format_string(self, quality: str) -> str:
        """
        Generate intelligent format string for video downloads.
        
        Args:
            quality: Quality preference (e.g., "720p", "1080p", "4k", "high", "highest")
            
        Returns:
            yt-dlp format string optimized for quality
        """
        # Quality preset mappings - prioritize separate video+audio for highest quality
        quality_presets = {
            "highest": "bestvideo+bestaudio/best[height>=2160]/best[height>=1440]/best[height>=1080]/best",
            "high": "bestvideo[height>=1080][fps<=60]+bestaudio/best[height>=1080][fps<=60]/best[height>=720]/best",
            "4k": "bestvideo[height>=2160]+bestaudio/best[height>=2160]/bestvideo[height=2160]+bestaudio/best[height>=1440]/best",
            "8k": "bestvideo[height>=4320]+bestaudio/best[height>=4320]/bestvideo[height=4320]+bestaudio/best[height>=2160]/best",
            "2160p": "bestvideo[height>=2160]+bestaudio/best[height>=2160]/bestvideo[height=2160]+bestaudio/best[height>=1440]/best",
            "1440p": "bestvideo[height>=1440]+bestaudio/best[height>=1440]/bestvideo[height=1440]+bestaudio/best[height>=1080]/best",
        }
        
        # Handle quality presets
        if quality in quality_presets:
            format_string = quality_presets[quality]
        elif quality == "best":
            format_string = "best"
        else:
            # Parse specific resolution (e.g., "720p", "1080p")
            height = quality.replace('p', '') if quality.endswith('p') else quality
            
            # Build format string with intelligent fallbacks
            format_parts = []
            
            # Codec preference logic
            if self.config.codec_preference == "vp9":
                codec_filter = "[vcodec*=vp9]"
            elif self.config.codec_preference == "av1":
                codec_filter = "[vcodec*=av01]"
            elif self.config.codec_preference == "h264":
                codec_filter = "[vcodec*=avc1]"
            else:
                codec_filter = ""
            
            # FPS preference
            fps_filter = "[fps<=60]" if self.config.prefer_60fps else ""
            
            # Container format preference
            if self.config.output_format == "any":
                container_filter = ""
            else:
                container_filter = f"[ext={self.config.output_format}]"
            
            # Build progressive format string with fallbacks
            if height.isdigit():
                height_num = int(height)
                
                # For high quality (720p+), prioritize separate video+audio streams
                if height_num >= 720:
                    # Primary: separate video+audio with all preferences (highest quality)
                    format_parts.append(f"bestvideo[height={height_num}]{codec_filter}{fps_filter}+bestaudio")
                    
                    # Fallback 1: separate video+audio with codec preference only
                    if codec_filter:
                        format_parts.append(f"bestvideo[height={height_num}]{codec_filter}+bestaudio")
                    
                    # Fallback 2: separate video+audio, any codec
                    format_parts.append(f"bestvideo[height={height_num}]+bestaudio")
                    
                    # Fallback 3: separate video+audio up to height
                    format_parts.append(f"bestvideo[height<={height_num}]{codec_filter}+bestaudio")
                    format_parts.append(f"bestvideo[height<={height_num}]+bestaudio")
                
                # Combined stream fallbacks
                format_parts.append(f"best[height={height_num}]{codec_filter}{fps_filter}{container_filter}")
                
                if codec_filter:
                    format_parts.append(f"best[height={height_num}]{codec_filter}")
                
                format_parts.append(f"best[height={height_num}]")
                format_parts.append(f"best[height<={height_num}]")
            else:
                # Non-numeric quality, use best available
                format_parts.append("best")
            
            # Final fallback
            format_parts.append("best")
            
            format_string = "/".join(format_parts)
        
        return format_string
    
    def get_video_info(self, url: str) -> Optional[VideoInfo]:
        """
        Get information about a YouTube video without downloading.
        
        Args:
            url: YouTube URL
            
        Returns:
            VideoInfo object with video details, None if failed
        """
        try:
            # Validate YouTube URL
            if not self._is_valid_youtube_url(url):
                print(f"❌ Invalid YouTube URL: {url}")
                return None
            
            print(f"📹 Getting video info for: {url}")
            
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if not info:
                    return None
                
                # Extract available formats
                formats = []
                if 'formats' in info:
                    for fmt in info['formats']:
                        if fmt.get('height'):  # Video format
                            formats.append(f"{fmt['height']}p")
                        elif fmt.get('abr'):   # Audio format
                            formats.append(f"{fmt['abr']}kbps")
                
                return VideoInfo(
                    id=info.get('id', ''),
                    title=info.get('title', 'Unknown'),
                    description=info.get('description', ''),
                    duration=info.get('duration', 0),
                    view_count=info.get('view_count', 0),
                    uploader=info.get('uploader', 'Unknown'),
                    upload_date=info.get('upload_date', ''),
                    thumbnail_url=info.get('thumbnail', ''),
                    url=url,
                    available_formats=list(set(formats))
                )
                
        except Exception as e:
            print(f"❌ Failed to get video info: {e}")
            return None
    
    def download_video(
        self, 
        url: str, 
        output_filename: Optional[str] = None,
        quality: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download YouTube video for use as template.
        
        Args:
            url: YouTube URL
            output_filename: Custom output filename (without extension)
            quality: Video quality override ("720p", "1080p", etc.)
            
        Returns:
            Dictionary with download results
        """
        try:
            if not self._is_valid_youtube_url(url):
                return self._error_response("Invalid YouTube URL")
            
            quality = quality or self.config.video_quality
            print(f"📹 Downloading video: {quality} quality")
            print(f"🔗 URL: {url}")
            
            # Configure video download options
            ydl_opts = self.ydl_opts_base.copy()
            
            if output_filename:
                safe_filename = self._sanitize_filename(output_filename)
                ydl_opts['outtmpl'] = str(self.video_dir / f"{safe_filename}.%(ext)s")
            else:
                ydl_opts['outtmpl'] = str(self.video_dir / '%(title)s.%(ext)s')
            
            # Generate intelligent format string
            format_string = self._get_video_format_string(quality)
            ydl_opts['format'] = format_string
            
            print(f"🎯 Format selector: {format_string}")
            
            # Enable verbose logging to see what's actually happening
            ydl_opts['verbose'] = True
            
            # Configure for highest quality downloads
            ydl_opts['merge_output_format'] = 'mp4'  # Merge video+audio to mp4
            ydl_opts['keepvideo'] = False  # Don't keep separate streams after merge
            
            # Only add FFmpeg merger for video+audio streams (no re-encoding)
            if '+' in format_string:  # Only if we're using separate video+audio
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4'
                }]
            else:
                ydl_opts['postprocessors'] = []
            
            # Show available formats if requested
            if self.config.show_available_formats:
                print("🔍 Checking available formats...")
                try:
                    with yt_dlp.YoutubeDL({'quiet': True}) as temp_ydl:
                        info = temp_ydl.extract_info(url, download=False)
                        if info and 'formats' in info:
                            print("📋 Available formats:")
                            for fmt in info['formats'][:10]:  # Show first 10
                                height = fmt.get('height', 'N/A')
                                fps = fmt.get('fps', 'N/A')
                                vcodec = fmt.get('vcodec', 'N/A')
                                ext = fmt.get('ext', 'N/A')
                                filesize = fmt.get('filesize')
                                size_mb = f"{filesize/(1024*1024):.1f}MB" if filesize else "N/A"
                                print(f"   {height}p {fps}fps {vcodec} {ext} ({size_mb})")
                except Exception as e:
                    print(f"⚠️  Could not list formats: {e}")
            
            # Add subtitle options
            if self.config.include_subtitles:
                ydl_opts.update({
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': [self.config.subtitle_language],
                    'subtitlesformat': 'srt',
                })
            
            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                if not info:
                    return self._error_response("Failed to extract video info")
                
                # Find the downloaded file
                title = info.get('title', 'video')
                safe_title = self._sanitize_filename(title)
                
                if output_filename:
                    expected_filename = f"{self._sanitize_filename(output_filename)}.{self.config.output_format}"
                else:
                    expected_filename = f"{safe_title}.{self.config.output_format}"
                
                output_path = self.video_dir / expected_filename
                
                # Check if file exists (yt-dlp might have modified the name)
                if not output_path.exists():
                    # Try to find the actual downloaded file with various extensions
                    possible_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv', '.m4v']
                    
                    # First try exact title match with different extensions
                    for ext in possible_extensions:
                        test_path = self.video_dir / f"{safe_title}{ext}"
                        if test_path.exists():
                            output_path = test_path
                            print(f"📁 Found file: {output_path.name}")
                            break
                    
                    # If still not found, search for any video file with similar name
                    if not output_path.exists():
                        for file_path in self.video_dir.glob(f"{safe_title}*"):
                            if file_path.suffix.lower() in possible_extensions:
                                output_path = file_path
                                print(f"📁 Found similar file: {output_path.name}")
                                break
                        
                        # Last resort: find the most recent video file
                        if not output_path.exists():
                            video_files = []
                            for ext in possible_extensions:
                                video_files.extend(self.video_dir.glob(f"*{ext}"))
                            
                            if video_files:
                                output_path = max(video_files, key=lambda p: p.stat().st_mtime)
                                print(f"📁 Using most recent video: {output_path.name}")
                
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    
                    # Analyze the actual downloaded video quality
                    actual_quality = self._analyze_video_quality(output_path)
                    
                    return {
                        "success": True,
                        "output_path": str(output_path),
                        "file_size": file_size,
                        "title": info.get('title', 'Unknown'),
                        "duration": info.get('duration', 0),
                        "quality": quality,
                        "actual_quality": actual_quality,
                        "format": output_path.suffix[1:],  # Actual file extension
                        "download_type": "video"
                    }
                else:
                    return self._error_response("Downloaded file not found")
                    
        except Exception as e:
            return self._error_response(f"Download failed: {e}")
    
    def download_audio(
        self, 
        url: str, 
        output_filename: Optional[str] = None,
        quality: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download YouTube audio for use as background music.
        
        Args:
            url: YouTube URL
            output_filename: Custom output filename (without extension)
            quality: Audio quality override ("128", "192", etc.)
            
        Returns:
            Dictionary with download results
        """
        try:
            if not self._is_valid_youtube_url(url):
                return self._error_response("Invalid YouTube URL")
            
            quality = quality or self.config.audio_quality
            print(f"🎵 Downloading audio: {quality}kbps quality")
            print(f"🔗 URL: {url}")
            
            # Configure audio download options
            ydl_opts = self.ydl_opts_base.copy()
            
            # Set output template for audio
            if output_filename:
                safe_filename = self._sanitize_filename(output_filename)
                ydl_opts['outtmpl'] = str(self.audio_dir / f"{safe_filename}.%(ext)s")
            else:
                ydl_opts['outtmpl'] = str(self.audio_dir / f"%(title)s.%(ext)s")
            
            # Set audio-only format with better fallback
            if quality == "best":
                ydl_opts['format'] = 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best[height<=480]/best'
            else:
                # Try to get audio with specified quality, with good fallbacks
                ydl_opts['format'] = f'bestaudio[abr>={quality}][abr<={int(quality)+50}]/bestaudio[abr<={quality}]/bestaudio/best[height<=480]/best'
            
            # Post-processing to convert to desired audio format
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.config.audio_format,
                'preferredquality': quality if quality != "best" else "192",
            }]
            
            # Enable verbose logging for debugging
            ydl_opts['verbose'] = False
            ydl_opts['extractaudio'] = True
            ydl_opts['audioformat'] = self.config.audio_format
            
            print(f"🔧 Audio format selector: {ydl_opts['format']}")
            print(f"🎵 Target format: {self.config.audio_format}")
            
            # Download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                if not info:
                    return self._error_response("Failed to extract video info")
                
                print(f"📝 Video title: {info.get('title', 'Unknown')}")
                
                # More robust file finding approach
                title = info.get('title', 'audio')
                safe_title = self._sanitize_filename(title)
                
                # Look for downloaded files in multiple ways
                possible_files = []
                
                # Check exact expected filename
                if output_filename:
                    expected_name = self._sanitize_filename(output_filename)
                    for ext in [self.config.audio_format, 'm4a', 'mp3', 'webm', 'opus']:
                        possible_files.append(self.audio_dir / f"{expected_name}.{ext}")
                
                # Check title-based filenames
                for ext in [self.config.audio_format, 'm4a', 'mp3', 'webm', 'opus']:
                    possible_files.append(self.audio_dir / f"{safe_title}.{ext}")
                
                # Search for any audio files with similar names
                if self.audio_dir.exists():
                    for pattern in [f"{safe_title}*", f"*{safe_title[:20]}*"]:
                        for file_path in self.audio_dir.glob(pattern):
                            if file_path.suffix.lower() in ['.mp3', '.m4a', '.wav', '.flac', '.opus', '.webm']:
                                possible_files.append(file_path)
                
                # Find the actual downloaded file
                output_path = None
                for file_path in possible_files:
                    if file_path.exists():
                        output_path = file_path
                        print(f"✅ Found downloaded file: {output_path.name}")
                        break
                
                if not output_path:
                    # Last resort: find the most recent audio file
                    audio_files = []
                    for ext in ['.mp3', '.m4a', '.wav', '.flac', '.opus', '.webm']:
                        audio_files.extend(self.audio_dir.glob(f"*{ext}"))
                    
                    if audio_files:
                        # Get the most recently modified file
                        output_path = max(audio_files, key=lambda p: p.stat().st_mtime)
                        print(f"🔍 Using most recent audio file: {output_path.name}")
                
                if output_path and output_path.exists():
                    file_size = output_path.stat().st_size
                    
                    return {
                        "success": True,
                        "output_path": str(output_path),
                        "file_size": file_size,
                        "title": info.get('title', 'Unknown'),
                        "duration": info.get('duration', 0),
                        "quality": f"{quality}kbps",
                        "format": output_path.suffix[1:],  # Get actual format from file extension
                        "download_type": "audio"
                    }
                else:
                    # List files in download directory for debugging
                    if self.audio_dir.exists():
                        files = list(self.audio_dir.iterdir())
                        print(f"📁 Files in audio directory: {[f.name for f in files]}")
                    
                    return self._error_response("Downloaded audio file not found")
                    
        except Exception as e:
            print(f"🔍 Audio download error details: {str(e)}")
            return self._error_response(f"Audio download failed: {e}")
    
    def download_playlist_info(self, url: str) -> List[VideoInfo]:
        """
        Get information about all videos in a YouTube playlist.
        
        Args:
            url: YouTube playlist URL
            
        Returns:
            List of VideoInfo objects for each video in playlist
        """
        try:
            if not self._is_valid_youtube_url(url, allow_playlist=True):
                print(f"❌ Invalid YouTube playlist URL: {url}")
                return []
            
            print(f"📋 Getting playlist info for: {url}")
            
            with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if not info or 'entries' not in info:
                    return []
                
                playlist_videos = []
                for entry in info['entries']:
                    if entry:
                        video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                        video_info = self.get_video_info(video_url)
                        if video_info:
                            playlist_videos.append(video_info)
                
                print(f"📋 Found {len(playlist_videos)} videos in playlist")
                return playlist_videos
                
        except Exception as e:
            print(f"❌ Failed to get playlist info: {e}")
            return []
    
    def _is_valid_youtube_url(self, url: str, allow_playlist: bool = False) -> bool:
        """Check if URL is a valid YouTube URL."""
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        ]
        
        if allow_playlist:
            youtube_patterns.extend([
                r'(?:https?://)?(?:www\.)?youtube\.com/playlist\?list=[\w-]+',
                r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+&list=[\w-]+',
            ])
        
        return any(re.match(pattern, url) for pattern in youtube_patterns)
    
    def _analyze_video_quality(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze the actual quality of a downloaded video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with quality information
        """
        try:
            # Try to get video info using yt-dlp
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(str(video_path), download=False)
                
                if info:
                    return {
                        "resolution": f"{info.get('width', 'N/A')}x{info.get('height', 'N/A')}",
                        "height": info.get('height', 'Unknown'),
                        "fps": info.get('fps', 'Unknown'),
                        "vcodec": info.get('vcodec', 'Unknown'),
                        "acodec": info.get('acodec', 'Unknown'),
                        "bitrate": info.get('tbr', 'Unknown'),
                        "container": info.get('ext', 'Unknown')
                    }
        except Exception as e:
            print(f"⚠️  Could not analyze video quality: {e}")
        
        # Fallback: basic file info
        return {
            "resolution": "Unknown",
            "height": "Unknown", 
            "fps": "Unknown",
            "vcodec": "Unknown",
            "acodec": "Unknown",
            "bitrate": "Unknown",
            "container": video_path.suffix[1:] if video_path.suffix else "Unknown"
        }
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system storage."""
        # Remove or replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'[^\w\s-]', '', filename)
        filename = re.sub(r'[-\s]+', '_', filename)
        return filename.strip('_')[:100]  # Limit length
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        print(f"❌ {error_message}")
        return {
            "success": False,
            "error": error_message,
            "output_path": None,
            "file_size": 0,
            "download_type": "error"
        }
    
    def get_download_history(self) -> List[Dict[str, Any]]:
        """Get list of previously downloaded files."""
        downloads = []
        
        # Check video directory
        if self.video_dir.exists():
            for file_path in self.video_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.webm', '.mkv']:
                    file_size = file_path.stat().st_size
                    downloads.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": file_size,
                        "type": "video",
                        "modified": file_path.stat().st_mtime,
                        "category": "videos"
                    })
        
        # Check audio directory
        if self.audio_dir.exists():
            for file_path in self.audio_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.mp3', '.m4a', '.wav', '.flac', '.opus']:
                    file_size = file_path.stat().st_size
                    downloads.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": file_size,
                        "type": "audio",
                        "modified": file_path.stat().st_mtime,
                        "category": "audio"
                    })
        
        # Sort by modification time (newest first)
        downloads.sort(key=lambda x: x['modified'], reverse=True)
        return downloads
    
    def cleanup_downloads(self, older_than_days: int = 30) -> int:
        """
        Clean up old downloaded files.
        
        Args:
            older_than_days: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        import time
        
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        removed_count = 0
        
        # Clean up video files
        if self.video_dir.exists():
            for file_path in self.video_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        print(f"🗑️  Removed old video: {file_path.name}")
                    except Exception as e:
                        print(f"⚠️  Failed to remove {file_path.name}: {e}")
        
        # Clean up audio files
        if self.audio_dir.exists():
            for file_path in self.audio_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        print(f"🗑️  Removed old audio: {file_path.name}")
                    except Exception as e:
                        print(f"⚠️  Failed to remove {file_path.name}: {e}")
        
        return removed_count


def get_recommended_channels() -> List[Dict[str, str]]:
    """Get recommended YouTube channels for educational content templates."""
    return [
        {
            "name": "Kurzgesagt – In a Nutshell",
            "url": "https://www.youtube.com/@kurzgesagt",
            "type": "Science Animation",
            "description": "High-quality animated educational videos"
        },
        {
            "name": "3Blue1Brown", 
            "url": "https://www.youtube.com/@3blue1brown",
            "type": "Mathematics",
            "description": "Mathematical concepts with beautiful animations"
        },
        {
            "name": "SciShow",
            "url": "https://www.youtube.com/@SciShow",
            "type": "Science Education",
            "description": "Quick science facts and explanations"
        },
        {
            "name": "Crash Course",
            "url": "https://www.youtube.com/@crashcourse",
            "type": "Educational Series",
            "description": "Comprehensive educational series on various topics"
        },
        {
            "name": "TED-Ed",
            "url": "https://www.youtube.com/@TEDEd",
            "type": "Educational Animation",
            "description": "Animated lessons on diverse topics"
        }
    ]


def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None