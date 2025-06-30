"""
FFmpeg utilities for fast video processing operations.

Provides optimized video operations using FFmpeg stream copying
to avoid expensive re-encoding for simple operations like trimming.
"""

import os
import subprocess
import shutil
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)


class FFmpegError(Exception):
    """Exception raised when FFmpeg operations fail."""
    pass


class FFmpegUtils:
    """
    Utility class for optimized FFmpeg operations.
    
    Focuses on performance-critical operations like video trimming
    using stream copying instead of re-encoding.
    """
    
    def __init__(self):
        """Initialize FFmpeg utilities."""
        self.ffmpeg_path = self._find_ffmpeg()
        if not self.ffmpeg_path:
            raise FFmpegError("FFmpeg not found in PATH")
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable in PATH."""
        return shutil.which("ffmpeg")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            raise FFmpegError("FFprobe not found in PATH")
        
        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            
            # Extract useful information
            video_info = {
                'duration': float(info['format']['duration']),
                'size_bytes': int(info['format']['size']),
                'bitrate': int(info['format']['bit_rate']) if 'bit_rate' in info['format'] else None,
                'streams': []
            }
            
            for stream in info['streams']:
                stream_info = {
                    'codec_type': stream['codec_type'],
                    'codec_name': stream['codec_name'],
                    'index': stream['index']
                }
                
                if stream['codec_type'] == 'video':
                    stream_info.update({
                        'width': stream.get('width'),
                        'height': stream.get('height'),
                        'fps': eval(stream.get('r_frame_rate', '0/1'))
                    })
                
                video_info['streams'].append(stream_info)
            
            return video_info
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"Failed to get video info: {e.stderr}")
        except Exception as e:
            raise FFmpegError(f"Failed to parse video info: {e}")
    
    def trim_video_fast(self, input_path: str, output_path: str, 
                       start_time: float = 0.0, duration: Optional[float] = None) -> bool:
        """
        Trim video using stream copying for maximum speed.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            start_time: Start time in seconds (default: 0.0)
            duration: Duration in seconds (None = to end of video)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build FFmpeg command for stream copying
            cmd = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-ss", str(start_time),  # Seek to start time
                "-i", input_path,        # Input file
            ]
            
            # Add duration if specified
            if duration is not None:
                cmd.extend(["-t", str(duration)])
            
            # Use stream copying for maximum speed
            cmd.extend([
                "-c", "copy",           # Copy all streams without re-encoding
                "-avoid_negative_ts", "make_zero",  # Handle timestamp issues
                "-y",                   # Overwrite output file
                output_path
            ])
            
            logger.info(f"Fast trimming video: {Path(input_path).name} -> {Path(output_path).name}")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            # Execute command
            start_time_exec = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            exec_time = time.time() - start_time_exec
            
            logger.info(f"✅ Video trimmed in {exec_time:.2f}s (stream copy)")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg trim failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Video trim error: {e}")
            return False
    
    def detect_gpu_encoders(self) -> Dict[str, bool]:
        """
        Detect available GPU encoders that actually work with MoviePy.
        
        Returns:
            Dictionary with encoder availability tested for MoviePy compatibility
        """
        encoders = {
            'h264_nvenc': False,    # NVIDIA NVENC
            'hevc_nvenc': False,    # NVIDIA NVENC HEVC
            'h264_vaapi': False,    # Intel/AMD VAAPI
            'h264_qsv': False,      # Intel Quick Sync
        }
        
        try:
            # First check if encoders are listed
            cmd = [self.ffmpeg_path, "-hide_banner", "-encoders"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            encoder_list = result.stdout
            
            # Test each encoder with a small test encode to verify MoviePy compatibility
            for encoder in list(encoders.keys()):
                if encoder in encoder_list:
                    if self._test_encoder_compatibility(encoder):
                        encoders[encoder] = True
                        logger.debug(f"Verified working GPU encoder: {encoder}")
                    else:
                        logger.debug(f"GPU encoder found but not compatible with MoviePy: {encoder}")
            
            available_count = sum(encoders.values())
            logger.info(f"GPU encoders available: {available_count}/4")
            
            return encoders
            
        except Exception as e:
            logger.warning(f"Failed to detect GPU encoders: {e}")
            return encoders
    
    def _test_encoder_compatibility(self, encoder: str) -> bool:
        """
        Test if an encoder actually works by trying a small test encode.
        
        Args:
            encoder: Encoder name to test
            
        Returns:
            True if encoder works, False otherwise
        """
        try:
            # Create a minimal test video (1 frame, 1 second)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                test_output = tmp_file.name
            
            # Minimal test encode command
            cmd = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-f", "lavfi",
                "-i", "testsrc=duration=0.1:size=64x64:rate=1",
                "-c:v", encoder,
                "-t", "0.1",
                "-y",
                test_output
            ]
            
            # Add encoder-specific parameters if needed
            if encoder == 'h264_nvenc':
                cmd.extend(["-preset", "fast"])
            elif encoder == 'h264_qsv':
                cmd.extend(["-preset", "fast"])
            
            # Try the encode with a short timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Clean up test file
            try:
                os.unlink(test_output)
            except:
                pass
            
            # Check if encode succeeded
            return result.returncode == 0
            
        except Exception as e:
            logger.debug(f"Encoder test failed for {encoder}: {e}")
            return False
    
    def get_best_encoder(self, prefer_gpu: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best available encoder configuration.
        
        Args:
            prefer_gpu: Whether to prefer GPU encoders
            
        Returns:
            Tuple of (codec_name, encoder_settings)
        """
        if prefer_gpu:
            gpu_encoders = self.detect_gpu_encoders()
            
            # Priority order for GPU encoders
            gpu_priority = ['h264_nvenc', 'h264_qsv', 'h264_vaapi']
            
            for encoder in gpu_priority:
                if gpu_encoders.get(encoder):
                    settings = self._get_gpu_encoder_settings(encoder)
                    logger.info(f"Selected GPU encoder: {encoder}")
                    return encoder, settings
        
        # Fallback to optimized CPU encoder
        logger.info("Using optimized CPU encoder: libx264")
        return 'libx264', self._get_cpu_encoder_settings()
    
    def _get_gpu_encoder_settings(self, encoder: str) -> Dict[str, Any]:
        """Get optimized settings for GPU encoders compatible with MoviePy."""
        base_settings = {
            'preset': 'fast',
            'threads': 0,
        }
        
        if encoder == 'h264_nvenc':
            # MoviePy-compatible NVENC settings
            return {
                **base_settings,
                'preset': 'fast',    # Use standard preset for MoviePy compatibility
                'bitrate': '5000k',  # Use bitrate instead of CRF for GPU encoders
            }
        elif encoder == 'h264_qsv':
            return {
                **base_settings,
                'preset': 'fast',
                'bitrate': '5000k',
            }
        elif encoder == 'h264_vaapi':
            return {
                **base_settings,
                'bitrate': '5000k',
            }
        
        return base_settings
    
    def _get_cpu_encoder_settings(self) -> Dict[str, Any]:
        """Get optimized settings for CPU encoding."""
        return {
            'preset': 'fast',      # Faster than 'medium'
            'crf': 23,             # Good quality
            'threads': 0,          # Auto-detect threads
            'tune': 'fastdecode',  # Optimize for playback
        }
    
    def trim_audio_fast(self, input_path: str, output_path: str,
                       start_time: float = 0.0, duration: Optional[float] = None) -> bool:
        """
        Trim audio using stream copying for maximum speed.
        
        Args:
            input_path: Path to input audio
            output_path: Path to output audio
            start_time: Start time in seconds (default: 0.0)
            duration: Duration in seconds (None = to end of audio)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-ss", str(start_time),
                "-i", input_path,
            ]
            
            if duration is not None:
                cmd.extend(["-t", str(duration)])
            
            cmd.extend([
                "-c", "copy",  # Copy audio stream without re-encoding
                "-avoid_negative_ts", "make_zero",
                "-y",
                output_path
            ])
            
            logger.debug(f"Fast trimming audio: {Path(input_path).name} -> {Path(output_path).name}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug("✅ Audio trimmed (stream copy)")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg audio trim failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Audio trim error: {e}")
            return False


# Module-level convenience functions
def trim_video_fast(input_path: str, output_path: str, 
                   start_time: float = 0.0, duration: Optional[float] = None) -> bool:
    """
    Convenience function for fast video trimming.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video  
        start_time: Start time in seconds
        duration: Duration in seconds (None = to end)
        
    Returns:
        True if successful
    """
    utils = FFmpegUtils()
    return utils.trim_video_fast(input_path, output_path, start_time, duration)


def get_video_duration(video_path: str) -> float:
    """
    Get video duration quickly using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    utils = FFmpegUtils()
    info = utils.get_video_info(video_path)
    return info['duration']


# Import time for timing operations
import time