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
        Detect available GPU encoders.
        
        Returns:
            Dictionary with encoder availability
        """
        encoders = {
            'h264_nvenc': False,    # NVIDIA NVENC
            'hevc_nvenc': False,    # NVIDIA NVENC HEVC
            'h264_vaapi': False,    # Intel/AMD VAAPI
            'h264_qsv': False,      # Intel Quick Sync
        }
        
        try:
            # Get list of available encoders
            cmd = [self.ffmpeg_path, "-hide_banner", "-encoders"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            encoder_list = result.stdout
            
            # Check for each encoder
            for encoder in encoders.keys():
                if encoder in encoder_list:
                    encoders[encoder] = True
                    logger.debug(f"Found GPU encoder: {encoder}")
            
            available_count = sum(encoders.values())
            logger.info(f"GPU encoders available: {available_count}/4")
            
            return encoders
            
        except Exception as e:
            logger.warning(f"Failed to detect GPU encoders: {e}")
            return encoders
    
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
        """Get optimized settings for GPU encoders."""
        base_settings = {
            'preset': 'fast',
            'crf': 23,  # Good quality/speed balance
            'threads': 0,
        }
        
        if encoder == 'h264_nvenc':
            return {
                **base_settings,
                'preset': 'p4',  # NVENC fast preset
                'tune': 'hq',    # High quality
                'rc': 'vbr',     # Variable bitrate
                'cq': 23,        # Quality level
                'gpu': 0,        # Use first GPU
            }
        elif encoder == 'h264_qsv':
            return {
                **base_settings,
                'preset': 'fast',
                'global_quality': 23,
            }
        elif encoder == 'h264_vaapi':
            return {
                **base_settings,
                'qp': 23,
                'quality': 'balanced',
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