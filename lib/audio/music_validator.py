"""
Background music validation and analysis for video generation.

Provides comprehensive validation, duration analysis, and optimization recommendations
for background music files in the MoneyTree pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False


class MusicFormat(Enum):
    """Supported music formats with quality ratings."""
    MP3 = ("mp3", "Good", "Widely compatible, good compression")
    WAV = ("wav", "Excellent", "Uncompressed, best quality, large files")
    M4A = ("m4a", "Good", "AAC compression, good quality")
    AAC = ("aac", "Good", "Advanced Audio Coding, efficient")
    OGG = ("ogg", "Good", "Open source, good compression")
    FLAC = ("flac", "Excellent", "Lossless compression, perfect quality")
    WMA = ("wma", "Fair", "Windows Media Audio, limited compatibility")
    
    def __init__(self, extension: str, quality: str, description: str):
        self.extension = extension
        self.quality = quality
        self.description = description


@dataclass
class MusicAnalysis:
    """Analysis results for background music file."""
    file_path: str
    exists: bool
    format_supported: bool
    format_info: Optional[MusicFormat]
    duration: Optional[float]
    sample_rate: Optional[int]
    channels: Optional[int]
    file_size: Optional[int]
    
    # Duration analysis
    duration_vs_content: str  # "perfect", "short", "long", "unknown"
    duration_ratio: Optional[float]  # music_duration / content_duration
    
    # Quality analysis
    quality_score: float  # 0.0 to 1.0
    quality_warnings: List[str]
    recommendations: List[str]
    
    # Validation results
    is_valid: bool
    error_message: Optional[str]


class MusicValidator:
    """
    Validates and analyzes background music files for video generation.
    
    Provides format validation, duration analysis, and optimization recommendations.
    """
    
    def __init__(self):
        """Initialize the music validator."""
        self.supported_formats = {fmt.extension.lower() for fmt in MusicFormat}
        
    def validate_music_file(self, music_path: str, expected_duration: Optional[float] = None) -> MusicAnalysis:
        """
        Validate and analyze a background music file.
        
        Args:
            music_path: Path to the music file
            expected_duration: Expected content duration for comparison
            
        Returns:
            MusicAnalysis with validation and analysis results
        """
        if not music_path:
            return self._create_invalid_analysis("No music file provided")
        
        file_path = Path(music_path)
        
        # Check if file exists
        if not file_path.exists():
            return self._create_invalid_analysis(f"Music file not found: {music_path}")
        
        # Get file info
        file_size = file_path.stat().st_size
        extension = file_path.suffix.lower().lstrip('.')
        
        # Check format support
        format_info = self._get_format_info(extension)
        format_supported = extension in self.supported_formats
        
        if not format_supported:
            return MusicAnalysis(
                file_path=str(file_path),
                exists=True,
                format_supported=False,
                format_info=format_info,
                duration=None,
                sample_rate=None,
                channels=None,
                file_size=file_size,
                duration_vs_content="unknown",
                duration_ratio=None,
                quality_score=0.0,
                quality_warnings=[f"Unsupported format: {extension}"],
                recommendations=[f"Convert to supported format: {', '.join(sorted(self.supported_formats))}"],
                is_valid=False,
                error_message=f"Unsupported music format: {extension}"
            )
        
        # Analyze audio properties
        try:
            duration, sample_rate, channels = self._analyze_audio_properties(str(file_path))
        except Exception as e:
            return MusicAnalysis(
                file_path=str(file_path),
                exists=True,
                format_supported=True,
                format_info=format_info,
                duration=None,
                sample_rate=None,
                channels=None,
                file_size=file_size,
                duration_vs_content="unknown",
                duration_ratio=None,
                quality_score=0.0,
                quality_warnings=[f"Could not analyze audio: {e}"],
                recommendations=["Verify file is not corrupted"],
                is_valid=False,
                error_message=f"Audio analysis failed: {e}"
            )
        
        # Duration analysis
        duration_vs_content, duration_ratio = self._analyze_duration_compatibility(
            duration, expected_duration
        )
        
        # Quality analysis
        quality_score, quality_warnings, recommendations = self._analyze_audio_quality(
            file_path, duration, sample_rate, channels, format_info
        )
        
        return MusicAnalysis(
            file_path=str(file_path),
            exists=True,
            format_supported=True,
            format_info=format_info,
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            file_size=file_size,
            duration_vs_content=duration_vs_content,
            duration_ratio=duration_ratio,
            quality_score=quality_score,
            quality_warnings=quality_warnings,
            recommendations=recommendations,
            is_valid=True,
            error_message=None
        )
    
    def _create_invalid_analysis(self, error: str) -> MusicAnalysis:
        """Create an invalid analysis result."""
        return MusicAnalysis(
            file_path="",
            exists=False,
            format_supported=False,
            format_info=None,
            duration=None,
            sample_rate=None,
            channels=None,
            file_size=None,
            duration_vs_content="unknown",
            duration_ratio=None,
            quality_score=0.0,
            quality_warnings=[],
            recommendations=[],
            is_valid=False,
            error_message=error
        )
    
    def _get_format_info(self, extension: str) -> Optional[MusicFormat]:
        """Get format information for file extension."""
        for fmt in MusicFormat:
            if fmt.extension.lower() == extension.lower():
                return fmt
        return None
    
    def _analyze_audio_properties(self, file_path: str) -> Tuple[float, int, int]:
        """
        Analyze basic audio properties.
        
        Returns:
            Tuple of (duration, sample_rate, channels)
        """
        if HAS_LIBROSA:
            try:
                # Use librosa for accurate analysis
                y, sr = librosa.load(file_path, sr=None)
                duration = len(y) / sr
                channels = 1 if y.ndim == 1 else y.shape[0]
                return duration, sr, channels
            except Exception:
                pass
        
        if HAS_MOVIEPY:
            try:
                # Fallback to MoviePy
                with AudioFileClip(file_path) as audio_clip:
                    duration = audio_clip.duration
                    # MoviePy doesn't easily give us sample rate/channels
                    # Use reasonable defaults
                    return duration, 44100, 2
            except Exception:
                pass
        
        raise Exception("No audio analysis library available (install librosa or moviepy)")
    
    def _analyze_duration_compatibility(self, music_duration: Optional[float], 
                                      content_duration: Optional[float]) -> Tuple[str, Optional[float]]:
        """
        Analyze music duration vs content duration.
        
        Returns:
            Tuple of (compatibility_status, duration_ratio)
        """
        if not music_duration or not content_duration:
            return "unknown", None
        
        ratio = music_duration / content_duration
        
        if 0.8 <= ratio <= 1.2:
            return "perfect", ratio
        elif ratio < 0.5:
            return "short", ratio
        elif ratio > 2.0:
            return "long", ratio
        elif ratio < 0.8:
            return "short", ratio
        else:
            return "long", ratio
    
    def _analyze_audio_quality(self, file_path: Path, duration: Optional[float], 
                             sample_rate: Optional[int], channels: Optional[int],
                             format_info: Optional[MusicFormat]) -> Tuple[float, List[str], List[str]]:
        """
        Analyze audio quality and provide recommendations.
        
        Returns:
            Tuple of (quality_score, warnings, recommendations)
        """
        quality_score = 0.5  # Base score
        warnings = []
        recommendations = []
        
        # Format quality scoring
        if format_info:
            if format_info.quality == "Excellent":
                quality_score += 0.3
            elif format_info.quality == "Good":
                quality_score += 0.2
            elif format_info.quality == "Fair":
                quality_score += 0.1
                warnings.append(f"Format has limited compatibility: {format_info.extension}")
                recommendations.append("Consider converting to MP3 or WAV for better compatibility")
        
        # Sample rate analysis
        if sample_rate:
            if sample_rate >= 44100:
                quality_score += 0.1
            elif sample_rate < 22050:
                warnings.append(f"Low sample rate: {sample_rate}Hz")
                recommendations.append("Higher sample rate (44.1kHz+) recommended for better quality")
        
        # Channel analysis
        if channels:
            if channels == 2:
                quality_score += 0.05  # Stereo bonus
            elif channels > 2:
                warnings.append(f"Multi-channel audio ({channels} channels) will be downmixed")
                recommendations.append("Stereo (2-channel) audio is optimal for video backgrounds")
        
        # Duration analysis
        if duration:
            if duration < 30:
                warnings.append(f"Short music duration: {duration:.1f}s")
                recommendations.append("Longer music files (60s+) work better for video backgrounds")
            elif duration > 600:  # 10 minutes
                warnings.append(f"Very long music file: {duration/60:.1f} minutes")
                recommendations.append("Consider trimming music to match expected video length")
        
        # File size analysis
        file_size = file_path.stat().st_size
        if file_size > 50 * 1024 * 1024:  # 50MB
            warnings.append(f"Large file size: {file_size/(1024*1024):.1f}MB")
            recommendations.append("Consider compressing to reduce file size")
        elif file_size < 1024 * 1024:  # 1MB
            warnings.append("Small file size may indicate low quality")
        
        # Clamp quality score
        quality_score = min(1.0, max(0.0, quality_score))
        
        return quality_score, warnings, recommendations
    
    def get_format_help(self) -> Dict[str, Any]:
        """Get comprehensive format help and recommendations."""
        return {
            "supported_formats": [
                {
                    "extension": fmt.extension,
                    "quality": fmt.quality,
                    "description": fmt.description,
                    "recommended": fmt.quality in ["Excellent", "Good"]
                }
                for fmt in MusicFormat
            ],
            "recommendations": {
                "best_quality": ["wav", "flac"],
                "best_compatibility": ["mp3", "m4a"],
                "best_balance": ["mp3"],
                "optimal_settings": {
                    "sample_rate": "44.1kHz or higher",
                    "channels": "Stereo (2 channels)",
                    "bitrate": "128kbps+ for MP3, 256kbps+ for best quality",
                    "duration": "60+ seconds for video backgrounds"
                }
            },
            "conversion_tips": [
                "Use tools like FFmpeg, Audacity, or online converters",
                "For MP3: Use 192kbps+ bitrate for good quality",
                "For video backgrounds: Instrumental music works best",
                "Avoid music with sudden volume changes",
                "Consider royalty-free music for sharing videos"
            ]
        }
    
    def validate_volume_setting(self, volume: float) -> Tuple[bool, Optional[str], List[str]]:
        """
        Validate volume setting and provide recommendations.
        
        Args:
            volume: Volume level (0.0 to 1.0)
            
        Returns:
            Tuple of (is_valid, error_message, recommendations)
        """
        recommendations = []
        
        if not 0.0 <= volume <= 1.0:
            return False, f"Volume must be between 0.0 and 1.0, got {volume}", []
        
        # Volume recommendations
        if volume == 0.0:
            recommendations.append("Volume is muted - no background music will be heard")
        elif volume < 0.1:
            recommendations.append("Very quiet - music may be barely audible")
        elif 0.1 <= volume <= 0.3:
            recommendations.append("Optimal range for background music")
        elif 0.3 < volume <= 0.5: 
            recommendations.append("Moderate volume - may compete with narration")
        elif 0.5 < volume <= 0.7:
            recommendations.append("High volume - may overpower narration")
        else:
            recommendations.append("Very high volume - will likely overpower narration")
        
        return True, None, recommendations