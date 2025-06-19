"""
Audio processing and optimization modules for MoneyTree.

This package provides comprehensive audio processing capabilities including:
- Duration estimation for pipeline optimization
- Quality validation and enhancement
- Audio post-processing and artifact removal
- Smooth audio segment concatenation
- Consistent voice management for TTS
"""

from .duration_estimator import AudioDurationEstimator, AudioEstimate
from .quality_validator import AudioQualityValidator, AudioQualityReport
from .post_processor import AudioPostProcessor
from .segment_processor import AudioSegmentProcessor, ConsistentVoiceManager

__all__ = [
    'AudioDurationEstimator',
    'AudioEstimate', 
    'AudioQualityValidator',
    'AudioQualityReport',
    'AudioPostProcessor',
    'AudioSegmentProcessor',
    'ConsistentVoiceManager'
]