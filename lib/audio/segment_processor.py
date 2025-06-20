"""
Audio segment processing for smooth concatenation and transitions.

Provides tools for handling audio segments from TTS generation,
including smooth concatenation, consistent voice management, and
natural pause insertion.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import librosa
    import scipy.signal
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    logger.warning("librosa/scipy not available, segment processing will be limited")


class AudioSegmentProcessor:
    """
    Handles smooth concatenation and transitions between audio segments.
    
    Designed specifically for TTS-generated audio segments that need to be
    joined seamlessly while maintaining natural speech flow.
    """
    
    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the audio segment processor.
        
        Args:
            sample_rate: Audio sample rate for processing
        """
        self.sample_rate = sample_rate
        
        # Processing parameters
        self.crossfade_duration = 0.1    # 100ms crossfade
        self.breath_pause_duration = 0.2  # 200ms breathing pause
        self.sentence_pause_duration = 0.4  # 400ms sentence pause
        self.min_segment_energy_ratio = 0.1  # Minimum energy for valid segment
    
    def concatenate_with_crossfade(self, segments: List[np.ndarray], 
                                 fade_ms: int = 100) -> np.ndarray:
        """
        Concatenate audio segments with smooth crossfading transitions.
        
        Args:
            segments: List of audio segments (numpy arrays)
            fade_ms: Crossfade duration in milliseconds
            
        Returns:
            Smoothly concatenated audio array
        """
        if not segments:
            return np.array([])
        
        if len(segments) == 1:
            return segments[0]
        
        fade_samples = int((fade_ms / 1000.0) * self.sample_rate)
        logger.info(f"Concatenating {len(segments)} segments with {fade_ms}ms crossfade")
        
        # Normalize segments first
        normalized_segments = self.normalize_segment_levels(segments)
        
        # Apply crossfading
        result = normalized_segments[0]
        
        for i in range(1, len(normalized_segments)):
            current_segment = normalized_segments[i]
            
            # Determine optimal crossfade based on segment characteristics
            optimal_fade = self._determine_optimal_crossfade(
                result, current_segment, fade_samples
            )
            
            # Apply crossfade
            result = self._apply_segment_crossfade(result, current_segment, optimal_fade)
        
        logger.debug(f"Final concatenated audio: {len(result)} samples, {len(result)/self.sample_rate:.2f}s")
        return result
    
    def detect_natural_boundaries(self, audio_data: np.ndarray) -> List[int]:
        """
        Detect natural speech boundaries for intelligent segment splitting.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            List of sample indices where natural boundaries occur
        """
        if not HAS_AUDIO_LIBS or len(audio_data) == 0:
            return []
        
        boundaries = []
        
        try:
            # Calculate energy envelope
            hop_length = 512
            frame_length = 2048
            
            rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, 
                                    hop_length=hop_length)[0]
            
            # Find low-energy regions (potential boundaries)
            mean_energy = np.mean(rms)
            boundary_threshold = mean_energy * 0.2  # 20% of mean energy
            
            low_energy_frames = rms < boundary_threshold
            
            # Find transitions from speech to silence and back
            energy_diff = np.diff(low_energy_frames.astype(int))
            
            # Speech-to-silence transitions (potential sentence ends)
            silence_starts = np.where(energy_diff == 1)[0]
            
            # Convert frame indices to sample indices
            for frame_idx in silence_starts:
                sample_idx = frame_idx * hop_length
                if sample_idx < len(audio_data):
                    boundaries.append(sample_idx)
            
            logger.debug(f"Detected {len(boundaries)} natural boundaries")
            return boundaries
            
        except Exception as e:
            logger.warning(f"Failed to detect natural boundaries: {e}")
            return []
    
    def add_breathing_pauses(self, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add natural breathing pauses between segments.
        
        Args:
            segments: List of audio segments
            
        Returns:
            List of segments with breathing pauses inserted
        """
        if len(segments) <= 1:
            return segments
        
        pause_samples = int(self.breath_pause_duration * self.sample_rate)
        breathing_pause = self._generate_breathing_pause(pause_samples)
        
        result_segments = []
        
        for i, segment in enumerate(segments):
            result_segments.append(segment)
            
            # Add breathing pause between segments (not after the last one)
            if i < len(segments) - 1:
                # Determine if pause is needed based on segment characteristics
                if self._needs_breathing_pause(segment, segments[i + 1]):
                    result_segments.append(breathing_pause)
        
        logger.debug(f"Added breathing pauses between {len(segments)} segments")
        return result_segments
    
    def normalize_segment_levels(self, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize volume levels across segments for consistency.
        
        Args:
            segments: List of audio segments
            
        Returns:
            List of volume-normalized segments
        """
        if not segments:
            return segments
        
        # Calculate RMS energy for each segment
        segment_energies = []
        for segment in segments:
            if len(segment) > 0:
                rms = np.sqrt(np.mean(segment ** 2))
                segment_energies.append(rms)
            else:
                segment_energies.append(0.0)
        
        # Find target energy (median to avoid outliers)
        valid_energies = [e for e in segment_energies if e > 0]
        if not valid_energies:
            return segments
        
        target_energy = np.median(valid_energies)
        
        # Normalize each segment
        normalized_segments = []
        for segment, energy in zip(segments, segment_energies):
            if energy > 0 and len(segment) > 0:
                normalization_factor = target_energy / energy
                # Limit normalization to prevent extreme adjustments
                normalization_factor = np.clip(normalization_factor, 0.5, 2.0)
                normalized_segment = segment * normalization_factor
                normalized_segments.append(normalized_segment)
            else:
                normalized_segments.append(segment)
        
        logger.debug(f"Normalized {len(segments)} segments to target energy {target_energy:.4f}")
        return normalized_segments
    
    def _determine_optimal_crossfade(self, segment1: np.ndarray, segment2: np.ndarray, 
                                   default_fade: int) -> int:
        """
        Determine optimal crossfade duration based on segment characteristics.
        
        Args:
            segment1: First audio segment
            segment2: Second audio segment
            default_fade: Default crossfade duration in samples
            
        Returns:
            Optimal crossfade duration in samples
        """
        # Analyze ending of first segment and beginning of second segment
        analysis_length = min(default_fade * 2, len(segment1), len(segment2))
        
        if analysis_length <= 0:
            return default_fade
        
        # Get tail of first segment and head of second segment
        tail = segment1[-analysis_length:] if len(segment1) >= analysis_length else segment1
        head = segment2[:analysis_length] if len(segment2) >= analysis_length else segment2
        
        # Calculate energy levels
        tail_energy = np.sqrt(np.mean(tail ** 2)) if len(tail) > 0 else 0
        head_energy = np.sqrt(np.mean(head ** 2)) if len(head) > 0 else 0
        
        # Adjust crossfade based on energy levels
        if tail_energy < 0.01 or head_energy < 0.01:
            # One segment ends/starts quietly - shorter crossfade
            return default_fade // 2
        elif abs(tail_energy - head_energy) > 0.1:
            # Energy mismatch - longer crossfade for smoother transition
            return int(default_fade * 1.5)
        else:
            # Similar energy levels - standard crossfade
            return default_fade
    
    def _apply_segment_crossfade(self, segment1: np.ndarray, segment2: np.ndarray, 
                               fade_samples: int) -> np.ndarray:
        """
        Apply crossfade between two audio segments.
        
        Args:
            segment1: First audio segment
            segment2: Second audio segment
            fade_samples: Crossfade duration in samples
            
        Returns:
            Combined audio with crossfade applied
        """
        if fade_samples <= 0 or len(segment1) == 0:
            return np.concatenate([segment1, segment2])
        
        if len(segment2) == 0:
            return segment1
        
        # Ensure fade doesn't exceed segment lengths
        actual_fade = min(fade_samples, len(segment1), len(segment2))
        
        if actual_fade <= 0:
            return np.concatenate([segment1, segment2])
        
        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, actual_fade)
        fade_in = np.linspace(0.0, 1.0, actual_fade)
        
        # Apply fades
        overlap_start = len(segment1) - actual_fade
        
        # Extract overlapping regions
        tail = segment1[overlap_start:].copy()
        head = segment2[:actual_fade].copy()
        
        # Apply fade curves
        tail *= fade_out
        head *= fade_in
        
        # Mix overlapping parts
        mixed_overlap = tail + head
        
        # Combine: first segment (without overlap) + mixed overlap + second segment (without overlap)
        result = np.concatenate([
            segment1[:overlap_start],
            mixed_overlap,
            segment2[actual_fade:]
        ])
        
        return result
    
    def _generate_breathing_pause(self, pause_samples: int) -> np.ndarray:
        """
        Generate a natural breathing pause with subtle ambient characteristics.
        
        Args:
            pause_samples: Duration of pause in samples
            
        Returns:
            Audio array representing a breathing pause
        """
        # Create mostly silent pause with very subtle background
        pause = np.zeros(pause_samples)
        
        # Add very subtle, quiet breathing-like sound
        if pause_samples > 0:
            # Generate very quiet pink noise for natural feel
            noise = np.random.normal(0, 0.001, pause_samples)
            
            # Apply gentle envelope
            envelope = np.exp(-np.linspace(0, 2, pause_samples))
            pause = noise * envelope
            
            # Apply gentle low-pass filter for more natural sound
            if HAS_AUDIO_LIBS and pause_samples > 100:
                try:
                    # Simple smoothing
                    pause = scipy.signal.savgol_filter(pause, min(51, pause_samples//10), 3)
                except:
                    pass  # Skip filtering if it fails
        
        return pause
    
    def _needs_breathing_pause(self, segment1: np.ndarray, segment2: np.ndarray) -> bool:
        """
        Determine if a breathing pause is needed between two segments.
        
        Args:
            segment1: First audio segment
            segment2: Second audio segment
            
        Returns:
            True if breathing pause is recommended
        """
        # Analyze segment characteristics
        if len(segment1) == 0 or len(segment2) == 0:
            return False
        
        # Check energy levels at segment boundaries
        boundary_length = min(int(0.1 * self.sample_rate), len(segment1), len(segment2))
        
        tail_energy = np.sqrt(np.mean(segment1[-boundary_length:] ** 2))
        head_energy = np.sqrt(np.mean(segment2[:boundary_length] ** 2))
        
        # Add pause if both segments have significant energy (likely speech)
        # and the segments are reasonably long (likely complete phrases)
        min_duration = 1.0 * self.sample_rate  # 1 second minimum
        
        if (len(segment1) > min_duration and len(segment2) > min_duration and
            tail_energy > 0.01 and head_energy > 0.01):
            return True
        
        return False


class ConsistentVoiceManager:
    """
    Manages consistent voice generation across chunks for ChatTTS.
    
    Ensures voice characteristics remain consistent throughout the
    entire audio generation process.
    """
    
    def __init__(self):
        """Initialize the consistent voice manager."""
        self.current_speaker = None
        self.voice_cache = {}
        self.voice_consistency_threshold = 0.8
    
    def get_consistent_speaker(self, chat_tts_instance=None) -> Any:
        """
        Get a consistent speaker voice for the entire generation.
        
        Args:
            chat_tts_instance: ChatTTS instance for voice sampling
            
        Returns:
            Consistent speaker voice embedding
        """
        if self.current_speaker is None and chat_tts_instance is not None:
            self.current_speaker = self.sample_and_cache_speaker(chat_tts_instance)
        
        return self.current_speaker
    
    def sample_and_cache_speaker(self, chat_tts_instance, text_sample: str = None) -> Any:
        """
        Sample and cache a speaker voice for consistent use.
        
        Args:
            chat_tts_instance: ChatTTS instance
            text_sample: Optional text sample for voice optimization
            
        Returns:
            Cached speaker voice embedding
        """
        try:
            # Sample a random speaker
            speaker_voice = chat_tts_instance.sample_random_speaker()
            
            # Cache the voice
            voice_id = f"speaker_{len(self.voice_cache)}"
            self.voice_cache[voice_id] = speaker_voice
            self.current_speaker = speaker_voice
            
            logger.info(f"Sampled and cached consistent speaker voice: {voice_id}")
            return speaker_voice
            
        except Exception as e:
            logger.error(f"Failed to sample speaker voice: {e}")
            return None
    
    def validate_voice_consistency(self, audio_segments: List[np.ndarray]) -> float:
        """
        Validate voice consistency across audio segments.
        
        Args:
            audio_segments: List of generated audio segments
            
        Returns:
            Consistency score (0.0-1.0)
        """
        if len(audio_segments) < 2:
            return 1.0  # Single segment is always consistent
        
        # Simple consistency check based on spectral characteristics
        consistency_scores = []
        
        for i in range(len(audio_segments) - 1):
            score = self._compare_voice_characteristics(
                audio_segments[i], audio_segments[i + 1]
            )
            consistency_scores.append(score)
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        logger.debug(f"Voice consistency score: {overall_consistency:.3f}")
        return overall_consistency
    
    def _compare_voice_characteristics(self, segment1: np.ndarray, segment2: np.ndarray) -> float:
        """
        Compare voice characteristics between two audio segments.
        
        Args:
            segment1: First audio segment
            segment2: Second audio segment
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if not HAS_AUDIO_LIBS or len(segment1) == 0 or len(segment2) == 0:
            return 0.8  # Default reasonable consistency
        
        try:
            # Extract spectral features for comparison
            # Use MFCC features as a proxy for voice characteristics
            mfcc1 = librosa.feature.mfcc(y=segment1, sr=self.sample_rate, n_mfcc=13)
            mfcc2 = librosa.feature.mfcc(y=segment2, sr=self.sample_rate, n_mfcc=13)
            
            # Calculate mean MFCC vectors
            mean_mfcc1 = np.mean(mfcc1, axis=1)
            mean_mfcc2 = np.mean(mfcc2, axis=1)
            
            # Calculate cosine similarity
            dot_product = np.dot(mean_mfcc1, mean_mfcc2)
            norm1 = np.linalg.norm(mean_mfcc1)
            norm2 = np.linalg.norm(mean_mfcc2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                # Convert to 0-1 range (cosine similarity is -1 to 1)
                consistency_score = (similarity + 1) / 2
                return max(0.0, min(1.0, consistency_score))
            else:
                return 0.5  # Neutral score if calculation fails
                
        except Exception as e:
            logger.warning(f"Voice comparison failed: {e}")
            return 0.8  # Default good consistency on error