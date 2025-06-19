"""
Audio quality validation and analysis for detecting issues and artifacts.

Provides comprehensive quality analysis to detect silence, noise, clipping,
and other audio artifacts that can degrade the user experience.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa not available, audio analysis will be limited")


@dataclass
class AudioQualityReport:
    """
    Comprehensive audio quality analysis results.
    
    Attributes:
        silence_percentage: Percentage of audio that is silent
        noise_percentage: Percentage of audio with excessive noise
        speech_percentage: Percentage of audio with speech content
        dynamic_range_db: Dynamic range in decibels
        clipping_detected: Whether audio clipping was detected
        artifacts_detected: List of detected audio artifacts
        needs_processing: Whether post-processing is recommended
        recommended_fixes: List of recommended fixes
        quality_score: Overall quality score (0.0-1.0)
        sample_rate: Audio sample rate
        duration: Audio duration in seconds
        mean_energy: Mean RMS energy level
        energy_variance: Variance in energy levels
    """
    silence_percentage: float
    noise_percentage: float
    speech_percentage: float
    dynamic_range_db: float
    clipping_detected: bool
    artifacts_detected: List[str]
    needs_processing: bool
    recommended_fixes: List[str]
    quality_score: float
    sample_rate: int
    duration: float
    mean_energy: float
    energy_variance: float


class AudioQualityValidator:
    """
    Validates audio quality and detects various issues and artifacts.
    
    Provides comprehensive analysis of generated audio to identify problems
    that could affect user experience or indicate TTS generation issues.
    """
    
    def __init__(self):
        """Initialize the audio quality validator."""
        # Quality thresholds
        self.silence_threshold_ratio = 0.1  # 10% of mean energy for silence detection
        self.noise_threshold_std = 3.0      # 3 std dev above mean for noise detection
        self.min_speech_percentage = 30.0   # Minimum expected speech content
        self.max_silence_percentage = 50.0  # Maximum acceptable silence
        self.min_dynamic_range_db = 10.0    # Minimum acceptable dynamic range
        self.clipping_threshold = 0.95      # Threshold for clipping detection
        
        # Quality scoring weights
        self.quality_weights = {
            'speech_content': 0.3,    # 30% weight for speech content
            'silence_level': 0.25,    # 25% weight for appropriate silence
            'dynamic_range': 0.2,     # 20% weight for dynamic range
            'noise_level': 0.15,      # 15% weight for low noise
            'clipping': 0.1           # 10% weight for no clipping
        }
    
    def analyze_audio(self, audio_path: str) -> AudioQualityReport:
        """
        Perform comprehensive audio quality analysis.
        
        Args:
            audio_path: Path to the audio file to analyze
            
        Returns:
            AudioQualityReport with detailed quality metrics
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return self._create_error_report("Audio file not found")
        
        try:
            # Load audio data
            audio_data, sample_rate = self._load_audio(audio_path)
            if audio_data is None:
                return self._create_error_report("Failed to load audio data")
            
            duration = len(audio_data) / sample_rate
            
            # Perform individual analyses
            silence_analysis = self._analyze_silence(audio_data, sample_rate)
            noise_analysis = self._analyze_noise(audio_data)
            speech_analysis = self._analyze_speech_content(audio_data, sample_rate)
            clipping_analysis = self._analyze_clipping(audio_data)
            dynamic_range = self._calculate_dynamic_range(audio_data)
            
            # Detect artifacts
            artifacts = self._detect_artifacts(audio_data, sample_rate)
            
            # Calculate overall metrics
            mean_energy = float(np.mean(np.abs(audio_data)))
            energy_variance = float(np.var(np.abs(audio_data)))
            
            # Determine if processing is needed
            needs_processing, recommended_fixes = self._determine_processing_needs(
                silence_analysis, noise_analysis, speech_analysis, 
                clipping_analysis, dynamic_range, artifacts
            )
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                silence_analysis, noise_analysis, speech_analysis,
                clipping_analysis, dynamic_range
            )
            
            report = AudioQualityReport(
                silence_percentage=silence_analysis['percentage'],
                noise_percentage=noise_analysis['percentage'],
                speech_percentage=speech_analysis['percentage'],
                dynamic_range_db=dynamic_range,
                clipping_detected=clipping_analysis['detected'],
                artifacts_detected=artifacts,
                needs_processing=needs_processing,
                recommended_fixes=recommended_fixes,
                quality_score=quality_score,
                sample_rate=sample_rate,
                duration=duration,
                mean_energy=mean_energy,
                energy_variance=energy_variance
            )
            
            self._log_quality_summary(audio_path, report)
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return self._create_error_report(f"Analysis failed: {str(e)}")
    
    def detect_silence_issues(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Detect silence-related issues in audio."""
        return self._analyze_silence(audio_data, sample_rate)
    
    def detect_noise_artifacts(self, audio_data: np.ndarray) -> Dict:
        """Detect noise and artifacts in audio."""
        return self._analyze_noise(audio_data)
    
    def detect_clipping(self, audio_data: np.ndarray) -> bool:
        """Detect audio clipping."""
        return self._analyze_clipping(audio_data)['detected']
    
    def detect_speech_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze speech content quality."""
        return self._analyze_speech_content(audio_data, sample_rate)
    
    def generate_quality_score(self, metrics: Dict) -> float:
        """Generate overall quality score from metrics."""
        # This is a simplified version - full implementation in _calculate_quality_score
        return max(0.0, min(1.0, metrics.get('quality_score', 0.5)))
    
    def _load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load audio file using available libraries."""
        try:
            if HAS_LIBROSA:
                # Use librosa for high-quality loading
                y, sr = librosa.load(audio_path, sr=None)
                return y, sr
            else:
                # Fallback to wave module for WAV files
                import wave
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    
                    # Convert to numpy array
                    if sample_width == 1:
                        audio_data = np.frombuffer(frames, dtype=np.uint8)
                        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                    elif sample_width == 2:
                        audio_data = np.frombuffer(frames, dtype=np.int16)
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    else:
                        logger.error(f"Unsupported sample width: {sample_width}")
                        return None, 0
                    
                    # Handle stereo by taking first channel
                    if channels == 2:
                        audio_data = audio_data[::2]
                    
                    return audio_data, sample_rate
                    
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            return None, 0
    
    def _analyze_silence(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze silence content in audio."""
        # Set frame parameters
        frame_length = 2048
        hop_length = 512
        
        # Calculate RMS energy
        if HAS_LIBROSA:
            rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        else:
            # Simple RMS calculation
            rms = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                rms.append(np.sqrt(np.mean(frame ** 2)))
            rms = np.array(rms)
        
        if len(rms) == 0:
            return {'percentage': 100.0, 'segments': [], 'mean_energy': 0.0}
        
        # Determine silence threshold
        mean_energy = np.mean(rms)
        silence_threshold = mean_energy * self.silence_threshold_ratio
        
        # Find silent frames
        silent_frames = rms < silence_threshold
        silence_percentage = (np.sum(silent_frames) / len(silent_frames)) * 100.0
        
        # Find silence segments
        silence_segments = self._find_silence_segments(silent_frames, sample_rate, hop_length)
        
        return {
            'percentage': silence_percentage,
            'segments': silence_segments,
            'mean_energy': float(mean_energy),
            'threshold': float(silence_threshold)
        }
    
    def _analyze_noise(self, audio_data: np.ndarray) -> Dict:
        """Analyze noise levels in audio."""
        # Set frame parameters
        frame_length = 2048
        hop_length = 512
        
        # Calculate RMS energy
        if HAS_LIBROSA:
            rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        else:
            # Simple RMS calculation
            rms = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                rms.append(np.sqrt(np.mean(frame ** 2)))
            rms = np.array(rms)
        
        if len(rms) == 0:
            return {'percentage': 0.0, 'spikes': []}
        
        # Detect noise spikes (unusually high energy)
        mean_energy = np.mean(rms)
        std_energy = np.std(rms)
        noise_threshold = mean_energy + (self.noise_threshold_std * std_energy)
        
        noisy_frames = rms > noise_threshold
        noise_percentage = (np.sum(noisy_frames) / len(noisy_frames)) * 100.0
        
        # Find noise spikes
        noise_spikes = np.where(noisy_frames)[0]
        
        return {
            'percentage': noise_percentage,
            'spikes': noise_spikes.tolist(),
            'threshold': float(noise_threshold),
            'mean_energy': float(mean_energy),
            'std_energy': float(std_energy)
        }
    
    def _analyze_speech_content(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze speech content in audio."""
        # Set frame parameters
        frame_length = 2048
        hop_length = 512
        
        # Calculate RMS energy
        if HAS_LIBROSA:
            rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        else:
            # Simple RMS calculation
            rms = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                rms.append(np.sqrt(np.mean(frame ** 2)))
            rms = np.array(rms)
        
        if len(rms) == 0:
            return {'percentage': 0.0, 'segments': []}
        
        # Determine speech threshold (higher than silence, lower than noise)
        mean_energy = np.mean(rms)
        speech_threshold = mean_energy * 0.3  # 30% of mean energy
        
        # Find speech frames
        speech_frames = rms > speech_threshold
        speech_percentage = (np.sum(speech_frames) / len(speech_frames)) * 100.0
        
        return {
            'percentage': speech_percentage,
            'threshold': float(speech_threshold),
            'mean_energy': float(mean_energy)
        }
    
    def _analyze_clipping(self, audio_data: np.ndarray) -> Dict:
        """Detect audio clipping."""
        # Find samples near the maximum amplitude
        max_amplitude = np.max(np.abs(audio_data))
        clipped_samples = np.abs(audio_data) > (self.clipping_threshold * max_amplitude)
        
        clipping_detected = np.sum(clipped_samples) > 0
        clipping_percentage = (np.sum(clipped_samples) / len(audio_data)) * 100.0
        
        return {
            'detected': clipping_detected,
            'percentage': clipping_percentage,
            'max_amplitude': float(max_amplitude),
            'threshold': self.clipping_threshold
        }
    
    def _calculate_dynamic_range(self, audio_data: np.ndarray) -> float:
        """Calculate dynamic range in decibels."""
        if len(audio_data) == 0:
            return 0.0
        
        # Remove silence for accurate measurement
        non_silent = audio_data[np.abs(audio_data) > 0.001]  # Remove very quiet samples
        
        if len(non_silent) == 0:
            return 0.0
        
        max_amplitude = np.max(np.abs(non_silent))
        min_amplitude = np.min(np.abs(non_silent[non_silent > 0]))
        
        if min_amplitude > 0:
            dynamic_range_db = 20 * np.log10(max_amplitude / min_amplitude)
            return float(dynamic_range_db)
        else:
            return 0.0
    
    def _detect_artifacts(self, audio_data: np.ndarray, sample_rate: int) -> List[str]:
        """Detect various audio artifacts."""
        artifacts = []
        
        # Detect sudden volume changes
        if len(audio_data) > sample_rate:  # At least 1 second of audio
            # Calculate rolling RMS
            window_size = sample_rate // 10  # 100ms windows
            rms_values = []
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i:i + window_size]
                rms_values.append(np.sqrt(np.mean(window ** 2)))
            
            rms_values = np.array(rms_values)
            if len(rms_values) > 1:
                # Look for sudden changes
                rms_diff = np.abs(np.diff(rms_values))
                if np.max(rms_diff) > np.mean(rms_values) * 2:
                    artifacts.append('sudden_volume_changes')
        
        # Detect DC offset
        dc_offset = np.mean(audio_data)
        if abs(dc_offset) > 0.01:  # 1% DC offset
            artifacts.append('dc_offset')
        
        # Detect repetitive patterns (potential generation artifacts)
        if HAS_LIBROSA and len(audio_data) > sample_rate * 2:  # At least 2 seconds
            try:
                # Simple repetition detection using autocorrelation
                autocorr = np.correlate(audio_data[:sample_rate], audio_data[:sample_rate], mode='full')
                if np.max(autocorr[len(autocorr)//2 + sample_rate//10:]) > np.max(autocorr) * 0.7:
                    artifacts.append('repetitive_patterns')
            except:
                pass  # Skip if analysis fails
        
        return artifacts
    
    def _find_silence_segments(self, silent_frames: np.ndarray, sample_rate: int, hop_length: int) -> List[Dict]:
        """Find continuous silence segments."""
        segments = []
        in_silence = False
        start_frame = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_silence:
                # Start of silence segment
                start_frame = i
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence segment
                start_time = (start_frame * hop_length) / sample_rate
                end_time = (i * hop_length) / sample_rate
                duration = end_time - start_time
                
                if duration > 0.5:  # Only report segments longer than 0.5 seconds
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration
                    })
                in_silence = False
        
        # Handle case where silence continues to end
        if in_silence:
            start_time = (start_frame * hop_length) / sample_rate
            end_time = (len(silent_frames) * hop_length) / sample_rate
            duration = end_time - start_time
            
            if duration > 0.5:
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration
                })
        
        return segments
    
    def _determine_processing_needs(self, silence_analysis: Dict, noise_analysis: Dict, 
                                  speech_analysis: Dict, clipping_analysis: Dict,
                                  dynamic_range: float, artifacts: List[str]) -> Tuple[bool, List[str]]:
        """Determine if post-processing is needed and what fixes to apply."""
        needs_processing = False
        recommended_fixes = []
        
        # Check silence issues
        if silence_analysis['percentage'] > self.max_silence_percentage:
            needs_processing = True
            recommended_fixes.append('trim_silence')
        
        # Check noise issues
        if noise_analysis['percentage'] > 10.0:  # More than 10% noise
            needs_processing = True
            recommended_fixes.append('noise_reduction')
        
        # Check speech content
        if speech_analysis['percentage'] < self.min_speech_percentage:
            needs_processing = True
            recommended_fixes.append('enhance_speech')
        
        # Check clipping
        if clipping_analysis['detected']:
            needs_processing = True
            recommended_fixes.append('fix_clipping')
        
        # Check dynamic range
        if dynamic_range < self.min_dynamic_range_db:
            needs_processing = True
            recommended_fixes.append('normalize_levels')
        
        # Check artifacts
        if artifacts:
            needs_processing = True
            if 'dc_offset' in artifacts:
                recommended_fixes.append('remove_dc_offset')
            if 'sudden_volume_changes' in artifacts:
                recommended_fixes.append('smooth_volume')
        
        return needs_processing, recommended_fixes
    
    def _calculate_quality_score(self, silence_analysis: Dict, noise_analysis: Dict,
                                speech_analysis: Dict, clipping_analysis: Dict,
                                dynamic_range: float) -> float:
        """Calculate overall quality score (0.0-1.0)."""
        
        # Speech content score (higher is better)
        speech_score = min(1.0, speech_analysis['percentage'] / 70.0)  # Target 70%+ speech
        
        # Silence score (moderate silence is good)
        ideal_silence = 20.0  # 20% silence is ideal
        silence_diff = abs(silence_analysis['percentage'] - ideal_silence)
        silence_score = max(0.0, 1.0 - (silence_diff / 50.0))  # Penalize deviation
        
        # Dynamic range score
        dynamic_score = min(1.0, dynamic_range / 30.0)  # Target 30dB+ range
        
        # Noise score (lower noise is better)
        noise_score = max(0.0, 1.0 - (noise_analysis['percentage'] / 20.0))  # Penalize noise
        
        # Clipping score (binary: good or bad)
        clipping_score = 0.0 if clipping_analysis['detected'] else 1.0
        
        # Weighted average
        quality_score = (
            speech_score * self.quality_weights['speech_content'] +
            silence_score * self.quality_weights['silence_level'] +
            dynamic_score * self.quality_weights['dynamic_range'] +
            noise_score * self.quality_weights['noise_level'] +
            clipping_score * self.quality_weights['clipping']
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _log_quality_summary(self, audio_path: str, report: AudioQualityReport):
        """Log a summary of the quality analysis."""
        filename = Path(audio_path).name
        
        logger.info(f"Audio quality analysis for {filename}:")
        logger.info(f"  Quality score: {report.quality_score:.2f}/1.0")
        logger.info(f"  Duration: {report.duration:.1f}s")
        logger.info(f"  Speech content: {report.speech_percentage:.1f}%")
        logger.info(f"  Silence: {report.silence_percentage:.1f}%")
        logger.info(f"  Dynamic range: {report.dynamic_range_db:.1f}dB")
        
        if report.clipping_detected:
            logger.warning(f"  ⚠️  Clipping detected")
        
        if report.artifacts_detected:
            logger.warning(f"  ⚠️  Artifacts: {', '.join(report.artifacts_detected)}")
        
        if report.needs_processing:
            logger.info(f"  💡 Recommended fixes: {', '.join(report.recommended_fixes)}")
        else:
            logger.info(f"  ✅ Audio quality acceptable")
    
    def _create_error_report(self, error_message: str) -> AudioQualityReport:
        """Create an error report when analysis fails."""
        return AudioQualityReport(
            silence_percentage=0.0,
            noise_percentage=0.0,
            speech_percentage=0.0,
            dynamic_range_db=0.0,
            clipping_detected=False,
            artifacts_detected=[],
            needs_processing=False,
            recommended_fixes=[],
            quality_score=0.0,
            sample_rate=0,
            duration=0.0,
            mean_energy=0.0,
            energy_variance=0.0
        )