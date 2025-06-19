"""
Audio post-processing for quality enhancement and artifact removal.

Provides tools to fix common audio issues like silence gaps, noise,
clipping, and volume inconsistencies.
"""

import os
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import librosa
    import scipy.signal
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    logger.warning("librosa/scipy not available, post-processing will be limited")


class AudioPostProcessor:
    """
    Post-processes audio to fix quality issues and enhance output.
    
    Provides various enhancement techniques including normalization,
    silence removal, noise reduction, and artifact correction.
    """
    
    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the audio post-processor.
        
        Args:
            sample_rate: Default sample rate for processing
        """
        self.sample_rate = sample_rate
        
        # Processing parameters
        self.silence_threshold = 0.01   # Threshold for silence detection
        self.crossfade_duration = 0.1   # 100ms crossfade
        self.noise_reduction_factor = 0.8  # Gentle noise reduction
    
    def enhance_audio(self, audio_path: str, fixes: List[str]) -> str:
        """
        Apply recommended fixes to enhance audio quality.
        
        Args:
            audio_path: Path to input audio file
            fixes: List of recommended fixes to apply
            
        Returns:
            Path to enhanced audio file
        """
        if not HAS_AUDIO_LIBS:
            logger.warning("Audio enhancement libraries not available, returning original")
            return audio_path
        
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=None)
            logger.info(f"Loaded audio for enhancement: {len(audio_data)} samples at {sr}Hz")
            
            # Apply fixes in order
            enhanced_audio = audio_data.copy()
            
            for fix in fixes:
                if fix == 'trim_silence':
                    enhanced_audio = self.remove_silence_gaps(enhanced_audio, sr)
                elif fix == 'noise_reduction':
                    enhanced_audio = self.apply_noise_reduction(enhanced_audio)
                elif fix == 'normalize_levels':
                    enhanced_audio = self.normalize_audio_levels(enhanced_audio)
                elif fix == 'remove_dc_offset':
                    enhanced_audio = self.remove_dc_offset(enhanced_audio)
                elif fix == 'smooth_volume':
                    enhanced_audio = self.smooth_volume_changes(enhanced_audio, sr)
                else:
                    logger.warning(f"Unknown fix: {fix}")
            
            # Save enhanced audio
            output_path = self._generate_enhanced_path(audio_path)
            self._save_audio(enhanced_audio, output_path, sr)
            
            logger.info(f"Enhanced audio saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            return audio_path  # Return original on error
    
    def normalize_audio_levels(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio levels to prevent clipping and ensure consistent volume.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Normalized audio array
        """
        if len(audio_data) == 0:
            return audio_data
        
        # Find peak amplitude
        peak = np.max(np.abs(audio_data))
        
        if peak > 0:
            # Normalize to 85% of maximum to leave headroom
            target_peak = 0.85
            normalization_factor = target_peak / peak
            normalized = audio_data * normalization_factor
            
            logger.debug(f"Normalized audio: peak {peak:.3f} -> {target_peak:.3f}")
            return normalized
        else:
            return audio_data
    
    def remove_silence_gaps(self, audio_data: np.ndarray, sample_rate: int, 
                          max_silence_duration: float = 1.0) -> np.ndarray:
        """
        Remove excessive silence gaps while preserving natural pauses.
        
        Args:
            audio_data: Input audio array
            sample_rate: Audio sample rate
            max_silence_duration: Maximum allowed silence duration in seconds
            
        Returns:
            Audio with trimmed silence gaps
        """
        if not HAS_AUDIO_LIBS or len(audio_data) == 0:
            return audio_data
        
        try:
            # Detect silence using RMS energy
            frame_length = 2048
            hop_length = 512
            
            rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, 
                                    hop_length=hop_length)[0]
            
            # Determine silence threshold
            mean_energy = np.mean(rms)
            silence_threshold = mean_energy * 0.1
            
            # Find non-silent frames
            non_silent = rms > silence_threshold
            
            # Convert frame indices to sample indices
            non_silent_samples = []
            for i, is_speech in enumerate(non_silent):
                start_sample = i * hop_length
                end_sample = min(start_sample + hop_length, len(audio_data))
                
                if is_speech:
                    non_silent_samples.extend(range(start_sample, end_sample))
                else:
                    # Keep some silence (up to max duration)
                    silence_samples = min(end_sample - start_sample, 
                                        int(max_silence_duration * sample_rate))
                    non_silent_samples.extend(range(start_sample, 
                                                  start_sample + silence_samples))
            
            # Extract non-silent audio
            if non_silent_samples:
                trimmed_audio = audio_data[non_silent_samples]
                logger.debug(f"Trimmed silence: {len(audio_data)} -> {len(trimmed_audio)} samples")
                return trimmed_audio
            else:
                return audio_data
                
        except Exception as e:
            logger.warning(f"Failed to trim silence: {e}")
            return audio_data
    
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply gentle noise reduction to improve audio quality.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Noise-reduced audio array
        """
        if not HAS_AUDIO_LIBS or len(audio_data) == 0:
            return audio_data
        
        try:
            # Simple spectral subtraction approach
            # Get noise profile from quiet sections
            rms = np.sqrt(np.convolve(audio_data**2, np.ones(1024)/1024, mode='same'))
            noise_threshold = np.percentile(rms, 20)  # Bottom 20% as noise estimate
            
            # Apply gentle high-pass filter to remove low-frequency noise
            if HAS_AUDIO_LIBS:
                # Design high-pass filter
                cutoff = 80  # 80 Hz cutoff
                nyquist = self.sample_rate / 2
                normal_cutoff = cutoff / nyquist
                
                b, a = scipy.signal.butter(2, normal_cutoff, btype='high')
                filtered_audio = scipy.signal.filtfilt(b, a, audio_data)
                
                # Gentle noise gate
                noise_gate = np.where(rms > noise_threshold, 1.0, self.noise_reduction_factor)
                enhanced_audio = filtered_audio * noise_gate
                
                logger.debug("Applied noise reduction filter")
                return enhanced_audio
            else:
                return audio_data
                
        except Exception as e:
            logger.warning(f"Failed to apply noise reduction: {e}")
            return audio_data
    
    def add_natural_pauses(self, audio_data: np.ndarray, 
                          sentence_boundaries: List[int]) -> np.ndarray:
        """
        Add natural pauses at sentence boundaries.
        
        Args:
            audio_data: Input audio array
            sentence_boundaries: Sample indices of sentence boundaries
            
        Returns:
            Audio with added natural pauses
        """
        if len(sentence_boundaries) == 0:
            return audio_data
        
        pause_duration = int(0.3 * self.sample_rate)  # 300ms pause
        pause_samples = np.zeros(pause_duration)
        
        # Insert pauses at boundaries
        segments = []
        last_boundary = 0
        
        for boundary in sorted(sentence_boundaries):
            if boundary > last_boundary and boundary < len(audio_data):
                # Add segment
                segments.append(audio_data[last_boundary:boundary])
                # Add pause
                segments.append(pause_samples)
                last_boundary = boundary
        
        # Add final segment
        if last_boundary < len(audio_data):
            segments.append(audio_data[last_boundary:])
        
        if segments:
            result = np.concatenate(segments)
            logger.debug(f"Added {len(sentence_boundaries)} natural pauses")
            return result
        else:
            return audio_data
    
    def apply_crossfade(self, segments: List[np.ndarray], fade_duration: float = 0.1) -> np.ndarray:
        """
        Apply crossfading between audio segments for smooth transitions.
        
        Args:
            segments: List of audio segments to concatenate
            fade_duration: Crossfade duration in seconds
            
        Returns:
            Smoothly concatenated audio
        """
        if len(segments) <= 1:
            return segments[0] if segments else np.array([])
        
        fade_samples = int(fade_duration * self.sample_rate)
        result_segments = []
        
        for i, segment in enumerate(segments):
            if i == 0:
                # First segment: fade in at start
                if len(segment) > fade_samples:
                    fade_in = np.linspace(0, 1, fade_samples)
                    segment[:fade_samples] *= fade_in
                result_segments.append(segment)
            elif i == len(segments) - 1:
                # Last segment: crossfade with previous and fade out
                if len(segment) > fade_samples and len(result_segments) > 0:
                    # Crossfade with previous segment
                    prev_segment = result_segments[-1]
                    if len(prev_segment) >= fade_samples:
                        fade_out = np.linspace(1, 0, fade_samples)
                        fade_in = np.linspace(0, 1, fade_samples)
                        
                        # Apply fades and overlap
                        overlap_start = len(prev_segment) - fade_samples
                        prev_segment[overlap_start:] *= fade_out
                        segment[:fade_samples] *= fade_in
                        
                        # Mix overlapping parts
                        mixed_overlap = prev_segment[overlap_start:] + segment[:fade_samples]
                        
                        # Combine: previous without overlap + mixed overlap + rest of current
                        combined = np.concatenate([
                            prev_segment[:overlap_start],
                            mixed_overlap,
                            segment[fade_samples:]
                        ])
                        
                        # Replace previous segment with combined
                        result_segments[-1] = combined
                    else:
                        result_segments.append(segment)
                else:
                    result_segments.append(segment)
            else:
                # Middle segments: crossfade with previous
                if len(segment) > fade_samples and len(result_segments) > 0:
                    prev_segment = result_segments[-1]
                    if len(prev_segment) >= fade_samples:
                        fade_out = np.linspace(1, 0, fade_samples)
                        fade_in = np.linspace(0, 1, fade_samples)
                        
                        # Apply crossfade
                        overlap_start = len(prev_segment) - fade_samples
                        prev_segment[overlap_start:] *= fade_out
                        segment[:fade_samples] *= fade_in
                        
                        # Mix overlapping parts
                        mixed_overlap = prev_segment[overlap_start:] + segment[:fade_samples]
                        
                        # Combine
                        combined = np.concatenate([
                            prev_segment[:overlap_start],
                            mixed_overlap,
                            segment[fade_samples:]
                        ])
                        
                        result_segments[-1] = combined
                    else:
                        result_segments.append(segment)
                else:
                    result_segments.append(segment)
        
        # Concatenate all segments
        if result_segments:
            final_audio = np.concatenate(result_segments)
            logger.debug(f"Applied crossfade to {len(segments)} segments")
            return final_audio
        else:
            return np.array([])
    
    def remove_dc_offset(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from audio.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Audio with DC offset removed
        """
        if len(audio_data) == 0:
            return audio_data
        
        dc_offset = np.mean(audio_data)
        corrected = audio_data - dc_offset
        
        if abs(dc_offset) > 0.001:
            logger.debug(f"Removed DC offset: {dc_offset:.4f}")
        
        return corrected
    
    def smooth_volume_changes(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Smooth sudden volume changes in audio.
        
        Args:
            audio_data: Input audio array
            sample_rate: Audio sample rate
            
        Returns:
            Audio with smoothed volume changes
        """
        if not HAS_AUDIO_LIBS or len(audio_data) == 0:
            return audio_data
        
        try:
            # Calculate rolling RMS
            window_size = sample_rate // 20  # 50ms windows
            rms_values = []
            
            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i:i + window_size]
                rms_values.append(np.sqrt(np.mean(window ** 2)))
            
            if len(rms_values) < 2:
                return audio_data
            
            rms_values = np.array(rms_values)
            
            # Smooth RMS values
            smoothed_rms = scipy.signal.savgol_filter(rms_values, 
                                                     min(len(rms_values), 9), 3)
            
            # Apply smoothed volume envelope
            result = audio_data.copy()
            for i, (original_rms, smooth_rms) in enumerate(zip(rms_values, smoothed_rms)):
                start_idx = i * (window_size // 2)
                end_idx = min(start_idx + window_size, len(result))
                
                if original_rms > 0:
                    volume_adjustment = smooth_rms / original_rms
                    # Limit adjustment to prevent artifacts
                    volume_adjustment = np.clip(volume_adjustment, 0.5, 2.0)
                    result[start_idx:end_idx] *= volume_adjustment
            
            logger.debug("Applied volume smoothing")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to smooth volume changes: {e}")
            return audio_data
    
    def _generate_enhanced_path(self, original_path: str) -> str:
        """Generate output path for enhanced audio."""
        path = Path(original_path)
        enhanced_name = f"{path.stem}_enhanced{path.suffix}"
        return str(path.parent / enhanced_name)
    
    def _save_audio(self, audio_data: np.ndarray, output_path: str, sample_rate: int):
        """Save audio data to file."""
        try:
            import soundfile as sf
            sf.write(output_path, audio_data, sample_rate)
        except ImportError:
            # Fallback to scipy
            try:
                import scipy.io.wavfile as wavfile
                # Convert to 16-bit int for WAV
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wavfile.write(output_path, sample_rate, audio_int16)
            except:
                # Last resort: use torchaudio if available
                try:
                    import torch
                    import torchaudio
                    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
                    torchaudio.save(output_path, audio_tensor, sample_rate)
                except:
                    raise RuntimeError("No suitable audio saving library available")