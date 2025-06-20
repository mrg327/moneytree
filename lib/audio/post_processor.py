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
                elif fix == 'fix_clipping':
                    enhanced_audio = self.fix_audio_clipping(enhanced_audio)
                elif fix == 'fix_tts_syllable_artifacts':
                    enhanced_audio = self.fix_tts_syllable_artifacts(enhanced_audio)
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
        Apply gentle noise reduction with click removal to improve audio quality.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Noise-reduced audio array with clicks removed
        """
        if not HAS_AUDIO_LIBS or len(audio_data) == 0:
            return audio_data
        
        try:
            # Step 1: Remove clicks and pops first
            click_free_audio = self._remove_clicks_with_gentle_gate(audio_data)
            
            # Step 2: Apply traditional noise reduction
            # Get noise profile from quiet sections
            rms = np.sqrt(np.convolve(click_free_audio**2, np.ones(1024)/1024, mode='same'))
            noise_threshold = np.percentile(rms, 20)  # Bottom 20% as noise estimate
            
            # Apply gentle high-pass filter to remove low-frequency noise
            if HAS_AUDIO_LIBS:
                # Design high-pass filter
                cutoff = 80  # 80 Hz cutoff
                nyquist = self.sample_rate / 2
                normal_cutoff = cutoff / nyquist
                
                b, a = scipy.signal.butter(2, normal_cutoff, btype='high')
                filtered_audio = scipy.signal.filtfilt(b, a, click_free_audio)
                
                # Gentle noise gate
                noise_gate = np.where(rms > noise_threshold, 1.0, self.noise_reduction_factor)
                enhanced_audio = filtered_audio * noise_gate
                
                logger.debug("Applied noise reduction with click removal")
                return enhanced_audio
            else:
                return click_free_audio
                
        except Exception as e:
            logger.warning(f"Failed to apply noise reduction: {e}")
            return audio_data
    
    def _remove_clicks_with_gentle_gate(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Remove clicks and pops using a gentle gating approach.
        
        This method detects sudden amplitude spikes and applies gentle smoothing
        to remove clicking sounds without affecting speech quality.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Audio with clicks removed
        """
        if len(audio_data) == 0:
            return audio_data
        
        try:
            # Calculate differences to detect sudden changes (clicks)
            diff = np.diff(audio_data)
            
            # Create a gentle rolling window for analysis
            window_size = 32  # Small window for click detection
            
            # Calculate rolling standard deviation of differences
            diff_padded = np.pad(diff, (window_size//2, window_size//2), mode='edge')
            diff_std = np.array([
                np.std(diff_padded[i:i+window_size]) 
                for i in range(len(diff))
            ])
            
            # Adaptive threshold based on overall audio characteristics
            median_std = np.median(diff_std)
            click_threshold = median_std * 4.0  # 4x median as threshold
            
            # Find potential clicks
            click_mask = np.abs(diff) > click_threshold
            click_indices = np.where(click_mask)[0]
            
            if len(click_indices) == 0:
                logger.debug("No clicks detected")
                return audio_data
            
            logger.debug(f"Found {len(click_indices)} potential clicks to smooth")
            
            # Apply gentle smoothing to click regions
            result = audio_data.copy()
            
            for click_idx in click_indices:
                # Define a small smoothing window around the click
                smooth_radius = 8  # 8 samples on each side
                start_idx = max(0, click_idx - smooth_radius)
                end_idx = min(len(result), click_idx + smooth_radius + 1)
                
                if end_idx - start_idx > 3:  # Need at least 3 samples
                    # Get the region to smooth
                    region = result[start_idx:end_idx].copy()
                    
                    # Apply gentle smoothing using a simple moving average
                    smoothed_region = self._apply_gentle_smoothing(region)
                    
                    # Blend smoothed region back with gentle fade
                    blend_factor = 0.7  # 70% smoothed, 30% original
                    result[start_idx:end_idx] = (
                        region * (1 - blend_factor) + 
                        smoothed_region * blend_factor
                    )
            
            logger.debug(f"Applied gentle click removal to {len(click_indices)} regions")
            return result
            
        except Exception as e:
            logger.warning(f"Click removal failed: {e}")
            return audio_data
    
    def _apply_gentle_smoothing(self, audio_region: np.ndarray) -> np.ndarray:
        """
        Apply gentle smoothing to a small audio region.
        
        Args:
            audio_region: Small audio segment to smooth
            
        Returns:
            Gently smoothed audio segment
        """
        if len(audio_region) < 3:
            return audio_region
        
        try:
            # Simple moving average with edge preservation
            smoothed = audio_region.copy()
            
            # Apply a 3-point moving average to the middle samples
            for i in range(1, len(audio_region) - 1):
                smoothed[i] = (audio_region[i-1] + audio_region[i] + audio_region[i+1]) / 3.0
            
            # Preserve the first and last samples to avoid boundary artifacts
            # This maintains the overall energy and prevents new clicks at boundaries
            
            return smoothed
            
        except Exception as e:
            logger.warning(f"Gentle smoothing failed: {e}")
            return audio_region
    
    def fix_audio_clipping(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Fix audio clipping by applying soft limiting and compression.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Audio with clipping artifacts removed
        """
        if len(audio_data) == 0:
            return audio_data
        
        try:
            result = audio_data.copy()
            
            # Step 1: Detect clipped regions
            clipping_threshold = 0.95  # 95% of max amplitude
            clipped_samples = np.abs(result) >= clipping_threshold
            
            if np.any(clipped_samples):
                clipped_count = np.sum(clipped_samples)
                logger.debug(f"Found {clipped_count} clipped samples ({clipped_count/len(result)*100:.1f}%)")
                
                # Step 2: Apply gentle compression to prevent further clipping
                # Use soft knee compression
                compression_ratio = 0.3  # Gentle compression
                threshold = 0.7  # Start compressing at 70%
                
                # Calculate compression
                abs_audio = np.abs(result)
                over_threshold = abs_audio > threshold
                
                if np.any(over_threshold):
                    # Apply soft compression to loud parts
                    compressed_magnitude = threshold + (abs_audio - threshold) * compression_ratio
                    compressed_magnitude = np.where(over_threshold, compressed_magnitude, abs_audio)
                    
                    # Maintain original sign
                    result = np.sign(result) * compressed_magnitude
                
                # Step 3: Apply soft limiting to clipped regions
                for i in range(len(clipped_samples)):
                    if clipped_samples[i]:
                        # Apply soft limiting with surrounding context
                        window_start = max(0, i - 4)
                        window_end = min(len(result), i + 5)
                        
                        # Calculate target value based on surrounding samples
                        surrounding = result[window_start:window_end]
                        non_clipped = surrounding[np.abs(surrounding) < clipping_threshold]
                        
                        if len(non_clipped) > 0:
                            # Use median of non-clipped surrounding samples
                            target = np.median(non_clipped) * np.sign(result[i])
                            # Blend with a soft transition
                            result[i] = result[i] * 0.3 + target * 0.7
                        else:
                            # Fallback: reduce amplitude
                            result[i] *= 0.8
                
                # Step 4: Final normalization to prevent new clipping
                max_amplitude = np.max(np.abs(result))
                if max_amplitude > 0.9:
                    result *= 0.85 / max_amplitude
                
                logger.debug("Applied clipping repair with soft limiting")
            else:
                logger.debug("No clipping detected")
            
            return result
            
        except Exception as e:
            logger.warning(f"Clipping fix failed: {e}")
            return audio_data
    
    def fix_tts_syllable_artifacts(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Fix TTS-specific syllable artifacts like weird pronunciations and uncomfortable sounds.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Audio with TTS syllable artifacts smoothed
        """
        if not HAS_AUDIO_LIBS or len(audio_data) == 0:
            return audio_data
        
        try:
            # Step 1: Detect unnatural frequency spikes (robotic artifacts)
            result = self._smooth_frequency_spikes(audio_data)
            
            # Step 2: Fix volume inconsistencies within syllables
            result = self._smooth_syllable_volumes(result)
            
            # Step 3: Remove harsh consonant artifacts
            result = self._soften_harsh_consonants(result)
            
            logger.debug("Applied TTS syllable artifact fixes")
            return result
            
        except Exception as e:
            logger.warning(f"TTS syllable artifact fix failed: {e}")
            return audio_data
    
    def _smooth_frequency_spikes(self, audio_data: np.ndarray) -> np.ndarray:
        """Smooth unnatural frequency spikes that create robotic sounds."""
        try:
            if not HAS_AUDIO_LIBS:
                return audio_data
            
            # Apply gentle low-pass filtering to reduce harsh high frequencies
            nyquist = self.sample_rate / 2
            cutoff = 6000  # 6kHz cutoff to remove harsh artifacts
            
            if cutoff < nyquist:
                normal_cutoff = cutoff / nyquist
                b, a = scipy.signal.butter(2, normal_cutoff, btype='low')
                smoothed = scipy.signal.filtfilt(b, a, audio_data)
                
                # Blend with original to preserve speech clarity
                blend_factor = 0.3  # 30% smoothed, 70% original
                result = audio_data * (1 - blend_factor) + smoothed * blend_factor
                
                logger.debug("Applied frequency spike smoothing")
                return result
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"Frequency spike smoothing failed: {e}")
            return audio_data
    
    def _smooth_syllable_volumes(self, audio_data: np.ndarray) -> np.ndarray:
        """Smooth volume inconsistencies within syllables."""
        try:
            # Use shorter windows to target syllable-level variations
            window_size = self.sample_rate // 50  # 20ms windows for syllable analysis
            
            if window_size >= len(audio_data):
                return audio_data
            
            # Calculate envelope
            envelope = []
            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i:i + window_size]
                envelope.append(np.sqrt(np.mean(window ** 2)))
            
            if len(envelope) < 3:
                return audio_data
            
            # Smooth the envelope to remove sudden changes
            if HAS_AUDIO_LIBS:
                smoothed_envelope = scipy.signal.savgol_filter(envelope, min(len(envelope), 5), 2)
            else:
                # Simple moving average fallback
                smoothed_envelope = np.convolve(envelope, np.ones(3)/3, mode='same')
            
            # Apply smoothed envelope
            result = audio_data.copy()
            for i, (orig_rms, smooth_rms) in enumerate(zip(envelope, smoothed_envelope)):
                start_idx = i * (window_size // 2)
                end_idx = min(start_idx + window_size, len(result))
                
                if orig_rms > 0:
                    adjustment = smooth_rms / orig_rms
                    # Limit adjustment to prevent artifacts
                    adjustment = np.clip(adjustment, 0.7, 1.4)
                    result[start_idx:end_idx] *= adjustment
            
            logger.debug("Applied syllable volume smoothing")
            return result
            
        except Exception as e:
            logger.warning(f"Syllable volume smoothing failed: {e}")
            return audio_data
    
    def _soften_harsh_consonants(self, audio_data: np.ndarray) -> np.ndarray:
        """Soften harsh consonant sounds that can be uncomfortable."""
        try:
            # Detect high-energy, short-duration events (harsh consonants)
            window_size = 64  # Very small window for consonant detection
            
            energy = []
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i:i + window_size]
                energy.append(np.sum(window ** 2))
            
            if len(energy) < 2:
                return audio_data
            
            # Find high-energy spikes
            median_energy = np.median(energy)
            harsh_threshold = median_energy * 8  # 8x median energy
            
            result = audio_data.copy()
            
            for i, window_energy in enumerate(energy):
                if window_energy > harsh_threshold:
                    # Soften this region
                    start_idx = i * window_size
                    end_idx = min(start_idx + window_size, len(result))
                    
                    # Apply gentle compression to harsh region
                    region = result[start_idx:end_idx]
                    softened = region * 0.8  # Reduce by 20%
                    
                    # Smooth transition
                    result[start_idx:end_idx] = softened
            
            logger.debug("Applied harsh consonant softening")
            return result
            
        except Exception as e:
            logger.warning(f"Harsh consonant softening failed: {e}")
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