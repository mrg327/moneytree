"""
Voice cloning utilities and management for XTTS-v2.

Provides tools for managing reference voices, quality validation,
and optimizing voice cloning performance.
"""

import os
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import librosa
    import numpy as np
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    logger.warning("Audio libraries not available - install with: uv add librosa numpy")


@dataclass
class VoiceReference:
    """Information about a voice reference audio file."""
    name: str
    file_path: str
    duration: float
    sample_rate: int
    quality_score: float
    language: str = "en"
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class VoiceQualityReport:
    """Quality analysis report for voice reference audio."""
    overall_score: float  # 0.0 to 1.0
    duration_score: float
    clarity_score: float
    consistency_score: float
    noise_level: float
    recommendations: List[str]
    is_suitable: bool
    issues: List[str]


class VoiceManager:
    """
    Manages voice references and provides voice cloning utilities.
    
    Handles voice library management, quality validation, and optimization
    for XTTS-v2 voice cloning.
    """
    
    def __init__(self, voices_dir: str = "voice_library"):
        """
        Initialize the voice manager.
        
        Args:
            voices_dir: Directory to store voice references
        """
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(exist_ok=True)
        
        self.voice_references: Dict[str, VoiceReference] = {}
        self._load_voice_library()
    
    def _load_voice_library(self):
        """Load existing voice references from the library."""
        try:
            # Look for audio files in the voices directory
            for audio_file in self.voices_dir.glob("*.wav"):
                if audio_file.is_file():
                    voice_ref = self._create_voice_reference_from_file(audio_file)
                    if voice_ref:
                        self.voice_references[voice_ref.name] = voice_ref
            
            logger.info(f"Loaded {len(self.voice_references)} voice references")
            
        except Exception as e:
            logger.warning(f"Failed to load voice library: {e}")
    
    def _create_voice_reference_from_file(self, file_path: Path) -> Optional[VoiceReference]:
        """Create a VoiceReference from an audio file."""
        if not HAS_AUDIO_LIBS:
            return None
            
        try:
            # Load and analyze audio
            audio, sr = librosa.load(str(file_path), sr=None)
            duration = len(audio) / sr
            
            # Basic quality analysis
            quality_score = self._calculate_basic_quality(audio, sr)
            
            voice_ref = VoiceReference(
                name=file_path.stem,
                file_path=str(file_path),
                duration=duration,
                sample_rate=sr,
                quality_score=quality_score
            )
            
            return voice_ref
            
        except Exception as e:
            logger.warning(f"Failed to create voice reference from {file_path}: {e}")
            return None
    
    def _calculate_basic_quality(self, audio: np.ndarray, sr: int) -> float:
        """Calculate a basic quality score for audio."""
        try:
            # Simple quality metrics
            duration = len(audio) / sr
            
            # Duration score (6+ seconds is ideal)
            duration_score = min(1.0, duration / 6.0) if duration > 0 else 0.0
            
            # RMS energy (avoid too quiet or too loud)
            rms = np.sqrt(np.mean(audio**2))
            energy_score = 1.0 - abs(rms - 0.1) / 0.1  # Target around 0.1 RMS
            energy_score = max(0.0, min(1.0, energy_score))
            
            # Simple noise estimation (high frequency content)
            if len(audio) > sr:  # At least 1 second
                high_freq = librosa.stft(audio)
                noise_score = 1.0 - min(1.0, np.mean(np.abs(high_freq[-10:])) * 10)
            else:
                noise_score = 0.5
            
            # Combine scores
            overall_score = (duration_score * 0.4 + energy_score * 0.3 + noise_score * 0.3)
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return 0.5  # Default moderate score
    
    def add_voice_reference(self, source_path: str, name: str, 
                           language: str = "en", description: str = "", 
                           tags: List[str] = None) -> VoiceReference:
        """
        Add a new voice reference to the library.
        
        Args:
            source_path: Path to source audio file
            name: Name for the voice reference
            language: Language of the voice
            description: Description of the voice
            tags: Tags for categorization
            
        Returns:
            VoiceReference object for the added voice
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source audio file not found: {source_path}")
        
        # Create safe filename
        safe_name = "".join(c for c in name if c.isalnum() or c in "_-").lower()
        target_path = self.voices_dir / f"{safe_name}.wav"
        
        # Copy and potentially optimize the audio file
        optimized_path = self._optimize_voice_reference(source_path, str(target_path))
        
        # Create voice reference
        voice_ref = self._create_voice_reference_from_file(Path(optimized_path))
        if voice_ref:
            voice_ref.name = name
            voice_ref.language = language
            voice_ref.description = description
            voice_ref.tags = tags or []
            
            self.voice_references[name] = voice_ref
            logger.info(f"Added voice reference: {name}")
            
            return voice_ref
        else:
            raise ValueError(f"Failed to create voice reference from {source_path}")
    
    def _optimize_voice_reference(self, source_path: str, target_path: str) -> str:
        """Optimize audio file for voice cloning."""
        if not HAS_AUDIO_LIBS:
            # Just copy if no audio processing available
            shutil.copy2(source_path, target_path)
            return target_path
        
        try:
            # Load audio
            audio, sr = librosa.load(source_path, sr=22050)  # Standardize sample rate
            
            # Basic optimizations
            # 1. Normalize volume
            audio = librosa.util.normalize(audio)
            
            # 2. Trim silence from beginning and end
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # 3. Ensure minimum length (pad if too short)
            min_length = int(3.0 * sr)  # 3 seconds minimum
            if len(audio) < min_length:
                audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
            
            # 4. Apply light noise reduction if needed
            # (This is a simple approach - could be enhanced)
            
            # Save optimized audio
            import soundfile as sf
            sf.write(target_path, audio, sr)
            
            logger.info(f"Optimized voice reference: {Path(source_path).name} -> {Path(target_path).name}")
            return target_path
            
        except Exception as e:
            logger.warning(f"Voice optimization failed, copying original: {e}")
            shutil.copy2(source_path, target_path)
            return target_path
    
    def analyze_voice_quality(self, voice_path: str) -> VoiceQualityReport:
        """
        Analyze the quality of a voice reference for cloning.
        
        Args:
            voice_path: Path to voice audio file
            
        Returns:
            VoiceQualityReport with detailed analysis
        """
        if not HAS_AUDIO_LIBS:
            return VoiceQualityReport(
                overall_score=0.5,
                duration_score=0.5,
                clarity_score=0.5,
                consistency_score=0.5,
                noise_level=0.5,
                recommendations=["Install audio libraries for detailed analysis"],
                is_suitable=True,
                issues=["Limited analysis without librosa"]
            )
        
        if not os.path.exists(voice_path):
            return VoiceQualityReport(
                overall_score=0.0,
                duration_score=0.0,
                clarity_score=0.0,
                consistency_score=0.0,
                noise_level=1.0,
                recommendations=["File not found"],
                is_suitable=False,
                issues=["Audio file does not exist"]
            )
        
        try:
            # Load audio
            audio, sr = librosa.load(voice_path, sr=None)
            duration = len(audio) / sr
            
            # Analyze different quality aspects
            duration_score = self._analyze_duration(duration)
            clarity_score = self._analyze_clarity(audio, sr)
            consistency_score = self._analyze_consistency(audio, sr)
            noise_level = self._analyze_noise_level(audio, sr)
            
            # Calculate overall score
            overall_score = (
                duration_score * 0.3 +
                clarity_score * 0.3 +
                consistency_score * 0.2 +
                (1.0 - noise_level) * 0.2
            )
            
            # Generate recommendations and identify issues
            recommendations = []
            issues = []
            
            if duration < 3.0:
                issues.append("Audio too short (< 3 seconds)")
                recommendations.append("Use audio of at least 6 seconds for best results")
            elif duration > 30.0:
                recommendations.append("Consider trimming to 10-20 seconds for faster processing")
            
            if clarity_score < 0.6:
                issues.append("Low audio clarity")
                recommendations.append("Use cleaner audio with less background noise")
            
            if noise_level > 0.3:
                issues.append("High noise level detected")
                recommendations.append("Reduce background noise and improve recording quality")
            
            if consistency_score < 0.5:
                issues.append("Inconsistent audio levels")
                recommendations.append("Use audio with consistent volume and speaking pace")
            
            is_suitable = overall_score >= 0.5 and len(issues) == 0
            
            return VoiceQualityReport(
                overall_score=overall_score,
                duration_score=duration_score,
                clarity_score=clarity_score,
                consistency_score=consistency_score,
                noise_level=noise_level,
                recommendations=recommendations,
                is_suitable=is_suitable,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Voice quality analysis failed: {e}")
            return VoiceQualityReport(
                overall_score=0.0,
                duration_score=0.0,
                clarity_score=0.0,
                consistency_score=0.0,
                noise_level=1.0,
                recommendations=["Analysis failed - check audio file format"],
                is_suitable=False,
                issues=[f"Analysis error: {e}"]
            )
    
    def _analyze_duration(self, duration: float) -> float:
        """Analyze duration suitability (0.0 to 1.0)."""
        if duration < 3.0:
            return duration / 3.0  # Penalty for being too short
        elif duration >= 6.0:
            return 1.0  # Optimal length
        else:
            return 0.5 + (duration - 3.0) / 6.0  # Linear improvement from 3-6s
    
    def _analyze_clarity(self, audio: np.ndarray, sr: int) -> float:
        """Analyze audio clarity (0.0 to 1.0)."""
        try:
            # Spectral centroid as a proxy for clarity
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroids)
            
            # Normalize to expected speech range (typically 1000-4000 Hz)
            clarity_score = min(1.0, avg_centroid / 3000.0)
            return max(0.0, clarity_score)
            
        except Exception:
            return 0.5  # Default if analysis fails
    
    def _analyze_consistency(self, audio: np.ndarray, sr: int) -> float:
        """Analyze volume and energy consistency (0.0 to 1.0)."""
        try:
            # RMS energy over time
            frame_length = int(0.1 * sr)  # 100ms frames
            hop_length = frame_length // 2
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Calculate coefficient of variation (std/mean)
            if np.mean(rms) > 0:
                cv = np.std(rms) / np.mean(rms)
                consistency_score = max(0.0, 1.0 - cv)  # Lower variation = higher score
            else:
                consistency_score = 0.0
            
            return min(1.0, consistency_score)
            
        except Exception:
            return 0.5  # Default if analysis fails
    
    def _analyze_noise_level(self, audio: np.ndarray, sr: int) -> float:
        """Analyze background noise level (0.0 = no noise, 1.0 = very noisy)."""
        try:
            # Simple noise estimation using spectral features
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Look at high frequency content as noise indicator
            high_freq_energy = np.mean(magnitude[-magnitude.shape[0]//4:, :])
            total_energy = np.mean(magnitude)
            
            if total_energy > 0:
                noise_ratio = high_freq_energy / total_energy
                return min(1.0, noise_ratio * 5.0)  # Scale and cap at 1.0
            else:
                return 1.0  # Silent audio is problematic
                
        except Exception:
            return 0.3  # Default moderate noise level if analysis fails
    
    def get_voice_references(self, language: str = None, 
                           min_quality: float = 0.0) -> List[VoiceReference]:
        """
        Get list of voice references matching criteria.
        
        Args:
            language: Filter by language (None for all)
            min_quality: Minimum quality score required
            
        Returns:
            List of matching VoiceReference objects
        """
        references = []
        
        for voice_ref in self.voice_references.values():
            # Apply filters
            if language and voice_ref.language != language:
                continue
            if voice_ref.quality_score < min_quality:
                continue
                
            references.append(voice_ref)
        
        # Sort by quality score (highest first)
        references.sort(key=lambda x: x.quality_score, reverse=True)
        return references
    
    def get_best_voice_for_language(self, language: str = "en") -> Optional[VoiceReference]:
        """Get the highest quality voice reference for a language."""
        voices = self.get_voice_references(language=language)
        return voices[0] if voices else None
    
    def remove_voice_reference(self, name: str) -> bool:
        """Remove a voice reference from the library."""
        if name in self.voice_references:
            voice_ref = self.voice_references[name]
            
            # Remove file if it exists
            if os.path.exists(voice_ref.file_path):
                try:
                    os.remove(voice_ref.file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove voice file: {e}")
            
            # Remove from memory
            del self.voice_references[name]
            logger.info(f"Removed voice reference: {name}")
            return True
        
        return False
    
    def list_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of all voice references."""
        summary = {}
        
        for name, voice_ref in self.voice_references.items():
            summary[name] = {
                "language": voice_ref.language,
                "duration": voice_ref.duration,
                "quality_score": voice_ref.quality_score,
                "description": voice_ref.description,
                "tags": voice_ref.tags,
                "file_path": voice_ref.file_path
            }
        
        return summary


def create_voice_sample_from_text(text: str, voice_ref: VoiceReference, 
                                 output_path: str) -> Dict[str, Any]:
    """
    Create a voice sample using XTTS-v2 voice cloning.
    
    Args:
        text: Text to convert to speech
        voice_ref: Voice reference to clone
        output_path: Path to save generated audio
        
    Returns:
        Dictionary with generation results
    """
    try:
        from lib.tts.coqui_speech_generator import CoquiSpeechGenerator, CoquiTTSConfig
        
        # Create XTTS-v2 configuration
        config = CoquiTTSConfig.for_xtts_v2(
            speaker_wav=voice_ref.file_path,
            language=voice_ref.language
        )
        
        # Generate speech
        generator = CoquiSpeechGenerator(config)
        result = generator.clone_voice_from_audio(
            reference_audio=voice_ref.file_path,
            text=text,
            output_path=output_path,
            language=voice_ref.language
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Voice sample generation failed: {e}"
        }