"""
Text-to-Speech generator using Coqui TTS for high-quality audio.

Uses Coqui TTS for offline, neural network-based speech synthesis.
"""

import os
import tempfile
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from lib.audio.quality_validator import AudioQualityValidator
from lib.audio.post_processor import AudioPostProcessor

try:
    from TTS.api import TTS
    HAS_TTS = True
except ImportError:
    HAS_TTS = False


@dataclass
class CoquiTTSConfig:
    """Configuration for Coqui TTS generation."""
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"  # Default English model
    output_format: str = "wav"
    sample_rate: int = 22050
    quality: str = "medium"  # low, medium, high
    device: str = "cpu"  # cpu or cuda


class CoquiSpeechGenerator:
    """
    Converts text content to speech using Coqui TTS.
    
    Provides high-quality neural TTS with various voice models.
    """
    
    def __init__(self, config: Optional[CoquiTTSConfig] = None):
        """
        Initialize the Coqui TTS generator.
        
        Args:
            config: TTS configuration, uses defaults if None
        """
        self.config = config or CoquiTTSConfig()
        self.tts = None
        
        # Initialize audio processing components
        self.quality_validator = AudioQualityValidator()
        self.post_processor = AudioPostProcessor(sample_rate=self.config.sample_rate)
        
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize the Coqui TTS engine."""
        if not HAS_TTS:
            print("❌ Coqui TTS not available. Install with: pip install TTS")
            return
        
        try:
            print(f"🤖 Initializing Coqui TTS...")
            print(f"   Model: {self.config.model_name}")
            print(f"   Device: {self.config.device}")
            
            # Initialize TTS with the specified model
            self.tts = TTS(
                model_name=self.config.model_name,
                progress_bar=False,
                gpu=self.config.device == "cuda"
            )
            
            print(f"✅ Coqui TTS initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Coqui TTS: {e}")
            print("💡 Try a different model or check internet connection for model download")
            self.tts = None
    
    def generate_speech_from_monologue(
        self, 
        monologue: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate speech from a monologue dictionary using Coqui TTS.
        
        Args:
            monologue: Monologue dictionary from discussion generator
            output_path: Optional output file path, generates temp file if None
            
        Returns:
            Dictionary with speech generation results
        """
        if not self.tts:
            return self._fallback_response("Coqui TTS engine not available")
        
        # Extract text from monologue script
        full_text = self._extract_text_from_script(monologue.get('script', []))
        
        if not full_text.strip():
            return self._fallback_response("No text content to convert")
        
        # Generate output path if not provided
        if not output_path:
            # Create audio output directory
            project_dir = Path(__file__).parent.parent.parent
            audio_dir = project_dir / "audio_output"
            audio_dir.mkdir(exist_ok=True)
            
            # Safe filename from topic
            topic = monologue.get('topic', 'unknown').replace(' ', '_').replace('/', '_')
            safe_topic = ''.join(c for c in topic if c.isalnum() or c in '_-')[:50]
            output_path = audio_dir / f"{safe_topic}_coqui.{self.config.output_format}"
        
        # Generate speech
        try:
            print(f"🎤 Generating high-quality speech: {output_path}")
            
            # Use Coqui TTS to generate audio
            self.tts.tts_to_file(
                text=full_text,
                file_path=str(output_path)
            )
            
            # Check if file was created and perform quality validation
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"🎵 Initial audio generated: {file_size:,} bytes")
                
                # Perform quality analysis
                print("🔍 Analyzing audio quality...")
                quality_report = self.quality_validator.analyze_audio(str(output_path))
                
                # Apply post-processing if needed
                final_output_path = str(output_path)
                if quality_report.needs_processing:
                    print(f"🔧 Applying enhancements: {', '.join(quality_report.recommended_fixes)}")
                    enhanced_path = self.post_processor.enhance_audio(str(output_path), quality_report.recommended_fixes)
                    
                    if os.path.exists(enhanced_path) and enhanced_path != str(output_path):
                        # Replace original with enhanced version
                        os.replace(enhanced_path, str(output_path))
                        print("✅ Audio enhancement applied")
                    else:
                        print("ℹ️  Using original audio (enhancement not needed)")
                else:
                    print("✅ Audio quality acceptable, no enhancement needed")
                
                # Get accurate duration measurement
                actual_duration = self._get_actual_audio_duration(str(output_path))
                final_file_size = os.path.getsize(output_path)
                
                print(f"📊 Final audio metrics:")
                print(f"   Duration: {actual_duration:.2f}s")
                print(f"   Quality score: {quality_report.quality_score:.2f}/1.0")
                print(f"   Speech content: {quality_report.speech_percentage:.1f}%")
                print(f"   Dynamic range: {quality_report.dynamic_range_db:.1f}dB")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "file_size": final_file_size,
                    "estimated_duration": actual_duration,
                    "text_word_count": len(full_text.split()),
                    "quality_metrics": {
                        "quality_score": quality_report.quality_score,
                        "silence_percentage": quality_report.silence_percentage,
                        "speech_percentage": quality_report.speech_percentage,
                        "dynamic_range_db": quality_report.dynamic_range_db,
                        "clipping_detected": quality_report.clipping_detected,
                        "enhancements_applied": quality_report.recommended_fixes if quality_report.needs_processing else []
                    },
                    "tts_config": {
                        "model": self.config.model_name,
                        "sample_rate": self.config.sample_rate,
                        "format": self.config.output_format,
                        "device": self.config.device,
                        "quality_validation": True
                    },
                    "engine": "coqui-tts"
                }
            else:
                return self._fallback_response("Audio file was not created")
                
        except Exception as e:
            return self._fallback_response(f"Coqui TTS generation failed: {e}")
    
    def _get_actual_audio_duration(self, audio_path: str) -> float:
        """
        Get the actual duration of the generated audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        try:
            # Try using librosa first (more accurate)
            try:
                import librosa
                y, sr = librosa.load(audio_path)
                duration = len(y) / sr
                print(f"🎵 Actual audio duration: {duration:.2f}s (measured with librosa)")
                return duration
            except ImportError:
                # Fallback to basic audio analysis
                pass
            
            # Try using wave module for WAV files
            if audio_path.lower().endswith('.wav'):
                import wave
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    print(f"🎵 Actual audio duration: {duration:.2f}s (measured with wave)")
                    return duration
            
            # Try using MoviePy as final fallback
            try:
                from moviepy import AudioFileClip
                with AudioFileClip(audio_path) as audio_clip:
                    duration = audio_clip.duration
                    print(f"🎵 Actual audio duration: {duration:.2f}s (measured with MoviePy)")
                    return duration
            except ImportError:
                pass
                
        except Exception as e:
            print(f"⚠️ Could not measure audio duration: {e}, using word-based estimate")
            
        # Fallback to word-based estimate if all else fails
        return len(self._extract_text_from_script([{'content': 'temp'}]).split()) / (180 / 60)
    
    def _extract_text_from_script(self, script: List[Any]) -> str:
        """Extract clean text from script turns (handles both dict and DiscussionTurn objects)."""
        text_parts = []
        
        for turn in script:
            # Handle both dict and DiscussionTurn object
            if hasattr(turn, 'content'):
                content = turn.content.strip()
            elif isinstance(turn, dict):
                content = turn.get('content', '').strip()
            else:
                content = str(turn).strip()
                
            if content:
                # Clean up the text for speech
                content = content.replace('&', 'and')
                content = content.replace('<', 'less than')
                content = content.replace('>', 'greater than')
                # Remove any markup that might interfere
                content = content.replace('*', '')
                content = content.replace('_', '')
                # Remove multiple spaces and normalize whitespace
                content = ' '.join(content.split())
                # Remove problematic characters that can cause TTS issues
                content = content.replace('|', '')
                content = content.replace('{', '')
                content = content.replace('}', '')
                content = content.replace('[', '')
                content = content.replace(']', '')
                
                # Add natural pauses at sentence boundaries
                if not content.endswith(('.', '!', '?')):
                    content += '.'
                text_parts.append(content)
        
        # Join with natural spacing for speech
        return ' '.join(text_parts)
    
    def _fallback_response(self, error_message: str) -> Dict[str, Any]:
        """Fallback response when speech generation fails."""
        print(f"❌ {error_message}")
        return {
            "success": False,
            "error": error_message,
            "output_path": None,
            "estimated_duration": 0,
            "text_word_count": 0,
            "engine": "coqui-tts"
        }
    
    def list_available_models(self) -> List[str]:
        """List available Coqui TTS models."""
        if not HAS_TTS:
            return []
        
        try:
            return TTS.list_models()
        except Exception as e:
            print(f"❌ Could not list models: {e}")
            return []
    
    def change_model(self, model_name: str) -> bool:
        """
        Change the TTS model.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            True if model was changed successfully
        """
        try:
            print(f"🔄 Changing to model: {model_name}")
            self.config.model_name = model_name
            self.tts = TTS(
                model_name=model_name,
                progress_bar=False,
                gpu=self.config.device == "cuda"
            )
            print(f"✅ Model changed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to change model: {e}")
            return False


def get_recommended_models() -> List[str]:
    """Get a list of recommended models for different use cases."""
    return [
        "tts_models/en/ljspeech/tacotron2-DDC",  # Fast, good quality
        "tts_models/en/ljspeech/fast_pitch",     # Very fast
        "tts_models/en/vctk/vits",               # Multiple speakers
        "tts_models/en/jenny/jenny",             # High quality female voice
    ]