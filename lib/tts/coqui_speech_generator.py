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
    
    # PyTorch 2.6+ compatibility: Add safe globals for XTTS-v2
    import torch
    if hasattr(torch.serialization, 'add_safe_globals'):
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            torch.serialization.add_safe_globals([XttsConfig])
        except ImportError:
            pass  # XttsConfig not available in this TTS version
except ImportError:
    HAS_TTS = False


@dataclass
class CoquiTTSConfig:
    """Configuration for Coqui TTS generation."""
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"  # Default English model
    output_format: str = "wav"
    sample_rate: int = 22050
    quality: str = "medium"  # low, medium, high
    device: str = None  # auto-detect, or specify "cpu" or "cuda"
    
    # XTTS-v2 specific parameters
    use_xtts: bool = False  # Enable XTTS-v2 voice cloning
    speaker_wav: Optional[str] = None  # Reference audio for voice cloning (6+ seconds)
    language: str = "en"  # Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko)
    gpt_cond_len: int = 3  # GPT conditioning length for voice cloning
    temperature: float = 0.75  # Temperature for generation randomness
    repetition_penalty: float = 5.0  # Penalty for repetitions
    length_penalty: float = 1.0  # Penalty for length
    speed: float = 1.0  # Speech speed multiplier
    enable_streaming: bool = False  # Enable streaming generation
    split_sentences: bool = True  # Split long text into sentences
    
    @classmethod
    def for_xtts_v2(cls, speaker_wav: str, language: str = "en", **kwargs) -> 'CoquiTTSConfig':
        """
        Create configuration optimized for XTTS-v2 voice cloning.
        
        Args:
            speaker_wav: Path to reference audio file (6+ seconds recommended)
            language: Language code for generation
            **kwargs: Additional configuration parameters
            
        Returns:
            CoquiTTSConfig configured for XTTS-v2
        """
        # Extract gpu parameter to avoid duplicate keyword argument
        gpu = kwargs.pop("gpu", False)
        return cls(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            sample_rate=24000,  # XTTS-v2 uses 24kHz for higher quality
            use_xtts=True,
            speaker_wav=speaker_wav,
            language=language,
            device="cuda" if gpu else "cpu",
            **kwargs
        )
    
    @classmethod
    def for_popular_model(cls, model_preset: str = "best_quality", **kwargs) -> 'CoquiTTSConfig':
        """
        Create configuration for popular pre-trained models.
        
        Args:
            model_preset: Model preset ('best_quality', 'fast', 'male_voice', 'female_voice')
            **kwargs: Additional configuration parameters
            
        Returns:
            CoquiTTSConfig for the specified preset
        """
        presets = {
            "best_quality": "tts_models/en/ljspeech/fast_pitch",
            "fast": "tts_models/en/ljspeech/tacotron2-DDC", 
            "male_voice": "tts_models/en/vctk/vits",  # Use p230 speaker
            "female_voice": "tts_models/en/jenny/jenny",
            "multilingual": "tts_models/multilingual/multi-dataset/your_tts"
        }
        
        model_name = presets.get(model_preset, presets["best_quality"])
        
        return cls(
            model_name=model_name,
            sample_rate=22050,
            **kwargs
        )


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
    
    def _detect_best_device(self) -> str:
        """
        Auto-detect the best available device for TTS.
        
        Returns:
            "cuda" if GPU available, "cpu" otherwise
        """
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print(f"🚀 GPU detected: {gpu_name}")
                print(f"   CUDA devices available: {gpu_count}")
                return "cuda"
        except ImportError:
            print("📦 PyTorch not available, cannot detect CUDA")
        except Exception as e:
            print(f"⚠️ GPU detection failed: {e}")
        
        print("🖥️  Using CPU for TTS")
        return "cpu"
    
    def _initialize_tts(self):
        """Initialize the Coqui TTS engine with XTTS-v2 support."""
        if not HAS_TTS:
            print("❌ Coqui TTS not available. Install with: pip install TTS")
            return
        
        try:
            # Auto-detect best device if not specified
            if self.config.device is None:
                self.config.device = self._detect_best_device()
            
            print(f"🤖 Initializing Coqui TTS...")
            print(f"   Model: {self.config.model_name}")
            print(f"   Device: {self.config.device}")
            
            if self.config.use_xtts:
                print(f"   XTTS-v2 Mode: Voice cloning enabled")
                print(f"   Language: {self.config.language}")
                if self.config.speaker_wav:
                    print(f"   Reference audio: {Path(self.config.speaker_wav).name}")
            
            # Initialize TTS with the specified model
            self.tts = TTS(
                model_name=self.config.model_name,
                progress_bar=False,
                gpu=self.config.device == "cuda"
            )
            
            # Validate speaker reference for XTTS-v2
            if self.config.use_xtts and self.config.speaker_wav:
                self._validate_speaker_reference()
            
            print(f"✅ Coqui TTS initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Coqui TTS: {e}")
            print("💡 Try a different model or check internet connection for model download")
            self.tts = None
    
    def _validate_speaker_reference(self):
        """Validate and optimize speaker reference audio for voice cloning."""
        if not self.config.speaker_wav or not os.path.exists(self.config.speaker_wav):
            print(f"⚠️ Warning: Speaker reference audio not found: {self.config.speaker_wav}")
            return False
        
        try:
            import librosa
            
            # Load and analyze the reference audio
            audio, sr = librosa.load(self.config.speaker_wav, sr=None)
            duration = len(audio) / sr
            
            print(f"📊 Speaker reference analysis:")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Sample rate: {sr}Hz")
            
            # Validate duration (XTTS-v2 works best with 6+ seconds)
            if duration < 3.0:
                print(f"⚠️ Warning: Reference audio is short ({duration:.1f}s). Recommend 6+ seconds for best quality.")
            elif duration >= 6.0:
                print(f"✅ Reference audio duration is optimal ({duration:.1f}s)")
            else:
                print(f"⚡ Reference audio duration is acceptable ({duration:.1f}s)")
                
            return True
            
        except Exception as e:
            print(f"⚠️ Could not analyze speaker reference: {e}")
            return False
    
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
            
            if self.config.use_xtts:
                # Use XTTS-v2 with voice cloning
                self._generate_xtts_speech(full_text, str(output_path))
            else:
                # Use traditional TTS models
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
    
    def _generate_xtts_speech(self, text: str, output_path: str):
        """
        Generate speech using XTTS-v2 with voice cloning.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the generated audio
        """
        print(f"🎭 Using XTTS-v2 voice cloning...")
        print(f"   Language: {self.config.language}")
        print(f"   Temperature: {self.config.temperature}")
        
        try:
            # XTTS-v2 generation parameters
            generation_kwargs = {
                "text": text,
                "file_path": output_path,
                "language": self.config.language,
                "split_sentences": self.config.split_sentences
            }
            
            # Add speaker reference for voice cloning
            if self.config.speaker_wav and os.path.exists(self.config.speaker_wav):
                generation_kwargs["speaker_wav"] = self.config.speaker_wav
                print(f"   🎯 Cloning voice from: {Path(self.config.speaker_wav).name}")
            else:
                print(f"   🎤 Using default XTTS-v2 voice (no reference provided)")
            
            # Generate speech with XTTS-v2
            self.tts.tts_to_file(**generation_kwargs)
            
            print(f"✅ XTTS-v2 generation complete")
            
        except Exception as e:
            print(f"❌ XTTS-v2 generation failed: {e}")
            print(f"💡 Falling back to default voice...")
            
            # Fallback to basic XTTS without speaker reference
            try:
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=self.config.language
                )
                print(f"✅ Fallback generation successful")
            except Exception as fallback_error:
                raise Exception(f"Both XTTS-v2 and fallback failed: {fallback_error}")
    
    def clone_voice_from_audio(self, reference_audio: str, text: str, output_path: str, 
                              language: str = "en") -> Dict[str, Any]:
        """
        Clone a voice from reference audio and generate speech.
        
        Args:
            reference_audio: Path to reference audio file (6+ seconds recommended)
            text: Text to convert to speech
            output_path: Path to save the generated audio
            language: Language code for generation
            
        Returns:
            Dictionary with voice cloning results
        """
        if not self.config.use_xtts:
            return {"success": False, "error": "XTTS-v2 not enabled in configuration"}
        
        if not os.path.exists(reference_audio):
            return {"success": False, "error": f"Reference audio not found: {reference_audio}"}
        
        try:
            print(f"🎭 Voice cloning with XTTS-v2...")
            print(f"   Reference: {Path(reference_audio).name}")
            print(f"   Target language: {language}")
            
            # Generate speech with voice cloning
            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                language=language,
                file_path=output_path,
                split_sentences=self.config.split_sentences
            )
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                duration = self._get_actual_audio_duration(output_path)
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "file_size": file_size,
                    "duration": duration,
                    "language": language,
                    "reference_audio": reference_audio,
                    "voice_cloning": True,
                    "engine": "xtts-v2"
                }
            else:
                return {"success": False, "error": "Voice cloning failed to generate audio file"}
                
        except Exception as e:
            return {"success": False, "error": f"Voice cloning failed: {e}"}
    
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


def get_recommended_models() -> List[Dict[str, Any]]:
    """Get a list of recommended models for different use cases."""
    return [
        {
            "name": "XTTS-v2 (Voice Cloning)",
            "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
            "description": "Advanced voice cloning with 6s audio, 17 languages",
            "features": ["voice_cloning", "multilingual", "high_quality"],
            "sample_rate": 24000,
            "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"],
            "use_case": "Custom voices, multilingual content",
            "quality": "excellent",
            "speed": "medium"
        },
        {
            "name": "FastPitch (Best Quality)",
            "model_name": "tts_models/en/ljspeech/fast_pitch",
            "description": "High quality English TTS with fast inference",
            "features": ["high_quality", "fast", "stable"],
            "sample_rate": 22050,
            "languages": ["en"],
            "use_case": "English content, production use",
            "quality": "high",
            "speed": "fast"
        },
        {
            "name": "VITS Multi-Speaker",
            "model_name": "tts_models/en/vctk/vits",
            "description": "Multiple English speakers including male voices",
            "features": ["multi_speaker", "natural"],
            "sample_rate": 22050,
            "languages": ["en"],
            "use_case": "Varied voices, character speech",
            "quality": "high", 
            "speed": "medium",
            "speakers": ["p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232"]
        },
        {
            "name": "Jenny (Female Voice)",
            "model_name": "tts_models/en/jenny/jenny",
            "description": "High quality female English voice",
            "features": ["female_voice", "clear"],
            "sample_rate": 22050,
            "languages": ["en"],
            "use_case": "Female narrator, clear speech",
            "quality": "high",
            "speed": "medium"
        },
        {
            "name": "Tacotron2 (Fast)",
            "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
            "description": "Fast and reliable English TTS",
            "features": ["fast", "reliable", "lightweight"],
            "sample_rate": 22050,
            "languages": ["en"],
            "use_case": "Quick generation, testing",
            "quality": "good",
            "speed": "very_fast"
        }
    ]


def get_xtts_supported_languages() -> List[Dict[str, str]]:
    """Get list of languages supported by XTTS-v2."""
    return [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"}, 
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "pl", "name": "Polish"},
        {"code": "tr", "name": "Turkish"},
        {"code": "ru", "name": "Russian"},
        {"code": "nl", "name": "Dutch"},
        {"code": "cs", "name": "Czech"},
        {"code": "ar", "name": "Arabic"},
        {"code": "zh-cn", "name": "Chinese (Simplified)"},
        {"code": "ja", "name": "Japanese"},
        {"code": "hu", "name": "Hungarian"},
        {"code": "ko", "name": "Korean"}
    ]


def create_voice_cloning_config(reference_audio: str, language: str = "en", 
                               gpu: bool = False, **kwargs) -> CoquiTTSConfig:
    """
    Create a configuration optimized for voice cloning with XTTS-v2.
    
    Args:
        reference_audio: Path to reference audio file
        language: Target language for generation
        gpu: Whether to use GPU acceleration
        **kwargs: Additional configuration parameters
        
    Returns:
        CoquiTTSConfig configured for voice cloning
    """
    return CoquiTTSConfig.for_xtts_v2(
        speaker_wav=reference_audio,
        language=language,
        gpu=gpu,
        **kwargs
    )