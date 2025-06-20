"""
Text-to-Speech generator using ChatTTS for natural conversational audio.

ChatTTS is optimized for dialogue scenarios with natural expression and prosody.
"""

import os
import tempfile
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import scipy.io.wavfile as wavfile

from lib.utils.logging_config import get_logger, LoggedOperation, log_execution_time
from lib.audio.quality_validator import AudioQualityValidator
from lib.audio.segment_processor import AudioSegmentProcessor, ConsistentVoiceManager
from lib.audio.post_processor import AudioPostProcessor

logger = get_logger(__name__)

try:
    import ChatTTS
    import torch
    HAS_CHATTTS = True
except ImportError:
    HAS_CHATTTS = False


@dataclass
class ChatTTSConfig:
    """Configuration for ChatTTS generation."""
    sample_rate: int = 24000
    output_format: str = "wav"
    temperature: float = 0.4  # Slightly higher for more natural variation
    top_k: int = 25           # Slightly wider for more natural vocabulary
    top_p: float = 0.75       # Slightly higher for more conversational flow
    device: str = "cpu"       # cpu or cuda
    use_decoder: bool = True  # Better quality but slower


class ChatTTSSpeechGenerator:
    """
    Converts text content to natural speech using ChatTTS.
    
    Provides conversational, expressive TTS with natural prosody.
    """
    
    def __init__(self, config: Optional[ChatTTSConfig] = None):
        """
        Initialize the ChatTTS generator.
        
        Args:
            config: TTS configuration, uses defaults if None
        """
        self.config = config or ChatTTSConfig()
        self.chat = None
        
        # Initialize audio processing components
        self.quality_validator = AudioQualityValidator()
        self.segment_processor = AudioSegmentProcessor(sample_rate=self.config.sample_rate)
        self.voice_manager = ConsistentVoiceManager()
        self.post_processor = AudioPostProcessor(sample_rate=self.config.sample_rate)
        
        self._initialize_chattts()
    
    def _initialize_chattts(self):
        """Initialize the ChatTTS engine with quality optimizations."""
        if not HAS_CHATTTS:
            logger.critical("ChatTTS not available. Install with: pip install ChatTTS")
            logger.info("Also requires: pip install torch numpy scipy")
            return
        
        with LoggedOperation(logger, "ChatTTS initialization", "INFO"):
            try:
                logger.info(f"Initializing ChatTTS on device: {self.config.device}")
                logger.debug(f"Sample rate: {self.config.sample_rate}Hz")
                
                # Optimize PyTorch for better quality and performance
                torch._dynamo.config.cache_size_limit = 64
                torch._dynamo.config.suppress_errors = True
                torch.set_float32_matmul_precision('high')
                
                # Initialize ChatTTS
                self.chat = ChatTTS.Chat()
                
                # Load models (this may take some time on first run)
                logger.info("Loading ChatTTS models (this may take a moment)...")
                self.chat.load()
                
                logger.info("ChatTTS initialized successfully with quality optimizations")
                
            except Exception as e:
                logger.critical(f"Failed to initialize ChatTTS: {e}")
                logger.info("Try installing dependencies: pip install ChatTTS torch numpy scipy")
                self.chat = None
                raise
    
    def generate_speech_from_monologue(
        self, 
        monologue: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate natural speech from a monologue dictionary using ChatTTS.
        
        Args:
            monologue: Monologue dictionary from discussion generator
            output_path: Optional output file path, generates temp file if None
            
        Returns:
            Dictionary with speech generation results
        """
        if not self.chat:
            return self._fallback_response("ChatTTS engine not available")
        
        # Extract text from monologue
        try:
            full_text = self._extract_text_from_monologue(monologue)
            print(f"   Extracted {len(full_text)} characters of text")
        except Exception as e:
            return self._fallback_response(f"Failed to extract text: {e}")
        
        if not full_text.strip():
            return self._fallback_response("No text content to convert")
        
        # Generate output path if not provided
        if not output_path:
            project_dir = Path(__file__).parent.parent.parent
            audio_dir = project_dir / "audio_output"
            audio_dir.mkdir(exist_ok=True)
            
            topic = monologue.get('topic', 'unknown').replace(' ', '_').replace('/', '_')
            safe_topic = ''.join(c for c in topic if c.isalnum() or c in '_-')[:50]
            output_path = audio_dir / f"{safe_topic}_chattts.{self.config.output_format}"
        
        # Generate speech using enhanced pipeline with quality optimization
        try:
            print(f"🗣️  Generating natural speech: {output_path}")
            print(f"   Config: temperature={self.config.temperature}, top_k={self.config.top_k}, top_p={self.config.top_p}")
            
            # Clean and normalize text with improved handling
            clean_text = self._normalize_text_for_tts(full_text)
            
            # Split into manageable chunks with natural boundaries
            text_chunks = self._split_text_for_tts(clean_text, max_length=150)
            print(f"   Processing {len(text_chunks)} text chunks")
            
            # Get consistent speaker for all chunks
            print("   Using consistent speaker voice...")
            consistent_speaker = self.voice_manager.get_consistent_speaker(self.chat)
            if consistent_speaker is None:
                consistent_speaker = self.chat.sample_random_speaker()
                logger.warning("Fallback to new speaker - consistency may be affected")
            
            # Set up inference parameters
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=consistent_speaker,
                temperature=self.config.temperature,
                top_P=self.config.top_p,
                top_K=self.config.top_k
            )
            
            # Enhanced text processing with natural pauses
            enhanced_chunks = self._prepare_chunks_with_natural_pauses(text_chunks)
            
            # Generate audio segments
            print(f"   Generating speech segments with consistent voice...")
            audio_segments = []
            for i, chunk in enumerate(enhanced_chunks):
                # Generate individual segment
                segment_wavs = self.chat.infer([chunk], params_infer_code=params_infer_code)
                if segment_wavs and len(segment_wavs) > 0:
                    audio_segments.append(segment_wavs[0])
                    print(f"   ✓ Generated segment {i+1}/{len(enhanced_chunks)}")
                else:
                    logger.warning(f"Failed to generate segment {i+1}")
            
            if not audio_segments:
                return self._fallback_response("No audio segments generated")
            
            # Validate voice consistency across segments
            consistency_score = self.voice_manager.validate_voice_consistency(audio_segments)
            print(f"   Voice consistency: {consistency_score:.2f}")
            
            # Apply smooth concatenation with crossfading
            print("   Applying smooth concatenation with crossfading...")
            final_audio = self.segment_processor.concatenate_with_crossfade(
                audio_segments, fade_ms=100
            )
            
            # Apply quality validation and enhancement
            print("   Validating and enhancing audio quality...")
            
            # Save initial version for quality analysis
            temp_path = str(output_path).replace('.wav', '_temp.wav')
            self._save_audio_array(final_audio, temp_path)
            
            # Perform quality analysis
            quality_report = self.quality_validator.analyze_audio(temp_path)
            
            # Apply post-processing if needed
            if quality_report.needs_processing:
                print(f"   Applying audio enhancements: {', '.join(quality_report.recommended_fixes)}")
                enhanced_path = self.post_processor.enhance_audio(temp_path, quality_report.recommended_fixes)
                
                # Use enhanced version if successful
                if os.path.exists(enhanced_path):
                    final_output_path = enhanced_path
                    # Rename to final output path
                    if enhanced_path != str(output_path):
                        os.rename(enhanced_path, str(output_path))
                else:
                    os.rename(temp_path, str(output_path))
            else:
                # Use original version
                os.rename(temp_path, str(output_path))
                print("   Audio quality acceptable, no enhancement needed")
            
            # Final verification
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                duration_estimate = len(final_audio) / self.config.sample_rate
                
                print(f"✅ Enhanced audio completed:")
                print(f"   File: {file_size:,} bytes, {duration_estimate:.1f}s duration")
                print(f"   Quality score: {quality_report.quality_score:.2f}/1.0")
                print(f"   Voice consistency: {consistency_score:.2f}")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "file_size": file_size,
                    "estimated_duration": duration_estimate,
                    "text_word_count": len(full_text.split()),
                    "quality_metrics": {
                        "quality_score": quality_report.quality_score,
                        "voice_consistency": consistency_score,
                        "silence_percentage": quality_report.silence_percentage,
                        "speech_percentage": quality_report.speech_percentage,
                        "dynamic_range_db": quality_report.dynamic_range_db,
                        "enhancements_applied": quality_report.recommended_fixes
                    },
                    "tts_config": {
                        "model": "ChatTTS",
                        "sample_rate": self.config.sample_rate,
                        "format": self.config.output_format,
                        "temperature": self.config.temperature,
                        "chunks_processed": len(text_chunks),
                        "crossfade_applied": True
                    },
                    "engine": "chattts"
                }
            else:
                return self._fallback_response("Audio file was not created")
                
        except Exception as e:
            return self._fallback_response(f"ChatTTS generation failed: {e}")
    
    def _extract_text_from_monologue(self, monologue: Dict[str, Any]) -> str:
        """Extract text from monologue, handling both script and direct text formats."""
        # Handle LLM-generated text format
        if 'generated_text' in monologue:
            return monologue['generated_text'].strip()
        
        # Handle script format (rule-based or LLM with script)
        if 'script' in monologue:
            return self._extract_text_from_script(monologue['script'])
        
        # Fallback: look for any text content
        for key in ['content', 'text', 'body']:
            if key in monologue and monologue[key]:
                return str(monologue[key]).strip()
        
        return ""
    
    def _extract_text_from_script(self, script: List[Any]) -> str:
        """Extract clean text from script turns."""
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
                # Clean up the text for natural speech
                content = content.replace('&', 'and')
                content = content.replace('<', 'less than')
                content = content.replace('>', 'greater than')
                # Remove markup
                content = content.replace('*', '')
                content = content.replace('_', '')
                
                text_parts.append(content)
        
        return ' '.join(text_parts)
    
    def _normalize_text_for_tts(self, text: str) -> str:
        """Normalize text to fix ChatTTS issues with numbers, dates, and special characters."""
        import re
        
        # Convert common problematic characters and patterns  
        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')
        text = text.replace('#', ' number ')
        
        # Handle Roman numerals (fix for "It" vs "II" issues)
        roman_numerals = {
            'II': 'the second', 'III': 'the third', 'IV': 'the fourth', 'VI': 'the sixth',
            'VII': 'the seventh', 'VIII': 'the eighth', 'IX': 'the ninth', 'XI': 'the eleventh',
            'XII': 'the twelfth', 'XIII': 'the thirteenth', 'XIV': 'the fourteenth',
            'XV': 'the fifteenth', 'XVI': 'the sixteenth', 'XVII': 'the seventeenth',
            'XVIII': 'the eighteenth', 'XIX': 'the nineteenth', 'XX': 'the twentieth'
        }
        
        # Replace Roman numerals in context (World War II, etc.)
        for roman, spoken in roman_numerals.items():
            # Match Roman numeral when it's a standalone word or at word boundaries
            pattern = r'\b' + re.escape(roman) + r'\b'
            text = re.sub(pattern, spoken, text, flags=re.IGNORECASE)
        
        # Handle currency (do before large number processing)
        text = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', r'\1 dollars', text)
        
        # Fix years (4-digit numbers) - convert to spelled out form
        def replace_year(match):
            year = match.group(0)
            if 1000 <= int(year) <= 2100:  # Reasonable year range
                return f"the year {year}"
            return year
        
        text = re.sub(r'\b(1[0-9]{3}|20[0-9]{2})\b', replace_year, text)
        
        # Fix large numbers with commas
        def replace_large_number(match):
            num = match.group(0).replace(',', '')
            try:
                val = int(num)
                if val >= 1000000:
                    return f"{val // 1000000} million"
                elif val >= 1000:
                    return f"{val // 1000} thousand"
                else:
                    return str(val)
            except:
                return match.group(0)
        
        text = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', replace_large_number, text)
        
        # Fix dates like "January 1, 2023" or "1/1/2023"  
        text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', r'month \1 day \2 \3', text)
        
        # Fix percentages (do this BEFORE decimal fix)
        text = re.sub(r'(\d+(?:\.\d+)?)%', r'\1 percent', text)
        
        # Fix decimal numbers
        text = re.sub(r'\b(\d+)\.(\d+)\b', r'\1 point \2', text)
        
        # Remove or replace other problematic characters that ChatTTS warns about
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _split_text_for_tts(self, text: str, max_length: int = 150) -> List[str]:
        """Split long text into chunks suitable for TTS processing with natural sentence boundaries."""
        import re
        
        # Enhanced sentence splitting with better boundary detection
        # Split on sentence endings followed by space and capital letter or end of string
        sentence_pattern = r'([.!?]+)\s+(?=[A-Z]|$)'
        sentences = re.split(sentence_pattern, text)
        
        # Reconstruct sentences by combining text with their punctuation
        clean_sentences = []
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]  # Add punctuation back
                clean_sentences.append(sentence.strip())
        
        # Remove empty sentences
        clean_sentences = [s for s in clean_sentences if s]
        
        # If no sentences found, fallback to original splitting
        if not clean_sentences:
            sentences = text.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
            clean_sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in clean_sentences:
            # If adding this sentence would exceed max_length, start new chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _fallback_response(self, error_message: str) -> Dict[str, Any]:
        """Fallback response when speech generation fails."""
        print(f"❌ {error_message}")
        return {
            "success": False,
            "error": error_message,
            "output_path": None,
            "estimated_duration": 0,
            "text_word_count": 0,
            "engine": "chattts"
        }
    
    def change_voice_settings(self, temperature: float = None, top_k: int = None, top_p: float = None):
        """
        Change voice generation settings for different expressiveness.
        
        Args:
            temperature: Controls randomness (0.1-1.0, higher = more expressive)
            top_k: Limits vocabulary selection 
            top_p: Controls diversity in generation
        """
        if temperature is not None:
            self.config.temperature = max(0.1, min(1.0, temperature))
        if top_k is not None:
            self.config.top_k = max(1, top_k)
        if top_p is not None:
            self.config.top_p = max(0.1, min(1.0, top_p))
        
        print(f"🎚️  Voice settings updated:")
        print(f"   Temperature: {self.config.temperature}")
        print(f"   Top-K: {self.config.top_k}")
        print(f"   Top-P: {self.config.top_p}")
    
    def _prepare_chunks_with_natural_pauses(self, text_chunks: List[str]) -> List[str]:
        """
        Prepare text chunks with natural pauses and improved prosody.
        
        Args:
            text_chunks: List of text chunks to enhance
            
        Returns:
            List of enhanced text chunks with natural pauses
        """
        enhanced_chunks = []
        
        for i, chunk in enumerate(text_chunks):
            enhanced_chunk = chunk
            
            # Add natural speech characteristics only to first chunk
            if i == 0:
                # Very subtle oral characteristic for natural start
                enhanced_chunk = f"[oral_2]{enhanced_chunk}"
            
            # Add natural pause tokens between sentences within chunks
            enhanced_chunk = enhanced_chunk.replace('. ', '. [uv_break] ')
            enhanced_chunk = enhanced_chunk.replace('! ', '! [uv_break] ')
            enhanced_chunk = enhanced_chunk.replace('? ', '? [uv_break] ')
            
            # Add breath pause at the end of each chunk (except last)
            if i < len(text_chunks) - 1:
                enhanced_chunk += " [uv_break]"
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _save_audio_array(self, audio_data: np.ndarray, output_path: str):
        """
        Save audio array to file using available libraries.
        
        Args:
            audio_data: Audio data as numpy array
            output_path: Path to save the audio file
        """
        try:
            # Try torchaudio first
            try:
                import torchaudio
                audio_tensor = torch.from_numpy(audio_data)
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
                torchaudio.save(output_path, audio_tensor, self.config.sample_rate)
                return
            except ImportError:
                pass
            
            # Fallback to scipy
            try:
                # Ensure audio is in the right format for WAV
                if audio_data.dtype != np.int16:
                    # Convert float to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_data
                
                wavfile.write(output_path, self.config.sample_rate, audio_int16)
                return
            except Exception as e:
                logger.warning(f"scipy write failed: {e}")
            
            # Last resort: basic file write (not recommended but works)
            logger.warning("Using basic numpy save as fallback")
            np.save(output_path.replace('.wav', '.npy'), audio_data)
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise


def get_recommended_voice_settings() -> List[Dict[str, Any]]:
    """Get recommended voice settings for different use cases."""
    return [
        {
            "name": "natural",
            "description": "Conversational, natural-sounding speech (default)",
            "temperature": 0.4,  # Slightly higher for natural variation
            "top_k": 25,         # Wider vocabulary for natural speech
            "top_p": 0.75        # More conversational flow
        },
        {
            "name": "conversational",
            "description": "Natural conversation style with good consistency",
            "temperature": 0.45, # Sweet spot for natural but consistent
            "top_k": 28,         # Good vocabulary range
            "top_p": 0.8         # Natural conversational flow
        },
        {
            "name": "expressive", 
            "description": "More animated and expressive",
            "temperature": 0.6,  # Higher for more variation
            "top_k": 30,         # Wider vocabulary
            "top_p": 0.85
        },
        {
            "name": "calm",
            "description": "Calm, measured delivery",
            "temperature": 0.25, # Lower for calm delivery
            "top_k": 18,         # More conservative
            "top_p": 0.65
        },
        {
            "name": "consistent",
            "description": "Very consistent, predictable speech",
            "temperature": 0.2,  # Low for maximum consistency
            "top_k": 15,         # Highly focused
            "top_p": 0.6         # Conservative sampling
        },
        {
            "name": "high_quality",
            "description": "GitHub example settings for best quality",
            "temperature": 0.3,  # GitHub example
            "top_k": 20,         # GitHub example
            "top_p": 0.7         # GitHub example
        }
    ]