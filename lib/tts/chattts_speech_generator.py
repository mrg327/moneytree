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
        
        # Generate speech using clean basic approach
        try:
            print(f"🗣️  Generating natural speech: {output_path}")
            print(f"   Config: temperature={self.config.temperature}, top_k={self.config.top_k}, top_p={self.config.top_p}")
            
            # Clean and normalize text
            clean_text = self._normalize_text_for_tts(full_text)
            
            # Split into manageable chunks
            text_chunks = self._split_text_for_tts(clean_text, max_length=150)
            print(f"   Processing {len(text_chunks)} text chunks")
            
            # Sample a consistent speaker for all chunks
            print("   Sampling consistent speaker voice...")
            rand_spk = self.chat.sample_random_speaker()
            
            # Set up inference parameters
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=rand_spk,
                temperature=self.config.temperature,
                top_P=self.config.top_p,
                top_K=self.config.top_k
            )
            
            # Add minimal prosodic enhancement for more natural speech
            enhanced_chunks = []
            for i, chunk in enumerate(text_chunks):
                # Add very subtle natural speech tokens (minimal and safe)
                if i == 0:
                    # Only add slight oral characteristic to first chunk
                    enhanced_chunk = f"[oral_2]{chunk}"
                else:
                    enhanced_chunk = chunk
                enhanced_chunks.append(enhanced_chunk)
            
            # Generate audio for all chunks at once
            print(f"   Generating speech for all chunks with minimal enhancement...")
            wavs = self.chat.infer(enhanced_chunks, params_infer_code=params_infer_code)
            
            if not wavs or len(wavs) == 0:
                return self._fallback_response("No audio generated from text")
            
            # Concatenate all audio chunks
            if len(wavs) > 1:
                final_audio = np.concatenate(wavs)
            else:
                final_audio = wavs[0]
            
            # Save to file using torchaudio for better compatibility
            try:
                # Convert to torch tensor for torchaudio
                audio_tensor = torch.from_numpy(final_audio)
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
                
                # Try torchaudio first
                try:
                    import torchaudio
                    torchaudio.save(str(output_path), audio_tensor, self.config.sample_rate)
                    print(f"✅ Audio saved with torchaudio")
                except:
                    # Fallback to scipy
                    wavfile.write(str(output_path), self.config.sample_rate, final_audio)
                    print(f"✅ Audio saved with scipy")
                    
            except Exception as save_error:
                print(f"⚠️ Save error: {save_error}, checking if file exists...")
                
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                duration_estimate = len(final_audio) / self.config.sample_rate
                print(f"✅ Audio verified: {file_size:,} bytes, {duration_estimate:.1f}s duration")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "file_size": file_size,
                    "estimated_duration": duration_estimate,
                    "text_word_count": len(full_text.split()),
                    "tts_config": {
                        "model": "ChatTTS",
                        "sample_rate": self.config.sample_rate,
                        "format": self.config.output_format,
                        "temperature": self.config.temperature,
                        "chunks_processed": len(text_chunks)
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
        """Split long text into chunks suitable for TTS processing."""
        # Split by sentences
        sentences = text.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
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