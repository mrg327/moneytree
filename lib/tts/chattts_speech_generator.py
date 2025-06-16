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
    temperature: float = 0.7  # Controls randomness/expressiveness
    top_k: int = 20
    top_p: float = 0.7
    device: str = "cpu"  # cpu or cuda
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
        """Initialize the ChatTTS engine."""
        if not HAS_CHATTTS:
            logger.critical("ChatTTS not available. Install with: pip install ChatTTS")
            logger.info("Also requires: pip install torch numpy scipy")
            return
        
        with LoggedOperation(logger, "ChatTTS initialization", "INFO"):
            try:
                logger.info(f"Initializing ChatTTS on device: {self.config.device}")
                logger.debug(f"Sample rate: {self.config.sample_rate}Hz")
                
                # Initialize ChatTTS
                self.chat = ChatTTS.Chat()
                
                # Load models (this may take some time on first run)
                logger.info("Loading ChatTTS models (this may take a moment)...")
                self.chat.load()
                
                logger.info("ChatTTS initialized successfully")
                
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
        
        # Extract text from monologue - make sure we don't mix up dict and config
        try:
            full_text = self._extract_text_from_monologue(monologue)
            print(f"   Extracted {len(full_text)} characters of text")
        except Exception as e:
            return self._fallback_response(f"Failed to extract text: {e}")
        
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
            output_path = audio_dir / f"{safe_topic}_chattts.{self.config.output_format}"
        
        # Generate speech
        try:
            print(f"🗣️  Generating natural conversational speech: {output_path}")
            print(f"   Config type: {type(self.config)}")
            print(f"   Config: {self.config}")
            
            # Split text into manageable chunks for better processing
            text_chunks = self._split_text_for_tts(full_text)
            
            # Generate consistent speaker for all chunks (use fixed seed for consistency)
            print("   Setting up consistent voice parameters...")
            torch.manual_seed(42)  # Fixed seed for reproducible voice
            
            all_audio = []
            
            # Generate a consistent speaker sample for the first chunk, then reuse it
            speaker_wav = None
            
            for i, chunk in enumerate(text_chunks):
                print(f"   Processing chunk {i+1}/{len(text_chunks)} with consistent voice")
                
                # Generate audio for this chunk using consistent parameters
                params_infer_code = self.chat.InferCodeParams()
                params_infer_code.temperature = self.config.temperature
                params_infer_code.top_K = self.config.top_k
                params_infer_code.top_P = self.config.top_p
                
                # Use deterministic seed for each chunk to maintain voice consistency
                params_infer_code.manual_seed = 42 + i  # Slight variation but consistent pattern
                
                print(f"   Using params: temp={params_infer_code.temperature}, top_K={params_infer_code.top_K}, top_P={params_infer_code.top_P}")
                
                # Generate audio with consistent parameters
                wavs = self.chat.infer([chunk], params_infer_code=params_infer_code)
                
                if wavs and len(wavs) > 0:
                    all_audio.append(wavs[0])
            
            if not all_audio:
                return self._fallback_response("No audio generated from text")
            
            # Concatenate all audio chunks
            if len(all_audio) > 1:
                final_audio = np.concatenate(all_audio)
            else:
                final_audio = all_audio[0]
            
            # Save to file (handle WSL permission issues)
            try:
                wavfile.write(str(output_path), self.config.sample_rate, final_audio)
                print(f"✅ Audio saved successfully")
            except PermissionError as pe:
                # Common WSL issue - file is often created despite the error
                print(f"⚠️ Permission warning during save, checking if file exists...")
                
            # Check if file was created and get stats (even if wavfile.write raised an error)
            if os.path.exists(output_path):
                try:
                    file_size = os.path.getsize(output_path)
                    duration_estimate = len(final_audio) / self.config.sample_rate
                    print(f"✅ Audio file verified: {file_size:,} bytes, {duration_estimate:.1f}s duration")
                    
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
                            "device": self.config.device,
                            "temperature": self.config.temperature
                        },
                        "engine": "chattts"
                    }
                except Exception as e:
                    print(f"⚠️ File exists but cannot read stats: {e}")
                    # File exists but we can't get stats - still return success
                    return {
                        "success": True,
                        "output_path": str(output_path),
                        "file_size": len(final_audio) * 4,  # Estimate for float32
                        "estimated_duration": len(final_audio) / self.config.sample_rate,
                        "text_word_count": len(full_text.split()),
                        "tts_config": {
                            "model": "ChatTTS",
                            "sample_rate": self.config.sample_rate,
                            "format": self.config.output_format,
                            "device": self.config.device,
                            "temperature": self.config.temperature
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

    def _split_text_for_tts(self, text: str, max_length: int = 200) -> List[str]:
        """Split long text into chunks suitable for TTS processing."""
        # First normalize the text to fix number/date issues
        text = self._normalize_text_for_tts(text)
        
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
            "description": "Balanced, natural-sounding speech",
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.7
        },
        {
            "name": "expressive", 
            "description": "More animated and expressive",
            "temperature": 0.9,
            "top_k": 30,
            "top_p": 0.8
        },
        {
            "name": "calm",
            "description": "Calm, measured delivery",
            "temperature": 0.5,
            "top_k": 15,
            "top_p": 0.6
        },
        {
            "name": "consistent",
            "description": "Very consistent, predictable speech",
            "temperature": 0.2,
            "top_k": 5,
            "top_p": 0.3
        }
    ]