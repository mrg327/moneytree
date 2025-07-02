"""
Whisper-based caption generation for accurate audio-synchronized captions.

Uses OpenAI's Whisper ASR model to generate precisely timed captions
from audio content, ensuring perfect synchronization with the spoken words.
"""

import os
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import whisper
    HAS_WHISPER = True
    WHISPER_TYPE = "openai"
except ImportError:
    try:
        from faster_whisper import WhisperModel
        HAS_WHISPER = True
        WHISPER_TYPE = "faster"
        logger.info("Using faster-whisper implementation")
    except ImportError:
        HAS_WHISPER = False
        WHISPER_TYPE = None
        logger.warning("No Whisper implementation available - install with: uv add openai-whisper or uv add faster-whisper")


@dataclass
class WhisperConfig:
    """Configuration for Whisper caption generation."""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: str = "en"      # Language code (en, fr, es, etc.)
    temperature: float = 0.0  # Temperature for transcription (0.0 = deterministic)
    word_timestamps: bool = True  # Enable word-level timestamps
    verbose: bool = False     # Verbose output during processing


@dataclass  
class CaptionSegment:
    """A single caption segment with precise timing."""
    text: str
    start_time: float
    end_time: float
    duration: float
    words: List[Dict[str, Any]]  # Word-level timing data
    confidence: float = 1.0


class WhisperCaptionGenerator:
    """
    Generates accurate captions using OpenAI Whisper ASR.
    
    Provides precise timing synchronization with audio content and
    handles various caption formatting options.
    """
    
    def __init__(self, config: Optional[WhisperConfig] = None):
        """
        Initialize the Whisper caption generator.
        
        Args:
            config: Whisper configuration, uses defaults if None
        """
        self.config = config or WhisperConfig()
        self.model = None
        
        if HAS_WHISPER:
            self._load_whisper_model()
        else:
            logger.error("Whisper not available - caption generation will be limited")
    
    def _load_whisper_model(self):
        """Load the Whisper model for transcription."""
        try:
            logger.info(f"Loading Whisper model: {self.config.model_size}")
            
            if WHISPER_TYPE == "faster":
                # Use faster-whisper implementation
                self.model = WhisperModel(self.config.model_size, device="cpu")
                logger.info("Faster-Whisper model loaded successfully")
            else:
                # Use OpenAI Whisper implementation
                self.model = whisper.load_model(self.config.model_size)
                logger.info("OpenAI Whisper model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None
    
    def generate_captions_from_audio(self, audio_path: str, 
                                   max_caption_length: int = 50,
                                   caption_duration: float = 3.0) -> List[CaptionSegment]:
        """
        Generate captions from audio using Whisper ASR.
        
        Args:
            audio_path: Path to audio file
            max_caption_length: Maximum characters per caption
            caption_duration: Preferred caption duration in seconds
            
        Returns:
            List of caption segments with precise timing
        """
        if not self.model:
            logger.error("Whisper model not available")
            return []
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return []
        
        try:
            logger.info(f"Transcribing audio with Whisper: {audio_path}")
            
            if WHISPER_TYPE == "faster":
                # Use faster-whisper API
                segments_iter, info = self.model.transcribe(
                    audio_path,
                    language=self.config.language,
                    temperature=self.config.temperature,
                    word_timestamps=self.config.word_timestamps,
                    vad_filter=True  # Voice activity detection
                )
                
                # Convert to list format
                segments = list(segments_iter)
                
                # Convert faster-whisper format to OpenAI format
                formatted_segments = []
                for segment in segments:
                    formatted_segment = {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                        "words": []
                    }
                    
                    # Add word-level timestamps if available
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            formatted_segment["words"].append({
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "probability": getattr(word, 'probability', 1.0)
                            })
                    
                    formatted_segments.append(formatted_segment)
                
                segments = formatted_segments
                
            else:
                # Use OpenAI Whisper API
                result = self.model.transcribe(
                    audio_path,
                    language=self.config.language,
                    temperature=self.config.temperature,
                    word_timestamps=self.config.word_timestamps,
                    verbose=self.config.verbose
                )
                
                # Extract segments and words
                segments = result.get("segments", [])
            
            if not segments:
                logger.warning("No speech segments detected in audio")
                return []
            
            logger.info(f"Found {len(segments)} speech segments")
            
            # Process segments into caption format
            captions = self._process_whisper_segments(
                segments, max_caption_length, caption_duration
            )
            
            logger.info(f"Generated {len(captions)} caption segments")
            return captions
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return []
    
    def _process_whisper_segments(self, segments: List[Dict], 
                                max_caption_length: int,
                                caption_duration: float) -> List[CaptionSegment]:
        """
        Process Whisper segments into properly formatted captions.
        
        Args:
            segments: Raw Whisper segments
            max_caption_length: Maximum caption length
            caption_duration: Preferred caption duration
            
        Returns:
            List of processed caption segments
        """
        captions = []
        current_caption_text = ""
        current_start_time = None
        current_words = []
        
        for segment in segments:
            segment_text = segment.get("text", "").strip()
            segment_start = segment.get("start", 0.0)
            segment_end = segment.get("end", 0.0)
            segment_words = segment.get("words", [])
            
            if not segment_text:
                continue
            
            # Initialize caption start time
            if current_start_time is None:
                current_start_time = segment_start
            
            # Check if adding this segment would exceed limits
            potential_text = (current_caption_text + " " + segment_text).strip()
            potential_duration = segment_end - current_start_time
            
            if (len(potential_text) <= max_caption_length and 
                potential_duration <= caption_duration and
                len(captions) == 0 or potential_duration <= caption_duration * 1.5):
                
                # Add to current caption
                current_caption_text = potential_text
                current_words.extend(segment_words)
                
            else:
                # Finalize current caption and start new one
                if current_caption_text:
                    caption = CaptionSegment(
                        text=current_caption_text,
                        start_time=current_start_time,
                        end_time=segment_start,
                        duration=segment_start - current_start_time,
                        words=current_words,
                        confidence=self._calculate_segment_confidence(current_words)
                    )
                    captions.append(caption)
                
                # Start new caption
                current_caption_text = segment_text
                current_start_time = segment_start
                current_words = segment_words.copy()
        
        # Add final caption
        if current_caption_text and segments:
            final_end = segments[-1].get("end", current_start_time + 1.0)
            caption = CaptionSegment(
                text=current_caption_text,
                start_time=current_start_time,
                end_time=final_end,
                duration=final_end - current_start_time,
                words=current_words,
                confidence=self._calculate_segment_confidence(current_words)
            )
            captions.append(caption)
        
        return captions
    
    def _calculate_segment_confidence(self, words: List[Dict]) -> float:
        """Calculate average confidence for a segment."""
        if not words:
            return 1.0
        
        confidences = [word.get("probability", 1.0) for word in words]
        return sum(confidences) / len(confidences) if confidences else 1.0
    
    def generate_moviepy_timing_data(self, captions: List[CaptionSegment]) -> List[Dict[str, Any]]:
        """
        Convert caption segments to MoviePy-compatible timing data.
        
        Args:
            captions: List of caption segments
            
        Returns:
            List of timing dictionaries for MoviePy TextClip creation
        """
        timing_data = []
        
        for caption in captions:
            timing_data.append({
                'text': caption.text,
                'start': caption.start_time,
                'end': caption.end_time,
                'duration': caption.duration,
                'confidence': caption.confidence,
                'word_count': len(caption.words),
                'words': caption.words  # Include word-level timing data for single-word captions
            })
        
        return timing_data
    
    def adjust_caption_timing(self, captions: List[CaptionSegment], 
                            min_duration: float = 1.0,
                            max_duration: float = 5.0) -> List[CaptionSegment]:
        """
        Adjust caption timing for better readability.
        
        Args:
            captions: Original caption segments
            min_duration: Minimum caption duration
            max_duration: Maximum caption duration
            
        Returns:
            Adjusted caption segments
        """
        adjusted = []
        
        for caption in captions:
            duration = caption.duration
            
            # Adjust duration if needed
            if duration < min_duration:
                # Extend short captions
                new_end = caption.start_time + min_duration
                duration = min_duration
            elif duration > max_duration:
                # Limit long captions
                new_end = caption.start_time + max_duration
                duration = max_duration
            else:
                new_end = caption.end_time
            
            adjusted_caption = CaptionSegment(
                text=caption.text,
                start_time=caption.start_time,
                end_time=new_end,
                duration=duration,
                words=caption.words,
                confidence=caption.confidence
            )
            
            adjusted.append(adjusted_caption)
        
        return adjusted
    
    def get_caption_statistics(self, captions: List[CaptionSegment]) -> Dict[str, Any]:
        """
        Get statistics about the generated captions.
        
        Args:
            captions: List of caption segments
            
        Returns:
            Dictionary with caption statistics
        """
        if not captions:
            return {"error": "No captions provided"}
        
        durations = [c.duration for c in captions]
        confidences = [c.confidence for c in captions]
        text_lengths = [len(c.text) for c in captions]
        word_counts = [len(c.words) for c in captions]
        
        total_duration = sum(durations)
        total_text = " ".join(c.text for c in captions)
        
        return {
            "caption_count": len(captions),
            "total_duration": total_duration,
            "average_duration": total_duration / len(captions),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "average_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "average_text_length": sum(text_lengths) / len(text_lengths),
            "average_words_per_caption": sum(word_counts) / len(word_counts),
            "total_words": sum(word_counts),
            "total_characters": len(total_text),
            "words_per_minute": (sum(word_counts) / total_duration) * 60 if total_duration > 0 else 0
        }


def get_recommended_whisper_models() -> List[Dict[str, Any]]:
    """Get recommended Whisper models for different use cases."""
    return [
        {
            "name": "tiny",
            "description": "Fastest, least accurate (39 MB)",
            "speed": "very_fast",
            "accuracy": "basic",
            "memory": "low"
        },
        {
            "name": "base", 
            "description": "Good balance of speed and accuracy (74 MB)",
            "speed": "fast",
            "accuracy": "good",
            "memory": "low"
        },
        {
            "name": "small",
            "description": "Better accuracy, slower (244 MB)",
            "speed": "medium",
            "accuracy": "better", 
            "memory": "medium"
        },
        {
            "name": "medium",
            "description": "High accuracy for most uses (769 MB)",
            "speed": "slow",
            "accuracy": "high",
            "memory": "high"
        },
        {
            "name": "large",
            "description": "Best accuracy, slowest (1550 MB)",
            "speed": "very_slow", 
            "accuracy": "best",
            "memory": "very_high"
        }
    ]