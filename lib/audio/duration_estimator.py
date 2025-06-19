"""
Audio duration estimation for pipeline optimization.

Provides early duration estimates to prevent resource waste on excess video/music processing.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AudioEstimate:
    """
    Audio duration and quality estimates for pipeline optimization.
    
    Attributes:
        estimated_duration: Predicted audio duration in seconds
        confidence_level: Confidence in estimate (0.0-1.0)
        word_count: Number of words in the text
        estimation_method: Method used for estimation
        quality_warnings: Potential quality issues detected in text
        buffer_factor: Recommended buffer factor for pre-processing
    """
    estimated_duration: float
    confidence_level: float
    word_count: int
    estimation_method: str
    quality_warnings: List[str]
    buffer_factor: float = 1.1


class AudioDurationEstimator:
    """
    Estimates audio duration before generation to optimize pipeline resource usage.
    
    Uses engine-specific speaking rates and text analysis to provide accurate
    duration estimates that enable early media trimming and optimization.
    """
    
    def __init__(self):
        """Initialize the duration estimator with engine-specific rates."""
        # Speaking rates in words per minute for different engines
        self.engine_rates = {
            'chattts': 160,      # Natural conversational rate
            'coqui': 180,        # Slightly faster synthetic speech
            'default': 170       # Conservative average
        }
        
        # Rate adjustments for different content types
        self.content_adjustments = {
            'technical': 0.85,   # Slower for technical content
            'numbers_heavy': 0.75, # Much slower for number-heavy content
            'conversational': 1.05, # Slightly faster for casual content
            'narrative': 0.95    # Slightly slower for storytelling
        }
    
    def estimate_from_monologue(self, monologue: Dict[str, Any], engine: str = 'chattts') -> AudioEstimate:
        """
        Estimate audio duration from a monologue dictionary.
        
        Args:
            monologue: Monologue dictionary from content generators
            engine: TTS engine being used ('chattts', 'coqui')
            
        Returns:
            AudioEstimate with duration prediction and metadata
        """
        # Extract text from monologue
        text = self._extract_text_from_monologue(monologue)
        
        if not text.strip():
            logger.warning("Empty text provided for duration estimation")
            return AudioEstimate(
                estimated_duration=0.0,
                confidence_level=0.0,
                word_count=0,
                estimation_method='empty_text',
                quality_warnings=['No text content found'],
                buffer_factor=1.0
            )
        
        return self.estimate_from_text(text, engine)
    
    def estimate_from_text(self, text: str, engine: str = 'chattts') -> AudioEstimate:
        """
        Estimate audio duration from raw text.
        
        Args:
            text: Text content to analyze
            engine: TTS engine being used
            
        Returns:
            AudioEstimate with duration prediction and metadata
        """
        # Clean and analyze text
        clean_text = self._clean_text_for_analysis(text)
        word_count = len(clean_text.split())
        
        # Analyze content type for rate adjustment
        content_analysis = self._analyze_content_type(clean_text)
        
        # Get base speaking rate for engine
        base_rate = self.engine_rates.get(engine, self.engine_rates['default'])
        
        # Apply content-based adjustment
        adjusted_rate = base_rate * content_analysis['rate_multiplier']
        
        # Calculate base duration (words per minute to seconds)
        base_duration = (word_count / adjusted_rate) * 60
        
        # Add pause adjustments for punctuation and structure
        pause_adjustment = self._calculate_pause_adjustments(clean_text)
        
        # Final duration estimate
        estimated_duration = base_duration + pause_adjustment
        
        # Calculate confidence based on content characteristics
        confidence = self._calculate_confidence(clean_text, content_analysis)
        
        # Determine buffer factor based on content complexity
        buffer_factor = self._calculate_buffer_factor(content_analysis, confidence)
        
        logger.info(f"Duration estimate: {estimated_duration:.1f}s for {word_count} words "
                   f"(rate: {adjusted_rate:.0f} wpm, confidence: {confidence:.2f})")
        
        return AudioEstimate(
            estimated_duration=estimated_duration,
            confidence_level=confidence,
            word_count=word_count,
            estimation_method=f'{engine}_rate_analysis',
            quality_warnings=content_analysis['warnings'],
            buffer_factor=buffer_factor
        )
    
    def validate_estimate_accuracy(self, estimate: AudioEstimate, actual_path: str) -> float:
        """
        Validate estimate accuracy against actual generated audio.
        
        Args:
            estimate: Original duration estimate
            actual_path: Path to actual generated audio file
            
        Returns:
            Accuracy percentage (0.0-100.0)
        """
        try:
            # Try to get actual duration using librosa
            try:
                import librosa
                y, sr = librosa.load(actual_path)
                actual_duration = len(y) / sr
            except ImportError:
                # Fallback to wave module
                import wave
                with wave.open(actual_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    actual_duration = frames / sample_rate
            
            # Calculate accuracy
            if actual_duration > 0:
                accuracy = (1 - abs(estimate.estimated_duration - actual_duration) / actual_duration) * 100
                accuracy = max(0, min(100, accuracy))  # Clamp to 0-100%
                
                logger.info(f"Duration accuracy: {accuracy:.1f}% "
                           f"(estimated: {estimate.estimated_duration:.1f}s, "
                           f"actual: {actual_duration:.1f}s)")
                
                return accuracy
            else:
                logger.warning("Could not determine actual audio duration")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error validating duration accuracy: {e}")
            return 0.0
    
    def get_engine_speaking_rate(self, engine: str) -> float:
        """
        Get the speaking rate for a specific TTS engine.
        
        Args:
            engine: TTS engine name
            
        Returns:
            Speaking rate in words per minute
        """
        return self.engine_rates.get(engine, self.engine_rates['default'])
    
    def _extract_text_from_monologue(self, monologue: Dict[str, Any]) -> str:
        """Extract text from monologue, handling different formats."""
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
        """Extract text from script format."""
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
                text_parts.append(content)
        
        return ' '.join(text_parts)
    
    def _clean_text_for_analysis(self, text: str) -> str:
        """Clean text for accurate word counting and analysis."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove markdown and formatting
        text = re.sub(r'[*_`]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        return text
    
    def _analyze_content_type(self, text: str) -> Dict[str, Any]:
        """Analyze content type to determine speaking rate adjustments."""
        warnings = []
        rate_multiplier = 1.0
        content_type = 'conversational'  # default
        
        # Count numbers and technical terms
        number_matches = len(re.findall(r'\b\d+(?:[.,]\d+)*\b', text))
        technical_terms = len(re.findall(r'\b(?:API|CPU|GPU|HTTP|JSON|XML|SQL|URL|IP|DNS|TCP|UDP|SSL|TLS)\b', text, re.IGNORECASE))
        
        # Calculate ratios
        word_count = len(text.split())
        if word_count > 0:
            number_ratio = number_matches / word_count
            technical_ratio = technical_terms / word_count
            
            # Determine content type and adjustment
            if number_ratio > 0.15:  # More than 15% numbers
                content_type = 'numbers_heavy'
                rate_multiplier = self.content_adjustments['numbers_heavy']
                warnings.append('High number content detected - may affect speech timing')
            elif technical_ratio > 0.1:  # More than 10% technical terms
                content_type = 'technical'
                rate_multiplier = self.content_adjustments['technical']
                warnings.append('Technical content detected - may require slower delivery')
            elif len(re.findall(r'[.!?]', text)) / word_count > 0.08:  # Many sentences
                content_type = 'narrative'
                rate_multiplier = self.content_adjustments['narrative']
        
        return {
            'type': content_type,
            'rate_multiplier': rate_multiplier,
            'number_ratio': number_ratio if word_count > 0 else 0,
            'technical_ratio': technical_ratio if word_count > 0 else 0,
            'warnings': warnings
        }
    
    def _calculate_pause_adjustments(self, text: str) -> float:
        """Calculate additional time for natural pauses."""
        # Count sentence boundaries
        sentences = len(re.findall(r'[.!?]', text))
        
        # Count paragraph breaks
        paragraphs = len(re.findall(r'\n\s*\n', text))
        
        # Count commas (shorter pauses)
        commas = len(re.findall(r',', text))
        
        # Calculate pause time
        sentence_pause_time = sentences * 0.75  # 0.75 seconds per sentence
        paragraph_pause_time = paragraphs * 1.5  # 1.5 seconds per paragraph
        comma_pause_time = commas * 0.25  # 0.25 seconds per comma
        
        total_pause_time = sentence_pause_time + paragraph_pause_time + comma_pause_time
        
        return total_pause_time
    
    def _calculate_confidence(self, text: str, content_analysis: Dict) -> float:
        """Calculate confidence in the duration estimate."""
        base_confidence = 0.85  # Start with good confidence
        
        # Reduce confidence for problematic content
        if content_analysis['number_ratio'] > 0.2:
            base_confidence -= 0.15  # High number content is unpredictable
        
        if content_analysis['technical_ratio'] > 0.15:
            base_confidence -= 0.1   # Technical content may have pronunciation issues
        
        # Reduce confidence for very short or very long texts
        word_count = len(text.split())
        if word_count < 20:
            base_confidence -= 0.2   # Very short texts are unpredictable
        elif word_count > 500:
            base_confidence -= 0.1   # Very long texts may have more variation
        
        # Ensure confidence stays in valid range
        return max(0.3, min(1.0, base_confidence))
    
    def _calculate_buffer_factor(self, content_analysis: Dict, confidence: float) -> float:
        """Calculate recommended buffer factor for pre-processing."""
        base_buffer = 1.1  # 10% buffer by default
        
        # Increase buffer for low confidence estimates
        if confidence < 0.6:
            base_buffer = 1.2  # 20% buffer for low confidence
        elif confidence < 0.8:
            base_buffer = 1.15  # 15% buffer for medium confidence
        
        # Increase buffer for problematic content types
        if content_analysis['type'] == 'numbers_heavy':
            base_buffer += 0.05  # Numbers can cause longer pauses
        elif content_analysis['type'] == 'technical':
            base_buffer += 0.03  # Technical terms may need careful pronunciation
        
        return min(1.3, base_buffer)  # Cap at 30% buffer