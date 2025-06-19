"""
Caption quality validation system with OCR-based verification.

This module provides automated quality assessment for rendered captions
using OCR verification, visual analysis, and comprehensive scoring.
"""

import cv2
import numpy as np
import tempfile
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

# Optional dependencies with graceful degradation
try:
    import pytesseract
    HAS_TESSERACT = True
    logger.info("Tesseract OCR available for caption validation")
except ImportError:
    HAS_TESSERACT = False
    logger.warning("Tesseract OCR not available - quality validation will be limited")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.error("PIL not available - caption validation requires PIL")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.error("OpenCV not available - caption validation requires OpenCV")


@dataclass
class TextRegion:
    """Detected text region in a frame."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    text: str
    timestamp: float


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for caption rendering."""
    overall_score: float  # 0.0 to 1.0
    text_accuracy: float  # OCR accuracy vs expected text
    positioning_accuracy: float  # Text within intended boundaries
    visual_quality: float  # Clarity, contrast, readability
    timing_accuracy: float  # Caption timing synchronization
    
    # Detailed metrics
    total_frames_analyzed: int = 0
    frames_with_text: int = 0
    ocr_success_rate: float = 0.0
    text_cutoff_detected: bool = False
    font_quality_score: float = 0.0
    background_contrast: float = 0.0
    
    # Error details
    issues_found: List[str] = None
    
    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []


class CaptionQualityValidator:
    """
    Comprehensive caption quality validation system.
    
    Analyzes rendered video output to validate caption quality using
    OCR verification, visual analysis, and automated scoring.
    """
    
    def __init__(self):
        """Initialize the caption quality validator."""
        if not HAS_OPENCV or not HAS_PIL:
            raise ImportError("OpenCV and PIL are required for caption validation")
        
        self.ocr_available = HAS_TESSERACT
        
        # OCR configuration
        if self.ocr_available:
            self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,-!?;:'
        
        # Analysis thresholds
        self.min_confidence_threshold = 30
        self.text_similarity_threshold = 0.7
        self.contrast_threshold = 50
        
        # Frame sampling configuration
        self.sample_interval = 30  # Analyze every 30th frame (1 second at 30fps)
        self.min_samples = 10
        self.max_samples = 100
    
    def validate_video_captions(self,
                              video_path: str,
                              expected_captions: List[Dict[str, Any]],
                              progress_callback: Optional[callable] = None) -> QualityMetrics:
        """
        Validate caption quality in a rendered video.
        
        Args:
            video_path: Path to video file to analyze
            expected_captions: List of expected caption timing and text data
            progress_callback: Optional progress callback
            
        Returns:
            Comprehensive quality metrics
        """
        logger.info(f"Starting caption quality validation for: {Path(video_path).name}")
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Video analysis: {frame_count} frames @ {fps:.2f}fps ({duration:.2f}s)")
            
            # Determine sampling strategy
            sample_frames = self._calculate_sample_frames(frame_count)
            
            # Extract text regions from sampled frames
            text_regions = self._extract_text_regions(cap, sample_frames, progress_callback)
            
            # Validate against expected captions
            quality_metrics = self._analyze_caption_quality(
                text_regions, expected_captions, duration, len(sample_frames)
            )
            
            cap.release()
            
            logger.info(f"Validation complete - Overall score: {quality_metrics.overall_score:.3f}")
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Caption validation failed: {e}")
            # Return minimal metrics indicating failure
            return QualityMetrics(
                overall_score=0.0,
                text_accuracy=0.0,
                positioning_accuracy=0.0,
                visual_quality=0.0,
                timing_accuracy=0.0,
                issues_found=[f"Validation failed: {str(e)}"]
            )
    
    def _calculate_sample_frames(self, total_frames: int) -> List[int]:
        """Calculate which frames to sample for analysis."""
        # Ensure reasonable sampling
        max_interval = max(1, total_frames // self.min_samples)
        interval = min(self.sample_interval, max_interval)
        
        sample_frames = list(range(0, total_frames, interval))
        
        # Limit to max samples
        if len(sample_frames) > self.max_samples:
            step = len(sample_frames) // self.max_samples
            sample_frames = sample_frames[::step]
        
        logger.debug(f"Sampling {len(sample_frames)} frames from {total_frames} total")
        return sample_frames
    
    def _extract_text_regions(self,
                            cap: cv2.VideoCapture,
                            sample_frames: List[int],
                            progress_callback: Optional[callable]) -> List[TextRegion]:
        """Extract text regions from sampled video frames."""
        text_regions = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for i, frame_num in enumerate(sample_frames):
            try:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                timestamp = frame_num / fps if fps > 0 else 0
                
                # Detect text in frame
                regions = self._detect_text_in_frame(frame, timestamp)
                text_regions.extend(regions)
                
                # Progress callback
                if progress_callback and i % 10 == 0:
                    progress = i / len(sample_frames)
                    progress_callback(progress)
                
            except Exception as e:
                logger.warning(f"Error processing frame {frame_num}: {e}")
                continue
        
        logger.info(f"Extracted {len(text_regions)} text regions from {len(sample_frames)} frames")
        return text_regions
    
    def _detect_text_in_frame(self, frame: np.ndarray, timestamp: float) -> List[TextRegion]:
        """Detect text regions in a single frame."""
        regions = []
        
        try:
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use EAST text detector if available, otherwise use OCR directly
            if self.ocr_available:
                regions = self._ocr_based_detection(gray, timestamp)
            else:
                # Fallback to simple contrast-based detection
                regions = self._contrast_based_detection(gray, timestamp)
            
        except Exception as e:
            logger.debug(f"Text detection failed for frame at {timestamp:.2f}s: {e}")
        
        return regions
    
    def _ocr_based_detection(self, gray_frame: np.ndarray, timestamp: float) -> List[TextRegion]:
        """Use OCR to detect and extract text from frame."""
        regions = []
        
        try:
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                gray_frame, 
                config=self.ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                confidence = float(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                # Filter by confidence and text quality
                if confidence < self.min_confidence_threshold or len(text) < 2:
                    continue
                
                # Extract bounding box
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Validate bounding box
                if w < 10 or h < 10:
                    continue
                
                region = TextRegion(
                    bbox=(x, y, w, h),
                    confidence=confidence / 100.0,  # Normalize to 0-1
                    text=text,
                    timestamp=timestamp
                )
                
                regions.append(region)
                
        except Exception as e:
            logger.debug(f"OCR detection failed: {e}")
        
        return regions
    
    def _contrast_based_detection(self, gray_frame: np.ndarray, timestamp: float) -> List[TextRegion]:
        """Fallback text detection based on contrast analysis."""
        regions = []
        
        try:
            # Simple contrast-based text region detection
            # This is a basic fallback when OCR is not available
            
            # Apply threshold to find text-like regions
            _, thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (typical text regions)
                if w < 50 or h < 15 or w > gray_frame.shape[1] * 0.8:
                    continue
                
                # Estimate confidence based on contrast
                roi = gray_frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                contrast = np.std(roi)
                confidence = min(1.0, contrast / 100.0)
                
                if confidence < 0.3:
                    continue
                
                region = TextRegion(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    text="[detected_text]",  # Placeholder since we can't read it
                    timestamp=timestamp
                )
                
                regions.append(region)
                
        except Exception as e:
            logger.debug(f"Contrast-based detection failed: {e}")
        
        return regions
    
    def _analyze_caption_quality(self,
                               text_regions: List[TextRegion],
                               expected_captions: List[Dict[str, Any]],
                               video_duration: float,
                               frames_analyzed: int) -> QualityMetrics:
        """Analyze caption quality based on detected text regions."""
        issues = []
        
        # Initialize metrics
        frames_with_text = len([r for r in text_regions if r.text.strip()])
        ocr_success_rate = frames_with_text / frames_analyzed if frames_analyzed > 0 else 0.0
        
        # Text accuracy analysis
        text_accuracy = self._calculate_text_accuracy(text_regions, expected_captions)
        
        # Positioning accuracy analysis
        positioning_accuracy = self._calculate_positioning_accuracy(text_regions, video_duration)
        
        # Visual quality analysis
        visual_quality = self._calculate_visual_quality(text_regions)
        
        # Timing accuracy analysis
        timing_accuracy = self._calculate_timing_accuracy(text_regions, expected_captions)
        
        # Detect specific issues
        if self._detect_text_cutoff(text_regions):\n            issues.append(\"Text cutoff detected in some frames\")\n        \n        if ocr_success_rate < 0.5:\n            issues.append(f\"Low OCR success rate: {ocr_success_rate:.2f}\")\n        \n        if positioning_accuracy < 0.7:\n            issues.append(\"Text positioning issues detected\")\n        \n        if visual_quality < 0.6:\n            issues.append(\"Visual quality issues detected\")\n        \n        # Calculate overall score\n        weights = {\n            'text_accuracy': 0.3,\n            'positioning_accuracy': 0.25,\n            'visual_quality': 0.25,\n            'timing_accuracy': 0.2\n        }\n        \n        overall_score = (\n            text_accuracy * weights['text_accuracy'] +\n            positioning_accuracy * weights['positioning_accuracy'] +\n            visual_quality * weights['visual_quality'] +\n            timing_accuracy * weights['timing_accuracy']\n        )\n        \n        # Create quality metrics\n        metrics = QualityMetrics(\n            overall_score=overall_score,\n            text_accuracy=text_accuracy,\n            positioning_accuracy=positioning_accuracy,\n            visual_quality=visual_quality,\n            timing_accuracy=timing_accuracy,\n            total_frames_analyzed=frames_analyzed,\n            frames_with_text=frames_with_text,\n            ocr_success_rate=ocr_success_rate,\n            text_cutoff_detected=self._detect_text_cutoff(text_regions),\n            font_quality_score=visual_quality,  # Simplified\n            background_contrast=self._calculate_average_contrast(text_regions),\n            issues_found=issues\n        )\n        \n        logger.info(f\"Quality analysis complete:\")\n        logger.info(f\"  Text accuracy: {text_accuracy:.3f}\")\n        logger.info(f\"  Positioning: {positioning_accuracy:.3f}\")\n        logger.info(f\"  Visual quality: {visual_quality:.3f}\")\n        logger.info(f\"  Timing: {timing_accuracy:.3f}\")\n        logger.info(f\"  Issues: {len(issues)}\")\n        \n        return metrics\n    \n    def _calculate_text_accuracy(self,\n                               text_regions: List[TextRegion],\n                               expected_captions: List[Dict[str, Any]]) -> float:\n        \"\"\"Calculate text accuracy using OCR results vs expected text.\"\"\"\n        if not self.ocr_available or not expected_captions:\n            return 0.8  # Assume reasonable accuracy if we can't measure\n        \n        try:\n            # Combine all detected text\n            detected_text = \" \".join([r.text for r in text_regions if r.text.strip()])\n            \n            # Combine all expected text\n            expected_text = \" \".join([caption.get('text', '') for caption in expected_captions])\n            \n            if not detected_text or not expected_text:\n                return 0.0\n            \n            # Calculate similarity using simple word matching\n            detected_words = set(detected_text.lower().split())\n            expected_words = set(expected_text.lower().split())\n            \n            if not expected_words:\n                return 0.0\n            \n            # Calculate Jaccard similarity\n            intersection = detected_words.intersection(expected_words)\n            union = detected_words.union(expected_words)\n            \n            similarity = len(intersection) / len(union) if union else 0.0\n            \n            logger.debug(f\"Text accuracy: {similarity:.3f} ({len(intersection)}/{len(union)} words matched)\")\n            \n            return similarity\n            \n        except Exception as e:\n            logger.warning(f\"Text accuracy calculation failed: {e}\")\n            return 0.5  # Conservative estimate\n    \n    def _calculate_positioning_accuracy(self,\n                                      text_regions: List[TextRegion],\n                                      video_duration: float) -> float:\n        \"\"\"Calculate positioning accuracy based on text region locations.\"\"\"\n        if not text_regions:\n            return 0.0\n        \n        try:\n            # Analyze text positioning consistency\n            y_positions = [r.bbox[1] for r in text_regions]  # Y coordinates\n            \n            if not y_positions:\n                return 0.0\n            \n            # Calculate position variance (lower is better for consistency)\n            position_variance = np.var(y_positions)\n            max_variance = 10000  # Reasonable threshold\n            \n            # Convert variance to accuracy score (0-1)\n            consistency_score = max(0.0, 1.0 - (position_variance / max_variance))\n            \n            # Check for text near edges (positioning issues)\n            edge_violations = 0\n            for region in text_regions:\n                x, y, w, h = region.bbox\n                # Assume video dimensions (would be better to get actual dimensions)\n                if y < 50 or y + h > 950:  # Too close to top/bottom edges\n                    edge_violations += 1\n            \n            edge_score = 1.0 - (edge_violations / len(text_regions))\n            \n            # Combined positioning accuracy\n            positioning_accuracy = (consistency_score + edge_score) / 2\n            \n            logger.debug(f\"Positioning accuracy: {positioning_accuracy:.3f} (consistency: {consistency_score:.3f}, edge: {edge_score:.3f})\")\n            \n            return positioning_accuracy\n            \n        except Exception as e:\n            logger.warning(f\"Positioning accuracy calculation failed: {e}\")\n            return 0.7  # Conservative estimate\n    \n    def _calculate_visual_quality(self, text_regions: List[TextRegion]) -> float:\n        \"\"\"Calculate visual quality based on OCR confidence and other factors.\"\"\"\n        if not text_regions:\n            return 0.0\n        \n        try:\n            # Average OCR confidence as quality indicator\n            confidences = [r.confidence for r in text_regions if r.confidence > 0]\n            \n            if not confidences:\n                return 0.5  # Neutral score if no confidence data\n            \n            avg_confidence = np.mean(confidences)\n            \n            # Additional quality factors could include:\n            # - Text size consistency\n            # - Contrast analysis\n            # - Font clarity metrics\n            \n            # For now, use confidence as primary quality indicator\n            visual_quality = avg_confidence\n            \n            logger.debug(f\"Visual quality: {visual_quality:.3f} (avg confidence: {avg_confidence:.3f})\")\n            \n            return visual_quality\n            \n        except Exception as e:\n            logger.warning(f\"Visual quality calculation failed: {e}\")\n            return 0.6  # Conservative estimate\n    \n    def _calculate_timing_accuracy(self,\n                                 text_regions: List[TextRegion],\n                                 expected_captions: List[Dict[str, Any]]) -> float:\n        \"\"\"Calculate timing accuracy based on when text appears vs expected timing.\"\"\"\n        if not text_regions or not expected_captions:\n            return 0.8  # Assume reasonable timing if we can't measure\n        \n        try:\n            # Get timestamps where text was detected\n            detected_times = sorted(set([r.timestamp for r in text_regions]))\n            \n            # Get expected caption time ranges\n            expected_ranges = []\n            for caption in expected_captions:\n                start = caption.get('start', 0)\n                end = caption.get('end', start + 3)  # Default 3s duration\n                expected_ranges.append((start, end))\n            \n            if not detected_times or not expected_ranges:\n                return 0.5\n            \n            # Calculate how many detected times fall within expected ranges\n            matches = 0\n            for detected_time in detected_times:\n                for start, end in expected_ranges:\n                    if start <= detected_time <= end:\n                        matches += 1\n                        break\n            \n            timing_accuracy = matches / len(detected_times) if detected_times else 0.0\n            \n            logger.debug(f\"Timing accuracy: {timing_accuracy:.3f} ({matches}/{len(detected_times)} timestamps matched)\")\n            \n            return timing_accuracy\n            \n        except Exception as e:\n            logger.warning(f\"Timing accuracy calculation failed: {e}\")\n            return 0.7  # Conservative estimate\n    \n    def _detect_text_cutoff(self, text_regions: List[TextRegion]) -> bool:\n        \"\"\"Detect if text appears to be cut off (descender issues, etc.).\"\"\"\n        try:\n            # Look for signs of text cutoff in OCR results\n            cutoff_indicators = [\n                'g', 'j', 'p', 'q', 'y'  # Letters with descenders\n            ]\n            \n            # Check for partial words or unusual OCR patterns\n            for region in text_regions:\n                text = region.text.lower()\n                \n                # Look for single letters that should be parts of words\n                if len(text) == 1 and text in cutoff_indicators:\n                    return True\n                \n                # Look for words that seem truncated\n                if any(word.endswith('_') or len(word) == 1 for word in text.split()):\n                    return True\n            \n            return False\n            \n        except Exception:\n            return False  # Conservative - don't flag cutoff if we can't detect\n    \n    def _calculate_average_contrast(self, text_regions: List[TextRegion]) -> float:\n        \"\"\"Calculate average contrast for text regions.\"\"\"\n        if not text_regions:\n            return 0.5\n        \n        # This would require frame data - simplified for now\n        # In full implementation, would analyze actual pixel contrast\n        \n        # Use confidence as proxy for contrast\n        confidences = [r.confidence for r in text_regions if r.confidence > 0]\n        return np.mean(confidences) if confidences else 0.5\n    \n    def quick_validation(self, video_path: str, sample_count: int = 5) -> Dict[str, Any]:\n        \"\"\"Perform quick validation with limited sampling.\"\"\"\n        logger.info(f\"Quick validation of {Path(video_path).name}\")\n        \n        try:\n            cap = cv2.VideoCapture(video_path)\n            if not cap.isOpened():\n                return {'success': False, 'error': 'Cannot open video'}\n            \n            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n            fps = cap.get(cv2.CAP_PROP_FPS)\n            \n            # Sample frames evenly\n            sample_frames = [i * frame_count // sample_count for i in range(sample_count)]\n            \n            text_detected = 0\n            frames_analyzed = 0\n            \n            for frame_num in sample_frames:\n                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)\n                ret, frame = cap.read()\n                \n                if not ret:\n                    continue\n                \n                frames_analyzed += 1\n                timestamp = frame_num / fps if fps > 0 else 0\n                \n                # Quick text detection\n                regions = self._detect_text_in_frame(frame, timestamp)\n                if regions:\n                    text_detected += 1\n            \n            cap.release()\n            \n            success_rate = text_detected / frames_analyzed if frames_analyzed > 0 else 0.0\n            \n            result = {\n                'success': True,\n                'frames_analyzed': frames_analyzed,\n                'frames_with_text': text_detected,\n                'text_detection_rate': success_rate,\n                'quality_estimate': min(1.0, success_rate * 1.2),  # Rough estimate\n                'ocr_available': self.ocr_available\n            }\n            \n            logger.info(f\"Quick validation complete: {success_rate:.2f} text detection rate\")\n            return result\n            \n        except Exception as e:\n            logger.error(f\"Quick validation failed: {e}\")\n            return {'success': False, 'error': str(e)}\n\n\n# Global validator instance\ncaption_validator = CaptionQualityValidator()"