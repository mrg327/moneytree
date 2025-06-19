"""
Video frame processor for handling video processing with caption overlays.

This module provides frame-by-frame video processing capabilities with
audio synchronization preservation and progress tracking.
"""

import cv2
import numpy as np
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import time

from lib.utils.logging_config import get_logger
from lib.video.clip import CaptionStyle
from lib.video.pil_text_engine import pil_text_engine, WrappedText

logger = get_logger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.error("OpenCV not available - frame processor will not work")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.error("PIL not available - frame processor will not work")


@dataclass
class VideoInfo:
    """Video file information."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    has_audio: bool


@dataclass
class ProcessingStats:
    """Video processing statistics."""
    frames_processed: int
    processing_time: float
    fps_average: float
    errors_count: int
    warnings_count: int


class VideoFrameProcessor:
    """
    Frame-by-frame video processor with audio preservation.
    
    Handles video processing while maintaining audio synchronization
    and providing detailed progress tracking.
    """
    
    def __init__(self, input_video_path: str, output_video_path: str):
        """
        Initialize the frame processor.
        
        Args:
            input_video_path: Path to input video file
            output_video_path: Path to output video file
        """
        if not HAS_OPENCV or not HAS_PIL:
            raise ImportError("OpenCV and PIL are required for frame processing")
        
        self.input_path = Path(input_video_path)
        self.output_path = Path(output_video_path)
        
        # Video objects
        self.video_cap = None
        self.video_writer = None
        
        # Video info
        self.video_info = None
        
        # Processing state
        self.current_frame = 0
        self.current_time = 0.0
        self.processing_stats = ProcessingStats(0, 0.0, 0.0, 0, 0)
        
        # Temporary files for audio handling
        self.temp_dir = None
        self.temp_video_path = None
        self.temp_audio_path = None
        
        self._load_video_info()
    
    def _load_video_info(self) -> VideoInfo:
        """Load and analyze video file information."""
        try:
            cap = cv2.VideoCapture(str(self.input_path))
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {self.input_path}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            
            # Get codec info
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            # Check for audio using ffprobe
            has_audio = self._check_audio_stream()
            
            self.video_info = VideoInfo(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                codec=codec.strip(),
                has_audio=has_audio
            )
            
            logger.info(f"Video info: {width}x{height} @ {fps:.2f}fps, {duration:.2f}s")
            logger.info(f"Codec: {codec}, Audio: {has_audio}, Frames: {frame_count}")
            
            return self.video_info
            
        except Exception as e:
            logger.error(f"Failed to load video info: {e}")
            raise
    
    def _check_audio_stream(self) -> bool:
        """Check if video has audio stream using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_streams', 
                '-select_streams', 'a', '-of', 'csv=p=0',
                str(self.input_path)
            ], capture_output=True, text=True, timeout=10)
            
            return bool(result.stdout.strip())
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not check audio stream - assuming no audio")
            return False
    
    def _setup_temp_files(self):
        """Setup temporary files for processing."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='moneytree_video_'))
        self.temp_video_path = self.temp_dir / 'temp_video.mp4'
        self.temp_audio_path = self.temp_dir / 'temp_audio.aac'
        
        logger.debug(f"Temp directory: {self.temp_dir}")
    
    def _extract_audio(self) -> bool:
        """Extract audio from input video."""
        if not self.video_info.has_audio:
            logger.info("No audio stream to extract")
            return True
        
        try:
            cmd = [
                'ffmpeg', '-y', '-i', str(self.input_path),
                '-vn', '-acodec', 'aac', '-ab', '128k',
                str(self.temp_audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("Audio extracted successfully")
                return True
            else:
                logger.warning(f"Audio extraction failed: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.warning(f"Audio extraction error: {e}")
            return False
    
    def _merge_audio_video(self) -> bool:
        """Merge processed video with original audio."""
        if not self.video_info.has_audio or not self.temp_audio_path.exists():
            # Just copy video file if no audio
            shutil.copy2(self.temp_video_path, self.output_path)
            logger.info("Video copied without audio")
            return True
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.temp_video_path),  # Video input
                '-i', str(self.temp_audio_path),  # Audio input
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',   # Re-encode audio
                '-shortest',     # Match shortest stream
                str(self.output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("Audio and video merged successfully")
                return True
            else:
                logger.error(f"Audio/video merge failed: {result.stderr}")
                # Fallback to video-only
                shutil.copy2(self.temp_video_path, self.output_path)
                return True
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"Audio/video merge error: {e}")
            # Fallback to video-only
            try:
                shutil.copy2(self.temp_video_path, self.output_path)
                return True
            except Exception as copy_error:
                logger.error(f"Fallback copy failed: {copy_error}")
                return False
    
    def process_video_with_captions(
        self,
        caption_timings: List[Dict[str, Any]],
        caption_style: CaptionStyle,
        progress_callback: Optional[Callable[[float], None]] = None,
        frame_callback: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    ) -> bool:
        """
        Process video with caption overlays and custom frame processing.
        
        Args:
            caption_timings: List of caption timing data
            caption_style: Caption styling configuration
            progress_callback: Optional progress callback (receives progress 0.0-1.0)
            frame_callback: Optional custom frame processing callback
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting video processing with captions")
            start_time = time.time()
            
            # Setup temporary files
            self._setup_temp_files()
            
            # Extract audio if present
            if not self._extract_audio():
                logger.warning("Audio extraction failed, continuing without audio")
            
            # Setup video capture and writer
            if not self._setup_video_processing():
                return False
            
            # Process frames
            success = self._process_frames(
                caption_timings, caption_style, progress_callback, frame_callback
            )
            
            if success:
                # Merge audio and video
                success = self._merge_audio_video()
            
            # Calculate final stats
            processing_time = time.time() - start_time
            self.processing_stats.processing_time = processing_time
            if self.processing_stats.frames_processed > 0:
                self.processing_stats.fps_average = self.processing_stats.frames_processed / processing_time
            
            logger.info(f"Processing complete: {self.processing_stats.frames_processed} frames in {processing_time:.2f}s")
            logger.info(f"Average FPS: {self.processing_stats.fps_average:.2f}")
            
            return success
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return False
        
        finally:
            self._cleanup()
    
    def _setup_video_processing(self) -> bool:
        """Setup video capture and writer."""
        try:
            # Open video capture
            self.video_cap = cv2.VideoCapture(str(self.input_path))
            if not self.video_cap.isOpened():
                raise ValueError("Cannot open input video")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.temp_video_path),
                fourcc,
                self.video_info.fps,
                (self.video_info.width, self.video_info.height)
            )
            
            if not self.video_writer.isOpened():
                raise ValueError("Cannot open video writer")
            
            logger.debug("Video processing setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Video setup failed: {e}")
            return False
    
    def _process_frames(
        self,
        caption_timings: List[Dict[str, Any]],
        caption_style: CaptionStyle,
        progress_callback: Optional[Callable[[float], None]],
        frame_callback: Optional[Callable[[np.ndarray, float], np.ndarray]]
    ) -> bool:
        """Process all video frames with captions."""
        try:
            frame_count = 0
            last_progress_report = 0
            
            while True:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                # Calculate current time
                current_time = frame_count / self.video_info.fps
                
                # Apply custom frame processing if provided
                if frame_callback:
                    try:
                        frame = frame_callback(frame, current_time)
                    except Exception as e:
                        logger.warning(f"Frame callback error: {e}")
                        self.processing_stats.warnings_count += 1
                
                # Apply captions
                frame = self._apply_captions_to_frame(frame, current_time, caption_timings, caption_style)
                
                # Write processed frame
                self.video_writer.write(frame)
                
                frame_count += 1
                self.processing_stats.frames_processed = frame_count
                
                # Progress reporting
                if progress_callback and frame_count - last_progress_report >= 30:  # Every ~1 second
                    progress = frame_count / self.video_info.frame_count
                    progress_callback(progress)
                    last_progress_report = frame_count
                
                # Periodic logging
                if frame_count % 300 == 0:  # Every ~10 seconds
                    progress = frame_count / self.video_info.frame_count * 100
                    elapsed = time.time() - self.processing_stats.processing_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{self.video_info.frame_count}) @ {current_fps:.1f}fps")
            
            logger.info(f"Frame processing complete: {frame_count} frames")
            return True
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            self.processing_stats.errors_count += 1
            return False
    
    def _apply_captions_to_frame(
        self,
        frame: np.ndarray,
        current_time: float,
        caption_timings: List[Dict[str, Any]],
        caption_style: CaptionStyle
    ) -> np.ndarray:
        """Apply captions to a single frame."""
        try:
            # Find active captions for current time
            active_captions = []
            for timing in caption_timings:
                if timing['start'] <= current_time <= timing['end']:
                    active_captions.append(timing['text'])
            
            if not active_captions:
                return frame
            
            # Convert frame to PIL for text rendering
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Render each active caption
            for caption_text in active_captions:
                pil_frame = self._render_caption_on_pil_frame(pil_frame, caption_text, caption_style)
            
            # Convert back to OpenCV format
            frame_with_captions = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            
            return frame_with_captions
            
        except Exception as e:
            logger.warning(f"Caption rendering error: {e}")
            self.processing_stats.warnings_count += 1
            return frame  # Return original frame on error
    
    def _render_caption_on_pil_frame(
        self,
        pil_frame: Image.Image,
        text: str,
        style: CaptionStyle
    ) -> Image.Image:
        """Render a single caption on PIL frame using the text engine."""
        try:
            # Calculate maximum text area
            frame_width, frame_height = pil_frame.size
            max_text_width = int(frame_width * 0.8)  # 80% of frame width
            max_text_height = int(frame_height * 0.3)  # 30% of frame height
            
            # Optimize text for readability
            optimized_style, wrapped_text = pil_text_engine.optimize_text_for_readability(
                text, style, max_text_width, max_text_height
            )
            
            # Calculate position based on style
            text_x = frame_width // 2  # Center horizontally
            
            if style.position == 'top':
                text_y = int(frame_height * 0.1)
            elif style.position == 'bottom':
                text_y = int(frame_height * 0.8)
            else:  # center
                text_y = frame_height // 2
            
            # Render text using PIL text engine
            pil_frame = pil_text_engine.render_text(
                pil_frame, text, (text_x, text_y), optimized_style, wrapped_text
            )
            
            return pil_frame
            
        except Exception as e:
            logger.warning(f"PIL caption rendering error: {e}")
            return pil_frame
    
    def _cleanup(self):
        """Clean up resources and temporary files."""
        try:
            # Close video objects
            if self.video_cap:
                self.video_cap.release()
            if self.video_writer:
                self.video_writer.release()
            
            cv2.destroyAllWindows()
            
            # Clean up temp files
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug("Temporary files cleaned up")
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def get_processing_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self.processing_stats
    
    def estimate_processing_time(self, sample_frames: int = 100) -> float:
        """
        Estimate total processing time by processing a sample of frames.
        
        Args:
            sample_frames: Number of frames to sample for estimation
            
        Returns:
            Estimated processing time in seconds
        """
        if not self.video_info:
            return 0.0
        
        try:
            # Create dummy caption for testing
            test_caption = [{'text': 'Test Caption', 'start': 0, 'end': 999, 'duration': 999}]
            test_style = CaptionStyle()
            
            # Setup video for sampling
            cap = cv2.VideoCapture(str(self.input_path))
            if not cap.isOpened():
                return 0.0
            
            # Sample frames at regular intervals
            sample_interval = max(1, self.video_info.frame_count // sample_frames)
            sample_times = []
            
            for i in range(0, min(sample_frames, self.video_info.frame_count), sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                start_time = time.time()
                current_time = i / self.video_info.fps
                
                # Apply caption processing (without writing)
                self._apply_captions_to_frame(frame, current_time, test_caption, test_style)
                
                sample_times.append(time.time() - start_time)
            
            cap.release()
            
            if sample_times:
                avg_frame_time = sum(sample_times) / len(sample_times)
                estimated_total = avg_frame_time * self.video_info.frame_count
                
                # Add overhead for I/O operations
                estimated_total *= 1.2
                
                logger.info(f"Processing time estimate: {estimated_total:.2f}s for {self.video_info.frame_count} frames")
                return estimated_total
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Processing time estimation failed: {e}")
            return 0.0


def process_video_with_frame_processor(
    input_video: str,
    output_video: str,
    caption_timings: List[Dict[str, Any]],
    caption_style: CaptionStyle,
    progress_callback: Optional[Callable[[float], None]] = None
) -> bool:
    """
    Convenience function for video processing with captions.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
        caption_timings: Caption timing data
        caption_style: Caption styling
        progress_callback: Optional progress callback
        
    Returns:
        True if successful, False otherwise
    """
    try:
        processor = VideoFrameProcessor(input_video, output_video)
        return processor.process_video_with_captions(
            caption_timings, caption_style, progress_callback
        )
    except Exception as e:
        logger.error(f"Frame processor convenience function failed: {e}")
        return False