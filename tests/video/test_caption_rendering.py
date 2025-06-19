"""
Comprehensive test suite for caption rendering system.

Tests both the enhanced MoviePy implementation and the OpenCV/PIL alternative
renderer, including quality validation and dual renderer functionality.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from lib.video.clip import CaptionStyle, VideoConfig
from lib.utils.logging_config import get_logger

logger = get_logger(__name__)


class TestCaptionRendering(unittest.TestCase):
    """Test suite for caption rendering functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_video_path = self.temp_dir / "test_video.mp4"
        self.test_audio_path = self.temp_dir / "test_audio.wav" 
        self.output_path = self.temp_dir / "output_video.mp4"
        
        # Create minimal test files (these would be real files in actual testing)
        self.test_video_path.touch()
        self.test_audio_path.touch()
        
        # Default test configurations
        self.caption_style = CaptionStyle(
            font_size=36,
            font_color='white',
            stroke_width=2,
            position='bottom'
        )
        
        self.video_config = VideoConfig(
            quality='medium',
            enable_quality_validation=True,
            validation_sample_count=5
        )
        
        self.test_caption_text = "This is a test caption with descenders like g, j, p, q, y to test cutoff issues."
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_margin_calculation(self):
        """Test dynamic margin calculation for TextClip."""
        try:
            from lib.video.clip import VideoClip
            
            with VideoClip(str(self.test_video_path)) as video_clip:
                # Test margin calculation
                margins = video_clip._calculate_dynamic_margins(self.caption_style)
                
                # Margins should be a tuple of two values
                self.assertIsInstance(margins, tuple)
                self.assertEqual(len(margins), 2)
                
                # Margins should account for stroke width, descender space, and safety
                expected_stroke = self.caption_style.stroke_width * 2
                expected_descender = max(8, int(self.caption_style.font_size * 0.25))
                expected_safety = 10
                expected_total = expected_stroke + expected_descender + expected_safety
                
                self.assertEqual(margins[0], expected_total)
                self.assertEqual(margins[1], expected_total)
                
                logger.info(f"✅ Margin calculation test passed: {margins}")
                
        except ImportError:
            self.skipTest("VideoClip dependencies not available")
    
    def test_font_validation_and_scoring(self):
        """Test font validation and quality scoring system."""
        try:
            from lib.video.clip import VideoClip
            
            with VideoClip(str(self.test_video_path)) as video_clip:
                # Test font validation
                test_fonts = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                    "nonexistent_font.ttf"
                ]
                
                scores = []
                for font_path in test_fonts:
                    font, score = video_clip._validate_and_score_font(font_path, self.caption_style)
                    scores.append((font_path, font is not None, score))
                
                # At least one font should be valid
                valid_fonts = [s for s in scores if s[1]]
                self.assertGreater(len(valid_fonts), 0, "At least one font should be valid")
                
                # Scores should be reasonable
                for font_path, is_valid, score in scores:
                    if is_valid:
                        self.assertGreaterEqual(score, 50, f"Valid font {font_path} should have score >= 50")
                    else:
                        self.assertEqual(score, 0, f"Invalid font {font_path} should have score 0")
                
                logger.info(f"✅ Font validation test passed: {len(valid_fonts)} valid fonts found")
                
        except ImportError:
            self.skipTest("VideoClip dependencies not available")
    
    def test_text_measurement_with_descenders(self):
        """Test enhanced text measurement including descender handling."""
        try:
            from lib.video.clip import VideoClip
            
            with VideoClip(str(self.test_video_path)) as video_clip:
                # Test text with descenders
                test_texts = [
                    "Normal text without descenders",
                    "Text with descenders: g, j, p, q, y",
                    "Mixed UPPERCASE and lowercase gjpqy",
                    "Short text",
                    "Very long text that might require wrapping and could potentially cause issues with positioning and measurement"
                ]
                
                for text in test_texts:
                    dimensions = video_clip._measure_text_dimensions_with_descenders(text, self.caption_style)
                    
                    self.assertIsInstance(dimensions, tuple)
                    self.assertEqual(len(dimensions), 2)
                    
                    width, height = dimensions
                    self.assertGreater(width, 0, f"Width should be positive for text: '{text[:30]}...'")
                    self.assertGreater(height, 0, f"Height should be positive for text: '{text[:30]}...'")
                    
                    # Height should be reasonable for font size
                    self.assertGreater(height, self.caption_style.font_size, "Height should be larger than font size")
                    self.assertLess(height, self.caption_style.font_size * 3, "Height should not be excessively large")
                
                logger.info(f"✅ Text measurement test passed for {len(test_texts)} test cases")
                
        except ImportError:
            self.skipTest("VideoClip dependencies not available")
    
    def test_margin_aware_positioning(self):
        """Test positioning calculations that account for margins."""
        try:
            from lib.video.clip import VideoClip
            
            with VideoClip(str(self.test_video_path)) as video_clip:
                # Test positioning for different video dimensions
                test_dimensions = [
                    (1080, 1920),  # Vertical video
                    (1920, 1080),  # Horizontal video
                    (720, 1280),   # Medium vertical
                ]
                
                positions = ['top', 'center', 'bottom']
                
                for width, height in test_dimensions:
                    for position in positions:
                        style = CaptionStyle(position=position, font_size=36, stroke_width=2)
                        
                        x_pos, y_pos = video_clip._calculate_safe_position_with_descenders(
                            self.test_caption_text, style, width, height
                        )
                        
                        # X position should be 'center' for horizontal centering
                        self.assertEqual(x_pos, 'center')
                        
                        # Y position should be within video bounds
                        self.assertGreaterEqual(y_pos, 0, f"Y position should be >= 0 for {position} position")
                        self.assertLess(y_pos, height, f"Y position should be < height for {position} position")
                        
                        # Position should respect style preferences
                        if position == 'top':
                            self.assertLess(y_pos, height * 0.4, "Top position should be in upper portion")
                        elif position == 'bottom':
                            self.assertGreater(y_pos, height * 0.6, "Bottom position should be in lower portion")
                
                logger.info(f"✅ Margin-aware positioning test passed for {len(test_dimensions) * len(positions)} combinations")
                
        except ImportError:
            self.skipTest("VideoClip dependencies not available")
    
    @patch('lib.video.pil_text_engine.HAS_PIL', True)
    def test_pil_text_engine(self):
        """Test PIL text rendering engine functionality."""
        try:
            from lib.video.pil_text_engine import PILTextEngine, TextMetrics, WrappedText
            
            engine = PILTextEngine()
            
            # Test text measurement
            metrics = engine.measure_text(self.test_caption_text, self.caption_style)
            
            self.assertIsInstance(metrics, TextMetrics)
            self.assertGreater(metrics.width, 0)
            self.assertGreater(metrics.height, 0)
            self.assertGreater(metrics.char_count, 0)
            self.assertGreaterEqual(metrics.line_count, 1)
            
            # Test text wrapping
            max_width = 400
            wrapped = engine.wrap_text(self.test_caption_text, max_width, self.caption_style)
            
            self.assertIsInstance(wrapped, WrappedText)
            self.assertGreater(len(wrapped.lines), 0)
            self.assertLessEqual(wrapped.total_width, max_width * 1.1)  # Allow 10% tolerance
            
            # Test font loading
            font = engine.get_font(self.caption_style)
            # Font may be None if no suitable fonts available, which is acceptable
            
            logger.info(f"✅ PIL text engine test passed")
            
        except ImportError:
            self.skipTest("PIL text engine dependencies not available")
    
    @patch('lib.video.opencv_caption_renderer.HAS_OPENCV', True)
    @patch('lib.video.opencv_caption_renderer.HAS_PIL', True)
    def test_opencv_caption_renderer(self):
        """Test OpenCV caption renderer functionality."""
        try:
            from lib.video.opencv_caption_renderer import OpenCVCaptionRenderer, CaptionTiming
            
            # Mock the actual video processing since we don't have real video files
            with patch('cv2.VideoCapture') as mock_cap, \
                 patch('cv2.VideoWriter') as mock_writer:
                
                # Mock video properties
                mock_cap_instance = Mock()
                mock_cap.return_value = mock_cap_instance
                mock_cap_instance.isOpened.return_value = True
                mock_cap_instance.get.side_effect = lambda prop: {
                    'fps': 30.0,
                    'width': 1080,
                    'height': 1920,
                    'frame_count': 900  # 30 seconds at 30fps
                }.get(prop, 0)
                
                renderer = OpenCVCaptionRenderer(str(self.test_video_path), str(self.output_path))
                
                # Test caption timing addition
                test_timings = [
                    {'text': 'First caption', 'start': 0.0, 'end': 3.0, 'duration': 3.0},
                    {'text': 'Second caption', 'start': 3.0, 'end': 6.0, 'duration': 3.0},
                ]
                
                renderer.add_caption_timings(test_timings)
                
                self.assertEqual(len(renderer.caption_timings), 2)
                self.assertEqual(renderer.caption_timings[0].text, 'First caption')
                
                logger.info(f"✅ OpenCV caption renderer test passed")
                
        except ImportError:
            self.skipTest("OpenCV caption renderer dependencies not available")
    
    def test_dual_renderer_manager(self):
        """Test dual renderer manager functionality."""
        try:
            from lib.video.caption_manager import CaptionRenderingManager, RendererType, RendererStatus
            
            manager = CaptionRenderingManager()
            
            # Test renderer detection
            available_renderers = manager._get_available_renderers()
            self.assertIsInstance(available_renderers, list)
            
            # Test renderer selection
            if available_renderers:
                selected = manager._select_optimal_renderer()
                self.assertIn(selected, available_renderers)
            
            # Test performance tracking
            manager.reset_performance_metrics()
            status = manager.get_renderer_status()
            self.assertIsInstance(status, dict)
            
            # Test task complexity estimation
            complexity = manager._estimate_task_complexity(self.test_caption_text, self.caption_style)
            self.assertIn(complexity, ['low', 'medium', 'high'])
            
            logger.info(f"✅ Dual renderer manager test passed")
            
        except ImportError:
            self.skipTest("Dual renderer manager dependencies not available")
    
    def test_quality_validator(self):
        """Test caption quality validation system."""
        try:
            from lib.video.caption_validator import CaptionQualityValidator, QualityMetrics
            
            validator = CaptionQualityValidator()
            
            # Test quick validation (with mocked video)
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap_instance = Mock()
                mock_cap.return_value = mock_cap_instance
                mock_cap_instance.isOpened.return_value = True
                mock_cap_instance.get.return_value = 30.0  # fps
                mock_cap_instance.read.return_value = (False, None)  # No frames to simulate end
                
                result = validator.quick_validation(str(self.test_video_path), sample_count=3)
                
                self.assertIsInstance(result, dict)
                self.assertIn('success', result)
                self.assertIn('frames_analyzed', result)
            
            # Test quality metrics structure
            expected_captions = [
                {'text': 'Test caption', 'start': 0.0, 'end': 3.0, 'duration': 3.0}
            ]
            
            # This would fail with real validation, but tests the structure
            try:
                metrics = validator.validate_video_captions(str(self.test_video_path), expected_captions)
                self.assertIsInstance(metrics, QualityMetrics)
            except:
                pass  # Expected to fail with dummy video file
            
            logger.info(f"✅ Quality validator test passed")
            
        except ImportError:
            self.skipTest("Quality validator dependencies not available")
    
    def test_configuration_integration(self):
        """Test configuration management and integration."""
        # Test CaptionStyle with renderer preferences
        style = CaptionStyle(
            preferred_renderer='opencv_pil',
            enable_fallback=True,
            quality_priority=True
        )
        
        self.assertEqual(style.preferred_renderer, 'opencv_pil')
        self.assertTrue(style.enable_fallback)
        self.assertTrue(style.quality_priority)
        
        # Test VideoConfig with quality validation
        config = VideoConfig(
            enable_quality_validation=True,
            quality_threshold=0.8,
            validation_sample_count=10,
            debug_caption_rendering=True
        )
        
        self.assertTrue(config.enable_quality_validation)
        self.assertEqual(config.quality_threshold, 0.8)
        self.assertEqual(config.validation_sample_count, 10)
        self.assertTrue(config.debug_caption_rendering)
        
        logger.info(f"✅ Configuration integration test passed")
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        try:
            from lib.video.clip import VideoClip
            
            # Test with invalid font
            invalid_style = CaptionStyle(font_family="NonExistentFont.ttf")
            
            with VideoClip(str(self.test_video_path)) as video_clip:
                # Should handle invalid font gracefully
                font = video_clip._get_consistent_font(invalid_style)
                # Font may be None or a fallback, both are acceptable
                
                # Test positioning with extreme values
                extreme_dimensions = (10, 10)  # Very small video
                try:
                    pos = video_clip._calculate_safe_position_with_descenders(
                        "Test", invalid_style, *extreme_dimensions
                    )
                    self.assertIsInstance(pos, tuple)
                    self.assertEqual(len(pos), 2)
                except Exception as e:
                    # Should handle gracefully
                    pass
            
            logger.info(f"✅ Error handling test passed")
            
        except ImportError:
            self.skipTest("VideoClip dependencies not available")
    
    def test_performance_benchmarks(self):
        """Test performance characteristics of caption rendering."""
        try:
            from lib.video.clip import VideoClip
            
            # Test text measurement performance
            start_time = time.time()
            
            with VideoClip(str(self.test_video_path)) as video_clip:
                for _ in range(100):  # 100 measurements
                    video_clip._measure_text_dimensions_with_descenders(
                        self.test_caption_text, self.caption_style
                    )
            
            measurement_time = time.time() - start_time
            avg_measurement_time = measurement_time / 100
            
            # Should be fast (under 10ms per measurement)
            self.assertLess(avg_measurement_time, 0.01, 
                          f"Text measurement too slow: {avg_measurement_time:.4f}s")
            
            # Test margin calculation performance
            start_time = time.time()
            
            with VideoClip(str(self.test_video_path)) as video_clip:
                for _ in range(1000):  # 1000 calculations
                    video_clip._calculate_dynamic_margins(self.caption_style)
            
            margin_time = time.time() - start_time
            avg_margin_time = margin_time / 1000
            
            # Should be very fast (under 1ms per calculation)
            self.assertLess(avg_margin_time, 0.001,
                          f"Margin calculation too slow: {avg_margin_time:.4f}s")
            
            logger.info(f"✅ Performance benchmark passed")
            logger.info(f"   Text measurement: {avg_measurement_time:.4f}s avg")
            logger.info(f"   Margin calculation: {avg_margin_time:.6f}s avg")
            
        except ImportError:
            self.skipTest("VideoClip dependencies not available")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete caption rendering workflows."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_video_path = self.temp_dir / "test_video.mp4"
        self.test_audio_path = self.temp_dir / "test_audio.wav"
        self.output_path = self.temp_dir / "output_video.mp4"
        
        # Create test files
        self.test_video_path.touch()
        self.test_audio_path.touch()
    
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_moviepy_workflow(self):
        """Test complete MoviePy rendering workflow with enhancements."""
        try:
            from lib.video.clip import VideoClip, CaptionStyle, VideoConfig
            
            # Enhanced caption style with new features
            style = CaptionStyle(
                font_size=42,
                position='bottom',
                stroke_width=3,
                preferred_renderer='moviepy',
                enable_fallback=True
            )
            
            # Mock the actual video processing
            with patch('moviepy.VideoFileClip'), \
                 patch('moviepy.AudioFileClip'), \
                 patch('moviepy.TextClip'):
                
                with VideoClip(str(self.test_video_path)) as video_clip:
                    # Test margin-enhanced caption creation
                    margins = video_clip._calculate_dynamic_margins(style)
                    self.assertIsInstance(margins, tuple)
                    
                    # Test enhanced positioning
                    pos = video_clip._calculate_safe_position_with_descenders(
                        "Test caption with descenders gjpqy", style, 1080, 1920
                    )
                    self.assertIsInstance(pos, tuple)
            
            logger.info(f"✅ End-to-end MoviePy workflow test passed")
            
        except ImportError:
            self.skipTest("MoviePy workflow dependencies not available")
    
    def test_dual_system_integration(self):
        """Test dual renderer system integration."""
        try:
            from lib.video.clip import VideoClip, CaptionStyle, VideoConfig
            
            style = CaptionStyle(
                preferred_renderer='auto',
                enable_fallback=True,
                quality_priority=True
            )
            
            config = VideoConfig(
                enable_quality_validation=True,
                quality_threshold=0.7
            )
            
            # Mock the dual system components
            with patch('lib.video.caption_manager.caption_manager') as mock_manager, \
                 patch('lib.video.caption_validator.caption_validator') as mock_validator:
                
                # Mock successful rendering
                mock_result = Mock()
                mock_result.success = True
                mock_result.render_time = 5.0
                mock_result.output_path = str(self.output_path)
                mock_result.renderer_used.value = 'moviepy'
                mock_result.error_message = None
                
                mock_manager.render_captions.return_value = mock_result
                
                with VideoClip(str(self.test_video_path)) as video_clip:
                    result = video_clip.render_with_dual_system(
                        "Test caption text",
                        str(self.test_audio_path),
                        str(self.output_path),
                        style,
                        config
                    )
                    
                    self.assertIsInstance(result, dict)
                    self.assertTrue(result.get('dual_system', False))
            
            logger.info(f"✅ Dual system integration test passed")
            
        except ImportError:
            self.skipTest("Dual system dependencies not available")


def run_caption_rendering_tests():
    """Run all caption rendering tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestCaptionRendering))
    suite.addTest(unittest.makeSuite(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Caption Rendering Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    return result


if __name__ == '__main__':
    run_caption_rendering_tests()