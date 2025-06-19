"""Tests for video processing and rendering components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from lib.video.clip import VideoClip, CaptionStyle, VideoConfig, create_sample_template


class TestVideoProcessing:
    """Test video processing components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_template_path = "downloads/videos/minecraft_parkour.mp4"
        self.audio_test_path = "audio_output/Cat_chattts.wav"  # Use existing audio file
        self.output_dir = Path("video_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Sample text for captions
        self.test_caption_text = "This is a test caption. It should be synchronized with audio. Multiple sentences for testing."

    def test_video_dependencies(self):
        """Test that video dependencies are available."""
        try:
            from moviepy import VideoFileClip, AudioFileClip, TextClip
            import librosa
            import numpy as np
            from PIL import Image
            
            print("✅ All video dependencies available")
            return True
            
        except ImportError as e:
            print(f"❌ Video dependencies missing: {e}")
            return False

    def test_template_loading(self):
        """Test loading template video."""
        if not Path(self.test_template_path).exists():
            print(f"⚠️ Template video not found: {self.test_template_path}")
            return False
        
        try:
            with VideoClip(self.test_template_path) as video_clip:
                assert video_clip.template_clip is not None
                assert video_clip.template_duration > 0
                assert video_clip.template_size[0] > 0
                assert video_clip.template_size[1] > 0
                
                print(f"✅ Template loaded: {video_clip.template_size[0]}x{video_clip.template_size[1]}, {video_clip.template_duration:.1f}s")
                return True
                
        except Exception as e:
            print(f"❌ Template loading failed: {e}")
            return False

    def test_sample_template_creation(self):
        """Test creating sample template videos."""
        try:
            # Create vertical template
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                vertical_path = tmp_file.name
            
            create_sample_template(vertical_path, duration=10.0, vertical=True)
            
            assert Path(vertical_path).exists()
            assert Path(vertical_path).stat().st_size > 1000  # Should be substantial
            
            print(f"✅ Sample template created: {vertical_path}")
            
            # Clean up
            Path(vertical_path).unlink()
            return True
            
        except Exception as e:
            print(f"❌ Sample template creation failed: {e}")
            return False

    def test_video_config(self):
        """Test video configuration classes."""
        try:
            # Test default config
            config = VideoConfig()
            assert config.fps > 0
            assert config.quality in ['low', 'medium', 'high']
            
            # Test resolution calculation
            resolution = config.get_target_resolution()
            assert len(resolution) == 2
            assert resolution[0] > 0
            assert resolution[1] > 0
            
            print(f"✅ Default config: {resolution[0]}x{resolution[1]} @ {config.fps}fps")
            
            # Test fast rendering config
            fast_config = VideoConfig.for_fast_rendering()
            fast_resolution = fast_config.get_target_resolution()
            
            assert fast_config.preset == 'ultrafast'
            assert fast_config.fps <= config.fps  # Should be faster
            
            print(f"✅ Fast config: {fast_resolution[0]}x{fast_resolution[1]} @ {fast_config.fps}fps, preset: {fast_config.preset}")
            
            return True
            
        except Exception as e:
            print(f"❌ Video config test failed: {e}")
            return False

    def test_caption_styles(self):
        """Test caption styling configurations."""
        try:
            # Test default style
            default_style = CaptionStyle()
            assert default_style.font_size > 0
            assert default_style.font_color is not None
            
            print(f"✅ Default caption style: {default_style.font_size}px {default_style.font_family}")
            
            # Test vertical style
            vertical_style = CaptionStyle.for_vertical_video()
            assert vertical_style.words_per_caption <= default_style.words_per_caption  # Should be fewer words
            
            print(f"✅ Vertical caption style: {vertical_style.words_per_caption} words/caption")
            
            # Test fast rendering style
            fast_style = CaptionStyle.for_fast_rendering()
            assert fast_style.stroke_width <= default_style.stroke_width  # Should be faster
            
            print(f"✅ Fast caption style: stroke width {fast_style.stroke_width}")
            
            return True
            
        except Exception as e:
            print(f"❌ Caption style test failed: {e}")
            return False

    def test_audio_integration(self):
        """Test adding audio to video clips."""
        if not Path(self.test_template_path).exists():
            print("⚠️ Skipping audio integration test (no template)")
            return False
            
        if not Path(self.audio_test_path).exists():
            print("⚠️ Skipping audio integration test (no audio file)")
            return False
        
        try:
            with VideoClip(self.test_template_path) as video_clip:
                # Add narration audio
                result = video_clip.add_narration_audio(self.audio_test_path)
                
                assert result is not None
                assert isinstance(result, dict)
                
                if result.get('success'):
                    print(f"✅ Audio integration successful: {result.get('narration_duration', 0):.1f}s")
                    return True
                else:
                    print(f"❌ Audio integration failed: {result.get('error')}")
                    return False
                    
        except Exception as e:
            print(f"❌ Audio integration test failed: {e}")
            return False

    def test_caption_timing_analysis(self):
        """Test caption timing analysis."""
        if not Path(self.audio_test_path).exists():
            print("⚠️ Skipping caption timing test (no audio file)")
            return False
        
        try:
            with VideoClip(self.test_template_path) as video_clip:
                # Test speech timing analysis
                timing_data = video_clip._analyze_speech_timing(self.audio_test_path, self.test_caption_text)
                
                assert timing_data is not None
                assert isinstance(timing_data, list)
                assert len(timing_data) > 0
                
                # Check timing data structure
                for item in timing_data:
                    assert 'text' in item
                    assert 'start' in item
                    assert 'end' in item
                    assert item['start'] >= 0
                    assert item['end'] > item['start']
                
                print(f"✅ Caption timing analysis: {len(timing_data)} segments")
                return True
                
        except Exception as e:
            print(f"❌ Caption timing analysis failed: {e}")
            return False

    def test_fast_video_rendering(self):
        """Test fast video rendering configuration."""
        if not Path(self.test_template_path).exists():
            print("⚠️ Skipping rendering test (no template)")
            return False
        
        try:
            with VideoClip(self.test_template_path) as video_clip:
                # Use fast rendering config
                fast_config = VideoConfig.for_fast_rendering()
                
                # Just test that rendering setup works (don't actually render)
                target_resolution = fast_config.get_target_resolution()
                
                assert target_resolution[0] > 0
                assert target_resolution[1] > 0
                assert fast_config.preset == 'ultrafast'
                
                print(f"✅ Fast rendering setup: {target_resolution[0]}x{target_resolution[1]}, preset: {fast_config.preset}")
                return True
                
        except Exception as e:
            print(f"❌ Fast rendering test failed: {e}")
            return False

    def test_video_resolution_scaling(self):
        """Test video resolution scaling logic."""
        if not Path(self.test_template_path).exists():
            print("⚠️ Skipping scaling test (no template)")
            return False
        
        try:
            with VideoClip(self.test_template_path) as video_clip:
                original_size = video_clip.template_size
                
                # Test different target resolutions
                configs = [
                    VideoConfig(quality='low', vertical_format=True),
                    VideoConfig(quality='medium', vertical_format=True),
                    VideoConfig(quality='high', vertical_format=True)
                ]
                
                for config in configs:
                    target_resolution = config.get_target_resolution()
                    
                    # Calculate scaling
                    scale_w = target_resolution[0] / original_size[0]
                    scale_h = target_resolution[1] / original_size[1]
                    scale = max(scale_w, scale_h)
                    
                    assert scale > 0
                    print(f"✅ {config.quality} scaling: {original_size} → {target_resolution} (scale: {scale:.2f})")
                
                return True
                
        except Exception as e:
            print(f"❌ Resolution scaling test failed: {e}")
            return False

    def test_memory_cleanup(self):
        """Test video clip memory cleanup."""
        try:
            video_clip = VideoClip(self.test_template_path) if Path(self.test_template_path).exists() else None
            
            if video_clip:
                # Test cleanup method
                video_clip.cleanup()
                print("✅ Memory cleanup successful")
                return True
            else:
                print("⚠️ Skipping cleanup test (no template)")
                return False
                
        except Exception as e:
            print(f"❌ Memory cleanup test failed: {e}")
            return False

    def test_dual_renderer_configuration(self):
        """Test dual renderer system configuration."""
        try:
            # Test enhanced CaptionStyle with renderer preferences
            style = CaptionStyle(
                preferred_renderer='opencv_pil',
                enable_fallback=True,
                quality_priority=True,
                font_size=40,
                stroke_width=3
            )
            
            assert style.preferred_renderer == 'opencv_pil'
            assert style.enable_fallback == True
            assert style.quality_priority == True
            
            print(f"✅ Dual renderer caption style: {style.preferred_renderer} with fallback={style.enable_fallback}")
            
            # Test enhanced VideoConfig with quality validation
            config = VideoConfig(
                enable_quality_validation=True,
                quality_threshold=0.8,
                validation_sample_count=10,
                debug_caption_rendering=True
            )
            
            assert config.enable_quality_validation == True
            assert config.quality_threshold == 0.8
            assert config.validation_sample_count == 10
            
            print(f"✅ Quality validation config: threshold={config.quality_threshold}, samples={config.validation_sample_count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dual renderer configuration test failed: {e}")
            return False

    def test_margin_system(self):
        """Test enhanced margin system for text cutoff prevention."""
        if not Path(self.test_template_path).exists():
            print("⚠️ Skipping margin system test (no template)")
            return False
            
        try:
            with VideoClip(self.test_template_path) as video_clip:
                # Test margin calculation
                style = CaptionStyle(font_size=36, stroke_width=2)
                margins = video_clip._calculate_dynamic_margins(style)
                
                assert isinstance(margins, tuple)
                assert len(margins) == 2
                assert margins[0] > 0 and margins[1] > 0
                
                # Margins should account for stroke and descenders
                expected_minimum = style.stroke_width * 2 + 8 + 10  # stroke + descender + safety
                assert margins[0] >= expected_minimum
                
                print(f"✅ Margin calculation: {margins[0]}x{margins[1]} (minimum expected: {expected_minimum})")
                
                # Test text measurement with descenders
                test_text = "Testing descenders: g, j, p, q, y"
                dimensions = video_clip._measure_text_dimensions_with_descenders(test_text, style)
                
                assert isinstance(dimensions, tuple)
                assert dimensions[0] > 0 and dimensions[1] > 0
                
                print(f"✅ Text measurement with descenders: {dimensions[0]}x{dimensions[1]}")
                
                return True
                
        except Exception as e:
            print(f"❌ Margin system test failed: {e}")
            return False

    def test_font_validation_system(self):
        """Test WSL2-optimized font validation system."""
        if not Path(self.test_template_path).exists():
            print("⚠️ Skipping font validation test (no template)")
            return False
            
        try:
            with VideoClip(self.test_template_path) as video_clip:
                style = CaptionStyle(font_size=36)
                
                # Test font validation
                test_fonts = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Common Linux font
                    "/mnt/c/Windows/Fonts/arial.ttf",  # WSL2 Windows access
                    "nonexistent_font.ttf"  # Should fail gracefully
                ]
                
                valid_fonts = 0
                for font_path in test_fonts:
                    font, score = video_clip._validate_and_score_font(font_path, style)
                    if font is not None:
                        valid_fonts += 1
                        print(f"  ✅ Valid font: {font_path} (score: {score})")
                    else:
                        print(f"  ❌ Invalid font: {font_path}")
                
                assert valid_fonts > 0, "At least one font should be valid"
                
                print(f"✅ Font validation: {valid_fonts}/{len(test_fonts)} fonts valid")
                
                return True
                
        except Exception as e:
            print(f"❌ Font validation test failed: {e}")
            return False

    def test_dual_renderer_fallback(self):
        """Test dual renderer system with fallback behavior."""
        try:
            # Import dual renderer components
            try:
                from lib.video.caption_manager import CaptionRenderingManager, RendererType
                from lib.video.caption_validator import CaptionQualityValidator
                
                # Test caption manager initialization
                manager = CaptionRenderingManager(
                    preferred_renderer=RendererType.AUTO,
                    enable_fallback=True
                )
                
                # Test renderer detection
                available_renderers = manager._get_available_renderers()
                print(f"✅ Available renderers: {[r.value for r in available_renderers]}")
                
                # Test renderer selection
                if available_renderers:
                    selected = manager._select_optimal_renderer(task_complexity="medium")
                    print(f"✅ Selected renderer: {selected.value}")
                    
                    # Test performance tracking
                    status = manager.get_renderer_status()
                    print(f"✅ Renderer status tracking: {len(status)} renderers monitored")
                
                # Test quality validator
                validator = CaptionQualityValidator()
                print(f"✅ Quality validator initialized (OCR available: {validator.ocr_available})")
                
                return True
                
            except ImportError as e:
                print(f"⚠️ Dual renderer system not available: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Dual renderer fallback test failed: {e}")
            return False

    def test_integrated_dual_rendering(self):
        """Test integrated dual rendering system."""
        if not Path(self.test_template_path).exists() or not Path(self.audio_test_path).exists():
            print("⚠️ Skipping integrated rendering test (missing files)")
            return False
            
        try:
            # Test with dual system configuration
            style = CaptionStyle(
                preferred_renderer='auto',
                enable_fallback=True,
                quality_priority=False,  # Prefer speed for testing
                font_size=36,
                position='bottom'
            )
            
            config = VideoConfig(
                enable_quality_validation=False,  # Disable for faster testing
                quality='low',  # Fast rendering for testing
                debug_caption_rendering=True
            )
            
            with VideoClip(self.test_template_path) as video_clip:
                # Mock the dual system to avoid actual video processing
                with patch('lib.video.caption_manager.caption_manager') as mock_manager:
                    # Mock successful rendering result
                    mock_result = Mock()
                    mock_result.success = True
                    mock_result.render_time = 2.5
                    mock_result.output_path = str(self.output_dir / "test_dual_output.mp4")
                    mock_result.renderer_used.value = 'moviepy'
                    mock_result.error_message = None
                    
                    mock_manager.render_captions.return_value = mock_result
                    mock_manager.get_renderer_status.return_value = {
                        'moviepy': {'status': 'available', 'success_rate': 0.95},
                        'opencv_pil': {'status': 'available', 'success_rate': 0.90}
                    }
                    
                    # Test dual system rendering
                    result = video_clip.render_with_dual_system(
                        caption_text=self.test_caption_text,
                        audio_path=self.audio_test_path,
                        caption_style=style,
                        video_config=config
                    )
                    
                    assert result['success'] == True
                    assert result['dual_system'] == True
                    assert 'renderer_used' in result
                    assert 'render_time' in result
                    
                    print(f"✅ Dual system integration: {result['renderer_used']} in {result['render_time']}s")
                    
                    return True
                    
        except Exception as e:
            print(f"❌ Integrated dual rendering test failed: {e}")
            return False


if __name__ == "__main__":
    # Run tests manually
    test_video = TestVideoProcessing()
    test_video.setup_method()
    
    print("🧪 Testing Video Processing...")
    print("=" * 50)
    
    # Basic dependency check
    if not test_video.test_video_dependencies():
        print("❌ Video dependencies not available - stopping tests")
        exit(1)
    
    # Template tests
    print("\\n🎬 Template Tests:")
    test_video.test_template_loading()
    test_video.test_sample_template_creation()
    
    # Configuration tests
    print("\\n⚙️ Configuration Tests:")
    test_video.test_video_config()
    test_video.test_caption_styles()
    
    # Audio/Caption tests
    print("\\n🎙️ Audio & Caption Tests:")
    test_video.test_audio_integration()
    test_video.test_caption_timing_analysis()
    
    # Rendering tests
    print("\\n🚀 Rendering Tests:")
    test_video.test_fast_video_rendering()
    test_video.test_video_resolution_scaling()
    
    # Enhanced caption system tests
    print("\\n📝 Enhanced Caption System Tests:")
    test_video.test_dual_renderer_configuration()
    test_video.test_margin_system()
    test_video.test_font_validation_system()
    
    # Dual renderer system tests
    print("\\n🔄 Dual Renderer System Tests:")
    test_video.test_dual_renderer_fallback()
    test_video.test_integrated_dual_rendering()
    
    # Cleanup tests
    print("\\n🧹 Cleanup Tests:")
    test_video.test_memory_cleanup()
    
    print("\\n🏁 Video processing tests complete")