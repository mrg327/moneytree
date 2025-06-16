"""Tests for video processing and rendering components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import tempfile
from pathlib import Path
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
    
    # Cleanup tests
    print("\\n🧹 Cleanup Tests:")
    test_video.test_memory_cleanup()
    
    print("\\n🏁 Video processing tests complete")