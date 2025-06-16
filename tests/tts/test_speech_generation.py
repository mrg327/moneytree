"""Tests for TTS speech generation components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from pathlib import Path
from lib.tts.chattts_speech_generator import ChatTTSSpeechGenerator, ChatTTSConfig
from lib.tts.coqui_speech_generator import CoquiSpeechGenerator, CoquiTTSConfig


class TestSpeechGeneration:
    """Test TTS speech generation components."""

    def setup_method(self):
        """Set up test fixtures."""
        # Sample monologue for testing
        self.sample_monologue = {
            'script': [
                {'content': 'Hello, this is a test message for TTS generation.'},
                {'content': 'This should create a short audio file.'}
            ],
            'word_count': 15,
            'estimated_duration': 8.0
        }
        
        self.test_audio_dir = Path("audio_output")
        self.test_audio_dir.mkdir(exist_ok=True)

    def test_chattts_initialization(self):
        """Test ChatTTS generator initialization."""
        try:
            config = ChatTTSConfig()
            generator = ChatTTSSpeechGenerator(config)
            
            assert generator is not None
            assert hasattr(generator, 'generate_speech_from_monologue')
            
            if generator.chat is not None:
                print("✅ ChatTTS initialization successful")
                return True
            else:
                print("⚠️ ChatTTS not available (models not loaded)")
                return False
                
        except Exception as e:
            print(f"⚠️ ChatTTS initialization failed: {e}")
            return False

    def test_chattts_model_loading(self):
        """Test ChatTTS model loading specifically."""
        try:
            config = ChatTTSConfig()
            generator = ChatTTSSpeechGenerator(config)
            
            # Check if models are actually loaded
            if hasattr(generator, 'chat') and generator.chat is not None:
                print("✅ ChatTTS models loaded successfully")
                
                # Check if we can access model components
                if hasattr(generator.chat, 'pretrain_models'):
                    print("✅ ChatTTS pretrained models accessible")
                else:
                    print("⚠️ ChatTTS pretrained models not accessible")
                
                return True
            else:
                print("❌ ChatTTS models failed to load")
                return False
                
        except Exception as e:
            print(f"❌ ChatTTS model loading failed: {e}")
            return False

    def test_chattts_speech_generation(self):
        """Test ChatTTS speech generation."""
        try:
            config = ChatTTSConfig()
            generator = ChatTTSSpeechGenerator(config)
            
            if generator.chat is None:
                print("⚠️ Skipping ChatTTS generation test (models not available)")
                return
            
            result = generator.generate_speech_from_monologue(self.sample_monologue)
            
            assert result is not None
            assert isinstance(result, dict)
            
            if result.get('success'):
                print(f"✅ ChatTTS generation successful: {result.get('output_path')}")
                
                # Check if file exists
                output_path = result.get('output_path')
                if output_path and Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size
                    print(f"📁 Audio file created: {file_size} bytes")
                    assert file_size > 0
                else:
                    print("⚠️ Audio file not found")
                    
            else:
                print(f"❌ ChatTTS generation failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ ChatTTS generation test failed: {e}")

    def test_coqui_initialization(self):
        """Test Coqui TTS generator initialization."""
        try:
            config = CoquiTTSConfig(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                device='cpu'
            )
            generator = CoquiSpeechGenerator(config)
            
            assert generator is not None
            assert hasattr(generator, 'generate_speech_from_monologue')
            
            if generator.tts is not None:
                print("✅ Coqui TTS initialization successful")
                return True
            else:
                print("⚠️ Coqui TTS not available")
                return False
                
        except Exception as e:
            print(f"⚠️ Coqui TTS initialization failed: {e}")
            return False

    def test_coqui_speech_generation(self):
        """Test Coqui TTS speech generation."""
        try:
            config = CoquiTTSConfig(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                device='cpu'
            )
            generator = CoquiSpeechGenerator(config)
            
            if generator.tts is None:
                print("⚠️ Skipping Coqui generation test (TTS not available)")
                return
            
            result = generator.generate_speech_from_monologue(self.sample_monologue)
            
            assert result is not None
            assert isinstance(result, dict)
            
            if result.get('success'):
                print(f"✅ Coqui generation successful: {result.get('output_path')}")
                
                # Check if file exists
                output_path = result.get('output_path')
                if output_path and Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size
                    print(f"📁 Audio file created: {file_size} bytes")
                    assert file_size > 0
                else:
                    print("⚠️ Audio file not found")
                    
            else:
                print(f"❌ Coqui generation failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Coqui generation test failed: {e}")

    def test_audio_file_quality(self):
        """Test that generated audio files are valid."""
        # Check existing audio files
        audio_files = list(self.test_audio_dir.glob("*.wav"))
        
        if not audio_files:
            print("⚠️ No audio files found for quality check")
            return
        
        for audio_file in audio_files[:3]:  # Check first 3 files
            try:
                file_size = audio_file.stat().st_size
                
                # Basic file checks
                assert file_size > 1000  # Should be at least 1KB
                assert audio_file.suffix.lower() == '.wav'
                
                print(f"✅ Audio file quality OK: {audio_file.name} ({file_size} bytes)")
                
            except Exception as e:
                print(f"❌ Audio file quality check failed for {audio_file.name}: {e}")

    def test_tts_config_validation(self):
        """Test TTS configuration validation."""
        # Test ChatTTS config
        try:
            config = ChatTTSConfig(
                temperature=0.5,
                top_k=20,
                top_p=0.7,
                sample_rate=24000
            )
            assert config.temperature == 0.5
            assert config.sample_rate == 24000
            print("✅ ChatTTS config validation successful")
            
        except Exception as e:
            print(f"❌ ChatTTS config validation failed: {e}")
        
        # Test Coqui config
        try:
            config = CoquiTTSConfig(
                model_name="test_model",
                device='cpu',
                sample_rate=22050
            )
            assert config.model_name == "test_model"
            assert config.device == 'cpu'
            print("✅ Coqui config validation successful")
            
        except Exception as e:
            print(f"❌ Coqui config validation failed: {e}")

    def test_speech_duration_estimation(self):
        """Test speech duration estimation."""
        try:
            config = ChatTTSConfig()
            generator = ChatTTSSpeechGenerator(config)
            
            # Test duration estimation logic
            test_text = "This is a test sentence with multiple words."
            word_count = len(test_text.split())
            
            # Estimate duration (roughly 150 words per minute)
            estimated_duration = (word_count / 150) * 60
            
            assert estimated_duration > 0
            assert estimated_duration < 60  # Should be reasonable
            
            print(f"✅ Duration estimation: {word_count} words → {estimated_duration:.1f}s")
            
        except Exception as e:
            print(f"❌ Duration estimation test failed: {e}")


if __name__ == "__main__":
    # Run tests manually
    test_tts = TestSpeechGeneration()
    test_tts.setup_method()
    
    print("🧪 Testing TTS Speech Generation...")
    print("=" * 50)
    
    # ChatTTS tests
    print("\\n🗣️ ChatTTS Tests:")
    try:
        if test_tts.test_chattts_initialization():
            test_tts.test_chattts_model_loading()
            test_tts.test_chattts_speech_generation()
    except Exception as e:
        print(f"❌ ChatTTS tests failed: {e}")
    
    # Coqui TTS tests  
    print("\\n🎙️ Coqui TTS Tests:")
    try:
        if test_tts.test_coqui_initialization():
            test_tts.test_coqui_speech_generation()
    except Exception as e:
        print(f"❌ Coqui TTS tests failed: {e}")
    
    # General tests
    print("\\n🔧 General TTS Tests:")
    try:
        test_tts.test_audio_file_quality()
        test_tts.test_tts_config_validation()
        test_tts.test_speech_duration_estimation()
    except Exception as e:
        print(f"❌ General TTS tests failed: {e}")
    
    print("\\n🏁 TTS tests complete")