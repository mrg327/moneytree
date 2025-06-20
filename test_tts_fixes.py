#!/usr/bin/env python3
"""
Test the TTS artifact fixes specifically.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lib.audio.post_processor import AudioPostProcessor
from lib.audio.quality_validator import AudioQualityValidator
import librosa
import numpy as np

def test_tts_fixes():
    """Test TTS artifact fixes on the problematic audio."""
    
    audio_path = "/mnt/c/Programming/moneytree/audio_output/Patrick_Stewart_coqui.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"🔧 Testing TTS artifact fixes on: {os.path.basename(audio_path)}")
    print("=" * 60)
    
    # Load audio
    print("📁 Loading audio...")
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    print(f"   Loaded: {len(audio_data)} samples at {sample_rate}Hz")
    print(f"   Duration: {len(audio_data)/sample_rate:.2f}s")
    
    # Initialize processor
    processor = AudioPostProcessor(sample_rate=sample_rate)
    
    # Test individual fixes
    print("\n🔍 Applying individual fixes...")
    
    # Test 1: Click removal
    print("1. Testing gentle click removal...")
    click_free = processor._remove_clicks_with_gentle_gate(audio_data)
    click_improvement = np.mean(np.abs(audio_data - click_free))
    print(f"   Average change: {click_improvement:.6f}")
    
    # Test 2: Clipping fix
    print("2. Testing clipping repair...")
    clipping_free = processor.fix_audio_clipping(audio_data)
    max_original = np.max(np.abs(audio_data))
    max_fixed = np.max(np.abs(clipping_free))
    print(f"   Peak amplitude: {max_original:.3f} → {max_fixed:.3f}")
    
    # Test 3: TTS syllable artifacts
    print("3. Testing TTS syllable artifact fixes...")
    syllable_fixed = processor.fix_tts_syllable_artifacts(audio_data)
    syllable_improvement = np.mean(np.abs(audio_data - syllable_fixed))
    print(f"   Average change: {syllable_improvement:.6f}")
    
    # Test 4: Combined processing
    print("4. Testing combined processing...")
    enhanced_audio = audio_data.copy()
    enhanced_audio = processor._remove_clicks_with_gentle_gate(enhanced_audio)
    enhanced_audio = processor.fix_audio_clipping(enhanced_audio)
    enhanced_audio = processor.fix_tts_syllable_artifacts(enhanced_audio)
    enhanced_audio = processor.smooth_volume_changes(enhanced_audio, sample_rate)
    
    total_improvement = np.mean(np.abs(audio_data - enhanced_audio))
    print(f"   Total average change: {total_improvement:.6f}")
    
    # Save processed version
    output_path = "/mnt/c/Programming/moneytree/audio_output/Patrick_Stewart_processed.wav"
    print(f"\n💾 Saving processed audio to: {output_path}")
    
    import soundfile as sf
    sf.write(output_path, enhanced_audio, sample_rate)
    
    file_size = os.path.getsize(output_path)
    print(f"✅ Processed audio saved: {file_size:,} bytes")
    
    # Quick quality comparison
    print(f"\n📊 Quality comparison:")
    validator = AudioQualityValidator()
    
    # Analyze original
    print("   Original audio analysis...")
    # Save temp original for analysis
    temp_original = "/tmp/temp_original.wav"
    sf.write(temp_original, audio_data, sample_rate)
    original_report = validator.analyze_audio(temp_original)
    
    # Analyze processed
    print("   Processed audio analysis...")
    processed_report = validator.analyze_audio(output_path)
    
    print(f"   Quality score: {original_report.quality_score:.3f} → {processed_report.quality_score:.3f}")
    print(f"   Clipping: {'Yes' if original_report.clipping_detected else 'No'} → {'Yes' if processed_report.clipping_detected else 'No'}")
    print(f"   Dynamic range: {original_report.dynamic_range_db:.1f}dB → {processed_report.dynamic_range_db:.1f}dB")
    
    # Clean up
    if os.path.exists(temp_original):
        os.remove(temp_original)
    
    print(f"\n✅ TTS artifact processing complete!")
    print(f"🎧 Listen to both files to compare:")
    print(f"   Original: {audio_path}")
    print(f"   Processed: {output_path}")

if __name__ == "__main__":
    test_tts_fixes()