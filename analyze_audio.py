#!/usr/bin/env python3
"""
Quick audio analysis script to identify TTS artifacts and issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lib.audio.quality_validator import AudioQualityValidator
from lib.audio.post_processor import AudioPostProcessor
import numpy as np

def analyze_recent_audio():
    """Analyze the most recent audio file for TTS issues."""
    
    # Most recent audio file
    audio_path = "/mnt/c/Programming/moneytree/audio_output/Patrick_Stewart_coqui.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"🔍 Analyzing audio: {os.path.basename(audio_path)}")
    print("=" * 60)
    
    # Initialize analyzers
    validator = AudioQualityValidator()
    post_processor = AudioPostProcessor()
    
    # Perform comprehensive analysis
    print("📊 Running quality analysis...")
    quality_report = validator.analyze_audio(audio_path)
    
    # Print detailed results
    print(f"\n📈 Quality Report:")
    print(f"   Overall Score: {quality_report.quality_score:.3f}/1.0")
    print(f"   Silence: {quality_report.silence_percentage:.1f}%")
    print(f"   Speech: {quality_report.speech_percentage:.1f}%")
    print(f"   Dynamic Range: {quality_report.dynamic_range_db:.1f} dB")
    print(f"   Duration: {quality_report.duration:.2f}s")
    
    if quality_report.clipping_detected:
        print("   ⚠️  CLIPPING DETECTED")
    
    if quality_report.artifacts_detected:
        print(f"   🚨 Artifacts: {', '.join(quality_report.artifacts_detected)}")
    
    print(f"\n💡 Recommended Fixes: {', '.join(quality_report.recommended_fixes)}")
    
    # Apply fixes and create enhanced version
    if quality_report.needs_processing:
        print(f"\n🔧 Applying audio enhancements...")
        enhanced_path = post_processor.enhance_audio(audio_path, quality_report.recommended_fixes)
        
        if os.path.exists(enhanced_path):
            enhanced_size = os.path.getsize(enhanced_path)
            print(f"✅ Enhanced audio created: {enhanced_path}")
            print(f"   Size: {enhanced_size:,} bytes")
            
            # Re-analyze enhanced version
            print(f"\n🔄 Re-analyzing enhanced audio...")
            enhanced_report = validator.analyze_audio(enhanced_path)
            print(f"   Quality improvement: {quality_report.quality_score:.3f} → {enhanced_report.quality_score:.3f}")
            
            return enhanced_path
        else:
            print("❌ Enhancement failed")
    else:
        print("✅ Audio quality is acceptable")
    
    return audio_path

if __name__ == "__main__":
    analyze_recent_audio()