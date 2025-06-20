#!/usr/bin/env python3
"""
Test script for Whisper-based caption generation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lib.video.whisper_captions import WhisperCaptionGenerator, WhisperConfig, get_recommended_whisper_models
from lib.video.clip import CaptionStyle

def test_whisper_captions():
    """Test Whisper caption generation on the most recent audio."""
    
    # Use the most recent audio file
    audio_path = "/mnt/c/Programming/moneytree/audio_output/Patrick_Stewart_coqui.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    # Test basic VideoClip integration first
    print("🎬 Testing VideoClip Whisper integration...")
    
    # Create a simple test to see if the integration works in VideoClip
    from lib.video.clip import VideoClip, CaptionStyle
    
    # Find a template video to test with
    template_path = None
    for ext in ['.mp4', '.mov', '.avi']:
        potential_path = f"/mnt/c/Programming/moneytree/caption_tests/caption_test_center{ext}"
        if os.path.exists(potential_path):
            template_path = potential_path
            break
    
    if template_path:
        print(f"📁 Using template: {os.path.basename(template_path)}")
        
        with VideoClip(template_path) as video_clip:
            # Test Whisper integration
            result = video_clip.add_synchronized_captions(
                "Sample text for testing", 
                audio_path, 
                CaptionStyle.for_vertical_video(),
                use_whisper=True
            )
            
            print(f"✅ VideoClip integration test: {result}")
            if result.get('whisper_used'):
                print("🎤 Whisper was successfully used!")
            else:
                print("📝 Fallback to speech analysis was used")
            
            return result
    else:
        print("⚠️ No template video found, testing Whisper directly...")
    
    print(f"🎤 Testing Whisper caption generation")
    print(f"📁 Audio: {os.path.basename(audio_path)}")
    print("=" * 60)
    
    # Show available models
    print("📋 Available Whisper models:")
    models = get_recommended_whisper_models()
    for model in models:
        print(f"   {model['name']}: {model['description']} ({model['speed']}, {model['accuracy']})")
    
    # Test with base model (good balance)
    print(f"\n🔄 Initializing Whisper (base model)...")
    
    config = WhisperConfig(
        model_size="base",
        language="en", 
        temperature=0.0,
        word_timestamps=True,
        verbose=False
    )
    
    generator = WhisperCaptionGenerator(config)
    
    if not generator.model:
        print("❌ Failed to load Whisper model")
        return
    
    print("✅ Whisper model loaded successfully")
    
    # Generate captions
    print(f"\n🎬 Generating captions...")
    style = CaptionStyle(position="bottom", font_size=36)
    
    max_length = 50  # characters
    duration = 3.0   # seconds
    
    print(f"   Max caption length: {max_length} characters")
    print(f"   Target duration: {duration} seconds")
    
    captions = generator.generate_captions_from_audio(
        audio_path,
        max_caption_length=max_length,
        caption_duration=duration
    )
    
    if not captions:
        print("❌ No captions generated")
        return
    
    # Show results
    print(f"\n📊 Caption Results:")
    print(f"   Generated {len(captions)} caption segments")
    
    # Show first few captions
    print(f"\n📝 Sample captions:")
    for i, caption in enumerate(captions[:5]):
        print(f"   {i+1:2d}. [{caption.start_time:6.2f}s - {caption.end_time:6.2f}s] \"{caption.text}\"")
        print(f"       Duration: {caption.duration:.2f}s, Confidence: {caption.confidence:.3f}")
    
    if len(captions) > 5:
        print(f"   ... and {len(captions) - 5} more captions")
    
    # Show statistics
    stats = generator.get_caption_statistics(captions)
    print(f"\n📈 Statistics:")
    print(f"   Total duration: {stats['total_duration']:.1f}s")
    print(f"   Average duration: {stats['average_duration']:.1f}s")
    print(f"   Average confidence: {stats['average_confidence']:.3f}")
    print(f"   Words per minute: {stats['words_per_minute']:.0f}")
    print(f"   Total words: {stats['total_words']}")
    print(f"   Average text length: {stats['average_text_length']:.1f} chars")
    
    # Test timing adjustment
    print(f"\n🔧 Testing timing adjustments...")
    adjusted = generator.adjust_caption_timing(captions, min_duration=1.5, max_duration=4.0)
    
    duration_changes = 0
    for orig, adj in zip(captions, adjusted):
        if abs(orig.duration - adj.duration) > 0.1:
            duration_changes += 1
    
    print(f"   Adjusted {duration_changes} caption durations")
    
    # Convert to MoviePy format
    print(f"\n🎬 Converting to MoviePy format...")
    timing_data = generator.generate_moviepy_timing_data(adjusted)
    
    print(f"   Created {len(timing_data)} timing entries")
    print(f"   Sample timing data:")
    for i, entry in enumerate(timing_data[:3]):
        print(f"      {i+1}. {entry['start']:.2f}s-{entry['end']:.2f}s: \"{entry['text'][:40]}...\"")
    
    print(f"\n✅ Whisper caption generation test completed!")
    print(f"🎯 Captions are now synchronized with actual audio content")
    
    return timing_data

if __name__ == "__main__":
    test_whisper_captions()