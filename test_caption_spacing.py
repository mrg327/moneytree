#!/usr/bin/env python3
"""
Test script for improved caption space utilization.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lib.video.clip import VideoClip, CaptionStyle

def test_caption_spacing():
    """Test the improved caption spacing with longer text."""
    
    # Use existing audio and template
    audio_path = "/mnt/c/Programming/moneytree/audio_output/Patrick_Stewart_coqui.wav"
    template_path = "/mnt/c/Programming/moneytree/caption_tests/caption_test_center.mp4"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    if not os.path.exists(template_path):
        print(f"❌ Template file not found: {template_path}")
        return
    
    print("🎬 Testing improved caption space utilization...")
    print(f"📁 Audio: {os.path.basename(audio_path)}")
    print(f"📁 Template: {os.path.basename(template_path)}")
    print("=" * 60)
    
    # Test text with longer content to see space utilization
    test_text = """
    The Python programming language is a high-level, interpreted programming language 
    with dynamic semantics and elegant syntax that makes it ideal for rapid application 
    development. Its built-in data structures and powerful libraries enable developers 
    to create sophisticated applications with minimal code complexity.
    """
    
    try:
        with VideoClip(template_path) as video_clip:
            # Test with vertical video style (most common format)
            print("🎯 Testing vertical video caption style...")
            vertical_style = CaptionStyle.for_vertical_video(font_size=36)
            print(f"   Max width: {vertical_style.max_width}%")
            print(f"   Words per caption: {vertical_style.words_per_caption}")
            
            result = video_clip.add_synchronized_captions(
                test_text.strip(),
                audio_path,
                vertical_style,
                use_whisper=False  # Use speech analysis for consistency
            )
            
            if result['success']:
                print(f"✅ Vertical captions: {result['caption_count']} segments")
                print(f"   Timing method: {result['timing_method']}")
                print(f"   Video dimensions: {result['dimensions']}")
                
                # Add narration
                video_clip.add_narration_audio(audio_path)
                
                # Render test video
                output_path = "/mnt/c/Programming/moneytree/video_output/caption_spacing_test.mp4"
                render_result = video_clip.render_video(output_path)
                
                if render_result['success']:
                    print(f"🎥 Test video created: {os.path.basename(output_path)}")
                    print(f"   File size: {render_result['file_size'] / 1024 / 1024:.1f} MB")
                    print(f"   Duration: {render_result['duration']:.1f}s")
                    print(f"   Resolution: {render_result['resolution']}")
                else:
                    print(f"❌ Render failed: {render_result.get('error', 'Unknown error')}")
            else:
                print(f"❌ Caption creation failed: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_caption_spacing()