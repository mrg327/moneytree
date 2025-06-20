#!/usr/bin/env python3
"""
Compare old vs new caption space utilization.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lib.video.clip import VideoClip, CaptionStyle

def compare_caption_styles():
    """Compare old vs new caption styling."""
    
    # Use existing audio and template
    audio_path = "/mnt/c/Programming/moneytree/audio_output/Patrick_Stewart_coqui.wav"
    template_path = "/mnt/c/Programming/moneytree/caption_tests/caption_test_center.mp4"
    
    if not os.path.exists(audio_path) or not os.path.exists(template_path):
        print("❌ Required files not found")
        return
    
    # Test text
    test_text = "The Python programming language is a high-level interpreted language with dynamic semantics."
    
    print("📊 Caption Style Comparison")
    print("=" * 50)
    
    # Old style (conservative)
    old_style = CaptionStyle(
        font_size=36,
        position='center',
        max_width=80,
        words_per_caption=6
    )
    
    # New style (improved space utilization)
    new_style = CaptionStyle.for_vertical_video(font_size=36)
    
    print("🔄 Old Style:")
    print(f"   Max width: {old_style.max_width}%")
    print(f"   Words per caption: {old_style.words_per_caption}")
    print(f"   Expected characters per line: ~25")
    
    print("\n✨ New Style:")
    print(f"   Max width: {new_style.max_width}%")
    print(f"   Words per caption: {new_style.words_per_caption}")
    print(f"   Expected characters per line: ~35")
    
    # Calculate space utilization
    old_text_area = 0.80  # 80% width usage
    new_text_area = 0.85  # 85% width usage
    
    improvement = ((new_text_area - old_text_area) / old_text_area) * 100
    
    print(f"\n📈 Space Utilization Improvement:")
    print(f"   Old: {old_text_area:.0%} of video width")
    print(f"   New: {new_text_area:.0%} of video width")
    print(f"   Improvement: +{improvement:.1f}% more text area")
    
    # Word capacity comparison
    old_words_per_segment = old_style.words_per_caption
    new_words_per_segment = new_style.words_per_caption
    word_improvement = ((new_words_per_segment - old_words_per_segment) / old_words_per_segment) * 100
    
    print(f"\n📝 Text Capacity Improvement:")
    print(f"   Old: {old_words_per_segment} words per caption")
    print(f"   New: {new_words_per_segment} words per caption")
    print(f"   Improvement: +{word_improvement:.1f}% more words per segment")
    
    print(f"\n🎯 Benefits:")
    print(f"   • Reduces number of caption segments by utilizing more space")
    print(f"   • Allows longer, more coherent text segments")
    print(f"   • Better utilizes the middle 80% of the video frame")
    print(f"   • Maintains readability while reducing wasted space")

if __name__ == "__main__":
    compare_caption_styles()