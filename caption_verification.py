#!/usr/bin/env python3
"""
Caption Verification Script
Extracts frames every 5 seconds and verifies caption text rendering
"""

import subprocess
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

def get_expected_captions() -> List[Dict[str, any]]:
    """Define the expected caption text based on the generation log."""
    # From the TTS generation log, we can extract the expected text
    full_text = """Consider Machine learning, which presents several noteworthy aspects. 
    The basic definition states that machine learning is study of algorithms that improve automatically through experience, providing our starting point. 
    Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalise to unseen data, and thus perform tasks without explicit instructions. 
    In related developments. 
    Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance. 
    Furthermore. 
    This overview provides a foundation for understanding the subject's key characteristics."""
    
    # Split into 6-word chunks as configured in the system
    words = full_text.split()
    words_per_caption = 6
    
    captions = []
    total_duration = 171.9  # From log
    caption_count = 19  # From log
    time_per_caption = total_duration / caption_count
    
    for i in range(0, len(words), words_per_caption):
        caption_text = ' '.join(words[i:i + words_per_caption])
        start_time = (i // words_per_caption) * time_per_caption
        end_time = start_time + time_per_caption
        
        captions.append({
            'text': caption_text.strip(),
            'start': start_time,
            'end': end_time,
            'index': i // words_per_caption
        })
        
        if len(captions) >= caption_count:
            break
    
    return captions

def extract_frames_every_5_seconds(video_path: str, output_dir: str) -> List[str]:
    """Extract frames every 5 seconds from the video."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get video duration
    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    
    frame_times = []
    extracted_files = []
    
    # Extract frames every 5 seconds
    t = 5  # Start at 5 seconds to skip intro
    while t < duration:
        frame_times.append(t)
        t += 5
    
    print(f"Extracting {len(frame_times)} frames from video (duration: {duration:.1f}s)")
    
    for i, timestamp in enumerate(frame_times):
        output_file = f"{output_dir}/frame_{timestamp:03.0f}s.png"
        cmd = [
            'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path, 
            '-frames:v', '1', '-update', '1', output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            extracted_files.append(output_file)
            print(f"✓ Extracted frame at {timestamp}s")
        else:
            print(f"✗ Failed to extract frame at {timestamp}s: {result.stderr}")
    
    return extracted_files

def find_expected_caption_at_time(timestamp: float, captions: List[Dict]) -> Optional[Dict]:
    """Find which caption should be displayed at a given timestamp."""
    for caption in captions:
        if caption['start'] <= timestamp <= caption['end']:
            return caption
    return None

def analyze_frame_with_ocr(frame_path: str) -> str:
    """Extract text from frame using OCR (simplified approach)."""
    # For this implementation, we'll use a simple visual analysis
    # In a production system, you'd use pytesseract or similar OCR tools
    
    # Since OCR setup might be complex, let's return the filename for manual verification
    # and implement a pattern-based approach
    
    try:
        # Try to use tesseract if available
        import pytesseract
        from PIL import Image
        
        image = Image.open(frame_path)
        # Configure tesseract for white text on dark background
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;()-\" '
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    except ImportError:
        print("⚠️ OCR not available (pytesseract not installed)")
        return "[OCR_NOT_AVAILABLE]"
    except Exception as e:
        print(f"⚠️ OCR failed: {e}")
        return "[OCR_FAILED]"

def verify_captions(video_path: str) -> Dict[str, any]:
    """Main verification function."""
    print("🔍 Starting Caption Verification Test")
    print("=" * 60)
    
    # Get expected captions
    expected_captions = get_expected_captions()
    print(f"📝 Expected {len(expected_captions)} captions")
    
    # Extract frames
    output_dir = "/tmp/caption_verification"
    frame_files = extract_frames_every_5_seconds(video_path, output_dir)
    print(f"🖼️ Extracted {len(frame_files)} frames")
    
    # Analyze each frame
    results = []
    successful_matches = 0
    
    for frame_file in frame_files:
        # Extract timestamp from filename
        filename = Path(frame_file).name
        timestamp_match = re.search(r'frame_(\d+)s\.png', filename)
        if not timestamp_match:
            continue
            
        timestamp = float(timestamp_match.group(1))
        
        # Find expected caption at this time
        expected_caption = find_expected_caption_at_time(timestamp, expected_captions)
        
        # Extract text from frame
        extracted_text = analyze_frame_with_ocr(frame_file)
        
        # Compare texts
        match_quality = "UNKNOWN"
        if extracted_text != "[OCR_NOT_AVAILABLE]" and extracted_text != "[OCR_FAILED]":
            if expected_caption:
                # Simple text similarity check
                expected_lower = expected_caption['text'].lower().replace('.', '').replace(',', '')
                extracted_lower = extracted_text.lower().replace('.', '').replace(',', '')
                
                # Check if key words match
                expected_words = set(expected_lower.split())
                extracted_words = set(extracted_lower.split())
                
                if len(expected_words) > 0:
                    overlap = len(expected_words.intersection(extracted_words))
                    similarity = overlap / len(expected_words)
                    
                    if similarity >= 0.7:
                        match_quality = "GOOD_MATCH"
                        successful_matches += 1
                    elif similarity >= 0.4:
                        match_quality = "PARTIAL_MATCH"
                    else:
                        match_quality = "POOR_MATCH"
                else:
                    match_quality = "NO_EXPECTED_TEXT"
            else:
                match_quality = "NO_CAPTION_EXPECTED"
        
        result = {
            'timestamp': timestamp,
            'frame_file': frame_file,
            'expected_text': expected_caption['text'] if expected_caption else None,
            'extracted_text': extracted_text,
            'match_quality': match_quality,
            'expected_caption_index': expected_caption['index'] if expected_caption else None
        }
        
        results.append(result)
        
        # Print progress
        status_emoji = "✅" if match_quality == "GOOD_MATCH" else "⚠️" if "MATCH" in match_quality else "❓"
        print(f"{status_emoji} Frame at {timestamp}s: {match_quality}")
        if expected_caption:
            print(f"   Expected: '{expected_caption['text'][:50]}...'")
        if extracted_text not in ["[OCR_NOT_AVAILABLE]", "[OCR_FAILED]"]:
            print(f"   Extracted: '{extracted_text[:50]}...'")
        print()
    
    # Generate summary
    summary = {
        'total_frames': len(results),
        'successful_matches': successful_matches,
        'success_rate': successful_matches / len(results) if results else 0,
        'results': results,
        'expected_captions': expected_captions
    }
    
    return summary

def main():
    """Main function."""
    # Find the most recent video file
    video_dir = Path("video_output")
    video_files = list(video_dir.glob("moneytree_video_*.mp4"))
    
    if not video_files:
        print("❌ No video files found in video_output directory")
        return
    
    # Get the most recent video
    latest_video = max(video_files, key=lambda f: f.stat().st_mtime)
    print(f"🎬 Analyzing video: {latest_video}")
    
    # Run verification
    results = verify_captions(str(latest_video))
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📊 CAPTION VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total frames analyzed: {results['total_frames']}")
    print(f"Successful matches: {results['successful_matches']}")
    print(f"Success rate: {results['success_rate']:.1%}")
    
    if results['success_rate'] >= 0.8:
        print("🎉 EXCELLENT: Caption rendering is working very well!")
    elif results['success_rate'] >= 0.6:
        print("👍 GOOD: Caption rendering is mostly working")
    elif results['success_rate'] >= 0.4:
        print("⚠️ FAIR: Caption rendering has some issues")
    else:
        print("❌ POOR: Caption rendering needs significant improvement")
    
    # Clean up temp files
    temp_dir = Path("/tmp/caption_verification")
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary files")

if __name__ == "__main__":
    main()