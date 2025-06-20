#!/usr/bin/env python3
"""
Set up sample voice references for XTTS-v2 voice cloning.

Downloads and prepares sample voice references from public sources.
"""

import os
import sys
import requests
from pathlib import Path

def download_file(url: str, local_path: str) -> bool:
    """Download a file from URL to local path."""
    try:
        print(f"📥 Downloading: {Path(local_path).name}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded: {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def setup_sample_voices():
    """Set up sample voice references."""
    print("🎤 Setting up sample voice references for XTTS-v2")
    print("=" * 50)
    
    # Create sample voices directory
    voices_dir = Path("sample_voices")
    voices_dir.mkdir(exist_ok=True)
    
    print("\n📚 Sample Voice Sources:")
    print("Since we need to respect copyright and licensing, here are the best")
    print("approaches to get reference audio for voice cloning:")
    print()
    
    print("🔗 1. Mozilla Common Voice (CC0 Public Domain)")
    print("   • Visit: https://commonvoice.mozilla.org/datasets")
    print("   • Download validated clips (already segmented)")
    print("   • Multiple languages and speakers available")
    print("   • Perfect for voice cloning experiments")
    print()
    
    print("📖 2. LibriVox Audiobooks (Public Domain)")
    print("   • Visit: https://librivox.org/")
    print("   • Download chapters from interesting speakers")
    print("   • Use extract_voice_samples.py to get clean segments")
    print("   • Great variety of voices and accents")
    print()
    
    print("🎭 3. Create Your Own Reference Audio")
    print("   • Record 10-20 seconds of clear speech")
    print("   • Use good quality microphone")
    print("   • Read text clearly and consistently")
    print("   • Save as WAV file (22kHz recommended)")
    print()
    
    # Create a sample script for recording
    sample_script = """
Hello, my name is [Your Name]. I'm creating a voice reference for text-to-speech generation.
This recording will be used to clone my voice using artificial intelligence.
The quick brown fox jumps over the lazy dog. 
I enjoy reading books and learning about new technologies.
This should provide enough speech data for accurate voice cloning.
Thank you for listening to this voice sample.
""".strip()
    
    script_path = voices_dir / "recording_script.txt"
    with open(script_path, 'w') as f:
        f.write(sample_script)
    
    print(f"📝 Created recording script: {script_path}")
    print("   Use this text to create your own voice reference")
    print()
    
    # Create instructions file
    instructions = """
# Voice Reference Setup Instructions

## Option 1: Download Mozilla Common Voice
1. Visit: https://commonvoice.mozilla.org/datasets
2. Accept the license terms
3. Download the English dataset (or preferred language)
4. Extract the .tar.gz file
5. Copy interesting clips from the 'clips' folder to this directory

## Option 2: Use LibriVox Audiobooks
1. Visit: https://librivox.org/
2. Find an audiobook with a voice you like
3. Download MP3 chapters
4. Run: python extract_voice_samples.py audiobook.mp3
5. Use the extracted samples from 'voice_samples' folder

## Option 3: Record Your Own Voice
1. Use the provided recording_script.txt
2. Record in a quiet environment
3. Speak clearly and consistently
4. Save as high-quality WAV file
5. Aim for 10-20 seconds of clean speech

## Testing Your Voice Reference
python demo_voice_cloning.py --analyze-voice your_reference.wav
python demo_voice_cloning.py "Test message" --reference-audio your_reference.wav

## Voice Quality Tips
- Use 6+ seconds of clean speech
- Avoid background noise or music
- Maintain consistent volume and pace
- Single speaker only (no conversations)
- Clear pronunciation and articulation
"""
    
    instructions_path = voices_dir / "README.md"
    with open(instructions_path, 'w') as f:
        f.write(instructions.strip())
    
    print(f"📖 Created instructions: {instructions_path}")
    print()
    
    print("🎯 Next Steps:")
    print("1. Follow one of the options above to get voice references")
    print("2. Place audio files (.wav) in the sample_voices/ directory")
    print("3. Test with: python demo_voice_cloning.py --analyze-voice sample_voices/your_file.wav")
    print("4. Start cloning: python demo_voice_cloning.py \"Hello world!\" --reference-audio sample_voices/your_file.wav")
    print()
    
    print("💡 Pro Tips:")
    print("• Start with Mozilla Common Voice for quick testing")
    print("• Use extract_voice_samples.py for longer audio files") 
    print("• Record your own voice for personalized content")
    print("• Multiple samples from same speaker = better quality")


if __name__ == "__main__":
    setup_sample_voices()