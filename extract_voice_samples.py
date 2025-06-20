#!/usr/bin/env python3
"""
Extract voice samples from longer audio files for XTTS-v2 voice cloning.

This script helps prepare reference audio by extracting clean segments
from longer recordings like LibriVox audiobooks or other sources.
"""

import os
import argparse
from pathlib import Path

try:
    import librosa
    import numpy as np
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    print("⚠️ Audio libraries not available. Install with: uv add librosa soundfile")

from lib.tts.voice_cloning import VoiceManager


def extract_voice_samples(input_audio: str, output_dir: str, 
                         target_duration: float = 10.0,
                         min_duration: float = 6.0,
                         max_samples: int = 5):
    """
    Extract clean voice samples from a longer audio file.
    
    Args:
        input_audio: Path to input audio file
        output_dir: Directory to save extracted samples
        target_duration: Target duration for each sample
        min_duration: Minimum duration for samples
        max_samples: Maximum number of samples to extract
    """
    if not HAS_AUDIO_LIBS:
        print("❌ Cannot extract samples without audio libraries")
        return []
    
    if not os.path.exists(input_audio):
        print(f"❌ Input audio not found: {input_audio}")
        return []
    
    print(f"🎤 Extracting voice samples from: {Path(input_audio).name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⏱️ Target duration: {target_duration}s")
    print("=" * 50)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load audio
        print("📊 Loading and analyzing audio...")
        audio, sr = librosa.load(input_audio, sr=22050)
        total_duration = len(audio) / sr
        
        print(f"   Total duration: {total_duration:.1f}s")
        print(f"   Sample rate: {sr}Hz")
        
        # Find speech segments using energy-based detection
        print("🔍 Detecting speech segments...")
        frame_length = 2048
        hop_length = 512
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Find speech segments (above threshold)
        threshold = np.mean(rms) * 0.4  # Adjusted threshold
        speech_frames = rms > threshold
        
        # Group consecutive speech frames
        speech_segments = []
        in_speech = False
        start_time = None
        
        for i, (time, is_speech) in enumerate(zip(times, speech_frames)):
            if is_speech and not in_speech:
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                if start_time is not None and (time - start_time) >= min_duration:
                    speech_segments.append((start_time, time))
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech and start_time is not None:
            if (times[-1] - start_time) >= min_duration:
                speech_segments.append((start_time, times[-1]))
        
        print(f"   Found {len(speech_segments)} speech segments >= {min_duration}s")
        
        # Extract and save the best segments
        voice_manager = VoiceManager()
        extracted_samples = []
        
        for i, (start, end) in enumerate(speech_segments[:max_samples]):
            duration = end - start
            
            # Limit to target duration if segment is too long
            if duration > target_duration:
                end = start + target_duration
                duration = target_duration
            
            # Extract audio segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = audio[start_sample:end_sample]
            
            # Normalize and clean up
            segment = librosa.util.normalize(segment)
            
            # Save sample
            sample_name = f"sample_{i+1:02d}_{duration:.1f}s.wav"
            sample_path = Path(output_dir) / sample_name
            
            sf.write(sample_path, segment, sr)
            
            # Analyze quality
            quality_report = voice_manager.analyze_voice_quality(str(sample_path))
            
            print(f"✅ Sample {i+1}: {sample_name}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Quality: {quality_report.overall_score:.3f}/1.000")
            print(f"   Suitable: {'Yes' if quality_report.is_suitable else 'No'}")
            
            if quality_report.issues:
                print(f"   Issues: {', '.join(quality_report.issues[:2])}")
            
            extracted_samples.append({
                "path": str(sample_path),
                "duration": duration,
                "quality_score": quality_report.overall_score,
                "suitable": quality_report.is_suitable
            })
            print()
        
        # Summary
        suitable_samples = [s for s in extracted_samples if s["suitable"]]
        print(f"📊 Extraction Summary:")
        print(f"   Total samples extracted: {len(extracted_samples)}")
        print(f"   Suitable for cloning: {len(suitable_samples)}")
        
        if suitable_samples:
            best_sample = max(suitable_samples, key=lambda x: x["quality_score"])
            print(f"   Best sample: {Path(best_sample['path']).name}")
            print(f"   Best quality: {best_sample['quality_score']:.3f}")
        
        return extracted_samples
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return []


def download_sample_audio():
    """Provide instructions for downloading sample audio."""
    print("📥 How to Get Reference Audio:")
    print("=" * 40)
    print()
    
    print("🎤 Mozilla Common Voice (Recommended):")
    print("   1. Visit: https://commonvoice.mozilla.org/datasets")
    print("   2. Accept the license terms")
    print("   3. Download English dataset (or your preferred language)")
    print("   4. Extract clips from the 'clips' folder")
    print()
    
    print("📚 LibriVox Audiobooks:")
    print("   1. Visit: https://librivox.org/")
    print("   2. Browse catalog and find an interesting reader")
    print("   3. Download MP3 chapters")
    print("   4. Use this script to extract clean segments")
    print()
    
    print("🌐 Internet Archive:")
    print("   1. Visit: https://archive.org/details/librivoxaudio")
    print("   2. Search for specific readers or books")
    print("   3. Download individual chapter files")
    print()
    
    print("💡 Tips for Good Reference Audio:")
    print("   • Look for clear, single-speaker recordings")
    print("   • Avoid background music or sound effects")
    print("   • Choose consistent speaking pace and volume")
    print("   • 6-20 seconds is optimal for voice cloning")
    print("   • Multiple samples from same speaker = better results")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Extract voice samples for XTTS-v2 cloning')
    parser.add_argument('input_audio', nargs='?', help='Input audio file to extract from')
    parser.add_argument('--output-dir', default='voice_samples', 
                       help='Output directory for extracted samples')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Target duration for each sample (seconds)')
    parser.add_argument('--min-duration', type=float, default=6.0,
                       help='Minimum duration for samples (seconds)')
    parser.add_argument('--max-samples', type=int, default=5,
                       help='Maximum number of samples to extract')
    parser.add_argument('--download-info', action='store_true',
                       help='Show information about downloading reference audio')
    
    args = parser.parse_args()
    
    if args.download_info or not args.input_audio:
        download_sample_audio()
        
        if not args.input_audio:
            return
    
    # Extract samples from provided audio
    samples = extract_voice_samples(
        input_audio=args.input_audio,
        output_dir=args.output_dir,
        target_duration=args.duration,
        min_duration=args.min_duration,
        max_samples=args.max_samples
    )
    
    if samples:
        print("🎯 Ready for Voice Cloning!")
        print("   Use the best quality samples with demo_voice_cloning.py")
        print("   Example:")
        best_sample = max((s for s in samples if s["suitable"]), 
                         key=lambda x: x["quality_score"], default=None)
        if best_sample:
            sample_path = best_sample["path"]
            print(f'   python demo_voice_cloning.py "Hello world!" --reference-audio "{sample_path}"')


if __name__ == "__main__":
    main()