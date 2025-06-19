#!/usr/bin/env python3
"""
Test script to analyze actual audio durations and check for audio quality issues.
"""

import os
import librosa
import numpy as np
from pathlib import Path
from lib.tts.coqui_speech_generator import CoquiSpeechGenerator, CoquiTTSConfig
from lib.wiki.crawler import WikipediaCrawler
from lib.llm.discussion_generator import DiscussionGenerator

def get_actual_audio_duration(audio_path: str) -> float:
    """
    Get the actual duration of an audio file using librosa.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0.0

def analyze_audio_quality(audio_path: str) -> dict:
    """
    Analyze audio quality including silence detection and noise analysis.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate duration
        duration = len(y) / sr
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Detect silence (frames with very low energy)
        silence_threshold = np.mean(rms) * 0.1  # 10% of mean energy
        silent_frames = rms < silence_threshold
        silence_percentage = np.sum(silent_frames) / len(silent_frames) * 100
        
        # Detect potential noise (very high energy spikes)
        noise_threshold = np.mean(rms) + 3 * np.std(rms)  # 3 standard deviations above mean
        noisy_frames = rms > noise_threshold
        noise_percentage = np.sum(noisy_frames) / len(noisy_frames) * 100
        
        # Calculate dynamic range
        max_energy = np.max(rms)
        min_energy = np.min(rms[rms > 0])  # Avoid log(0)
        dynamic_range_db = 20 * np.log10(max_energy / min_energy) if min_energy > 0 else 0
        
        # Detect actual speech segments
        speech_threshold = np.mean(rms) * 0.3
        speech_frames = rms > speech_threshold
        speech_percentage = np.sum(speech_frames) / len(speech_frames) * 100
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "total_samples": len(y),
            "silence_percentage": silence_percentage,
            "noise_percentage": noise_percentage,
            "speech_percentage": speech_percentage,
            "dynamic_range_db": dynamic_range_db,
            "mean_energy": np.mean(rms),
            "max_energy": max_energy,
            "min_energy": min_energy,
            "energy_std": np.std(rms)
        }
        
    except Exception as e:
        print(f"Error analyzing audio quality: {e}")
        return {"error": str(e)}

def test_existing_audio_files():
    """Test existing audio files in the audio_output directory."""
    audio_dir = Path("audio_output")
    if not audio_dir.exists():
        print("No audio_output directory found")
        return
    
    print("=== Analyzing Existing Audio Files ===")
    
    for audio_file in audio_dir.glob("*.wav"):
        print(f"\n📁 File: {audio_file.name}")
        print(f"   File size: {audio_file.stat().st_size / 1024:.1f} KB")
        
        # Get actual duration
        actual_duration = get_actual_audio_duration(str(audio_file))
        print(f"   Actual duration: {actual_duration:.2f} seconds")
        
        # Analyze quality
        quality_analysis = analyze_audio_quality(str(audio_file))
        if "error" not in quality_analysis:
            print(f"   Sample rate: {quality_analysis['sample_rate']} Hz")
            print(f"   Silence: {quality_analysis['silence_percentage']:.1f}%")
            print(f"   Speech: {quality_analysis['speech_percentage']:.1f}%")
            print(f"   Noise: {quality_analysis['noise_percentage']:.1f}%")
            print(f"   Dynamic range: {quality_analysis['dynamic_range_db']:.1f} dB")
            
            # Flag potential issues
            if quality_analysis['silence_percentage'] > 50:
                print("   ⚠️  HIGH SILENCE DETECTED")
            if quality_analysis['noise_percentage'] > 10:
                print("   ⚠️  HIGH NOISE DETECTED")
            if quality_analysis['speech_percentage'] < 30:
                print("   ⚠️  LOW SPEECH CONTENT")
            if quality_analysis['dynamic_range_db'] < 10:
                print("   ⚠️  LOW DYNAMIC RANGE")
        else:
            print(f"   ❌ Analysis failed: {quality_analysis['error']}")

def test_coqui_duration_accuracy():
    """Test Coqui TTS and compare estimated vs actual durations."""
    print("\n=== Testing Coqui TTS Duration Accuracy ===")
    
    # Create a test monologue
    test_monologue = {
        'topic': 'test_duration',
        'script': [
            {'content': 'This is a test sentence to check duration accuracy.'},
            {'content': 'We want to see how close the estimated duration is to the actual duration.'},
            {'content': 'This helps us understand if the current estimation method is working correctly.'}
        ]
    }
    
    # Generate speech
    config = CoquiTTSConfig(model_name="tts_models/en/ljspeech/fast_pitch")
    generator = CoquiSpeechGenerator(config)
    
    if generator.tts:
        result = generator.generate_speech_from_monologue(test_monologue)
        
        if result["success"]:
            print(f"✅ Generated audio: {result['output_path']}")
            print(f"   Estimated duration: {result['estimated_duration']:.2f} seconds")
            print(f"   Word count: {result['text_word_count']}")
            
            # Get actual duration
            actual_duration = get_actual_audio_duration(result['output_path'])
            print(f"   Actual duration: {actual_duration:.2f} seconds")
            
            # Calculate accuracy
            if actual_duration > 0:
                accuracy = (1 - abs(result['estimated_duration'] - actual_duration) / actual_duration) * 100
                print(f"   Duration accuracy: {accuracy:.1f}%")
                
                # Analyze quality
                quality = analyze_audio_quality(result['output_path'])
                if "error" not in quality:
                    print(f"   Audio quality analysis:")
                    print(f"     - Silence: {quality['silence_percentage']:.1f}%")
                    print(f"     - Speech: {quality['speech_percentage']:.1f}%")
                    print(f"     - Sample rate: {quality['sample_rate']} Hz")
            else:
                print("   ❌ Could not determine actual duration")
        else:
            print(f"❌ Failed to generate speech: {result.get('error', 'Unknown error')}")
    else:
        print("❌ Coqui TTS not available")

def create_improved_duration_function():
    """Create an improved duration calculation function."""
    print("\n=== Improved Duration Calculation ===")
    
    improved_code = '''
def get_actual_audio_duration(audio_path: str) -> float:
    """
    Get the actual duration of an audio file using librosa.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        return len(y) / sr
    except ImportError:
        # Fallback to file-based approach if librosa not available
        try:
            import wave
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                return frames / sample_rate
        except:
            # Last resort: estimate based on file size (rough approximation)
            import os
            file_size = os.path.getsize(audio_path)
            # Rough estimate: 16-bit mono at 22050 Hz = ~44KB per second
            return file_size / 44100
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0.0

def analyze_and_fix_duration_in_coqui_generator(result_dict: dict, audio_path: str) -> dict:
    """
    Enhance the result dictionary with actual audio duration and quality analysis.
    
    Args:
        result_dict: The original result dictionary from generate_speech_from_monologue
        audio_path: Path to the generated audio file
        
    Returns:
        Enhanced result dictionary with actual duration and quality metrics
    """
    # Get actual duration
    actual_duration = get_actual_audio_duration(audio_path)
    
    # Calculate accuracy of original estimate
    estimated_duration = result_dict.get('estimated_duration', 0)
    accuracy = 0
    if actual_duration > 0 and estimated_duration > 0:
        accuracy = (1 - abs(estimated_duration - actual_duration) / actual_duration) * 100
    
    # Update result dictionary
    result_dict.update({
        'actual_duration': actual_duration,
        'duration_accuracy': accuracy,
        'duration_method': 'librosa_analysis'
    })
    
    # Optional: Add quality analysis
    try:
        import librosa
        import numpy as np
        
        y, sr = librosa.load(audio_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        
        # Basic quality metrics
        silence_threshold = np.mean(rms) * 0.1
        silent_frames = rms < silence_threshold
        silence_percentage = np.sum(silent_frames) / len(silent_frames) * 100
        
        result_dict['audio_quality'] = {
            'silence_percentage': silence_percentage,
            'sample_rate': sr,
            'mean_energy': float(np.mean(rms))
        }
        
        # Flag potential issues
        if silence_percentage > 50:
            result_dict['quality_warnings'] = result_dict.get('quality_warnings', [])
            result_dict['quality_warnings'].append('High silence content detected')
            
    except Exception as e:
        result_dict['quality_analysis_error'] = str(e)
    
    return result_dict
'''
    
    print("Here's the improved code for the Coqui TTS generator:")
    print(improved_code)

if __name__ == "__main__":
    # Test existing files
    test_existing_audio_files()
    
    # Test Coqui TTS duration accuracy
    test_coqui_duration_accuracy()
    
    # Show improved duration function
    create_improved_duration_function()