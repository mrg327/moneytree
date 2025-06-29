#!/usr/bin/env python3
"""
Demo showcasing XTTS-v2 voice cloning capabilities.

Demonstrates voice cloning, quality analysis, and popular model usage.
"""

import sys
import argparse
import os
from pathlib import Path

from lib.utils.logging_config import setup_logging, get_logger
from lib.tts.coqui_speech_generator import (
    CoquiSpeechGenerator, CoquiTTSConfig, 
    get_recommended_models, get_xtts_supported_languages,
    create_voice_cloning_config
)
from lib.tts.voice_cloning import VoiceManager

# Setup logging
setup_logging(log_level="INFO", console_output=True)
logger = get_logger(__name__)


def main():
    """Run the voice cloning demo."""
    parser = argparse.ArgumentParser(description='MoneyTree: XTTS-v2 Voice Cloning Demo')
    parser.add_argument('text', nargs='?', default="Hello, this is a test of voice cloning using XTTS-v2. The quality should be quite impressive!", 
                       help='Text to convert to speech')
    parser.add_argument('--reference-audio', required=False,
                       help='Path to reference audio for voice cloning (6+ seconds recommended)')
    parser.add_argument('--language', default='en', choices=[lang['code'] for lang in get_xtts_supported_languages()],
                       help='Language for speech generation')
    parser.add_argument('--model-preset', choices=['xtts_v2', 'best_quality', 'fast', 'male_voice', 'female_voice'],
                       default='xtts_v2', help='Model preset to use')
    parser.add_argument('--list-models', action='store_true',
                       help='List available model presets and exit')
    parser.add_argument('--list-languages', action='store_true', 
                       help='List supported languages and exit')
    parser.add_argument('--analyze-voice', 
                       help='Analyze quality of a voice reference file')
    parser.add_argument('--voice-library', action='store_true',
                       help='Show voice library management demo')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration if available')
    parser.add_argument('--output', help='Output audio file path')
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_models:
        show_available_models()
        return
    
    if args.list_languages:
        show_supported_languages()
        return
    
    if args.analyze_voice:
        analyze_voice_quality(args.analyze_voice)
        return
    
    if args.voice_library:
        voice_library_demo()
        return
    
    # Generate output path if not provided
    if not args.output:
        output_dir = Path("audio_output")
        output_dir.mkdir(exist_ok=True)
        
        if args.reference_audio:
            ref_name = Path(args.reference_audio).stem
            args.output = output_dir / f"cloned_{ref_name}_{args.language}.wav"
        else:
            args.output = output_dir / f"demo_{args.model_preset}_{args.language}.wav"
    
    # Run the appropriate demo
    if args.model_preset == 'xtts_v2' or args.reference_audio:
        voice_cloning_demo(args)
    else:
        model_preset_demo(args)


def show_available_models():
    """Display available model presets."""
    print("🤖 Available Model Presets:")
    print("=" * 60)
    
    models = get_recommended_models()
    for model in models:
        print(f"\n📋 {model['name']}")
        print(f"   Model: {model['model_name']}")
        print(f"   Description: {model['description']}")
        print(f"   Quality: {model['quality']} | Speed: {model['speed']}")
        print(f"   Use case: {model['use_case']}")
        print(f"   Languages: {', '.join(model['languages'])}")
        if 'features' in model:
            print(f"   Features: {', '.join(model['features'])}")


def show_supported_languages():
    """Display supported languages for XTTS-v2."""
    print("🌍 XTTS-v2 Supported Languages:")
    print("=" * 40)
    
    languages = get_xtts_supported_languages()
    for lang in languages:
        print(f"   {lang['code']:6} - {lang['name']}")


def analyze_voice_quality(voice_path: str):
    """Analyze the quality of a voice reference."""
    print(f"🔍 Analyzing voice quality: {Path(voice_path).name}")
    print("=" * 50)
    
    if not os.path.exists(voice_path):
        print(f"❌ Error: Voice file not found: {voice_path}")
        return
    
    # Use voice manager for analysis
    voice_manager = VoiceManager()
    quality_report = voice_manager.analyze_voice_quality(voice_path)
    
    print(f"📊 Quality Analysis Results:")
    print(f"   Overall Score: {quality_report.overall_score:.3f}/1.000")
    print(f"   Duration Score: {quality_report.duration_score:.3f}")
    print(f"   Clarity Score: {quality_report.clarity_score:.3f}")
    print(f"   Consistency Score: {quality_report.consistency_score:.3f}")
    print(f"   Noise Level: {quality_report.noise_level:.3f}")
    
    print(f"\n✅ Suitable for cloning: {'Yes' if quality_report.is_suitable else 'No'}")
    
    if quality_report.issues:
        print(f"\n⚠️ Issues Found:")
        for issue in quality_report.issues:
            print(f"   • {issue}")
    
    if quality_report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in quality_report.recommendations:
            print(f"   • {rec}")


def voice_library_demo():
    """Demonstrate voice library management."""
    print("📚 Voice Library Management Demo")
    print("=" * 40)
    
    voice_manager = VoiceManager("voice_library")
    
    # Show current library
    voices = voice_manager.list_voices()
    print(f"📋 Current Voice Library ({len(voices)} voices):")
    
    if voices:
        for name, info in voices.items():
            print(f"   🎤 {name}")
            print(f"      Language: {info['language']}")
            print(f"      Duration: {info['duration']:.1f}s")
            print(f"      Quality: {info['quality_score']:.3f}")
            if info['description']:
                print(f"      Description: {info['description']}")
            print()
    else:
        print("   (No voices in library)")
    
    # Show how to add voices
    print("💡 To add voices to your library:")
    print("   voice_manager.add_voice_reference(")
    print("       source_path='path/to/reference.wav',")
    print("       name='Speaker Name',")
    print("       language='en',")
    print("       description='Clear female voice'")
    print("   )")


def voice_cloning_demo(args):
    """Demonstrate voice cloning with XTTS-v2."""
    print("🎭 Voice Cloning Demo with XTTS-v2")
    print("=" * 50)
    
    if not args.reference_audio:
        print("⚠️ No reference audio provided. Using default XTTS-v2 voice.")
        config = CoquiTTSConfig.for_xtts_v2(
            speaker_wav=None,
            language=args.language,
            gpu=args.gpu
        )
    else:
        print(f"📁 Reference audio: {Path(args.reference_audio).name}")
        print(f"🌍 Language: {args.language}")
        
        # Analyze reference audio quality first
        voice_manager = VoiceManager()
        quality_report = voice_manager.analyze_voice_quality(args.reference_audio)
        
        print(f"\n📊 Reference Audio Analysis:")
        print(f"   Quality Score: {quality_report.overall_score:.3f}/1.000")
        print(f"   Suitable: {'Yes' if quality_report.is_suitable else 'No'}")
        
        if quality_report.issues:
            print(f"   Issues: {', '.join(quality_report.issues)}")
        
        # Create voice cloning configuration
        config = create_voice_cloning_config(
            reference_audio=args.reference_audio,
            language=args.language,
            gpu=args.gpu
        )
    
    print(f"\n🎤 Generating speech...")
    print(f"   Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
    print(f"   Output: {args.output}")
    
    try:
        # Initialize TTS generator
        generator = CoquiSpeechGenerator(config)
        
        if args.reference_audio and os.path.exists(args.reference_audio):
            # Use direct voice cloning method
            result = generator.clone_voice_from_audio(
                reference_audio=args.reference_audio,
                text=args.text,
                output_path=str(args.output),
                language=args.language
            )
        else:
            # Use standard generation with XTTS-v2
            result = generator.generate_speech_from_monologue(
                {"script": [{"content": args.text}], "topic": "demo"},
                str(args.output)
            )
        
        if result.get('success', False):
            print(f"\n✅ Voice cloning successful!")
            print(f"   Output file: {result.get('output_path', args.output)}")
            print(f"   Duration: {result.get('duration', 0):.1f} seconds")
            print(f"   File size: {result.get('file_size', 0):,} bytes")
            
            if 'voice_cloning' in result:
                print(f"   Voice cloning: {result['voice_cloning']}")
                print(f"   Engine: {result.get('engine', 'unknown')}")
        else:
            print(f"❌ Voice cloning failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def model_preset_demo(args):
    """Demonstrate model presets."""
    print(f"🤖 Model Preset Demo: {args.model_preset}")
    print("=" * 50)
    
    # Get model info
    models = get_recommended_models()
    model_info = next((m for m in models if args.model_preset in m['model_name'].lower() or 
                      args.model_preset in m['name'].lower()), None)
    
    if model_info:
        print(f"📋 Using: {model_info['name']}")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Description: {model_info['description']}")
        print(f"   Quality: {model_info['quality']} | Speed: {model_info['speed']}")
    
    # Create configuration
    config = CoquiTTSConfig.for_popular_model(
        model_preset=args.model_preset,
        device="cuda" if args.gpu else "cpu"
    )
    
    print(f"\n🎤 Generating speech...")
    print(f"   Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
    print(f"   Output: {args.output}")
    
    try:
        # Initialize TTS generator
        generator = CoquiSpeechGenerator(config)
        
        # Generate speech
        result = generator.generate_speech_from_monologue(
            {"script": [{"content": args.text}], "topic": "demo"},
            str(args.output)
        )
        
        if result.get('success', False):
            print(f"\n✅ Speech generation successful!")
            print(f"   Output file: {result.get('output_path', args.output)}")
            print(f"   Duration: {result.get('duration', 0):.1f} seconds")
            print(f"   File size: {result.get('file_size', 0):,} bytes")
            print(f"   Engine: {result.get('engine', 'coqui-tts')}")
        else:
            print(f"❌ Speech generation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
