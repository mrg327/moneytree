#!/usr/bin/env python3
"""
Demo comparing natural ChatTTS vs Coqui TTS for speech generation.

Usage:
    python demo_natural_tts.py "Python programming language"
    python demo_natural_tts.py "Banana" --engine chattts --voice expressive
    python demo_natural_tts.py "Cat" --engine coqui --model fast_pitch
"""

import sys
import argparse
from pathlib import Path

from lib.wiki.crawler import WikipediaCrawler
from lib.llm.discussion_generator import HumorousDiscussionGenerator, DiscussionFormat
from lib.llm.llm_generator import LLMMonologueGenerator, LLMConfig
from lib.tts.coqui_speech_generator import CoquiSpeechGenerator, CoquiTTSConfig, get_recommended_models
from lib.tts.chattts_speech_generator import ChatTTSSpeechGenerator, ChatTTSConfig, get_recommended_voice_settings


def main():
    """Run the natural TTS comparison demo."""
    parser = argparse.ArgumentParser(description='MoneyTree: Natural vs Robotic TTS Comparison')
    parser.add_argument('topic', help='Wikipedia topic to process')
    parser.add_argument('--engine', choices=['chattts', 'coqui', 'both'], 
                       default='chattts', help='TTS engine to use')
    
    # ChatTTS options
    parser.add_argument('--voice', choices=['natural', 'expressive', 'calm', 'consistent'],
                       default='natural', help='ChatTTS voice style')
    
    # Coqui TTS options  
    parser.add_argument('--model', choices=['tacotron2', 'fast_pitch', 'vits', 'jenny', 'xtts_v2'], 
                       default='tacotron2', help='Coqui TTS model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', 
                       help='Device to use for TTS')
    
    # Voice cloning options (XTTS-v2)
    parser.add_argument('--clone-voice', help='Path to reference audio for voice cloning (6+ seconds)')
    parser.add_argument('--language', default='en', help='Language for XTTS-v2 generation')
    parser.add_argument('--analyze-reference', action='store_true',
                       help='Analyze quality of reference audio before cloning')
    
    # Generation options
    parser.add_argument('--use-rule-based', action='store_true', 
                       help='Use rule-based generator instead of LLM')
    parser.add_argument('--list-voices', action='store_true', help='List ChatTTS voice settings')
    parser.add_argument('--list-models', action='store_true', help='List Coqui TTS models')
    
    args = parser.parse_args()
    
    if args.list_voices:
        print("🗣️  ChatTTS Voice Settings:")
        for voice in get_recommended_voice_settings():
            print(f"   • {voice['name']}: {voice['description']}")
            print(f"     Temperature: {voice['temperature']}, Top-K: {voice['top_k']}, Top-P: {voice['top_p']}")
        return
    
    if args.list_models:
        print("🎙️  Coqui TTS Models:")
        for model in get_recommended_models():
            print(f"   • {model}")
        return
    
    print("🌳 MoneyTree")
    print("=" * 60)
    print(f"📖 Topic: {args.topic}")
    print(f"🎙️  Engine: {args.engine}")
    if args.engine in ['chattts', 'both']:
        print(f"🗣️  Voice style: {args.voice}")
    if args.engine in ['coqui', 'both']:
        print(f"🤖 Coqui model: {args.model}")
    
    # Step 1: Get Wikipedia content
    print("\\n1️⃣ Fetching Wikipedia content...")
    crawler = WikipediaCrawler()
    content = crawler.get_page_summary(args.topic)
    
    if not content:
        print(f"❌ Could not find Wikipedia page for '{args.topic}'")
        print("💡 Try a different topic or check the spelling!")
        return
    
    print(f"✅ Found: {content.get('title', args.topic)}")
    print(f"📝 Description: {content.get('description', 'No description')}")
    
    # Step 2: Generate content
    print("\\n2️⃣ Generating educational content...")
    
    if args.use_rule_based:
        print("📝 Using rule-based generation...")
        generator = HumorousDiscussionGenerator()
        monologue = generator.generate_discussion(content, DiscussionFormat.MONOLOGUE)
    else:
        print("🤖 Using LLM-based generation...")
        config = LLMConfig()
        generator = LLMMonologueGenerator(config)
        monologue = generator.generate_monologue(content, target_length=180)
        
        if monologue['model_used'] == 'fallback':
            print("⚠️  LLM not available, falling back to rule-based generation")
            generator = HumorousDiscussionGenerator()
            monologue = generator.generate_discussion(content, DiscussionFormat.MONOLOGUE)
    
    print(f"✅ Generated content ({monologue['word_count']} words)")
    print(f"⏱️  Estimated duration: {monologue['estimated_duration']:.1f} seconds")
    
    # Show generation method used
    if 'model_used' in monologue:
        method = "LLM" if monologue['model_used'] != 'fallback' else "Rule-based (LLM fallback)"
    else:
        method = "Rule-based"
    print(f"🧠 Generation method: {method}")
    
    # Step 3: Generate speech with selected engine(s)
    results = {}
    
    if args.engine in ['chattts', 'both']:
        print("\\n3️⃣ Generating natural speech with ChatTTS...")
        
        # Get voice settings
        voice_settings = next((v for v in get_recommended_voice_settings() if v['name'] == args.voice), 
                             get_recommended_voice_settings()[0])
        
        chattts_config = ChatTTSConfig(
            temperature=voice_settings['temperature'],
            top_k=voice_settings['top_k'],
            top_p=voice_settings['top_p'],
            device=args.device
        )
        
        chattts_gen = ChatTTSSpeechGenerator(chattts_config)
        
        if chattts_gen.chat:
            result = chattts_gen.generate_speech_from_monologue(monologue)
            results['chattts'] = result
            
            if result['success']:
                print(f"✅ Natural speech generated!")
                print(f"💾 Saved to: {result['output_path']}")
                print(f"📦 File size: {result['file_size']:,} bytes")
                print(f"⏱️  Duration: {result['estimated_duration']:.1f} seconds")
                print(f"🗣️  Voice style: {args.voice}")
                
                # Provide Windows path
                windows_path = str(result['output_path']).replace('/mnt/c/', 'C:\\\\').replace('/', '\\\\')
                print(f"🪟 Windows path: {windows_path}")
            else:
                print(f"❌ ChatTTS generation failed: {result.get('error', 'Unknown error')}")
        else:
            print("❌ ChatTTS engine not available")
            print("💡 Install with: uv pip install ChatTTS torch numpy scipy")
    
    if args.engine in ['coqui', 'both']:
        print("\\n4️⃣ Generating traditional speech with Coqui TTS...")
        
        # Handle voice cloning analysis first
        if args.analyze_reference and args.clone_voice:
            print(f"🔍 Analyzing reference audio quality...")
            from lib.tts.voice_cloning import VoiceManager
            
            voice_manager = VoiceManager()
            quality_report = voice_manager.analyze_voice_quality(args.clone_voice)
            
            print(f"📊 Quality Analysis:")
            print(f"   Overall Score: {quality_report.overall_score:.3f}/1.000")
            print(f"   Suitable for cloning: {'Yes' if quality_report.is_suitable else 'No'}")
            
            if quality_report.issues:
                print(f"   Issues: {', '.join(quality_report.issues)}")
            if quality_report.recommendations:
                print(f"   Recommendations: {quality_report.recommendations[0]}")
        
        # Configure Coqui TTS
        if args.model == 'xtts_v2' or args.clone_voice:
            # Use XTTS-v2 for voice cloning
            from lib.tts.coqui_speech_generator import create_voice_cloning_config
            
            if args.clone_voice:
                print(f"🎭 Using XTTS-v2 with voice cloning")
                print(f"   Reference: {Path(args.clone_voice).name}")
                print(f"   Language: {args.language}")
                
                coqui_config = create_voice_cloning_config(
                    reference_audio=args.clone_voice,
                    language=args.language,
                    gpu=(args.device == 'cuda')
                )
            else:
                print(f"🎭 Using XTTS-v2 with default voice")
                coqui_config = CoquiTTSConfig.for_xtts_v2(
                    speaker_wav=None,
                    language=args.language,
                    gpu=(args.device == 'cuda')
                )
        else:
            # Use traditional models
            model_map = {
                'tacotron2': "tts_models/en/ljspeech/tacotron2-DDC",
                'fast_pitch': "tts_models/en/ljspeech/fast_pitch", 
                'vits': "tts_models/en/vctk/vits",
                'jenny': "tts_models/en/jenny/jenny"
            }
            
            coqui_config = CoquiTTSConfig(
                model_name=model_map[args.model],
                device=args.device,
                sample_rate=22050,
                output_format="wav"
            )
        
        coqui_gen = CoquiSpeechGenerator(coqui_config)
        
        if coqui_gen.tts:
            result = coqui_gen.generate_speech_from_monologue(monologue)
            results['coqui'] = result
            
            if result['success']:
                print(f"✅ Traditional speech generated!")
                print(f"💾 Saved to: {result['output_path']}")
                print(f"📦 File size: {result['file_size']:,} bytes")
                print(f"⏱️  Duration: {result['estimated_duration']:.1f} seconds")
                print(f"🤖 Model: {args.model}")
                
                # Provide Windows path
                windows_path = str(result['output_path']).replace('/mnt/c/', 'C:\\\\').replace('/', '\\\\')
                print(f"🪟 Windows path: {windows_path}")
            else:
                print(f"❌ Coqui TTS generation failed: {result.get('error', 'Unknown error')}")
        else:
            print("❌ Coqui TTS engine not available")
    
    # Final comparison
    if len(results) > 1:
        print(f"\\n🎯 TTS Comparison:")
        print(f"   📖 Topic: {content.get('title', args.topic)}")
        print(f"   📝 Content: {method} generation")
        
        for engine, result in results.items():
            if result['success']:
                engine_name = "Natural (ChatTTS)" if engine == 'chattts' else "Traditional (Coqui)"
                print(f"   🎙️  {engine_name}:")
                print(f"      Duration: {result['estimated_duration']:.1f}s")
                print(f"      File size: {result['file_size']:,} bytes")
                print(f"      Quality: {'Conversational' if engine == 'chattts' else 'Synthetic'}")
    elif len(results) == 1:
        engine = list(results.keys())[0]
        result = results[engine]
        if result['success']:
            engine_name = "Natural ChatTTS" if engine == 'chattts' else "Traditional Coqui TTS"
            print(f"\\n🎉 {engine_name} Pipeline Complete!")
            print(f"   📖 Topic: {content.get('title', args.topic)}")
            print(f"   📝 Generator: {method}")
            print(f"   🎙️  Engine: {engine_name}")
            print(f"   ⏱️  Duration: {result['estimated_duration']:.1f} seconds")


if __name__ == "__main__":
    main()
