#!/usr/bin/env python3
"""
Demo using Coqui TTS for high-quality speech generation.

Usage:
    python demo_coqui_tts.py "Python programming language"
    python demo_coqui_tts.py "Banana" --model fast_pitch
"""

import sys
import argparse
from pathlib import Path

from lib.wiki.crawler import WikipediaCrawler
from lib.llm.discussion_generator import HumorousDiscussionGenerator, DiscussionFormat
from lib.llm.llm_generator import LLMMonologueGenerator, LLMConfig
from lib.tts.coqui_speech_generator import CoquiSpeechGenerator, CoquiTTSConfig, get_recommended_models


def main():
    """Run the Coqui TTS demo."""
    parser = argparse.ArgumentParser(description='MoneyTree: Wikipedia to Speech with Coqui TTS')
    parser.add_argument('topic', help='Wikipedia topic to process')
    parser.add_argument('--model', choices=['tacotron2', 'fast_pitch', 'vits', 'jenny'], 
                       default='tacotron2', help='TTS model to use')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', 
                       help='Device to use for TTS')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--use-rule-based', action='store_true', 
                       help='Use rule-based generator instead of LLM')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("🎙️  Recommended Coqui TTS Models:")
        for model in get_recommended_models():
            print(f"   • {model}")
        return
    
    print("🌳 MoneyTree: Wikipedia → LLM → High-Quality Speech (Coqui TTS)")
    print("=" * 65)
    print(f"📖 Topic: {args.topic}")
    if args.use_rule_based:
        print("⚙️  Mode: Rule-based generation")
    else:
        print("🤖 Mode: LLM-powered generation (with fallback)")
    
    # Step 1: Get Wikipedia content
    print("\n1️⃣ Fetching Wikipedia content...")
    crawler = WikipediaCrawler()
    content = crawler.get_page_summary(args.topic)
    
    if not content:
        print(f"❌ Could not find Wikipedia page for '{args.topic}'")
        print("💡 Try a different topic or check the spelling!")
        return
    
    print(f"✅ Found: {content.get('title', args.topic)}")
    print(f"📝 Description: {content.get('description', 'No description')}")
    
    # Step 2: Generate humorous content
    print("\n2️⃣ Generating educational content...")
    
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
    
    # Show the generated text
    print("\n📄 Generated Content:")
    print("-" * 40)
    
    if 'script' in monologue and monologue['script']:
        # Handle script format (rule-based or LLM with script)
        for turn in monologue['script']:
            if hasattr(turn, 'content'):
                print(turn.content)
            elif isinstance(turn, dict):
                print(turn.get('content', ''))
            print()
    elif 'generated_text' in monologue:
        # Handle LLM format (direct text)
        print(monologue['generated_text'])
        print()
    else:
        print("No content generated")
        print()
    
    print("-" * 40)
    
    # Step 3: Generate high-quality speech with Coqui TTS
    print("\n3️⃣ Generating high-quality speech with Coqui TTS...")
    
    # Configure Coqui TTS
    model_map = {
        'tacotron2': "tts_models/en/ljspeech/tacotron2-DDC",
        'fast_pitch': "tts_models/en/ljspeech/fast_pitch", 
        'vits': "tts_models/en/vctk/vits",
        'jenny': "tts_models/en/jenny/jenny"
    }
    
    tts_config = CoquiTTSConfig(
        model_name=model_map[args.model],
        device=args.device,
        sample_rate=22050,
        output_format="wav"
    )
    
    speech_gen = CoquiSpeechGenerator(tts_config)
    
    if speech_gen.tts:
        result = speech_gen.generate_speech_from_monologue(monologue)
        
        if result['success']:
            print(f"✅ High-quality speech generated!")
            print(f"💾 Saved to: {result['output_path']}")
            print(f"📦 File size: {result['file_size']:,} bytes")
            print(f"⏱️  Duration: {result['estimated_duration']:.1f} seconds")
            print(f"📊 Words: {result['text_word_count']}")
            print(f"🤖 Model: {result['tts_config']['model']}")
            
            # Provide Windows path
            windows_path = str(result['output_path']).replace('/mnt/c/', 'C:\\').replace('/', '\\')
            print(f"🪟 Windows path: {windows_path}")
            
        else:
            print(f"❌ Speech generation failed: {result.get('error', 'Unknown error')}")
    else:
        print("❌ Coqui TTS engine not available")
        print("💡 Check your installation or internet connection for model download")
    
    # Final summary
    print(f"\n🎉 Pipeline Summary:")
    print(f"   📖 Topic: {content.get('title', args.topic)}")
    print(f"   📝 Generator: {method}")
    print(f"   🎙️  TTS Engine: Coqui TTS")
    print(f"   🤖 TTS Model: {args.model}")
    print(f"   ⏱️  Total duration: ~{monologue['estimated_duration']:.1f} seconds")


if __name__ == "__main__":
    main()