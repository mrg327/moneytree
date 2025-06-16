"""Integration tests for the complete MoneyTree pipeline."""

import pytest
import os
from pathlib import Path
import tempfile

# Import all pipeline components
from lib.wiki.crawler import WikipediaCrawler
from lib.llm.discussion_generator import HumorousDiscussionGenerator, DiscussionFormat
from lib.tts.chattts_speech_generator import ChatTTSSpeechGenerator, ChatTTSConfig
from lib.video.clip import VideoClip, CaptionStyle, VideoConfig, create_sample_template


class TestFullPipeline:
    """Test complete pipeline integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_topic = "Cat"
        self.template_path = "downloads/videos/minecraft_parkour.mp4" 
        self.output_dir = Path("video_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create a temporary template if main one doesn't exist
        self.temp_template = None

    def test_wikipedia_to_content_pipeline(self):
        """Test Wikipedia → Content Generation pipeline."""
        try:
            print("🧪 Testing Wikipedia → Content pipeline...")
            
            # Step 1: Fetch Wikipedia content
            crawler = WikipediaCrawler()
            wiki_content = crawler.get_page_summary(self.test_topic)
            
            if not wiki_content:
                print(f"❌ Wikipedia fetch failed for '{self.test_topic}'")
                return False
            
            print(f"✅ Wikipedia: {wiki_content.get('title', 'Unknown')}")
            
            # Step 2: Generate content
            generator = HumorousDiscussionGenerator()
            monologue = generator.generate_discussion(wiki_content, DiscussionFormat.MONOLOGUE)
            
            assert monologue is not None
            assert monologue.get('word_count', 0) > 0
            
            print(f"✅ Content: {monologue['word_count']} words, {monologue['estimated_duration']:.1f}s")
            return True
            
        except Exception as e:
            print(f"❌ Wikipedia → Content pipeline failed: {e}")
            return False

    def test_content_to_audio_pipeline(self):
        """Test Content → Audio pipeline."""
        try:
            print("🧪 Testing Content → Audio pipeline...")
            
            # Generate content first
            crawler = WikipediaCrawler()
            wiki_content = crawler.get_page_summary(self.test_topic)
            
            if not wiki_content:
                print("⚠️ Skipping audio pipeline (no Wikipedia content)")
                return False
            
            generator = HumorousDiscussionGenerator()
            monologue = generator.generate_discussion(wiki_content, DiscussionFormat.MONOLOGUE)
            
            # Try ChatTTS generation
            try:
                config = ChatTTSConfig()
                speech_gen = ChatTTSSpeechGenerator(config)
                
                if speech_gen.chat is None:
                    print("⚠️ ChatTTS not available, skipping audio test")
                    return False
                
                audio_result = speech_gen.generate_speech_from_monologue(monologue)
                
                if audio_result.get('success'):
                    audio_path = audio_result['output_path']
                    if Path(audio_path).exists():
                        file_size = Path(audio_path).stat().st_size
                        print(f"✅ Audio: {Path(audio_path).name} ({file_size} bytes)")
                        return True
                    else:
                        print("❌ Audio file not created")
                        return False
                else:
                    print(f"❌ Audio generation failed: {audio_result.get('error')}")
                    return False
                    
            except Exception as e:
                print(f"❌ Audio generation error: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Content → Audio pipeline failed: {e}")
            return False

    def test_audio_to_video_pipeline(self):
        """Test Audio → Video pipeline."""
        try:
            print("🧪 Testing Audio → Video pipeline...")
            
            # Check for existing audio file
            audio_files = list(Path("audio_output").glob("*_chattts.wav"))
            if not audio_files:
                print("⚠️ No audio files found, skipping video pipeline")
                return False
            
            audio_path = str(audio_files[0])
            print(f"📁 Using audio: {Path(audio_path).name}")
            
            # Check template
            if not Path(self.template_path).exists():
                print("⚠️ Creating temporary template...")
                self.temp_template = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                create_sample_template(self.temp_template.name, duration=30.0, vertical=True)
                template_path = self.temp_template.name
            else:
                template_path = self.template_path
            
            # Test video creation (but don't actually render)
            with VideoClip(template_path) as video_clip:
                # Test audio integration
                narration_result = video_clip.add_narration_audio(audio_path)
                
                if not narration_result.get('success'):
                    print(f"❌ Audio integration failed: {narration_result.get('error')}")
                    return False
                
                print(f"✅ Audio integrated: {narration_result['narration_duration']:.1f}s")
                
                # Test caption generation
                test_text = "This is a test caption for video integration."
                caption_result = video_clip.add_synchronized_captions(
                    test_text, 
                    audio_path, 
                    CaptionStyle.for_fast_rendering()
                )
                
                if caption_result.get('success'):
                    print(f"✅ Captions: {caption_result['caption_count']} segments")
                else:
                    print(f"⚠️ Caption generation issues: {caption_result}")
                
                return True
                
        except Exception as e:
            print(f"❌ Audio → Video pipeline failed: {e}")
            return False
        
        finally:
            # Clean up temporary template
            if self.temp_template and Path(self.temp_template.name).exists():
                Path(self.temp_template.name).unlink()

    def test_fast_end_to_end_pipeline(self):
        """Test complete pipeline with fast settings."""
        try:
            print("🧪 Testing Fast End-to-End pipeline...")
            
            # Step 1: Wikipedia content
            crawler = WikipediaCrawler()
            wiki_content = crawler.get_page_summary(self.test_topic)
            
            if not wiki_content:
                print("❌ Wikipedia fetch failed")
                return False
            
            # Step 2: Generate content (rule-based for speed)
            generator = HumorousDiscussionGenerator()
            monologue = generator.generate_discussion(wiki_content, DiscussionFormat.MONOLOGUE)
            
            # Extract caption text
            text_parts = []
            if 'script' in monologue:
                for turn in monologue['script']:
                    if hasattr(turn, 'content'):
                        text_parts.append(turn.content)
                    elif isinstance(turn, dict):
                        text_parts.append(turn.get('content', ''))
            caption_text = ' '.join(text_parts) if text_parts else "Test caption text"
            
            print(f"✅ Content: {monologue['word_count']} words")
            
            # Step 3: Check if we can skip TTS (use existing audio)
            audio_files = list(Path("audio_output").glob(f"{self.test_topic}_chattts.wav"))
            if audio_files:
                audio_path = str(audio_files[0])
                print(f"✅ Using existing audio: {Path(audio_path).name}")
            else:
                print("⚠️ No existing audio, would need TTS generation")
                return False
            
            # Step 4: Video setup (fast config)
            template_path = self.template_path if Path(self.template_path).exists() else None
            
            if not template_path:
                print("⚠️ No template available")
                return False
            
            # Test fast video configuration
            fast_config = VideoConfig.for_fast_rendering()
            fast_caption_style = CaptionStyle.for_fast_rendering()
            
            print(f"✅ Fast config: {fast_config.get_target_resolution()}, {fast_config.fps}fps, {fast_config.preset}")
            
            # Test video component setup (don't render)
            with VideoClip(template_path) as video_clip:
                video_clip.video_config = fast_config
                video_clip.caption_style = fast_caption_style
                
                # Test all components can be added
                narration_result = video_clip.add_narration_audio(audio_path)
                caption_result = video_clip.add_synchronized_captions(
                    caption_text, audio_path, fast_caption_style
                )
                
                success = (narration_result.get('success', False) and 
                          caption_result.get('success', False))
                
                if success:
                    print("✅ Fast pipeline components ready")
                    print(f"   📊 Audio: {narration_result['narration_duration']:.1f}s")
                    print(f"   📝 Captions: {caption_result['caption_count']} segments")
                    return True
                else:
                    print("❌ Fast pipeline component setup failed")
                    return False
                    
        except Exception as e:
            print(f"❌ Fast end-to-end pipeline failed: {e}")
            return False

    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery."""
        try:
            print("🧪 Testing Pipeline Error Handling...")
            
            # Test with invalid Wikipedia topic
            crawler = WikipediaCrawler()
            invalid_result = crawler.get_page_summary("InvalidTopicThatDoesNotExist12345")
            
            if invalid_result is None or invalid_result == {}:
                print("✅ Invalid Wikipedia topic handled correctly")
            else:
                print("⚠️ Unexpected result for invalid topic")
            
            # Test with invalid audio path
            if Path(self.template_path).exists():
                try:
                    with VideoClip(self.template_path) as video_clip:
                        result = video_clip.add_narration_audio("nonexistent_audio.wav")
                        
                        if not result.get('success'):
                            print("✅ Invalid audio path handled correctly")
                        else:
                            print("⚠️ Invalid audio path not caught")
                            
                except Exception as e:
                    print(f"✅ Audio error handled: {e}")
            
            # Test content generation with minimal content
            minimal_content = {'title': 'Test', 'content': 'Short.', 'description': 'Test'}
            generator = HumorousDiscussionGenerator()
            
            try:
                result = generator.generate_discussion(minimal_content, DiscussionFormat.MONOLOGUE)
                if result and result.get('word_count', 0) > 0:
                    print("✅ Minimal content handled correctly")
                else:
                    print("⚠️ Minimal content processing issues")
            except Exception as e:
                print(f"⚠️ Minimal content error: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False

    def test_pipeline_performance_estimates(self):
        """Test pipeline performance and timing estimates."""
        try:
            print("🧪 Testing Pipeline Performance Estimates...")
            
            # Estimate Wikipedia fetch time
            import time
            start_time = time.time()
            
            crawler = WikipediaCrawler()
            wiki_content = crawler.get_page_summary(self.test_topic)
            
            wiki_time = time.time() - start_time
            print(f"✅ Wikipedia fetch: {wiki_time:.2f}s")
            
            if not wiki_content:
                return False
            
            # Estimate content generation time
            start_time = time.time()
            
            generator = HumorousDiscussionGenerator()
            monologue = generator.generate_discussion(wiki_content, DiscussionFormat.MONOLOGUE)
            
            content_time = time.time() - start_time
            print(f"✅ Content generation: {content_time:.2f}s")
            
            # Estimate configuration time
            start_time = time.time()
            
            fast_config = VideoConfig.for_fast_rendering()
            production_config = VideoConfig.for_production()
            target_res_fast = fast_config.get_target_resolution()
            target_res_prod = production_config.get_target_resolution()
            
            config_time = time.time() - start_time
            print(f"✅ Configuration setup: {config_time:.4f}s")
            
            # Performance comparison
            fast_pixels = target_res_fast[0] * target_res_fast[1]
            prod_pixels = target_res_prod[0] * target_res_prod[1]
            pixel_ratio = prod_pixels / fast_pixels
            
            print(f"📊 Performance estimates:")
            print(f"   Fast: {target_res_fast[0]}x{target_res_fast[1]} ({fast_pixels:,} pixels)")
            print(f"   Production: {target_res_prod[0]}x{target_res_prod[1]} ({prod_pixels:,} pixels)")
            print(f"   Production is {pixel_ratio:.1f}x more pixels than fast")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance estimates failed: {e}")
            return False


if __name__ == "__main__":
    # Run integration tests manually
    test_pipeline = TestFullPipeline()
    test_pipeline.setup_method()
    
    print("🧪 Testing Full Pipeline Integration...")
    print("=" * 60)
    
    # Individual pipeline segment tests
    print("\\n📊 Pipeline Segment Tests:")
    test_pipeline.test_wikipedia_to_content_pipeline()
    test_pipeline.test_content_to_audio_pipeline()
    test_pipeline.test_audio_to_video_pipeline()
    
    # Complete pipeline tests
    print("\\n🚀 Complete Pipeline Tests:")
    test_pipeline.test_fast_end_to_end_pipeline()
    
    # Error handling tests
    print("\\n🛡️ Error Handling Tests:")
    test_pipeline.test_pipeline_error_handling()
    
    # Performance analysis
    print("\\n⚡ Performance Analysis:")
    test_pipeline.test_pipeline_performance_estimates()
    
    print("\\n🏁 Integration tests complete")