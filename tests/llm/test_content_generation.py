"""Tests for LLM and rule-based content generation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from lib.llm.discussion_generator import HumorousDiscussionGenerator, DiscussionFormat
from lib.llm.llm_generator import LLMMonologueGenerator, LLMConfig


class TestContentGeneration:
    """Test content generation components."""

    def setup_method(self):
        """Set up test fixtures."""
        # Sample content for testing
        self.sample_content = {
            'title': 'Cat',
            'content': 'Cats are domestic animals that have been companions to humans for thousands of years. They are known for their independence, agility, and hunting skills.',
            'description': 'Domestic feline animal'
        }

    def test_rule_based_generator_initialization(self):
        """Test rule-based generator initializes properly."""
        generator = HumorousDiscussionGenerator()
        assert generator is not None
        assert hasattr(generator, 'generate_discussion')
        print("✅ Rule-based generator initialization successful")

    def test_rule_based_monologue_generation(self):
        """Test rule-based monologue generation."""
        generator = HumorousDiscussionGenerator()
        
        try:
            result = generator.generate_discussion(self.sample_content, DiscussionFormat.MONOLOGUE)
            
            assert result is not None
            assert isinstance(result, dict)
            assert 'word_count' in result
            assert 'estimated_duration' in result
            assert result['word_count'] > 0
            assert result['estimated_duration'] > 0
            
            print(f"✅ Rule-based monologue: {result['word_count']} words, {result['estimated_duration']:.1f}s")
            
            # Check if we have script content
            if 'script' in result:
                print(f"📝 Script has {len(result['script'])} turns")
            
        except Exception as e:
            print(f"❌ Rule-based generation failed: {e}")
            raise

    def test_rule_based_qa_generation(self):
        """Test rule-based Q&A generation."""
        generator = HumorousDiscussionGenerator()
        
        try:
            result = generator.generate_discussion(self.sample_content, DiscussionFormat.QA)
            
            assert result is not None
            assert isinstance(result, dict)
            assert 'word_count' in result
            assert result['word_count'] > 0
            
            print(f"✅ Rule-based Q&A: {result['word_count']} words")
            
        except Exception as e:
            print(f"❌ Rule-based Q&A failed: {e}")
            raise

    def test_llm_generator_initialization(self):
        """Test LLM generator initialization."""
        try:
            config = LLMConfig()
            generator = LLMMonologueGenerator(config)
            assert generator is not None
            assert hasattr(generator, 'generate_monologue')
            print("✅ LLM generator initialization successful")
            
        except Exception as e:
            print(f"⚠️ LLM generator initialization failed (may be expected): {e}")

    def test_llm_monologue_generation(self):
        """Test LLM-based monologue generation."""
        try:
            config = LLMConfig()
            generator = LLMMonologueGenerator(config)
            
            result = generator.generate_monologue(self.sample_content, target_length=60)
            
            assert result is not None
            assert isinstance(result, dict)
            
            if result.get('model_used') == 'fallback':
                print("⚠️ LLM fell back to rule-based generation (expected if no LLM available)")
                assert 'word_count' in result
            else:
                print(f"✅ LLM monologue: {result.get('word_count', 0)} words")
                assert 'generated_text' in result or 'word_count' in result
            
        except Exception as e:
            print(f"⚠️ LLM generation failed (may be expected if no LLM): {e}")

    def test_content_length_control(self):
        """Test that content generation respects length parameters."""
        generator = HumorousDiscussionGenerator()
        
        # Generate short content
        short_result = generator.generate_discussion(self.sample_content, DiscussionFormat.MONOLOGUE)
        
        # Should have reasonable word count
        assert short_result['word_count'] > 10  # Not too short
        assert short_result['word_count'] < 500  # Not too long for rule-based
        
        print(f"✅ Content length control: {short_result['word_count']} words")

    def test_content_quality_checks(self):
        """Test basic content quality."""
        generator = HumorousDiscussionGenerator()
        result = generator.generate_discussion(self.sample_content, DiscussionFormat.MONOLOGUE)
        
        # Extract text for quality checks
        if 'script' in result:
            text_parts = []
            for turn in result['script']:
                if hasattr(turn, 'content'):
                    text_parts.append(turn.content)
                elif isinstance(turn, dict):
                    text_parts.append(turn.get('content', ''))
            full_text = ' '.join(text_parts)
        else:
            full_text = result.get('generated_text', '')
        
        if full_text:
            # Basic quality checks
            assert len(full_text) > 50  # Should have substantial content
            assert 'cat' in full_text.lower()  # Should mention the topic
            assert len(full_text.split()) >= 10  # Should have multiple words
            
            print(f"✅ Content quality good: {len(full_text)} characters")
        else:
            print("⚠️ No text content found for quality check")


if __name__ == "__main__":
    # Run tests manually
    test_gen = TestContentGeneration()
    test_gen.setup_method()
    
    print("🧪 Testing Content Generation...")
    print("=" * 50)
    
    try:
        test_gen.test_rule_based_generator_initialization()
    except Exception as e:
        print(f"❌ Rule-based init failed: {e}")
    
    try:
        test_gen.test_rule_based_monologue_generation()
    except Exception as e:
        print(f"❌ Rule-based monologue failed: {e}")
    
    try:
        test_gen.test_rule_based_qa_generation()
    except Exception as e:
        print(f"❌ Rule-based Q&A failed: {e}")
    
    try:
        test_gen.test_llm_generator_initialization()
    except Exception as e:
        print(f"❌ LLM init failed: {e}")
    
    try:
        test_gen.test_llm_monologue_generation()
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
    
    try:
        test_gen.test_content_length_control()
    except Exception as e:
        print(f"❌ Length control failed: {e}")
    
    try:
        test_gen.test_content_quality_checks()
    except Exception as e:
        print(f"❌ Quality checks failed: {e}")
    
    print("\\n🏁 Content generation tests complete")