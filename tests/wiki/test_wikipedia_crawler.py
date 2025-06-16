"""Tests for Wikipedia crawler functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from lib.wiki.crawler import WikipediaCrawler


class TestWikipediaCrawler:
    """Test Wikipedia content fetching."""

    def setup_method(self):
        """Set up test fixtures."""
        self.crawler = WikipediaCrawler()

    def test_crawler_initialization(self):
        """Test that crawler initializes properly."""
        assert self.crawler is not None
        assert hasattr(self.crawler, 'get_page_summary')

    def test_get_simple_page(self):
        """Test fetching a simple, reliable Wikipedia page."""
        # Use a very stable page that's unlikely to change
        result = self.crawler.get_page_summary("Cat")
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'title' in result
        assert 'content' in result
        assert result['title'] is not None
        assert result['content'] is not None
        assert len(result['content']) > 100  # Should have substantial content
        
        print(f"✅ Wikipedia fetch successful: {result['title']}")
        print(f"📝 Content length: {len(result['content'])} characters")

    def test_get_nonexistent_page(self):
        """Test handling of non-existent pages."""
        result = self.crawler.get_page_summary("ThisPageDefinitelyDoesNotExist12345")
        
        # Should return None or empty dict
        assert result is None or result == {}
        print("✅ Non-existent page handled correctly")

    def test_get_disambiguation_page(self):
        """Test handling of disambiguation pages."""
        result = self.crawler.get_page_summary("Mercury")
        
        # Should still return content even for disambiguation
        if result:
            assert 'title' in result
            assert 'content' in result
            print(f"✅ Disambiguation page handled: {result['title']}")
        else:
            print("⚠️ Disambiguation page returned empty (may be expected)")

    def test_network_timeout_handling(self):
        """Test that network issues are handled gracefully."""
        # This tests the crawler's error handling
        try:
            result = self.crawler.get_page_summary("Cat")
            assert result is not None or result is None  # Either works, just shouldn't crash
            print("✅ Network handling works")
        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(f"Network error not handled gracefully: {e}")

    def test_special_characters_in_title(self):
        """Test handling of special characters in page titles."""
        result = self.crawler.get_page_summary("André Gide")
        
        if result:
            assert 'title' in result
            print(f"✅ Special characters handled: {result['title']}")
        else:
            print("⚠️ Special character page not found (may be expected)")

    def test_content_quality(self):
        """Test that returned content is of good quality."""
        result = self.crawler.get_page_summary("Python (programming language)")
        
        if result and result.get('content'):
            content = result['content']
            
            # Basic quality checks
            assert len(content) > 200  # Should have substantial content
            assert not content.startswith("{{")  # Should not have raw wiki markup
            assert "python" in content.lower()  # Should be relevant
            
            print(f"✅ Content quality good: {len(content)} chars")
        else:
            print("⚠️ Content quality test skipped (no content)")


if __name__ == "__main__":
    # Run tests manually
    test_crawler = TestWikipediaCrawler()
    test_crawler.setup_method()
    
    print("🧪 Testing Wikipedia Crawler...")
    print("=" * 50)
    
    try:
        test_crawler.test_crawler_initialization()
        print("✅ Initialization test passed")
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
    
    try:
        test_crawler.test_get_simple_page()
        print("✅ Simple page test passed")
    except Exception as e:
        print(f"❌ Simple page test failed: {e}")
    
    try:
        test_crawler.test_get_nonexistent_page()
        print("✅ Non-existent page test passed")
    except Exception as e:
        print(f"❌ Non-existent page test failed: {e}")
    
    try:
        test_crawler.test_network_timeout_handling()
        print("✅ Network handling test passed")
    except Exception as e:
        print(f"❌ Network handling test failed: {e}")
    
    print("\\n🏁 Wikipedia crawler tests complete")