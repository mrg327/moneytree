import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from lib.wiki.crawler import WikipediaCrawler


class TestWikipediaCrawler:
    
    @pytest.fixture
    def crawler(self):
        return WikipediaCrawler()
    
    @pytest.fixture
    def mock_search_response(self):
        return {
            "query": {
                "search": [
                    {
                        "title": "Python (programming language)",
                        "snippet": "Python is a high-level programming language",
                        "size": 12345,
                        "wordcount": 2000,
                        "timestamp": "2023-01-01T00:00:00Z"
                    },
                    {
                        "title": "Python",
                        "snippet": "Python may refer to various things",
                        "size": 5678,
                        "wordcount": 800,
                        "timestamp": "2023-01-02T00:00:00Z"
                    }
                ]
            }
        }
    
    @pytest.fixture
    def mock_summary_response(self):
        return {
            "title": "Python (programming language)",
            "description": "High-level programming language",
            "extract": "Python is a high-level, general-purpose programming language...",
            "content_urls": {
                "desktop": {
                    "page": "https://en.wikipedia.org/wiki/Python_(programming_language)"
                }
            },
            "thumbnail": {
                "source": "https://upload.wikimedia.org/python.png",
                "width": 200,
                "height": 150
            },
            "coordinates": {"lat": 0, "lon": 0}
        }
    
    @pytest.fixture
    def mock_content_response(self):
        return {
            "query": {
                "pages": {
                    "23862": {
                        "pageid": 23862,
                        "title": "Python (programming language)",
                        "extract": "Full content of the Python programming language article...",
                        "fullurl": "https://en.wikipedia.org/wiki/Python_(programming_language)"
                    }
                }
            }
        }
    
    def test_initialization_default_language(self):
        crawler = WikipediaCrawler()
        assert "en.wikipedia.org" in crawler.base_url
        assert "en.wikipedia.org" in crawler.api_url
        assert "MoneyTree" in crawler.session.headers["User-Agent"]
    
    def test_initialization_custom_language(self):
        crawler = WikipediaCrawler(language="fr")
        assert "fr.wikipedia.org" in crawler.base_url
        assert "fr.wikipedia.org" in crawler.api_url
    
    @patch('requests.Session.get')
    def test_search_success(self, mock_get, crawler, mock_search_response):
        mock_response = Mock()
        mock_response.json.return_value = mock_search_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = crawler.search("Python programming", limit=5)
        
        assert len(results) == 2
        assert results[0]["title"] == "Python (programming language)"
        assert "high-level programming language" in results[0]["snippet"]
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_search_empty_results(self, mock_get, crawler):
        mock_response = Mock()
        mock_response.json.return_value = {"query": {"search": []}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = crawler.search("nonexistent topic")
        
        assert results == []
    
    @patch('requests.Session.get')
    def test_get_page_summary_success(self, mock_get, crawler, mock_summary_response):
        mock_response = Mock()
        mock_response.json.return_value = mock_summary_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = crawler.get_page_summary("Python (programming language)")
        
        assert result is not None
        assert result["title"] == "Python (programming language)"
        assert result["description"] == "High-level programming language"
        assert "programming language" in result["extract"]
        assert "wikipedia.org" in result["url"]
        assert "thumbnail" in result
    
    @patch('requests.Session.get')
    def test_get_page_summary_request_error(self, mock_get, crawler):
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        result = crawler.get_page_summary("Test Page")
        
        assert result is None
    
    @patch('requests.Session.get')
    def test_get_page_content_success(self, mock_get, crawler, mock_summary_response, mock_content_response):
        mock_responses = [
            Mock(json=Mock(return_value=mock_summary_response)),
            Mock(json=Mock(return_value=mock_content_response))
        ]
        for response in mock_responses:
            response.raise_for_status.return_value = None
        mock_get.side_effect = mock_responses
        
        result = crawler.get_page_content("Python (programming language)")
        
        assert result is not None
        assert result["title"] == "Python (programming language)"
        assert result["description"] == "High-level programming language"
        assert "Full content" in result["full_content"]
        assert result["page_id"] == "23862"
        assert "wikipedia.org" in result["url"]
    
    @patch('requests.Session.get')
    def test_get_page_categories_success(self, mock_get, crawler):
        mock_response_data = {
            "query": {
                "pages": {
                    "12345": {
                        "categories": [
                            {"title": "Category:Programming languages"},
                            {"title": "Category:Python (programming language)"},
                            {"title": "Category:Object-oriented programming languages"}
                        ]
                    }
                }
            }
        }
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        categories = crawler.get_page_categories("Python (programming language)")
        
        assert len(categories) == 3
        assert "Programming languages" in categories
        assert "Python (programming language)" in categories
        assert "Object-oriented programming languages" in categories
    
    @patch('requests.Session.get')
    def test_get_random_page_success(self, mock_get, crawler, mock_summary_response, mock_content_response):
        random_response = {
            "query": {
                "random": [
                    {"title": "Random Article", "id": 12345}
                ]
            }
        }
        
        mock_responses = [
            Mock(json=Mock(return_value=random_response)),
            Mock(json=Mock(return_value=mock_summary_response)),
            Mock(json=Mock(return_value=mock_content_response))
        ]
        for response in mock_responses:
            response.raise_for_status.return_value = None
        mock_get.side_effect = mock_responses
        
        result = crawler.get_random_page()
        
        assert result is not None
        assert "title" in result
    
    @patch('requests.Session.get')
    def test_make_request_with_params(self, mock_get, crawler):
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        params = {"action": "query", "format": "json"}
        result = crawler._make_request("http://test.com", params)
        
        assert result == {"test": "data"}
        mock_get.assert_called_once()
        call_args = mock_get.call_args[0][0]
        assert "action=query" in call_args
        assert "format=json" in call_args
    
    @patch('requests.Session.get')
    def test_make_request_raises_for_status(self, mock_get, crawler):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(requests.exceptions.HTTPError):
            crawler._make_request("http://test.com")
    
    @patch('requests.Session.get')
    def test_get_extended_text_success(self, mock_get, crawler):
        # Mock wikitext response
        mock_wikitext_response = {
            "query": {
                "pages": {
                    "12345": {
                        "title": "Python (programming language)",
                        "fullurl": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                        "extract": "Full comprehensive text content of the Python article with all sections...",
                        "thumbnail": {"source": "https://example.com/thumb.jpg"},
                        "original": {"source": "https://example.com/original.jpg"},
                        "revisions": [{
                            "slots": {
                                "main": {
                                    "*": "{{Infobox programming language\n|name = Python\n|logo = Python-logo.png\n}}\n\n'''Python''' is a high-level programming language..."
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        # Mock parsed response
        mock_parsed_response = {
            "parse": {
                "text": {"*": "<div>HTML content of the full article...</div>"},
                "sections": [
                    {"line": "History", "level": 2, "index": 1},
                    {"line": "Features", "level": 2, "index": 2},
                    {"line": "Syntax", "level": 3, "index": 3}
                ],
                "links": [{"*": "Guido van Rossum"}, {"*": "Programming language"}],
                "categories": ["Category:Programming languages", "Category:Python"],
                "templates": [{"*": "Infobox programming language"}],
                "images": ["Python-logo.png", "Python-code-example.png"],
                "externallinks": ["https://python.org", "https://docs.python.org"]
            }
        }
        
        mock_responses = [
            Mock(json=Mock(return_value=mock_wikitext_response)),
            Mock(json=Mock(return_value=mock_parsed_response))
        ]
        for response in mock_responses:
            response.raise_for_status.return_value = None
        mock_get.side_effect = mock_responses
        
        result = crawler.get_extended_text("Python (programming language)")
        
        assert result is not None
        assert result["title"] == "Python (programming language)"
        assert result["page_id"] == "12345"
        assert "comprehensive text content" in result["full_text"]
        assert "raw_wikitext" in result
        assert "{{Infobox programming language" in result["raw_wikitext"]
        assert "html_content" in result
        assert len(result["sections"]) == 3
        assert "History" in [section["line"] for section in result["sections"]]
        assert "Programming languages" in result["categories"]
        assert "https://python.org" in result["external_links"]
        assert "Python-logo.png" in result["images"]
        assert result["sections_structure"]["History"]["level"] == 2
    
    @patch('requests.Session.get')
    def test_get_extended_text_request_error(self, mock_get, crawler):
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        result = crawler.get_extended_text("Test Page")
        
        assert result is None


class TestWikipediaCrawlerIntegration:
    """Integration tests that make real API calls - run sparingly"""
    
    @pytest.mark.integration
    def test_real_search(self):
        crawler = WikipediaCrawler()
        results = crawler.search("Python programming", limit=3)
        
        assert len(results) > 0
        assert any("Python" in result["title"] for result in results)
    
    @pytest.mark.integration  
    def test_real_page_summary(self):
        crawler = WikipediaCrawler()
        result = crawler.get_page_summary("Python (programming language)")
        
        assert result is not None
        assert "Python" in result["title"]
        assert len(result["extract"]) > 0
    
    @pytest.mark.integration
    def test_real_page_content(self):
        crawler = WikipediaCrawler()
        result = crawler.get_page_content("Python (programming language)")
        
        assert result is not None
        assert "Python" in result["title"]
        assert len(result["full_content"]) > len(result["extract"])
        assert "wikipedia.org" in result["url"]
    
    @pytest.mark.integration
    def test_real_extended_text(self):
        crawler = WikipediaCrawler()
        result = crawler.get_extended_text("Python (programming language)")
        
        assert result is not None
        assert "Python" in result["title"]
        assert len(result["full_text"]) > 0
        assert "raw_wikitext" in result
        assert "html_content" in result
        assert len(result["sections"]) > 0
        assert len(result["external_links"]) > 0
        assert "wikipedia.org" in result["url"]