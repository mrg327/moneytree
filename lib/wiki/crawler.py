import requests
from urllib.parse import urlencode
from typing import Dict, List, Optional, Any

from lib.utils.logging_config import get_logger, LoggedOperation, log_execution_time

logger = get_logger(__name__)


class WikipediaCrawler:
    """A Wikipedia API client for retrieving page content and information."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize the Wikipedia crawler.
        
        Args:
            language: Language code for Wikipedia (default: "en")
        """
        self.base_url = f"https://{language}.wikipedia.org/api/rest_v1"
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MoneyTree/1.0 (https://github.com/user/moneytree)"
        })
        logger.debug(f"Initialized Wikipedia crawler for language: {language}")
        logger.debug(f"Base URL: {self.base_url}")
        logger.debug(f"API URL: {self.api_url}")
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for Wikipedia articles matching the query.
        
        Args:
            query: Search term
            limit: Maximum number of results to return
            
        Returns:
            List of search results with title, description, and page info
        """
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet|titlesnippet|size|wordcount|timestamp"
        }
        
        response = self._make_request(self.api_url, params)
        if "query" in response and "search" in response["query"]:
            return response["query"]["search"]
        return []
    
    def get_page_content(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get the full content of a Wikipedia page.
        
        Args:
            title: Page title
            
        Returns:
            Dictionary containing page content and metadata
        """
        # First, get the page summary and basic info
        summary_url = f"{self.base_url}/page/summary/{requests.utils.quote(title)}"
        
        try:
            summary_response = self._make_request(summary_url)
            
            # Get the full page content
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts|info|pageimages",
                "exintro": False,
                "explaintext": True,
                "inprop": "url|displaytitle",
                "piprop": "original|thumbnail",
                "pithumbsize": 300
            }
            
            content_response = self._make_request(self.api_url, params)
            
            if "query" in content_response and "pages" in content_response["query"]:
                pages = content_response["query"]["pages"]
                page_id = next(iter(pages.keys()))
                page_data = pages[page_id]
                
                # Combine summary and content data
                result = {
                    "title": summary_response.get("title", title),
                    "description": summary_response.get("description", ""),
                    "extract": summary_response.get("extract", ""),
                    "full_content": page_data.get("extract", ""),
                    "url": page_data.get("fullurl", ""),
                    "page_id": page_id,
                    "last_modified": summary_response.get("timestamp", ""),
                    "coordinates": summary_response.get("coordinates", {}),
                    "thumbnail": summary_response.get("thumbnail", {}),
                    "original_image": summary_response.get("originalimage", {})
                }
                
                return result
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch page content for '{title}': {e}")
            return None
    
    @log_execution_time(logger)
    def get_page_summary(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a Wikipedia page.
        
        Args:
            title: Page title
            
        Returns:
            Dictionary containing page summary and basic info
        """
        summary_url = f"{self.base_url}/page/summary/{requests.utils.quote(title)}"
        
        with LoggedOperation(logger, f"fetching page summary for '{title}'"):
            try:
                response = self._make_request(summary_url)
                result = {
                    "title": response.get("title", ""),
                    "description": response.get("description", ""),
                    "extract": response.get("extract", ""),
                    "url": response.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "thumbnail": response.get("thumbnail", {}),
                    "coordinates": response.get("coordinates", {})
                }
                logger.info(f"Successfully fetched summary for '{result['title']}'")
                logger.debug(f"Summary length: {len(result['extract'])} characters")
                return result
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch page summary for '{title}': {e}")
                return None
    
    def get_random_page(self) -> Optional[Dict[str, Any]]:
        """
        Get a random Wikipedia page.
        
        Returns:
            Dictionary containing random page info
        """
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnlimit": 1,
            "rnnamespace": 0
        }
        
        response = self._make_request(self.api_url, params)
        if "query" in response and "random" in response["query"]:
            random_page = response["query"]["random"][0]
            return self.get_page_content(random_page["title"])
        return None
    
    def get_page_categories(self, title: str) -> List[str]:
        """
        Get categories for a Wikipedia page.
        
        Args:
            title: Page title
            
        Returns:
            List of category names
        """
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "categories",
            "cllimit": 100
        }
        
        response = self._make_request(self.api_url, params)
        categories = []
        
        if "query" in response and "pages" in response["query"]:
            pages = response["query"]["pages"]
            for page_id, page_data in pages.items():
                if "categories" in page_data:
                    categories = [cat["title"].replace("Category:", "") 
                                for cat in page_data["categories"]]
        
        return categories
    
    def get_extended_text(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get extended text content from a Wikipedia page with maximum detail.
        
        This method retrieves comprehensive content including:
        - Full article text (all sections)
        - Section structure and headings
        - References and external links
        - Infobox data when available
        - Extended metadata
        
        Args:
            title: Page title
            
        Returns:
            Dictionary containing extended text content and metadata
        """
        try:
            # Get full wikitext and parse it
            wikitext_params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "revisions|info|pageimages|extracts",
                "rvprop": "content",
                "rvslots": "main",
                "inprop": "url|displaytitle",
                "piprop": "original|thumbnail",
                "pithumbsize": 500,
                "exintro": False,
                "explaintext": True,
                "exsectionformat": "wiki"
            }
            
            # Get parsed content with sections
            parsed_params = {
                "action": "parse",
                "format": "json", 
                "page": title,
                "prop": "text|sections|links|categories|templates|images|externallinks",
                "disablelimitreport": True,
                "disableeditsection": True,
                "disabletoc": False
            }
            
            # Make both requests
            wikitext_response = self._make_request(self.api_url, wikitext_params)
            parsed_response = self._make_request(self.api_url, parsed_params)
            
            result = {}
            
            # Process wikitext response
            if "query" in wikitext_response and "pages" in wikitext_response["query"]:
                pages = wikitext_response["query"]["pages"]
                page_id = next(iter(pages.keys()))
                page_data = pages[page_id]
                
                result.update({
                    "title": page_data.get("title", title),
                    "page_id": page_id,
                    "url": page_data.get("fullurl", ""),
                    "full_text": page_data.get("extract", ""),
                    "thumbnail": page_data.get("thumbnail", {}),
                    "original_image": page_data.get("original", {})
                })
                
                # Get raw wikitext if available
                if "revisions" in page_data and page_data["revisions"]:
                    revision = page_data["revisions"][0]
                    if "slots" in revision and "main" in revision["slots"]:
                        result["raw_wikitext"] = revision["slots"]["main"].get("*", "")
            
            # Process parsed response
            if "parse" in parsed_response:
                parse_data = parsed_response["parse"]
                
                result.update({
                    "html_content": parse_data.get("text", {}).get("*", ""),
                    "sections": parse_data.get("sections", []),
                    "links": [link.get("*", "") for link in parse_data.get("links", [])],
                    "categories": [cat.replace("Category:", "") if isinstance(cat, str) else str(cat).replace("Category:", "") for cat in parse_data.get("categories", [])],
                    "templates": [tpl.get("*", "") for tpl in parse_data.get("templates", [])],
                    "images": parse_data.get("images", []),
                    "external_links": parse_data.get("externallinks", [])
                })
                
                # Extract section text organized by headings
                sections_text = {}
                for section in parse_data.get("sections", []):
                    if "line" in section:
                        sections_text[section["line"]] = {
                            "level": section.get("level", 0),
                            "index": section.get("index", 0)
                        }
                
                result["sections_structure"] = sections_text
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch extended text for '{title}': {e}")
            return None
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the Wikipedia API.
        
        Args:
            url: Request URL
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        if params:
            url = f"{url}?{urlencode(params)}"
        
        logger.debug(f"Making request to: {url}")
        response = self.session.get(url)
        response.raise_for_status()
        logger.debug(f"Request successful, response size: {len(response.content)} bytes")
        return response.json()