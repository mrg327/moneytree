"""
LLM-based YouTube description and tag generator using Ollama.

Creates YouTube-optimized descriptions and tags from Wikipedia content.
"""

import ollama
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from lib.utils.logging_config import get_logger, LoggedOperation

logger = get_logger(__name__)


@dataclass
class DescriptionConfig:
    """Configuration for LLM description generation."""
    model: str = "llama3.1:8b" 
    temperature: float = 0.7
    max_tokens: int = 400
    timeout: int = 30
    host: Optional[str] = None  # Will auto-detect Windows host from WSL
    
    # Description generation settings
    target_description_length: int = 175  # Target words for YouTube description
    max_tags: int = 12
    min_tags: int = 6


class LLMDescriptionGenerator:
    """
    Generates YouTube descriptions and tags using local LLMs via Ollama.
    
    Creates engaging, SEO-optimized descriptions and relevant tags based on
    Wikipedia content and video characteristics.
    """
    
    def __init__(self, config: Optional[DescriptionConfig] = None):
        """
        Initialize the description generator.
        
        Args:
            config: Description generation configuration, uses defaults if None
        """
        self.config = config or DescriptionConfig()
        self.ollama_host = self._detect_ollama_host()
        # We'll set up the client during generation to avoid hanging on init
    
    def _detect_ollama_host(self) -> str:
        """Detect the correct host for Ollama (handles WSL -> Windows)."""
        if self.config.host:
            return self.config.host
        
        # Check if we're running in WSL
        try:
            with open('/proc/version', 'r') as f:
                version_info = f.read()
                if 'Microsoft' in version_info or 'WSL' in version_info:
                    # WSL detected - try Windows host
                    logger.info("WSL detected, connecting to Windows Ollama host")
                    return "http://172.31.32.1:11434"  # Default WSL->Windows bridge
        except FileNotFoundError:
            pass
        
        # Default to localhost
        return "http://localhost:11434"
    
    def generate_youtube_metadata(self, 
                                wikipedia_content: Dict[str, Any],
                                video_duration: float,
                                wikipedia_categories: List[str] = None) -> Dict[str, Any]:
        """
        Generate YouTube description and tags from Wikipedia content.
        
        Args:
            wikipedia_content: Wikipedia page content from WikipediaCrawler
            video_duration: Duration of the generated video in seconds
            wikipedia_categories: List of Wikipedia categories for tag generation
            
        Returns:
            Dictionary with generated description, tags, and metadata
        """
        # Extract key information
        title = wikipedia_content.get("title", "Unknown Topic")
        description = wikipedia_content.get("description", "")
        extract = wikipedia_content.get("extract", "")
        
        logger.info(f"Generating YouTube metadata for: {title}")
        logger.debug(f"Video duration: {video_duration:.1f}s")
        
        # Create the prompt for description and tag generation
        prompt = self._create_youtube_prompt(title, description, extract, video_duration, wikipedia_categories)
        
        try:
            logger.info(f"🔌 Connecting to Ollama at: {self.ollama_host}")
            
            # Set up client for this request
            try:
                from ollama import Client
                client = Client(host=self.ollama_host)
                response = client.generate(
                    model=self.config.model,
                    prompt=prompt,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                    stream=False
                )
            except ImportError:
                # Fallback for older ollama versions
                response = ollama.generate(
                    model=self.config.model,
                    prompt=prompt,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                    stream=False
                )
            
            generated_text = response['response'].strip()
            
            # Parse the response to extract description and tags
            description_text, tags_list = self._parse_youtube_response(generated_text)
            
            # Enhance tags with Wikipedia categories
            if wikipedia_categories:
                tags_list = self._enhance_tags_with_categories(tags_list, wikipedia_categories, title)
            
            # Validate and clean the results
            description_text = self._validate_description(description_text, title)
            tags_list = self._validate_tags(tags_list)
            
            logger.info(f"✅ Generated description ({len(description_text.split())} words)")
            logger.info(f"✅ Generated {len(tags_list)} tags")
            
            return {
                "success": True,
                "title": title,
                "description": description_text,
                "tags": tags_list,
                "raw_response": generated_text,
                "word_count": len(description_text.split()),
                "tag_count": len(tags_list),
                "model_used": self.config.model,
                "video_duration": video_duration
            }
            
        except Exception as e:
            logger.error(f"❌ LLM description generation failed: {e}")
            logger.error(f"   Tried connecting to: {self.ollama_host}")
            # Return failure - no fallback as per requirements
            return {
                "success": False,
                "error": f"Ollama LLM unavailable: {e}",
                "title": title
            }
    
    def _create_youtube_prompt(self, 
                              title: str, 
                              description: str, 
                              extract: str, 
                              duration: float,
                              categories: List[str] = None) -> str:
        """Create a prompt for generating YouTube description and tags."""
        
        # Limit extract length to avoid token limits
        extract_snippet = extract[:800] + "..." if len(extract) > 800 else extract
        
        # Format duration for human readability
        duration_text = f"{int(duration//60)}:{int(duration%60):02d}" if duration >= 60 else f"{int(duration)} seconds"
        
        # Include categories if available
        categories_text = ""
        if categories:
            # Clean and filter categories
            clean_categories = [cat for cat in categories[:10] if not any(skip in cat.lower() 
                               for skip in ['wikipedia', 'articles', 'pages', 'commons', 'wikimedia'])]
            if clean_categories:
                categories_text = f"Wikipedia Categories: {', '.join(clean_categories[:8])}\n"
        
        prompt = f"""Create YouTube metadata for an educational video about this topic.

Topic: {title}
Brief Description: {description}
Key Content: {extract_snippet}
{categories_text}Video Duration: {duration_text}

Generate:

1. DESCRIPTION (150-200 words):
Write an engaging YouTube description that:
- Summarizes the key fascinating points from this topic
- Uses educational but accessible language
- Includes natural keywords for discoverability
- Mentions this is AI-generated educational content
- Has a clear structure with the most interesting facts first
- Ends with a call to action for educational content

2. TAGS (8-12 tags):
Create relevant tags including:
- Main topic and variations
- Educational keywords (education, learning, explained, facts)
- Related concepts and subtopics
- Subject area (science, history, technology, etc.)
- Popular educational formats (educational, tutorial, facts)

Format your response EXACTLY like this:

DESCRIPTION:
[Your description here]

TAGS:
tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8

Remember: Keep the description informative yet engaging, and ensure tags are specific and discoverable."""

        return prompt
    
    def _parse_youtube_response(self, generated_text: str) -> Tuple[str, List[str]]:
        """Parse LLM response to extract description and tags."""
        
        # Initialize defaults
        description = ""
        tags = []
        
        # Split the response into sections
        lines = generated_text.strip().split('\n')
        current_section = None
        description_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line.upper().startswith('DESCRIPTION:'):
                current_section = 'description'
                # Check if description is on the same line
                desc_text = line[12:].strip()  # Remove "DESCRIPTION:"
                if desc_text:
                    description_lines.append(desc_text)
                continue
            elif line.upper().startswith('TAGS:'):
                current_section = 'tags'
                # Check if tags are on the same line
                tags_text = line[5:].strip()  # Remove "TAGS:"
                if tags_text:
                    tags = self._parse_tags_line(tags_text)
                continue
            
            # Process content based on current section
            if current_section == 'description' and line:
                description_lines.append(line)
            elif current_section == 'tags' and line:
                tags.extend(self._parse_tags_line(line))
        
        # Join description lines
        description = ' '.join(description_lines).strip()
        
        # If parsing failed, try to extract from the raw text
        if not description:
            description = self._extract_description_fallback(generated_text)
        
        if not tags:
            tags = self._extract_tags_fallback(generated_text)
        
        return description, tags
    
    def _parse_tags_line(self, tags_text: str) -> List[str]:
        """Parse a line containing tags."""
        # Split by commas and clean up
        raw_tags = [tag.strip() for tag in tags_text.split(',')]
        # Filter out empty tags and clean up
        return [tag for tag in raw_tags if tag and len(tag) > 1]
    
    def _extract_description_fallback(self, text: str) -> str:
        """Fallback method to extract description from response."""
        # Try to find the largest coherent paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            # Return the longest paragraph that looks like a description
            candidates = [p for p in paragraphs if len(p.split()) > 30]
            if candidates:
                return max(candidates, key=len)
            else:
                return paragraphs[0]
        
        # Last resort: return first part of the text
        sentences = text.split('.')[:3]
        return '. '.join(sentences) + '.' if sentences else text[:200]
    
    def _extract_tags_fallback(self, text: str) -> List[str]:
        """Fallback method to extract tags from response."""
        # Look for comma-separated words in the text
        tag_patterns = [
            r'tags?[:\s]+([^.\n]+)',
            r'keywords?[:\s]+([^.\n]+)',
            r'hashtags?[:\s]+([^.\n]+)'
        ]
        
        for pattern in tag_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                tags_text = matches[0]
                tags = [tag.strip() for tag in tags_text.split(',')]
                return [tag for tag in tags if tag and len(tag) > 1][:12]
        
        return []
    
    def _enhance_tags_with_categories(self, existing_tags: List[str], 
                                    categories: List[str], 
                                    title: str) -> List[str]:
        """Enhance tag list with relevant Wikipedia categories."""
        
        # Convert existing tags to lowercase for comparison
        existing_lower = [tag.lower() for tag in existing_tags]
        
        # Process categories to create additional tags
        category_tags = []
        for category in categories[:15]:  # Limit to prevent overwhelm
            # Skip meta categories
            if any(skip in category.lower() for skip in 
                   ['wikipedia', 'articles', 'pages', 'commons', 'wikimedia', 'categories']):
                continue
            
            # Clean up category names
            clean_category = category.strip()
            
            # Extract meaningful words from categories
            words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_category)
            for word in words:
                if (len(word) >= 3 and 
                    word.lower() not in existing_lower and 
                    word.lower() not in [t.lower() for t in category_tags]):
                    category_tags.append(word.lower())
        
        # Combine tags, prioritizing generated tags
        combined_tags = existing_tags.copy()
        
        # Add category tags up to the limit
        for cat_tag in category_tags:
            if len(combined_tags) >= self.config.max_tags:
                break
            if cat_tag not in [t.lower() for t in combined_tags]:
                combined_tags.append(cat_tag)
        
        return combined_tags
    
    def _validate_description(self, description: str, title: str) -> str:
        """Validate and clean the generated description."""
        
        if not description or len(description.strip()) < 50:
            # Generate a basic fallback description
            return (f"Learn about {title} in this educational video. "
                   "This AI-generated content explores the key facts and interesting aspects "
                   "of this topic in an engaging and accessible way. "
                   "Perfect for students, educators, and anyone curious about learning something new!")
        
        # Clean up the description
        description = description.strip()
        
        # Ensure it mentions AI-generated content if not already present
        if 'ai' not in description.lower() and 'generated' not in description.lower():
            description += " This educational content is AI-generated to make learning accessible and engaging."
        
        # Ensure reasonable length (YouTube descriptions should be substantial but not too long)
        words = description.split()
        if len(words) > 250:
            description = ' '.join(words[:250]) + "..."
        elif len(words) < 80:
            description += f" Explore more educational content about {title} and related topics!"
        
        return description
    
    def _validate_tags(self, tags: List[str]) -> List[str]:
        """Validate and clean the generated tags."""
        
        if not tags:
            return ["education", "learning", "educational", "facts", "explained"]
        
        # Clean and filter tags
        cleaned_tags = []
        for tag in tags:
            tag = tag.strip().lower()
            # Remove hashtags and invalid characters
            tag = re.sub(r'[#@]', '', tag)
            tag = re.sub(r'[^\w\s-]', '', tag)
            
            # Skip if too short, too long, or contains numbers only
            if (len(tag) >= 2 and len(tag) <= 30 and 
                not tag.isdigit() and tag not in cleaned_tags):
                cleaned_tags.append(tag)
        
        # Ensure we have enough tags
        if len(cleaned_tags) < self.config.min_tags:
            default_tags = ["education", "learning", "educational", "facts", "explained", "tutorial"]
            for default_tag in default_tags:
                if default_tag not in cleaned_tags:
                    cleaned_tags.append(default_tag)
                if len(cleaned_tags) >= self.config.min_tags:
                    break
        
        # Limit to maximum tags
        return cleaned_tags[:self.config.max_tags]
    
    def format_metadata_summary(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for logging and display."""
        if not metadata.get('success'):
            return f"❌ Metadata generation failed: {metadata.get('error', 'Unknown error')}"
        
        lines = []
        lines.append(f"📝 YouTube Metadata Generated:")
        lines.append(f"   Title: {metadata['title']}")
        lines.append(f"   Description: {metadata['word_count']} words")
        lines.append(f"   Tags: {metadata['tag_count']} tags")
        lines.append(f"   Model: {metadata['model_used']}")
        lines.append(f"   Tags: {', '.join(metadata['tags'][:5])}{'...' if len(metadata['tags']) > 5 else ''}")
        
        return "\n".join(lines)