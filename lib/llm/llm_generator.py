"""
LLM-based humorous discussion generator using Ollama.

Creates natural, human-sounding monologues using local language models.
"""

import ollama
import json
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""
    model: str = "llama3.1:8b" 
    temperature: float = 0.8
    max_tokens: int = 300
    timeout: int = 30
    host: Optional[str] = None  # Will auto-detect Windows host from WSL


class LLMMonologueGenerator:
    """
    Generates humorous monologues using local LLMs via Ollama.
    
    Creates Sam O'Nella / Casually Explained style content that sounds
    natural and human rather than template-based.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM generator.
        
        Args:
            config: LLM configuration, uses defaults if None
        """
        self.config = config or LLMConfig()
        self.ollama_host = self._detect_ollama_host()
        # We'll set up the client during generation to avoid hanging on init
    
    def _detect_ollama_host(self) -> str:
        """Detect the correct host for Ollama (handles WSL -> Windows)."""
        if self.config.host:
            return self.config.host
        
        # Check if we're running in WSL
        try:
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                if 'microsoft' in version_info or 'wsl' in version_info:
                    # We're in WSL, Ollama is likely on Windows host
                    try:
                        # Get Windows host IP from WSL
                        result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            # Extract the gateway IP (Windows host)
                            for line in result.stdout.split('\n'):
                                if 'default via' in line:
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        host_ip = parts[2]
                                        print(f"🔍 Detected WSL environment, using Windows host: {host_ip}:11434")
                                        return f"http://{host_ip}:11434"
                    except subprocess.TimeoutExpired:
                        print("⚠️  Timeout getting host IP, using localhost")
                    except Exception as e:
                        print(f"⚠️  Could not detect Windows host IP: {e}")
                        print("🔄 Falling back to localhost")
        except Exception:
            # Not WSL or can't detect
            pass
        
        # Default to localhost
        return "http://localhost:11434"
    
    
    def generate_monologue(
        self, 
        topic_content: Dict[str, Any], 
        target_length: int = 180
    ) -> Dict[str, Any]:
        """
        Generate a humorous monologue about the topic using LLM.
        
        Args:
            topic_content: Content from Wikipedia crawler
            target_length: Target word count (~180 words = 1 minute)
            
        Returns:
            Dictionary with monologue script and metadata
        """
        # Extract key information
        title = topic_content.get("title", "Unknown Topic")
        description = topic_content.get("description", "")
        extract = topic_content.get("extract", "")
        
        # Create the prompt
        prompt = self._create_prompt(title, description, extract, target_length)
        
        try:
            print(f"🔌 Connecting to Ollama at: {self.ollama_host}")
            
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
            
            # Parse the response
            script = self._parse_monologue(generated_text)
            
            return {
                "format": "llm_monologue",
                "topic": title,
                "script": script,
                "generated_text": generated_text,
                "word_count": len(generated_text.split()),
                "estimated_duration": (len(generated_text.split()) / 150) * 60,
                "model_used": self.config.model
            }
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            print(f"   Tried connecting to: {self.ollama_host}")
            return self._fallback_response(title)
    
    def _create_prompt(
        self, 
        title: str, 
        description: str, 
        extract: str, 
        target_length: int
    ) -> str:
        """Create a prompt for the LLM to generate objective but amusing content."""
        
        # Limit extract length to avoid token limits
        extract_snippet = extract[:500] + "..." if len(extract) > 500 else extract
        
        prompt = f"""Tell the story of {title} in {target_length} words using straightforward storytelling.

Topic: {title}
Description: {description}
Source material: {extract_snippet}

Structure your narrative to:
- Juxtapose contrasting elements without explicit commentary
- Present scale and consequences in practical terms
- Show how formal processes interact with unusual circumstances
- Arrange events chronologically to reveal natural timing ironies
- Treat all subject matter with equal seriousness regardless of apparent triviality

Write as natural conversation. Let the facts and their relationships create any comedic effect through presentation rather than commentary.

Write ONLY the story text, no formatting:"""

        return prompt
    
    def _parse_monologue(self, generated_text: str) -> List[Dict[str, str]]:
        """Parse generated text into script format."""
        # Split into sentences for better formatting
        sentences = []
        current_sentence = ""
        
        for char in generated_text:
            current_sentence += char
            if char in '.!?' and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Group sentences into turns (2-3 sentences each)
        turns = []
        current_turn = ""
        sentence_count = 0
        
        for sentence in sentences:
            current_turn += sentence + " "
            sentence_count += 1
            
            # Create a new turn every 2-3 sentences or when turn gets long
            if sentence_count >= 2 or len(current_turn) > 100:
                if current_turn.strip():
                    turns.append({
                        "speaker": "Narrator",
                        "content": current_turn.strip(),
                        "tone": "conversational"
                    })
                current_turn = ""
                sentence_count = 0
        
        # Add any remaining content
        if current_turn.strip():
            turns.append({
                "speaker": "Narrator", 
                "content": current_turn.strip(),
                "tone": "conversational"
            })
        
        return turns
    
    def _fallback_response(self, title: str) -> Dict[str, Any]:
        """Fallback response when LLM generation fails."""
        fallback_text = f"""The topic of {title} presents an interesting case study in modern information access. While this system was designed to provide an entertaining explanation of the subject, it currently finds itself unable to connect to the language model that would generate such content.

This creates a peculiar situation where the attempt to explain something has become, itself, something that requires explanation. The original subject matter remains unexplored while we instead examine the mechanics of failed communication systems.

In a sense, this demonstrates the fragility of our increasingly complex technological frameworks - sophisticated enough to process human language and generate coherent explanations, yet vulnerable to simple connectivity issues. The intended content about {title} waits patiently in a Wikipedia database while various software components fail to coordinate their basic functions.

Perhaps there's something appropriately modern about this outcome."""
        
        return {
            "format": "llm_monologue",
            "topic": title,
            "script": [{
                "speaker": "Narrator",
                "content": fallback_text,
                "tone": "conversational"
            }],
            "generated_text": fallback_text,
            "word_count": len(fallback_text.split()),
            "estimated_duration": (len(fallback_text.split()) / 150) * 60,
            "model_used": "fallback"
        }
    
    def format_as_script(self, monologue: Dict[str, Any]) -> str:
        """Format monologue as a readable script."""
        script_lines = []
        script_lines.append(f"=== {monologue['topic']} ===")
        script_lines.append(f"Model: {monologue['model_used']}")
        script_lines.append(f"Estimated Duration: {monologue['estimated_duration']:.1f} seconds")
        script_lines.append("=" * 50)
        script_lines.append("")
        
        for turn in monologue['script']:
            script_lines.append(f"{turn['speaker']}: {turn['content']}")
            script_lines.append("")
        
        script_lines.append("=" * 50)
        script_lines.append(f"Word Count: {monologue['word_count']} words")
        
        return "\n".join(script_lines)
