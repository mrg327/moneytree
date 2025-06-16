"""
Humorous discussion generator for Wikipedia content.

Creates funny podcast-style discussions, debates, or interviews based on input text.
Designed to generate approximately 1 minute of spoken content (~150-200 words).
"""

import random
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class DiscussionFormat(Enum):
    """Available discussion formats."""
    MONOLOGUE = "monologue"  # Single-person educational humor (Sam O'Nella style)
    PODCAST = "podcast"
    DEBATE = "debate" 
    INTERVIEW = "interview"
    EXPERT_PANEL = "expert_panel"


@dataclass
class Character:
    """Character in the discussion."""
    name: str
    personality: str
    speaking_style: str
    catchphrases: List[str]
    knowledge_level: str  # "expert", "casual", "confused"


@dataclass
class DiscussionTurn:
    """Single turn in a discussion."""
    speaker: str
    content: str
    tone: str  # "excited", "skeptical", "confused", "dramatic"


class HumorousDiscussionGenerator:
    """
    Generates humorous discussions about Wikipedia topics.
    
    Creates engaging, funny dialogue between characters discussing
    the provided subject matter in various formats.
    """
    
    def __init__(self):
        """Initialize the discussion generator with character templates."""
        self.characters = self._load_character_templates()
        self.humor_techniques = self._load_humor_techniques()
        self.discussion_templates = self._load_discussion_templates()
    
    def generate_discussion(
        self, 
        topic_content: Dict[str, Any], 
        format_type: DiscussionFormat = DiscussionFormat.MONOLOGUE,
        target_length: int = 180  # Target ~180 words for ~1 minute
    ) -> Dict[str, Any]:
        """
        Generate a humorous discussion about the topic.
        
        Args:
            topic_content: Content from Wikipedia crawler
            format_type: Type of discussion format
            target_length: Target word count for output
            
        Returns:
            Dictionary with discussion script and metadata
        """
        # Extract key information from content
        key_facts = self._extract_key_facts(topic_content)
        
        # Select characters for this discussion
        characters = self._select_characters(format_type)
        
        # Generate discussion turns
        turns = self._generate_discussion_turns(
            key_facts, characters, format_type, target_length
        )
        
        return {
            "format": format_type.value,
            "topic": topic_content.get("title", "Unknown Topic"),
            "characters": [char.name for char in characters],
            "script": turns,
            "estimated_duration": self._estimate_duration(turns),
            "word_count": self._count_words(turns)
        }
    
    def _extract_key_facts(self, content: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract the most interesting/funny facts from content."""
        facts = []
        
        # Get basic info
        title = content.get("title", "")
        description = content.get("description", "")
        
        if title and description:
            facts.append({
                "type": "basic",
                "content": f"{title} is {description}",
                "humor_potential": "high" if any(word in description.lower() 
                    for word in ["unusual", "strange", "bizarre", "unique"]) else "medium"
            })
        
        # Extract from full text if available
        full_text = content.get("full_text", "") or content.get("extract", "")
        if full_text:
            # Simple sentence extraction (first few sentences)
            sentences = full_text.split('. ')[:5]
            for sentence in sentences:
                if len(sentence) > 20:  # Skip very short fragments
                    facts.append({
                        "type": "detail",
                        "content": sentence.strip(),
                        "humor_potential": self._assess_humor_potential(sentence)
                    })
        
        # Look for categories that might be funny
        categories = content.get("categories", [])
        if categories:
            funny_categories = [cat for cat in categories if 
                any(word in cat.lower() for word in 
                    ["death", "controversy", "failure", "bizarre", "unusual", "strange"])]
            for cat in funny_categories[:2]:  # Limit to 2
                facts.append({
                    "type": "category",
                    "content": f"This is categorized under '{cat}'",
                    "humor_potential": "high"
                })
        
        return facts[:8]  # Limit facts to keep discussion focused
    
    def _assess_humor_potential(self, text: str) -> str:
        """Assess how funny a piece of text might be."""
        funny_keywords = [
            "died", "failed", "exploded", "accidentally", "mistakenly",
            "controversy", "bizarre", "unusual", "strange", "weird",
            "million", "billion", "tiny", "enormous", "ancient"
        ]
        
        text_lower = text.lower()
        funny_count = sum(1 for word in funny_keywords if word in text_lower)
        
        if funny_count >= 2:
            return "high"
        elif funny_count >= 1:
            return "medium"
        else:
            return "low"
    
    def _select_characters(self, format_type: DiscussionFormat) -> List[Character]:
        """Select appropriate characters for the discussion format."""
        if format_type == DiscussionFormat.MONOLOGUE:
            return [
                Character(
                    name="Narrator",
                    personality="dry, educational humor with random tangents",
                    speaking_style="conversational like explaining to a friend, deadpan delivery",
                    catchphrases=["So anyway...", "Fun fact:", "Speaking of which...", "Which is weird because...", "Now here's where it gets interesting..."],
                    knowledge_level="expert"
                )
            ]
        elif format_type == DiscussionFormat.PODCAST:
            return [
                Character(
                    name="Dave",
                    personality="overly enthusiastic about everything",
                    speaking_style="uses lots of superlatives and gets excited",
                    catchphrases=["That's INCREDIBLE!", "Wait, what?", "Mind blown!"],
                    knowledge_level="casual"
                ),
                Character(
                    name="Sarah",
                    personality="skeptical fact-checker with dry humor",
                    speaking_style="deadpan delivery with occasional sarcasm",
                    catchphrases=["Actually...", "Hold on there", "That can't be right"],
                    knowledge_level="expert"
                )
            ]
        elif format_type == DiscussionFormat.DEBATE:
            return [
                Character(
                    name="Pro",
                    personality="passionate advocate",
                    speaking_style="dramatic and persuasive",
                    catchphrases=["Clearly!", "Obviously!", "The evidence shows..."],
                    knowledge_level="expert"
                ),
                Character(
                    name="Con", 
                    personality="contrarian nitpicker",
                    speaking_style="questions everything",
                    catchphrases=["But what about...", "That's debatable", "I disagree"],
                    knowledge_level="expert"
                )
            ]
        else:  # Default to monologue
            return self._select_characters(DiscussionFormat.MONOLOGUE)
    
    def _generate_discussion_turns(
        self, 
        facts: List[Dict[str, str]], 
        characters: List[Character],
        format_type: DiscussionFormat,
        target_length: int
    ) -> List[DiscussionTurn]:
        """Generate the actual discussion turns."""
        if format_type == DiscussionFormat.MONOLOGUE:
            return self._generate_monologue(facts, characters[0], target_length)
        else:
            return self._generate_dialogue(facts, characters, format_type, target_length)
    
    def _generate_monologue(
        self, 
        facts: List[Dict[str, str]], 
        narrator: Character,
        target_length: int
    ) -> List[DiscussionTurn]:
        """Generate a single-person monologue in Sam O'Nella / Casually Explained style."""
        turns = []
        current_length = 0
        fact_index = 0
        used_phrases = set()  # Track used phrases to avoid repetition
        
        # Opening - casual and conversational
        topic_name = facts[0]['content'].split(' is ')[0] if facts and ' is ' in facts[0]['content'] else "this thing"
        
        openings = [
            f"The subject of {topic_name} offers some interesting details worth examining.",
            f"Consider {topic_name}, which presents several noteworthy aspects.",
            f"{topic_name} represents a fascinating example of human endeavor and its consequences.",
            f"The story of {topic_name} involves circumstances that merit closer attention.",
            f"An examination of {topic_name} reveals some unexpected complexities.",
            f"{topic_name} demonstrates how seemingly straightforward subjects can contain surprising elements."
        ]
        
        turns.append(DiscussionTurn(
            speaker=narrator.name,
            content=random.choice(openings),
            tone="conversational"
        ))
        
        # Create varied phrase pools with objective tone
        basic_intros = [
            "The research indicates", "Documentation shows", "Historical records reveal", 
            "Studies have found", "Evidence suggests", "Examination reveals", 
            "Analysis demonstrates", "Investigation shows", "The facts indicate"
        ]
        
        transitions = [
            "Additionally", "Furthermore", "Moreover", "Notably", "Interestingly", 
            "Subsequently", "Consequently", "In related developments", "As it happens"
        ]
        
        observations = [
            "This represents a notable pattern.", "Such outcomes are well-documented.", 
            "These circumstances align with broader trends.", "This exemplifies a common phenomenon.",
            "Such results follow predictable patterns.", "This outcome reflects typical human behavior.",
            "These findings align with established research.", "This demonstrates recurring themes."
        ]
        
        # Main content - weave facts with varied humor
        while current_length < target_length * 0.85 and fact_index < len(facts):
            fact = facts[fact_index]
            fact_content = fact['content']
            
            # Choose intro style and avoid repeats
            available_intros = [intro for intro in basic_intros if intro not in used_phrases]
            if not available_intros:
                available_intros = basic_intros
                used_phrases.clear()
            
            intro = random.choice(available_intros)
            used_phrases.add(intro)
            
            # Vary the fact presentation with objective humor
            if fact['type'] == 'basic':
                presentations = [
                    f"{intro} that {fact_content.lower()}. {random.choice(observations)}",
                    f"{fact_content}. This establishes the fundamental parameters of the subject.",
                    f"The basic definition states that {fact_content.lower()}, providing our starting point."
                ]
            elif 'death' in fact_content.lower() or 'died' in fact_content.lower():
                presentations = [
                    f"The outcome proved less favorable: {fact_content}. Such conclusions were apparently not considered in the original planning.",
                    f"Historical records document that {fact_content}. This represents a significant deviation from intended outcomes.",
                    f"{fact_content}. The long-term consequences of these events became evident in retrospect."
                ]
            elif any(word in fact_content.lower() for word in ['million', 'billion', 'thousand']):
                presentations = [
                    f"The scale involved is noteworthy: {fact_content}. These figures provide context for the magnitude of the undertaking.",
                    f"Quantitative analysis reveals that {fact_content}. Such measurements offer perspective on the scope of the phenomenon.",
                    f"{fact_content}. The numerical scale suggests this exceeded typical parameters for such activities."
                ]
            else:
                presentations = [
                    f"{intro} that {fact_content.lower()}.",
                    f"{fact_content}. {random.choice(transitions)}",
                    f"Documentation establishes that {fact_content}."
                ]
            
            turn_content = random.choice(presentations)
            turns.append(DiscussionTurn(
                speaker=narrator.name,
                content=turn_content,
                tone="dry"
            ))
            
            # Add varied observations occasionally
            if fact_index < len(facts) - 1 and random.random() < 0.3:
                observation_types = [
                    ["The decision-making process behind this initiative remains unclear.", "The methodology employed here raises several questions.", "The rationale for this approach is not immediately evident."],
                    ["This exemplifies certain recurring patterns in human organizational behavior.", "Such outcomes align with established behavioral research.", "These results reflect well-documented social phenomena."],
                    ["The documentation of such events serves an important archival function.", "The systematic recording of these details enables future analysis.", "Such detailed record-keeping facilitates historical understanding."],
                    ["The broader implications of these events became apparent over time.", "The full scope of consequences emerged gradually.", "Long-term effects proved more significant than initially anticipated."]
                ]
                
                observation_category = random.choice(observation_types)
                observation = random.choice(observation_category)
                
                turns.append(DiscussionTurn(
                    speaker=narrator.name,
                    content=observation,
                    tone="analytical"
                ))
            
            current_length = self._count_words(turns)
            fact_index += 1
        
        # Varied closings with objective tone
        closings = [
            "This overview provides a foundation for understanding the subject's key characteristics.",
            "These details offer insight into the complexity underlying seemingly straightforward topics.",
            "The examination reveals patterns that extend beyond this particular case study.",
            "Such analysis demonstrates the value of detailed investigation into documented phenomena.",
            "This exploration illustrates how systematic examination can uncover unexpected dimensions.",
            "These findings contribute to our broader understanding of similar historical events."
        ]
        
        turns.append(DiscussionTurn(
            speaker=narrator.name,
            content=random.choice(closings),
            tone="conversational"
        ))
        
        return turns
    
    def _generate_dialogue(
        self, 
        facts: List[Dict[str, str]], 
        characters: List[Character],
        format_type: DiscussionFormat,
        target_length: int
    ) -> List[DiscussionTurn]:
        """Generate dialogue for podcast/debate formats (original logic)."""
        turns = []
        current_length = 0
        fact_index = 0
        
        # Opening
        if format_type == DiscussionFormat.PODCAST:
            turns.append(DiscussionTurn(
                speaker=characters[0].name,
                content=f"Welcome back to 'Things That Exist!' I'm {characters[0].name}, and today we're diving into something absolutely {random.choice(['fascinating', 'bizarre', 'incredible'])}!",
                tone="excited"
            ))
            
            turns.append(DiscussionTurn(
                speaker=characters[1].name,
                content=f"And I'm {characters[1].name}. Let me guess - you found another Wikipedia rabbit hole?",
                tone="skeptical"
            ))
        
        # Main discussion - alternate between characters
        while current_length < target_length * 0.8 and fact_index < len(facts):
            fact = facts[fact_index]
            
            # Character 1 presents fact
            enthusiasm = random.choice(["So get this:", "Here's the wild part:", "But wait, there's more:"])
            turns.append(DiscussionTurn(
                speaker=characters[0].name,
                content=f"{enthusiasm} {fact['content']}",
                tone="excited"
            ))
            
            # Character 2 responds with humor
            if fact_index < len(facts) - 1:
                responses = [
                    "That sounds made up.",
                    "Are you reading from Wikipedia again?",
                    "I need to fact-check this.",
                    "That's... actually kind of amazing.",
                    "Why do you know this?"
                ]
                turns.append(DiscussionTurn(
                    speaker=characters[1].name,
                    content=random.choice(responses),
                    tone="skeptical"
                ))
            
            current_length = self._count_words(turns)
            fact_index += 1
        
        # Closing
        closings = [
            "Well, that's our show for today!",
            "And that's why the internet is dangerous.",
            "I'm never googling anything again.",
            "Thanks for joining us on this journey into the weird!"
        ]
        turns.append(DiscussionTurn(
            speaker=characters[0].name,
            content=random.choice(closings),
            tone="excited"
        ))
        
        return turns
    
    def _count_words(self, turns: List[DiscussionTurn]) -> int:
        """Count total words in discussion turns."""
        return sum(len(turn.content.split()) for turn in turns)
    
    def _estimate_duration(self, turns: List[DiscussionTurn]) -> float:
        """Estimate duration in seconds (assumes ~150 words per minute)."""
        word_count = self._count_words(turns)
        return (word_count / 150) * 60
    
    def _load_character_templates(self) -> Dict:
        """Load character templates (placeholder for now)."""
        return {}
    
    def _load_humor_techniques(self) -> Dict:
        """Load humor techniques (placeholder for now).""" 
        return {}
    
    def _load_discussion_templates(self) -> Dict:
        """Load discussion templates (placeholder for now)."""
        return {}
    
    def format_as_script(self, discussion: Dict[str, Any]) -> str:
        """Format discussion as a readable script."""
        script_lines = []
        script_lines.append(f"=== {discussion['topic']} ===")
        script_lines.append(f"Format: {discussion['format'].title()}")
        script_lines.append(f"Estimated Duration: {discussion['estimated_duration']:.1f} seconds")
        script_lines.append("=" * 50)
        script_lines.append("")
        
        for turn in discussion['script']:
            script_lines.append(f"{turn.speaker}: {turn.content}")
            script_lines.append("")
        
        script_lines.append("=" * 50)
        script_lines.append(f"Word Count: {discussion['word_count']} words")
        
        return "\n".join(script_lines)