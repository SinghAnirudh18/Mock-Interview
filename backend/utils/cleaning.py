"""
Response cleaning utilities for LLM outputs.
Handles DeepSeek chain-of-thought removal, meta-commentary stripping,
and validation of interviewer responses.
"""
import re
from typing import Optional, Tuple


class ResponseCleaner:
    """
    Cleans LLM responses to remove unwanted content like:
    - Chain-of-thought tags (<think>, <thought>)
    - Meta-commentary and stage directions
    - Advice-giving patterns (when interviewer should only ask)
    """
    
    # Patterns that indicate AI is giving advice instead of asking questions
    ANSWER_INDICATORS = [
        "you should", "i recommend", "it's important to", "the best time",
        "generally speaking", "in my experience", "typically",
        "it depends on", "you'll want to", "you need to",
        "here's what", "let me explain", "the answer is",
        "you can", "i suggest", "my advice", "i think you should"
    ]
    
    # Thinking patterns to remove
    THINKING_PATTERNS = [
        r'<think>.*?</think>',
        r'<thought>.*?</thought>',
        r'<reasoning>.*?</reasoning>',
        r'<internal>.*?</internal>',
        r'\*thinks?\*.*?\*',
        r'\*internal.*?\*',
    ]
    
    # Meta-commentary patterns to remove
    META_PATTERNS = [
        r'^\s*(Wait,|Hmm,|Let me see,|Let me think,|I need to|Looking at|The candidate)',
        r'\(.*?internal.*?\)',
        r'\[.*?thinking.*?\]',
        r'^\s*\*.*?\*\s*$',
    ]
    
    @classmethod
    def clean_thinking_tags(cls, text: str) -> str:
        """Remove all thinking/reasoning tags from response."""
        cleaned = text
        for pattern in cls.THINKING_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()
    
    @classmethod
    def clean_meta_commentary(cls, text: str) -> str:
        """Remove meta-commentary and stage directions."""
        cleaned = text
        for pattern in cls.META_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove parenthetical and bracketed comments
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
        
        return cleaned.strip()
    
    @classmethod
    def is_giving_advice(cls, text: str) -> bool:
        """Check if the response is giving advice instead of asking a question."""
        text_lower = text.lower()
        advice_count = sum(1 for indicator in cls.ANSWER_INDICATORS if indicator in text_lower)
        
        # If multiple advice indicators or no question mark, likely giving advice
        has_question = '?' in text
        return advice_count >= 2 or (advice_count >= 1 and not has_question)
    
    @classmethod
    def extract_question(cls, text: str) -> Optional[str]:
        """Extract only the question part from a response."""
        # Try to find sentences ending with question marks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        questions = [s.strip() for s in sentences if '?' in s]
        
        if questions:
            # Return the most relevant question (usually the last one)
            return questions[-1]
        return None
    
    @classmethod
    def clean_interviewer_response(cls, text: str) -> Tuple[str, bool]:
        """
        Full cleaning pipeline for interviewer responses.
        
        Returns:
            Tuple of (cleaned_text, is_valid)
            If is_valid is False, fallback question should be used.
        """
        if not text:
            return "", False
        
        # Step 1: Remove thinking tags
        cleaned = cls.clean_thinking_tags(text)
        
        # Step 2: Remove meta-commentary
        cleaned = cls.clean_meta_commentary(cleaned)
        
        # Step 3: Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Step 4: Check if response is valid - be more lenient
        if not cleaned or len(cleaned) < 5:
            return "", False
        
        # Step 5: Check if AI is giving too much advice (only if very obvious)
        if cls.is_giving_advice(cleaned):
            # Try to extract just the question
            question = cls.extract_question(cleaned)
            if question and len(question) > 10:
                cleaned = question
            # Don't return False here - still use the cleaned content
        
        # Step 6: Ensure it's a question-like response
        # If it doesn't end properly, try to clean it up
        cleaned = cleaned.rstrip('.')
        if not cleaned.endswith('?') and not cleaned.endswith('!'):
            # Only add ? if it seems like a question
            question_starters = ['what', 'how', 'why', 'can', 'could', 'would', 'tell', 'describe', 'explain', 'when', 'where', 'who', 'do', 'did', 'have', 'are', 'is']
            if any(cleaned.lower().startswith(starter) for starter in question_starters):
                cleaned += '?'
        
        return cleaned, True
    
    @classmethod
    def clean_json_response(cls, text: str) -> str:
        """Clean response and extract JSON content."""
        # Remove thinking tags first
        cleaned = cls.clean_thinking_tags(text)
        
        # Find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if json_match:
            return json_match.group()
        
        # Try to find JSON array
        array_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', cleaned, re.DOTALL)
        if array_match:
            return array_match.group()
        
        return cleaned
    
    @classmethod
    def clean_analysis_response(cls, text: str) -> str:
        """Clean response for analysis outputs (JSON expected)."""
        return cls.clean_json_response(text)
