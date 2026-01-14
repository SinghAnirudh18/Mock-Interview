"""
Response cleaning utilities for LLM outputs.
Handles DeepSeek R1 chain-of-thought removal with AGGRESSIVE cleaning.

This module is specifically tuned for DeepSeek-R1-Distill models that 
include extensive <think> reasoning blocks before actual responses.
"""
import re
from typing import Optional, Tuple


class ResponseCleaner:
    """
    Aggressively cleans LLM responses to remove ALL reasoning content.
    Designed for DeepSeek R1 models with visible chain-of-thought.
    """
    
    # Patterns that indicate AI is giving advice instead of asking questions
    ANSWER_INDICATORS = [
        "you should", "i recommend", "it's important to", "the best time",
        "generally speaking", "in my experience", "typically",
        "it depends on", "you'll want to", "you need to",
        "here's what", "let me explain", "the answer is"
    ]
    
    @classmethod
    def aggressive_clean(cls, text: str) -> str:
        """
        AGGRESSIVE cleaning for DeepSeek R1 output.
        Removes ALL content before actual interviewer speech.
        """
        if not text:
            return ""
        
        # Step 1: Remove everything between <think> tags (including partial tags)
        # Handle various formats: <think>, </think>, <|think|>, etc.
        cleaned = re.sub(r'</?think[^>]*>.*?(?=</?think|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Step 2: Remove partial/broken think tags like "hink>" or "<thin" 
        cleaned = re.sub(r'<?h?t?h?i?n?k?>?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?\s*think\s*>?', '', cleaned, flags=re.IGNORECASE)
        
        # Step 3: Remove ALL content before common interview starters
        # This is the nuclear option - find where actual speech starts
        interview_starters = [
            r"(Hello[,!]?\s)",
            r"(Hi[,!]?\s)",
            r"(Good\s+(?:morning|afternoon|evening))",
            r"(Welcome[,!]?\s)",
            r"(Thank\s+you)",
            r"(Great[,!]?\s)",
            r"(That'?s?\s+(?:great|interesting|good))",
            r"(Could\s+you)",
            r"(Can\s+you)",
            r"(Would\s+you)",
            r"(Tell\s+me)",
            r"(Please\s+(?:tell|describe|explain))",
            r"(What\s+(?:is|are|do|did|would|made|brings|drew))",
            r"(How\s+(?:do|did|would|have))",
            r"(Why\s+(?:do|did|would|are))",
            r"(Describe\s)",
            r"(Explain\s)",
            r"(Walk\s+me)",
            r"(Share\s)",
            r"(I'?m\s+Alex)",
            r"(Nice\s+to\s+meet)",
        ]
        
        for pattern in interview_starters:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                # Keep only from this point forward
                cleaned = cleaned[match.start():]
                break
        
        # Step 4: Remove remaining internal reasoning phrases
        reasoning_patterns = [
            r"^.*?(?:okay|alright|let me|so|now|first|then)\s*[,.]?\s*",
            r"^.*?(?:I need to|I should|I will|I think|I assume)\s+.*?[.!]\s*",
            r"^.*?(?:looking at|based on|considering|given that)\s+.*?[.!]\s*",
            r"^.*?(?:the candidate|they said|they mentioned)\s+.*?[.!]\s*",
        ]
        
        for pattern in reasoning_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Step 5: Remove parenthetical and bracketed content
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\*[^*]*\*', '', cleaned)
        
        # Step 6: Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    @classmethod
    def extract_first_question(cls, text: str) -> Optional[str]:
        """Extract the first question from response."""
        # Find sentences ending with ?
        matches = re.findall(r'[^.!?]*\?', text)
        if matches:
            # Return the first substantial question
            for match in matches:
                match = match.strip()
                if len(match.split()) >= 3:  # At least 3 words
                    return match
        return None
    
    @classmethod
    def is_valid_interviewer_response(cls, text: str) -> bool:
        """Check if text looks like valid interviewer speech."""
        if not text or len(text) < 10:
            return False
        
        # Check for reasoning indicators that shouldn't be in final output
        bad_indicators = [
            'hink>', '<think', 'okay,', 'alright,', 'let me', 'i need to',
            'i should', 'looking at', 'based on', 'the candidate', 'they said',
            'my reasoning', 'first i', 'then i', 'so i', 'considering'
        ]
        
        text_lower = text.lower()
        for indicator in bad_indicators:
            if indicator in text_lower:
                return False
        
        return True
    
    @classmethod
    def clean_interviewer_response(cls, text: str) -> Tuple[str, bool]:
        """
        Full cleaning pipeline for interviewer responses.
        Uses AGGRESSIVE cleaning for DeepSeek R1 models.
        
        Returns:
            Tuple of (cleaned_text, is_valid)
        """
        if not text:
            return "", False
        
        # Aggressive cleaning
        cleaned = cls.aggressive_clean(text)
        
        # Validate the result
        if not cls.is_valid_interviewer_response(cleaned):
            # Try to extract just a question
            question = cls.extract_first_question(text)
            if question:
                cleaned = cls.aggressive_clean(question)
            else:
                return "", False
        
        # Final validation
        if not cleaned or len(cleaned) < 10:
            return "", False
        
        # Ensure it ends with proper punctuation
        if not cleaned.endswith(('?', '!', '.')):
            question_words = ['what', 'how', 'why', 'can', 'could', 'would', 
                            'tell', 'describe', 'explain', 'when', 'where', 'who']
            if any(cleaned.lower().startswith(w) for w in question_words):
                cleaned += '?'
            else:
                cleaned += '.'
        
        return cleaned, True
    
    @classmethod
    def clean_json_response(cls, text: str) -> str:
        """Clean response and extract JSON content."""
        # Remove ALL content before the first {
        cleaned = re.sub(r'^.*?(?=\{)', '', text, flags=re.DOTALL)
        
        # Find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if json_match:
            return json_match.group()
        
        return "{}"
    
    @classmethod
    def clean_analysis_response(cls, text: str) -> str:
        """Clean response for analysis outputs."""
        return cls.clean_json_response(text)
