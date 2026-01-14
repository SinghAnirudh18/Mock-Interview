"""
LLM Client wrapper for llama.cpp REST API.
Handles communication with local LLM server, response cleaning, and retries.
"""
import json
import time
import requests
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from utils.config import config
from utils.cleaning import ResponseCleaner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    is_valid: bool
    raw_response: Dict[str, Any]
    tokens_used: int = 0


class LLMClient:
    """
    Client for interacting with llama.cpp /completion endpoint.
    Handles DeepSeek/LLaMA models with chain-of-thought cleaning.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or config.llm.base_url
        self.completion_url = f"{self.base_url}/completion"
        self.timeout = config.llm.timeout
        self.max_retries = config.llm.max_retries
        logger.info(f"LLM Client initialized: {self.completion_url} (timeout={self.timeout}s)")
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to LLM server with retries."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.completion_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        
        raise ConnectionError(f"Failed to connect to LLM server after {self.max_retries + 1} attempts: {last_error}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = None,
        top_p: float = None,
        repeat_penalty: float = None,
        stop_sequences: Optional[list] = None,
        grammar: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate completion from LLM.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate (n_predict)
            temperature: Sampling temperature (None uses default)
            top_p: Top-p sampling (None uses default)
            repeat_penalty: Repetition penalty (None uses default)
            stop_sequences: List of strings that stop generation
            grammar: Optional grammar constraint
            
        Returns:
            LLMResponse with cleaned content
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature or config.llm.default_temperature,
            "top_p": top_p or config.llm.default_top_p,
            "repeat_penalty": repeat_penalty or config.llm.default_repeat_penalty,
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        if grammar:
            payload["grammar"] = grammar
        
        try:
            response = self._make_request(payload)
            content = response.get("content", "")
            tokens = response.get("tokens_predicted", 0)
            
            return LLMResponse(
                content=content,
                is_valid=bool(content.strip()),
                raw_response=response,
                tokens_used=tokens
            )
        except Exception as e:
            return LLMResponse(
                content="",
                is_valid=False,
                raw_response={"error": str(e)},
                tokens_used=0
            )
    
    def generate_question(
        self,
        prompt: str,
        max_tokens: int = 200,  # Increased for DeepSeek R1 which includes thinking
    ) -> Tuple[str, bool]:
        """
        Generate an interview question with full cleaning.
        
        Returns:
            Tuple of (cleaned_question, is_valid)
        """
        logger.info("Generating question via LLM...")
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop_sequences=["Candidate:", "They said:", "Answer:", "Q:", "A:"]
        )
        
        if not response.is_valid:
            logger.warning(f"LLM response invalid: {response.raw_response}")
            return "", False
        
        logger.info(f"Raw LLM response: {response.content[:200]}...")
        cleaned, is_valid = ResponseCleaner.clean_interviewer_response(response.content)
        logger.info(f"Cleaned response (valid={is_valid}): {cleaned[:100] if cleaned else 'EMPTY'}")
        
        return cleaned, is_valid
    
    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.3,
    ) -> Tuple[Optional[Dict], bool]:
        """
        Generate JSON response from LLM.
        
        Returns:
            Tuple of (parsed_json, is_valid)
        """
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=["\n\n\n"],
        )
        
        if not response.is_valid:
            return None, False
        
        # Clean and extract JSON
        cleaned = ResponseCleaner.clean_json_response(response.content)
        
        try:
            parsed = json.loads(cleaned)
            return parsed, True
        except json.JSONDecodeError:
            # Try to fix common issues
            try:
                # Fix trailing commas
                fixed = cleaned.replace(",}", "}").replace(",]", "]")
                parsed = json.loads(fixed)
                return parsed, True
            except json.JSONDecodeError:
                return None, False
    
    def generate_analysis(
        self,
        prompt: str,
        max_tokens: int = 500,
    ) -> Tuple[Optional[Dict], bool]:
        """
        Generate analysis response (JSON expected).
        Uses lower temperature for more consistent output.
        """
        return self.generate_json(prompt, max_tokens, temperature=0.3)
    
    def health_check(self) -> bool:
        """Check if LLM server is responding."""
        try:
            response = self.generate("Hello", max_tokens=5)
            return response.is_valid
        except Exception:
            return False


# Global client instance
llm_client = LLMClient()
