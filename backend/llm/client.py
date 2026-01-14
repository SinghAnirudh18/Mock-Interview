"""
LLM Client using OpenRouter API.
OpenRouter provides unified access to many models including Claude, GPT-4, Gemini, etc.

Set OPENROUTER_API_KEY environment variable or pass directly.
"""
import os
import json
import logging
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

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


class GeminiClient:
    """
    Client for OpenRouter API.
    Provides access to many models with a unified interface.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Use provided key, or env var, or hardcoded fallback
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or "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"
        
        # OpenRouter API settings
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "google/gemini-2.0-flash-001"  # Good balance of speed and quality
        
        logger.info(f"OpenRouter Client initialized with model: {self.model}")
    
    def _make_request(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        """Make request to OpenRouter API."""
        import requests
        
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",  # Required by OpenRouter
            "X-Title": "AI Interviewer"  # Optional - shows in OpenRouter dashboard
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return {"error": str(e)}
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs  # Accept but ignore extra params for compatibility
    ) -> LLMResponse:
        """Generate completion from OpenRouter."""
        response = self._make_request(prompt, max_tokens, temperature)
        
        if "error" in response:
            return LLMResponse(
                content="",
                is_valid=False,
                raw_response=response,
                tokens_used=0
            )
        
        try:
            content = response["choices"][0]["message"]["content"]
            tokens = response.get("usage", {}).get("total_tokens", 0)
            
            return LLMResponse(
                content=content.strip(),
                is_valid=bool(content.strip()),
                raw_response=response,
                tokens_used=tokens
            )
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse OpenRouter response: {e}")
            return LLMResponse(
                content="",
                is_valid=False,
                raw_response=response,
                tokens_used=0
            )
    
    def generate_question(
        self,
        prompt: str,
        max_tokens: int = 150,
    ) -> Tuple[str, bool]:
        """
        Generate an interview question.
        """
        logger.info("Generating question via OpenRouter...")
        
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        if not response.is_valid:
            logger.warning(f"OpenRouter response invalid: {response.raw_response}")
            return "", False
        
        # Clean the response
        cleaned = response.content.strip()
        
        # Remove any quotes wrapping the response
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        
        logger.info(f"OpenRouter response: {cleaned[:100]}...")
        
        return cleaned, bool(cleaned)
    
    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.3,
    ) -> Tuple[Optional[Dict], bool]:
        """Generate JSON response."""
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if not response.is_valid:
            return None, False
        
        content = response.content
        
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group()), True
            except json.JSONDecodeError:
                pass
        
        # Try parsing the whole content
        try:
            return json.loads(content), True
        except json.JSONDecodeError:
            return None, False
    
    def generate_analysis(
        self,
        prompt: str,
        max_tokens: int = 500,
    ) -> Tuple[Optional[Dict], bool]:
        """Generate analysis response (JSON expected)."""
        return self.generate_json(prompt, max_tokens, temperature=0.3)
    
    def health_check(self) -> bool:
        """Check if API is responding."""
        try:
            response = self.generate("Say hello", max_tokens=10)
            return response.is_valid
        except Exception:
            return False


# Global client instance
llm_client = GeminiClient()

# Alias for backward compatibility
LLMClient = GeminiClient
