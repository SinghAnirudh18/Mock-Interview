"""
Configuration settings for the AI Interviewer system.
All settings can be overridden via environment variables.
"""
import os
from datetime import timedelta
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM server configuration."""
    base_url: str = field(default_factory=lambda: os.getenv("LLM_URL", "http://localhost:9000"))
    completion_endpoint: str = "/completion"
    timeout: int = 60  # Increased for DeepSeek R1 thinking time
    max_retries: int = 3
    
    # Default generation parameters
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_repeat_penalty: float = 1.2
    
    @property
    def completion_url(self) -> str:
        return f"{self.base_url}{self.completion_endpoint}"


@dataclass
class WhisperConfig:
    """Whisper STT configuration."""
    model_path: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL_PATH", "../models/medium"))
    device: str = "cuda"
    compute_type: str = "float16"


@dataclass
class MemoryConfig:
    """Memory and vector DB configuration."""
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    collection_name: str = "interview_memories"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Memory retrieval settings
    short_term_turns: int = 5  # Number of recent turns to keep in context
    rag_top_k: int = 5  # Number of similar memories to retrieve


@dataclass
class PhaseConfig:
    """Configuration for a single interview phase."""
    min_questions: int
    max_questions: int
    time_limit_minutes: int
    
    @property
    def time_limit(self) -> timedelta:
        return timedelta(minutes=self.time_limit_minutes)


@dataclass
class InterviewConfig:
    """Interview flow configuration."""
    default_job_role: str = "Software Engineer"
    
    # Phase configurations
    phases: Dict[str, PhaseConfig] = field(default_factory=lambda: {
        "greeting": PhaseConfig(min_questions=1, max_questions=2, time_limit_minutes=2),
        "introduction": PhaseConfig(min_questions=3, max_questions=5, time_limit_minutes=5),
        "technical": PhaseConfig(min_questions=5, max_questions=10, time_limit_minutes=20),
        "behavioral": PhaseConfig(min_questions=3, max_questions=6, time_limit_minutes=10),
        "situational": PhaseConfig(min_questions=2, max_questions=4, time_limit_minutes=8),
        "closing": PhaseConfig(min_questions=1, max_questions=3, time_limit_minutes=3),
    })
    
    # Scoring weights
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "quality": 0.2,
        "relevance": 0.25,
        "completeness": 0.2,
        "technical_depth": 0.2,
        "communication": 0.15,
    })


class Config:
    """Main configuration class combining all config sections."""
    
    def __init__(self):
        self.llm = LLMConfig()
        self.whisper = WhisperConfig()
        self.memory = MemoryConfig()
        self.interview = InterviewConfig()
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls()


# Global config instance
config = Config()
