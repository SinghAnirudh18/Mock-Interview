"""
Interview phase definitions and transition logic.
"""
from datetime import timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from models.schemas import InterviewPhase
from utils.config import config


@dataclass
class PhaseInfo:
    """Information about a single interview phase."""
    phase: InterviewPhase
    min_questions: int
    max_questions: int
    time_limit: timedelta
    description: str
    focus_areas: List[str]
    
    def get_config(self) -> Dict[str, Any]:
        """Get phase configuration as dictionary."""
        return {
            "min_questions": self.min_questions,
            "max_questions": self.max_questions,
            "time_limit_minutes": self.time_limit.total_seconds() / 60,
            "description": self.description,
            "focus_areas": self.focus_areas
        }


# Phase order for progression
PHASE_ORDER = [
    InterviewPhase.GREETING,
    InterviewPhase.INTRODUCTION,
    InterviewPhase.TECHNICAL,
    InterviewPhase.BEHAVIORAL,
    InterviewPhase.SITUATIONAL,
    InterviewPhase.CLOSING,
]


class InterviewPhases:
    """
    Manages interview phase definitions and transitions.
    """
    
    # Phase definitions with detailed configuration
    PHASES: Dict[InterviewPhase, PhaseInfo] = {
        InterviewPhase.GREETING: PhaseInfo(
            phase=InterviewPhase.GREETING,
            min_questions=1,
            max_questions=2,
            time_limit=timedelta(minutes=2),
            description="Initial greeting and introduction",
            focus_areas=["welcome", "rapport building"]
        ),
        InterviewPhase.INTRODUCTION: PhaseInfo(
            phase=InterviewPhase.INTRODUCTION,
            min_questions=3,
            max_questions=5,
            time_limit=timedelta(minutes=5),
            description="Understanding candidate background",
            focus_areas=["background", "motivation", "career goals", "experience overview"]
        ),
        InterviewPhase.TECHNICAL: PhaseInfo(
            phase=InterviewPhase.TECHNICAL,
            min_questions=5,
            max_questions=10,
            time_limit=timedelta(minutes=20),
            description="Technical knowledge assessment",
            focus_areas=["coding", "architecture", "problem-solving", "tools", "best practices"]
        ),
        InterviewPhase.BEHAVIORAL: PhaseInfo(
            phase=InterviewPhase.BEHAVIORAL,
            min_questions=3,
            max_questions=6,
            time_limit=timedelta(minutes=10),
            description="Past behavior and experiences",
            focus_areas=["teamwork", "conflict resolution", "leadership", "challenges"]
        ),
        InterviewPhase.SITUATIONAL: PhaseInfo(
            phase=InterviewPhase.SITUATIONAL,
            min_questions=2,
            max_questions=4,
            time_limit=timedelta(minutes=8),
            description="Hypothetical scenario handling",
            focus_areas=["decision-making", "judgment", "priorities", "problem-solving approach"]
        ),
        InterviewPhase.CLOSING: PhaseInfo(
            phase=InterviewPhase.CLOSING,
            min_questions=1,
            max_questions=3,
            time_limit=timedelta(minutes=3),
            description="Wrapping up the interview",
            focus_areas=["questions", "next steps", "closing remarks"]
        ),
    }
    
    @classmethod
    def get_phase_info(cls, phase: InterviewPhase) -> Optional[PhaseInfo]:
        """Get information about a specific phase."""
        return cls.PHASES.get(phase)
    
    @classmethod
    def get_next_phase(cls, current: InterviewPhase) -> Optional[InterviewPhase]:
        """
        Get the next phase in the interview progression.
        
        Args:
            current: The current phase
            
        Returns:
            The next phase, or None if at the end
        """
        try:
            idx = PHASE_ORDER.index(current)
            if idx < len(PHASE_ORDER) - 1:
                return PHASE_ORDER[idx + 1]
            return InterviewPhase.ENDED
        except ValueError:
            return None
    
    @classmethod
    def get_previous_phase(cls, current: InterviewPhase) -> Optional[InterviewPhase]:
        """Get the previous phase."""
        try:
            idx = PHASE_ORDER.index(current)
            if idx > 0:
                return PHASE_ORDER[idx - 1]
            return None
        except ValueError:
            return None
    
    @classmethod
    def should_transition(
        cls,
        phase: InterviewPhase,
        questions_asked: int,
        time_elapsed: timedelta,
        performance_score: Optional[float] = None
    ) -> bool:
        """
        Determine if we should transition to the next phase.
        
        Args:
            phase: Current phase
            questions_asked: Number of questions asked in this phase
            time_elapsed: Time spent in this phase
            performance_score: Optional average score (0-10) for adaptive transition
            
        Returns:
            True if should transition
        """
        info = cls.get_phase_info(phase)
        if not info:
            return False
        
        # Must ask minimum questions
        if questions_asked < info.min_questions:
            return False
        
        # Transition if max questions reached
        if questions_asked >= info.max_questions:
            return True
        
        # Transition if time limit exceeded
        if time_elapsed >= info.time_limit:
            return True
        
        # Adaptive: transition earlier if performance is very high or very low
        if performance_score is not None and questions_asked >= info.min_questions:
            if performance_score >= 8.5:  # Very strong, can move on
                return True
            if performance_score <= 3.0 and questions_asked >= info.min_questions + 1:
                # Struggling, move on after extra chance
                return True
        
        return False
    
    @classmethod
    def get_phase_progress(
        cls,
        phase: InterviewPhase,
        questions_asked: int,
        time_elapsed: timedelta
    ) -> Dict[str, Any]:
        """
        Get progress information for the current phase.
        
        Returns:
            Dictionary with progress metrics
        """
        info = cls.get_phase_info(phase)
        if not info:
            return {}
        
        question_progress = questions_asked / info.max_questions
        time_progress = time_elapsed.total_seconds() / info.time_limit.total_seconds()
        
        return {
            "phase": phase.value,
            "questions_asked": questions_asked,
            "min_questions": info.min_questions,
            "max_questions": info.max_questions,
            "question_progress": min(question_progress, 1.0),
            "time_elapsed_minutes": time_elapsed.total_seconds() / 60,
            "time_limit_minutes": info.time_limit.total_seconds() / 60,
            "time_progress": min(time_progress, 1.0),
            "can_transition": questions_asked >= info.min_questions,
            "must_transition": questions_asked >= info.max_questions or time_progress >= 1.0
        }
    
    @classmethod
    def get_all_phases_info(cls) -> List[Dict[str, Any]]:
        """Get information about all phases."""
        return [
            {
                "phase": phase.value,
                **cls.PHASES[phase].get_config()
            }
            for phase in PHASE_ORDER
        ]
