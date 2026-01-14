"""
Interview state machine for managing interview flow.
Tracks conversation, phases, and candidate profile.
"""
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from models.schemas import (
    InterviewPhase,
    InterviewSession,
    CandidateProfile,
    QuestionRecord,
    AnswerRecord,
    ConversationTurn,
    AnswerAnalysis,
)
from interview.phases import InterviewPhases, PHASE_ORDER
from utils.config import config


class InterviewStateMachine:
    """
    Manages the state of an interview session.
    Handles phase transitions, conversation history, and candidate profile.
    """
    
    def __init__(self, job_role: str = "Software Engineer", session_id: Optional[str] = None):
        """
        Initialize a new interview state machine.
        
        Args:
            job_role: The job role being interviewed for
            session_id: Optional existing session ID for resuming
        """
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        self.job_role = job_role
        
        # Phase tracking
        self.phase = InterviewPhase.GREETING
        self.phase_start_time = datetime.now()
        
        # Timing
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        # Conversation tracking
        self.questions_asked: List[QuestionRecord] = []
        self.answers_received: List[AnswerRecord] = []
        self.conversation_history: List[ConversationTurn] = []
        
        # Candidate state
        self.candidate_profile = CandidateProfile()
        self.current_topic: Optional[str] = None
        self.difficulty_level: int = 3  # 1-5 scale
        
        # Focus and observations
        self.focus_areas: List[str] = []
        self.red_flags: List[str] = []
        self.positive_signs: List[str] = []
        
        # Topic tracking (to avoid duplicates)
        self.covered_topics: List[str] = []
    
    # ========================================
    # Conversation Management
    # ========================================
    
    def add_question(self, question: str, topic: Optional[str] = None) -> QuestionRecord:
        """
        Record a question asked by the interviewer.
        
        Args:
            question: The question text
            topic: Optional topic being addressed
            
        Returns:
            The created question record
        """
        record = QuestionRecord(
            question=question,
            phase=self.phase,
            topic=topic or self.current_topic,
            difficulty_level=self.difficulty_level,
            timestamp=datetime.now()
        )
        self.questions_asked.append(record)
        
        # Add to conversation history
        self.conversation_history.append(ConversationTurn(
            role="interviewer",
            content=question,
            phase=self.phase,
            timestamp=datetime.now(),
            metadata={"topic": topic, "difficulty": self.difficulty_level}
        ))
        
        # Track topic
        if topic and topic not in self.covered_topics:
            self.covered_topics.append(topic)
        
        return record
    
    def add_answer(self, answer: str, analysis: AnswerAnalysis) -> AnswerRecord:
        """
        Record a candidate's answer with its analysis.
        
        Args:
            answer: The answer text
            analysis: The analysis of the answer
            
        Returns:
            The created answer record
        """
        # Reference the last question if available
        question_ref = None
        if self.questions_asked:
            question_ref = self.questions_asked[-1].question[:50]
        
        record = AnswerRecord(
            answer=answer,
            analysis=analysis,
            phase=self.phase,
            question_ref=question_ref,
            timestamp=datetime.now()
        )
        self.answers_received.append(record)
        
        # Add to conversation history
        self.conversation_history.append(ConversationTurn(
            role="candidate",
            content=answer,
            phase=self.phase,
            timestamp=datetime.now(),
            metadata={
                "scores": {
                    "quality": analysis.quality_score,
                    "relevance": analysis.relevance_score,
                    "completeness": analysis.completeness_score
                }
            }
        ))
        
        return record
    
    def get_last_question(self) -> Optional[str]:
        """Get the most recent question asked."""
        if self.questions_asked:
            return self.questions_asked[-1].question
        return None
    
    def get_last_answer(self) -> Optional[str]:
        """Get the most recent answer received."""
        if self.answers_received:
            return self.answers_received[-1].answer
        return None
    
    def get_recent_context(self, num_turns: int = 5) -> List[ConversationTurn]:
        """Get the most recent conversation turns for short-term memory."""
        return self.conversation_history[-num_turns:]
    
    def get_context_string(self, num_turns: int = 3) -> str:
        """Get recent context as a formatted string for prompts."""
        recent = self.get_recent_context(num_turns)
        if not recent:
            return "No conversation yet."
        
        lines = []
        for turn in recent:
            role = "Interviewer" if turn.role == "interviewer" else "Candidate"
            content = turn.content[:150] + "..." if len(turn.content) > 150 else turn.content
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    # ========================================
    # Phase Management
    # ========================================
    
    def get_phase_question_count(self, phase: Optional[InterviewPhase] = None) -> int:
        """Count questions asked in a specific phase."""
        target = phase or self.phase
        return sum(1 for q in self.questions_asked if q.phase == target)
    
    def get_phase_time_elapsed(self) -> timedelta:
        """Get time elapsed in current phase."""
        return datetime.now() - self.phase_start_time
    
    def should_transition_phase(self) -> bool:
        """Check if we should transition to the next phase."""
        if self.phase == InterviewPhase.ENDED:
            return False
        
        # Get average recent score for adaptive transition
        recent_scores = [a.analysis.average_score for a in self.answers_received[-3:]]
        avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else None
        
        return InterviewPhases.should_transition(
            phase=self.phase,
            questions_asked=self.get_phase_question_count(),
            time_elapsed=self.get_phase_time_elapsed(),
            performance_score=avg_score
        )
    
    def transition_to_next_phase(self) -> bool:
        """
        Transition to the next interview phase.
        
        Returns:
            True if transitioned, False if interview ended
        """
        next_phase = InterviewPhases.get_next_phase(self.phase)
        
        if next_phase is None or next_phase == InterviewPhase.ENDED:
            self.phase = InterviewPhase.ENDED
            self.end_time = datetime.now()
            return False
        
        self.phase = next_phase
        self.phase_start_time = datetime.now()
        self.current_topic = None
        
        return True
    
    def end_interview(self):
        """End the interview immediately."""
        self.phase = InterviewPhase.ENDED
        self.end_time = datetime.now()
    
    # ========================================
    # Candidate Profile Management
    # ========================================
    
    def update_profile_from_analysis(self, analysis: AnswerAnalysis):
        """
        Update candidate profile based on answer analysis.
        
        Args:
            analysis: The answer analysis to incorporate
        """
        extracted = analysis.extracted_info
        
        # Update skills
        for skill in extracted.get("skills", []):
            if skill and skill not in self.candidate_profile.skills:
                self.candidate_profile.skills.append(skill)
        
        # Update technologies
        for tech in extracted.get("technologies", []):
            if tech and tech not in self.candidate_profile.technologies:
                self.candidate_profile.technologies.append(tech)
        
        # Update experience level inference
        exp_level = extracted.get("experience_level", "")
        if exp_level == "senior" and (self.candidate_profile.experience_years or 0) < 5:
            self.candidate_profile.experience_years = 8
        elif exp_level == "mid" and (self.candidate_profile.experience_years or 0) < 3:
            self.candidate_profile.experience_years = 4
        elif exp_level == "junior" and self.candidate_profile.experience_years is None:
            self.candidate_profile.experience_years = 1
        
        # Update communication style
        if analysis.communication_quality >= 7:
            self.candidate_profile.communication_style = "clear and articulate"
        elif analysis.communication_quality <= 4:
            self.candidate_profile.communication_style = "needs improvement"
        
        # Update confidence
        conf = extracted.get("confidence_indicator", "")
        if conf == "high":
            self.candidate_profile.confidence_level = 5
        elif conf == "low":
            self.candidate_profile.confidence_level = 2
        
        # Collect red flags and positive signs
        self.red_flags.extend(analysis.red_flags)
        self.positive_signs.extend(analysis.positive_signs)
        
        # Adjust difficulty based on performance
        avg_score = analysis.average_score
        if avg_score >= 8:
            self.difficulty_level = min(5, self.difficulty_level + 1)
        elif avg_score <= 4:
            self.difficulty_level = max(1, self.difficulty_level - 1)
    
    def get_profile_summary(self) -> str:
        """Get a summary of the candidate profile for prompts."""
        parts = []
        
        if self.candidate_profile.skills:
            parts.append(f"Skills: {', '.join(self.candidate_profile.skills[:5])}")
        
        if self.candidate_profile.technologies:
            parts.append(f"Technologies: {', '.join(self.candidate_profile.technologies[:5])}")
        
        if self.candidate_profile.experience_years:
            parts.append(f"Experience: ~{self.candidate_profile.experience_years} years")
        
        if self.candidate_profile.communication_style:
            parts.append(f"Communication: {self.candidate_profile.communication_style}")
        
        return "; ".join(parts) if parts else "No profile data yet."
    
    # ========================================
    # Serialization
    # ========================================
    
    def to_session(self) -> InterviewSession:
        """Convert state machine to InterviewSession model."""
        return InterviewSession(
            session_id=self.session_id,
            job_role=self.job_role,
            phase=self.phase,
            start_time=self.start_time,
            end_time=self.end_time,
            phase_start_time=self.phase_start_time,
            questions_asked=self.questions_asked,
            answers_received=self.answers_received,
            conversation_history=self.conversation_history,
            candidate_profile=self.candidate_profile,
            current_topic=self.current_topic,
            difficulty_level=self.difficulty_level,
            focus_areas=self.focus_areas,
            red_flags=list(set(self.red_flags)),  # Deduplicate
            positive_signs=list(set(self.positive_signs)),
            covered_topics=self.covered_topics,
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current interview status."""
        phase_progress = InterviewPhases.get_phase_progress(
            self.phase,
            self.get_phase_question_count(),
            self.get_phase_time_elapsed()
        )
        
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "job_role": self.job_role,
            "questions_asked_total": len(self.questions_asked),
            "phase_progress": phase_progress,
            "difficulty_level": self.difficulty_level,
            "candidate_profile": {
                "skills": self.candidate_profile.skills[:5],
                "technologies": self.candidate_profile.technologies[:5],
                "experience_years": self.candidate_profile.experience_years,
                "confidence_level": self.candidate_profile.confidence_level,
            },
            "is_ended": self.phase == InterviewPhase.ENDED,
        }
