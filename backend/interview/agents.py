"""
Agent orchestration for the AI interviewer system.
Coordinates all agents: Interviewer, Analysis, Extractor, Adaptive, Report.
"""
import random
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from llm.client import llm_client
from llm.prompts import Prompts, FALLBACK_QUESTIONS
from memory.rag import rag_pipeline
from memory.extractors import fact_extractor
from memory.vector_db import memory_store
from interview.scoring import AnswerScorer
from models.schemas import (
    InterviewPhase,
    AnswerAnalysis,
    CandidateProfile,
)

# Set up logging
logger = logging.getLogger(__name__)


class InterviewerAgent:
    """
    Generates interview questions based on phase and context.
    """
    
    def __init__(self):
        self.llm = llm_client
    
    def generate_question(
        self,
        phase: InterviewPhase,
        job_role: str,
        candidate_profile: CandidateProfile,
        covered_topics: List[str],
        difficulty_level: int,
        recent_context: str,
        rag_context: str = "",
    ) -> str:
        """
        Generate the next interview question.
        
        Args:
            phase: Current interview phase
            job_role: The job role being interviewed for
            candidate_profile: Current candidate profile
            covered_topics: Topics already covered
            difficulty_level: Current difficulty (1-5)
            recent_context: Recent conversation context
            rag_context: Context from RAG retrieval
            
        Returns:
            The generated question
        """
        logger.info(f"Generating question for phase: {phase.value}")
        
        # Build candidate info string
        candidate_info = []
        if candidate_profile.skills:
            candidate_info.append(f"Skills: {', '.join(candidate_profile.skills[:5])}")
        if candidate_profile.technologies:
            candidate_info.append(f"Technologies: {', '.join(candidate_profile.technologies[:5])}")
        candidate_str = "; ".join(candidate_info) if candidate_info else "No profile data yet"
        
        # Add RAG context if available
        if rag_context:
            candidate_str = f"{rag_context}\n\n{candidate_str}"
        
        # Generate phase-specific prompt
        if phase == InterviewPhase.GREETING:
            prompt = Prompts.interviewer_greeting(job_role)
        elif phase == InterviewPhase.INTRODUCTION:
            prompt = Prompts.interviewer_introduction(job_role, candidate_str)
        elif phase == InterviewPhase.TECHNICAL:
            covered = ", ".join(covered_topics[-5:]) if covered_topics else "None yet"
            techs = ", ".join(candidate_profile.technologies[:3]) if candidate_profile.technologies else ""
            prompt = Prompts.interviewer_technical(job_role, techs, difficulty_level, covered)
        elif phase == InterviewPhase.BEHAVIORAL:
            prompt = Prompts.interviewer_behavioral(recent_context)
        elif phase == InterviewPhase.SITUATIONAL:
            skills = ", ".join(candidate_profile.skills[:3]) if candidate_profile.skills else ""
            prompt = Prompts.interviewer_situational(job_role, skills)
        elif phase == InterviewPhase.CLOSING:
            prompt = Prompts.interviewer_closing()
        else:
            return "Thank you for participating. The interview is now complete."
        
        # Generate question using LLM
        question, is_valid = self.llm.generate_question(prompt, max_tokens=200)
        
        if not is_valid or not question:
            logger.warning(f"LLM failed for {phase.value}, using fallback")
            question = self._get_fallback(phase, difficulty_level)
        else:
            logger.info(f"LLM generated question: {question[:80]}...")
        
        return question
    
    def _get_fallback(self, phase: InterviewPhase, difficulty_level: int) -> str:
        """Get a fallback question for the phase."""
        phase_key = phase.value
        
        if phase_key == "technical":
            level_questions = FALLBACK_QUESTIONS.get(phase_key, {}).get(
                difficulty_level, 
                FALLBACK_QUESTIONS.get(phase_key, {}).get(3, ["Tell me about a technical challenge you faced."])
            )
            return random.choice(level_questions)
        else:
            questions = FALLBACK_QUESTIONS.get(phase_key, ["Could you tell me more about that?"])
            return random.choice(questions)


class AnalysisAgent:
    """
    Analyzes candidate answers for quality, relevance, and extracted info.
    """
    
    def __init__(self):
        self.llm = llm_client
    
    def analyze_answer(
        self,
        question: str,
        answer: str,
        phase: InterviewPhase,
        job_role: str
    ) -> AnswerAnalysis:
        """
        Analyze a candidate's answer.
        
        Args:
            question: The question that was asked
            answer: The candidate's answer
            phase: Current interview phase
            job_role: The job role
            
        Returns:
            AnswerAnalysis with scores and extracted info
        """
        if not answer or len(answer.strip()) < 5:
            return self._get_default_analysis()
        
        # Generate analysis prompt
        prompt = Prompts.analyze_answer(
            job_role=job_role,
            phase=phase.value,
            question=question,
            answer=answer
        )
        
        # Get LLM analysis
        result, is_valid = self.llm.generate_analysis(prompt, max_tokens=500)
        
        if is_valid and result:
            return AnswerScorer.validate_analysis(result)
        
        return self._get_default_analysis()
    
    def _get_default_analysis(self) -> AnswerAnalysis:
        """Return default analysis when LLM fails."""
        return AnswerAnalysis(
            quality_score=5,
            relevance_score=5,
            completeness_score=5,
            technical_depth=3,
            communication_quality=5,
            extracted_info={
                "skills": [],
                "technologies": [],
                "experience_level": "unknown",
                "communication_style": "unknown",
                "confidence_indicator": "medium",
                "key_points": []
            },
            suggested_follow_ups=["Can you tell me more about that?"],
            areas_to_probe=["Follow up on previous answer"]
        )


class ReportAgent:
    """
    Generates final interview reports and recommendations.
    """
    
    def __init__(self):
        self.llm = llm_client
    
    def generate_report(
        self,
        job_role: str,
        candidate_profile: CandidateProfile,
        all_analyses: List[AnswerAnalysis],
        red_flags: List[str],
        positive_signs: List[str]
    ) -> Dict[str, Any]:
        """
        Generate the final interview report.
        
        Args:
            job_role: The job role
            candidate_profile: Final candidate profile
            all_analyses: All answer analyses
            red_flags: Collected red flags
            positive_signs: Collected positive signs
            
        Returns:
            Report dictionary
        """
        # Calculate scores
        overall_scores = AnswerScorer.calculate_overall_scores(all_analyses)
        avg_weighted = sum(overall_scores.values()) / len(overall_scores) if overall_scores else 5
        
        # Build profile string
        profile_str = []
        profile_str.append(f"Skills: {', '.join(candidate_profile.skills) or 'None identified'}")
        profile_str.append(f"Technologies: {', '.join(candidate_profile.technologies) or 'None identified'}")
        profile_str.append(f"Experience: {candidate_profile.experience_years or 'Unknown'} years")
        profile_str.append(f"Communication: {candidate_profile.communication_style or 'Neutral'}")
        
        # Build score string
        score_str = ", ".join([f"{k}={v}/10" for k, v in overall_scores.items()])
        
        # Generate LLM report
        prompt = Prompts.generate_report(
            job_role=job_role,
            candidate_profile="\n".join(profile_str),
            avg_scores=score_str,
            red_flags=", ".join(red_flags[:5]) if red_flags else "None",
            positive_signs=", ".join(positive_signs[:5]) if positive_signs else "Several positive indicators"
        )
        
        result, is_valid = self.llm.generate_json(prompt, max_tokens=400)
        
        if is_valid and result:
            return result
        
        # Fallback report
        return {
            "recommendation": AnswerScorer.get_recommendation(avg_weighted),
            "fit_score": round(avg_weighted),
            "summary": "Interview completed. Manual review recommended for final assessment.",
            "strengths": positive_signs[:3] if positive_signs else ["Engaged participant"],
            "weaknesses": red_flags[:3] if red_flags else [],
            "improvement_areas": [],
            "next_steps": ["Technical assessment", "Team interview"],
            "technical_depth_estimate": "mid",
            "behavior_pattern_summary": "Standard interview behavior observed."
        }


class AgentController:
    """
    Orchestrates all agents and manages the interview flow.
    """
    
    def __init__(self):
        self.interviewer = InterviewerAgent()
        self.analyzer = AnalysisAgent()
        self.reporter = ReportAgent()
    
    def process_answer(
        self,
        session_id: str,
        question: str,
        answer: str,
        phase: InterviewPhase,
        job_role: str,
        candidate_profile: CandidateProfile
    ) -> Tuple[AnswerAnalysis, List[str]]:
        """
        Process a candidate's answer through all relevant agents.
        
        Args:
            session_id: The interview session ID
            question: The question that was asked
            answer: The candidate's answer
            phase: Current phase
            job_role: The job role
            candidate_profile: Current candidate profile
            
        Returns:
            Tuple of (AnswerAnalysis, list of stored fact IDs)
        """
        # Step 1: Analyze the answer
        analysis = self.analyzer.analyze_answer(
            question=question,
            answer=answer,
            phase=phase,
            job_role=job_role
        )
        
        # Step 2: Extract facts
        facts = fact_extractor.extract_facts(phase, question, answer)
        
        # Also extract from analysis results
        analysis_facts = fact_extractor.extract_from_analysis(analysis.model_dump())
        facts.extend(analysis_facts)
        
        # Step 3: Store facts in vector DB with embeddings
        fact_dicts = [f.to_dict() for f in facts]
        fact_ids = rag_pipeline.store_answer_facts(
            session_id=session_id,
            facts=fact_dicts,
            phase=phase
        )
        
        return analysis, fact_ids
    
    def generate_next_question(
        self,
        session_id: str,
        phase: InterviewPhase,
        job_role: str,
        candidate_profile: CandidateProfile,
        covered_topics: List[str],
        difficulty_level: int,
        recent_context: str
    ) -> str:
        """
        Generate the next interview question with RAG context.
        
        Args:
            session_id: The interview session ID
            phase: Current phase
            job_role: The job role
            candidate_profile: Current candidate profile
            covered_topics: Topics already covered
            difficulty_level: Current difficulty
            recent_context: Recent conversation context
            
        Returns:
            The next question
        """
        # Get RAG context
        rag_context = rag_pipeline.get_relevant_context_for_question(
            session_id=session_id,
            current_phase=phase,
            current_topic=covered_topics[-1] if covered_topics else None
        )
        
        # Generate question
        return self.interviewer.generate_question(
            phase=phase,
            job_role=job_role,
            candidate_profile=candidate_profile,
            covered_topics=covered_topics,
            difficulty_level=difficulty_level,
            recent_context=recent_context,
            rag_context=rag_context
        )
    
    def generate_final_report(
        self,
        job_role: str,
        candidate_profile: CandidateProfile,
        all_analyses: List[AnswerAnalysis],
        red_flags: List[str],
        positive_signs: List[str]
    ) -> Dict[str, Any]:
        """
        Generate the final interview report.
        
        Args:
            job_role: The job role
            candidate_profile: Final candidate profile
            all_analyses: All answer analyses
            red_flags: Collected red flags
            positive_signs: Collected positive signs
            
        Returns:
            Complete report dictionary
        """
        return self.reporter.generate_report(
            job_role=job_role,
            candidate_profile=candidate_profile,
            all_analyses=all_analyses,
            red_flags=red_flags,
            positive_signs=positive_signs
        )


# Global controller instance
agent_controller = AgentController()
