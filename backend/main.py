"""
Production AI Interviewer - FastAPI Backend

A production-grade AI interviewer system with:
- ChromaDB for long-term vector memory
- RAG for context retrieval
- Multi-agent architecture
- Adaptive questioning
- Comprehensive scoring and reporting

Compatible with DeepSeek / llama.cpp REST API.
"""
import sys
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import config
from models.schemas import (
    InterviewPhase,
    StartInterviewRequest,
    InterviewResponse,
    AnswerAnalysis,
)
from interview.state import InterviewStateMachine
from interview.agents import agent_controller
from interview.scoring import AnswerScorer
from memory.vector_db import memory_store

# ================================================================
# FastAPI App Initialization
# ================================================================

app = FastAPI(
    title="AI Interviewer API",
    description="Production-grade AI interviewer with vector memory and RAG",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# Whisper Model (GPU STT)
# ================================================================

# Lazy loading of Whisper model to avoid startup delay
_whisper_model = None

def get_whisper_model():
    """Lazy load the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            config.whisper.model_path,
            device=config.whisper.device,
            compute_type=config.whisper.compute_type
        )
    return _whisper_model

# ================================================================
# Session Management
# ================================================================

# Global interview state (single session for now)
# In production, use Redis or database for multi-session support
_current_session: Optional[InterviewStateMachine] = None


def get_current_session() -> InterviewStateMachine:
    """Get the current interview session."""
    global _current_session
    if _current_session is None:
        raise HTTPException(
            status_code=400,
            detail="No active interview session. Please start an interview first."
        )
    return _current_session


def create_new_session(job_role: str) -> InterviewStateMachine:
    """Create a new interview session."""
    global _current_session
    _current_session = InterviewStateMachine(job_role=job_role)
    return _current_session


def clear_session():
    """Clear the current session."""
    global _current_session
    if _current_session:
        # Clear memory for this session
        memory_store.clear_session(_current_session.session_id)
    _current_session = None


# ================================================================
# API Endpoints
# ================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "version": "2.0.0",
        "service": "AI Interviewer",
        "memory_stats": memory_store.get_collection_stats()
    }


@app.post("/start-interview")
async def start_interview(request: StartInterviewRequest):
    """
    Start a new interview session.
    
    Args:
        request: Contains job_role and optional focus_areas
        
    Returns:
        Interview session info with initial greeting
    """
    # Create new session
    session = create_new_session(request.job_role)
    
    if request.focus_areas:
        session.focus_areas = request.focus_areas
    
    # Generate initial greeting
    greeting = agent_controller.generate_next_question(
        session_id=session.session_id,
        phase=session.phase,
        job_role=session.job_role,
        candidate_profile=session.candidate_profile,
        covered_topics=session.covered_topics,
        difficulty_level=session.difficulty_level,
        recent_context=""
    )
    
    # Record the greeting as first question
    session.add_question(greeting, topic="introduction")
    
    return {
        "status": "Interview started",
        "session_id": session.session_id,
        "job_role": session.job_role,
        "phase": session.phase.value,
        "interviewer_message": greeting,
        "phases": [p.value for p in InterviewPhase.get_order()]
    }


@app.post("/interview-response")
async def interview_response(file: UploadFile = File(...)):
    """
    Process candidate's voice response during interview.
    
    Args:
        file: Audio file with candidate's response
        
    Returns:
        Transcript, analysis, and next question
    """
    session = get_current_session()
    
    if session.phase == InterviewPhase.ENDED:
        return {
            "error": "Interview has ended",
            "interviewer_message": "The interview has concluded. Thank you for your time.",
            "interview_ended": True
        }
    
    # Validate file type
    content_type = file.content_type or ""
    if "audio" not in content_type and "video" not in content_type and "webm" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"File must be audio or video. Got: {content_type}"
        )
    
    # Transcribe audio
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            audio_path = tmp.name
        
        whisper = get_whisper_model()
        segments, _ = whisper.transcribe(audio_path)
        user_text = " ".join([s.text for s in segments]).strip()
        
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except:
            pass
        
        if not user_text or len(user_text.split()) < 2:
            return {
                "interviewer_message": "I didn't catch that. Could you please repeat your answer?",
                "phase": session.phase.value,
                "transcript": "",
                "interview_ended": False
            }
    
    except Exception as e:
        return {
            "error": f"Transcription failed: {str(e)}",
            "interviewer_message": "There was an issue processing your audio. Could you please try again?",
            "phase": session.phase.value,
            "interview_ended": False
        }
    
    # Process the transcribed text
    return await _process_text_response(session, user_text)


@app.post("/text-response")
async def text_response(user_text: str = Query(..., min_length=1)):
    """
    Process text response from candidate (for testing).
    
    Args:
        user_text: The candidate's text response
        
    Returns:
        Analysis and next question
    """
    session = get_current_session()
    
    if session.phase == InterviewPhase.ENDED:
        return {
            "error": "Interview has ended",
            "interviewer_message": "The interview is complete. Thank you.",
            "interview_ended": True
        }
    
    return await _process_text_response(session, user_text)


async def _process_text_response(
    session: InterviewStateMachine,
    user_text: str
) -> Dict[str, Any]:
    """
    Process a text response from the candidate.
    
    Args:
        session: The interview session
        user_text: The candidate's response text
        
    Returns:
        Response dictionary with analysis and next question
    """
    # Get the last question
    last_question = session.get_last_question() or ""
    
    # Process answer through agents (analyze + extract facts + store in vector DB)
    analysis, fact_ids = agent_controller.process_answer(
        session_id=session.session_id,
        question=last_question,
        answer=user_text,
        phase=session.phase,
        job_role=session.job_role,
        candidate_profile=session.candidate_profile
    )
    
    # Record the answer
    session.add_answer(user_text, analysis)
    
    # Update candidate profile
    session.update_profile_from_analysis(analysis)
    
    # Check for phase transition
    if session.should_transition_phase():
        if not session.transition_to_next_phase():
            # Interview ended
            return {
                "transcript": user_text,
                "interviewer_message": "Thank you for your time. The interview is now complete. You can request your interview report.",
                "phase": session.phase.value,
                "analysis_scores": {
                    "quality": analysis.quality_score,
                    "relevance": analysis.relevance_score,
                    "completeness": analysis.completeness_score,
                    "technical_depth": analysis.technical_depth,
                    "communication": analysis.communication_quality
                },
                "interview_ended": True,
                "facts_stored": len(fact_ids)
            }
    
    # Generate next question with RAG context
    next_question = agent_controller.generate_next_question(
        session_id=session.session_id,
        phase=session.phase,
        job_role=session.job_role,
        candidate_profile=session.candidate_profile,
        covered_topics=session.covered_topics,
        difficulty_level=session.difficulty_level,
        recent_context=session.get_context_string(3)
    )
    
    # Record the question
    session.add_question(next_question)
    
    return {
        "transcript": user_text,
        "interviewer_message": next_question,
        "phase": session.phase.value,
        "analysis_scores": {
            "quality": analysis.quality_score,
            "relevance": analysis.relevance_score,
            "completeness": analysis.completeness_score,
            "technical_depth": analysis.technical_depth,
            "communication": analysis.communication_quality
        },
        "candidate_profile_update": {
            "skills": session.candidate_profile.skills[-5:],
            "technologies": session.candidate_profile.technologies[-5:],
            "difficulty_level": session.difficulty_level
        },
        "questions_asked": len(session.questions_asked),
        "interview_ended": False,
        "facts_stored": len(fact_ids)
    }


@app.get("/interview-status")
async def get_interview_status():
    """
    Get current interview status and candidate profile.
    
    Returns:
        Current phase, progress, and candidate info
    """
    session = get_current_session()
    return session.get_status()


@app.post("/end-interview")
async def end_interview():
    """
    End the interview early.
    
    Returns:
        Interview summary
    """
    session = get_current_session()
    session.end_interview()
    
    duration = None
    if session.start_time and session.end_time:
        duration = (session.end_time - session.start_time).total_seconds() / 60
    
    return {
        "status": "Interview ended",
        "message": "Interview terminated. You can request the report.",
        "duration_minutes": round(duration, 1) if duration else None,
        "total_questions": len(session.questions_asked),
        "phases_covered": list(set(q.phase.value for q in session.questions_asked))
    }


@app.get("/interview-report")
async def get_interview_report():
    """
    Generate comprehensive interview report.
    
    Returns:
        Full interview report with scores, analysis, and recommendations
    """
    session = get_current_session()
    
    if not session.answers_received:
        raise HTTPException(
            status_code=400,
            detail="No interview data available. Please complete at least one exchange."
        )
    
    # Get all analyses
    all_analyses = [a.analysis for a in session.answers_received]
    
    # Calculate overall scores
    overall_scores = AnswerScorer.calculate_overall_scores(all_analyses)
    
    # Calculate phase scores
    phase_scores = {}
    for phase in InterviewPhase.get_order():
        phase_analyses = [
            a.analysis for a in session.answers_received 
            if a.phase == phase
        ]
        if phase_analyses:
            phase_scores[phase.value] = AnswerScorer.aggregate_phase_scores(phase_analyses, phase)
    
    # Generate AI assessment
    ai_assessment = agent_controller.generate_final_report(
        job_role=session.job_role,
        candidate_profile=session.candidate_profile,
        all_analyses=all_analyses,
        red_flags=session.red_flags,
        positive_signs=session.positive_signs
    )
    
    # Calculate duration
    duration_minutes = None
    if session.start_time:
        end = session.end_time or datetime.now()
        duration_minutes = round((end - session.start_time).total_seconds() / 60, 1)
    
    # Get memory summary
    memory_summary = memory_store.get_session_summary(session.session_id)
    
    # Build skill graph from profile
    skill_graph = {}
    for tech in session.candidate_profile.technologies[:10]:
        skill_graph[tech] = session.candidate_profile.depth_of_knowledge.get(tech, 5)
    for skill in session.candidate_profile.skills[:10]:
        if skill not in skill_graph:
            skill_graph[skill] = 5
    
    # Build complete report
    report = {
        "interview_metadata": {
            "session_id": session.session_id,
            "job_role": session.job_role,
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "end_time": session.end_time.isoformat() if session.end_time else datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "total_questions": len(session.questions_asked),
            "phases_covered": list(set(q.phase.value for q in session.questions_asked))
        },
        "candidate_assessment": ai_assessment,
        "detailed_scores": {
            "overall": overall_scores,
            "phase_breakdown": phase_scores
        },
        "candidate_profile": {
            "skills": session.candidate_profile.skills,
            "technologies": session.candidate_profile.technologies,
            "experience_years": session.candidate_profile.experience_years,
            "communication_style": session.candidate_profile.communication_style,
            "confidence_level": session.candidate_profile.confidence_level,
            "strengths": session.candidate_profile.strengths,
            "weaknesses": session.candidate_profile.weaknesses,
        },
        "skill_graph": skill_graph,
        "memory_summary": {
            "total_facts_stored": memory_summary.get("total_facts", 0),
            "skills_identified": memory_summary.get("skills", []),
            "technologies_identified": memory_summary.get("technologies", [])
        },
        "qa_transcript": [
            {
                "phase": session.questions_asked[i].phase.value,
                "question": session.questions_asked[i].question,
                "answer": session.answers_received[i].answer if i < len(session.answers_received) else None,
                "scores": {
                    "quality": session.answers_received[i].analysis.quality_score,
                    "relevance": session.answers_received[i].analysis.relevance_score,
                    "completeness": session.answers_received[i].analysis.completeness_score,
                } if i < len(session.answers_received) else None
            }
            for i in range(len(session.questions_asked))
        ],
        "observations": {
            "red_flags": list(set(session.red_flags))[:10],
            "positive_signs": list(set(session.positive_signs))[:10],
            "difficulty_progression": session.difficulty_level
        }
    }
    
    return report


@app.post("/reset-interview")
async def reset_interview():
    """
    Reset interview state and start fresh.
    
    Returns:
        Confirmation message
    """
    clear_session()
    return {"status": "Interview reset successfully"}


@app.get("/debug/conversation")
async def debug_conversation():
    """
    Debug endpoint to see full conversation history.
    
    Returns:
        Full conversation state
    """
    try:
        session = get_current_session()
        return {
            "session_id": session.session_id,
            "phase": session.phase.value,
            "difficulty": session.difficulty_level,
            "current_topic": session.current_topic,
            "covered_topics": session.covered_topics,
            "conversation_history": [
                {
                    "role": turn.role,
                    "content": turn.content[:200] + "..." if len(turn.content) > 200 else turn.content,
                    "phase": turn.phase.value,
                    "timestamp": turn.timestamp.isoformat()
                }
                for turn in session.conversation_history
            ],
            "candidate_profile": session.get_profile_summary()
        }
    except HTTPException:
        return {"error": "No active session"}


@app.get("/debug/memory")
async def debug_memory():
    """
    Debug endpoint to see stored memories.
    
    Returns:
        Memory store statistics and recent facts
    """
    try:
        session = get_current_session()
        summary = memory_store.get_session_summary(session.session_id)
        return {
            "session_id": session.session_id,
            "collection_stats": memory_store.get_collection_stats(),
            "session_summary": summary
        }
    except HTTPException:
        return {
            "error": "No active session",
            "collection_stats": memory_store.get_collection_stats()
        }


# ================================================================
# Main Entry Point
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
