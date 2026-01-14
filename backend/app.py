from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile, requests, json
import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio
from enum import Enum

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# Load Whisper (GPU STT)
# ------------------------------------------------------
whisper_model = WhisperModel(
    "../models/medium",
    device="cuda",
    compute_type="float16"
)

# ------------------------------------------------------
# LLM Server URL (DeepSeek/LLaMA CPP)
# ------------------------------------------------------
LLM_URL = "http://localhost:9000/completion"

# ------------------------------------------------------
# Enhanced Interview State Management
# ------------------------------------------------------
class InterviewPhase(Enum):
    GREETING = "greeting"
    INTRODUCTION = "introduction"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SITUATIONAL = "situational"
    CLOSING = "closing"
    ENDED = "ended"

class CandidateProfile(BaseModel):
    skills: List[str] = []
    experience_years: Optional[int] = None
    projects: List[Dict] = []
    technologies: List[str] = []
    strengths: List[str] = []
    weaknesses: List[str] = []
    communication_style: str = ""
    confidence_level: int = 3  # 1-5 scale
    problem_solving_ability: Optional[int] = None
    depth_of_knowledge: Dict[str, int] = {}  # technology -> score 1-10
    
class AnswerAnalysis(BaseModel):
    quality_score: int  # 1-10
    relevance_score: int  # 1-10
    completeness_score: int  # 1-10
    technical_depth: int  # 1-10
    communication_quality: int  # 1-10
    extracted_info: Dict
    suggested_follow_ups: List[str]
    areas_to_probe: List[str]
    red_flags: List[str] = []
    positive_signs: List[str] = []

class InterviewState:
    def __init__(self):
        self.phase = InterviewPhase.GREETING
        self.job_role = "Software Engineer"
        self.questions_asked: List[Dict] = []  # [{question, phase, timestamp}]
        self.answers_received: List[Dict] = []  # [{answer, analysis, timestamp}]
        self.candidate_profile = CandidateProfile()
        self.conversation_history: List[Dict] = []
        self.current_topic: Optional[str] = None
        self.difficulty_level: int = 3  # 1-5 scale, adjusts based on performance
        self.start_time = None
        self.end_time = None
        self.phase_start_time = None
        self.interview_focus_areas: List[str] = []
        self.red_flags: List[str] = []
        self.positive_signs: List[str] = []
        
    def get_conversation_context(self, num_exchanges: int = 3) -> str:
        """Get recent conversation context for AI"""
        recent = []
        for i in range(max(0, len(self.conversation_history) - num_exchanges), len(self.conversation_history)):
            if i >= 0:
                recent.append(self.conversation_history[i])
        return json.dumps(recent, indent=2)

# Global interview state
interview_state = InterviewState()

# ------------------------------------------------------
# Interview Configuration
# ------------------------------------------------------
PHASE_CONFIG = {
    InterviewPhase.GREETING: {
        "min_questions": 1,
        "max_questions": 2,
        "time_limit": timedelta(minutes=2)
    },
    InterviewPhase.INTRODUCTION: {
        "min_questions": 3,
        "max_questions": 5,
        "time_limit": timedelta(minutes=5)
    },
    InterviewPhase.TECHNICAL: {
        "min_questions": 5,
        "max_questions": 10,
        "time_limit": timedelta(minutes=20)
    },
    InterviewPhase.BEHAVIORAL: {
        "min_questions": 3,
        "max_questions": 6,
        "time_limit": timedelta(minutes=10)
    },
    InterviewPhase.SITUATIONAL: {
        "min_questions": 2,
        "max_questions": 4,
        "time_limit": timedelta(minutes=8)
    },
    InterviewPhase.CLOSING: {
        "min_questions": 1,
        "max_questions": 3,
        "time_limit": timedelta(minutes=3)
    }
}

# ------------------------------------------------------
# Enhanced Helper Functions
# ------------------------------------------------------
def clean_llm_response(text: str) -> str:
    """Clean LLM response for interview conversation"""
    # Remove thinking tags
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<thought>.*?</thought>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove meta commentary and stage directions
    meta_patterns = [
        r'^.*?(?=(Hello|Hi|Good|Thank|So|Now|Next|Alright|Well|Can you|Could you|What|How|Why|Tell me|Describe|Would you))',
        r'^(Wait,|Hmm,|Let me see,|Let me think,|I need to|Looking at|The candidate)',
        r'\(.*?\)',  # Remove parenthetical comments
        r'\[.*?\]',  # Remove bracketed comments
    ]
    
    for pattern in meta_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Detect if AI is trying to answer a question instead of asking one
    answer_indicators = [
        "you should", "i recommend", "it's important to", "the best time",
        "generally speaking", "in my experience", "typically",
        "it depends on", "you'll want to", "you need to"
    ]
    
    cleaned_lower = cleaned.lower()
    if any(indicator in cleaned_lower for indicator in answer_indicators):
        # AI is trying to give advice instead of asking a question
        # Return empty to trigger fallback
        return ""
    
    # Remove excess whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n+', '\n', cleaned)
    
    # Extract only the interviewer's speech (not actions or thoughts)
    lines = cleaned.split('\n')
    speech_lines = []
    for line in lines:
        # Keep lines that sound like speech, not stage directions
        if line and not line.startswith(('(', '[', '*')) and len(line) > 10:
            speech_lines.append(line.strip())
    
    result = ' '.join(speech_lines[:2])  # Limit to 1-2 sentences
    
    # If result is too short or seems incomplete, return empty to trigger fallback
    if len(result.split()) < 3:
        return ""
    
    return result.strip()

def analyze_candidate_answer(question: str, answer: str, phase: InterviewPhase) -> AnswerAnalysis:
    """Use AI to deeply analyze the candidate's answer"""
    
    prompt = f"""
You are Alex, an expert interviewer analyzing a candidate's response.

JOB ROLE: {interview_state.job_role}
CURRENT PHASE: {phase.value}
CURRENT TOPIC: {interview_state.current_topic or 'General'}

QUESTION ASKED: "{question}"

CANDIDATE'S ANSWER: "{answer}"

Analyze this answer thoroughly and provide:

1. Quality (1-10): How well-structured and articulate is the answer?
2. Relevance (1-10): How relevant is it to the question asked?
3. Completeness (1-10): Does it fully address the question?
4. Technical Depth (1-10): If technical, how deep is the knowledge shown?
5. Communication (1-10): Clarity, conciseness, confidence.

Extract key information about:
- Skills/technologies mentioned
- Experience level indicators
- Problem-solving approach
- Communication style
- Any red flags or positive signs

Suggest follow-up questions to probe deeper into:
- Areas where answer was weak/vague
- Interesting points that need elaboration
- Contradictions or inconsistencies

Respond in this EXACT JSON format:
{{
    "quality_score": <int>,
    "relevance_score": <int>,
    "completeness_score": <int>,
    "technical_depth": <int>,
    "communication_quality": <int>,
    "extracted_info": {{
        "skills": [<list of skills>],
        "technologies": [<list of technologies>],
        "experience_level": "<junior/mid/senior>",
        "communication_style": "<formal/casual/technical/etc>",
        "confidence_indicator": "<high/medium/low>",
        "key_points": [<list of key points mentioned>]
    }},
    "suggested_follow_ups": [<list of follow-up questions>],
    "areas_to_probe": [<list of topics to explore further>],
    "red_flags": [<list of any concerning aspects>],
    "positive_signs": [<list of positive aspects>]
}}
"""

    try:
        payload = {
            "prompt": prompt,
            "n_predict": 400,
            "temperature": 0.3,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stop": ["\n\n", "}", "Candidate:"],
            "grammar": 'root ::= "{" [^}]+ "}"',
        }
        
        response = requests.post(LLM_URL, json=payload, timeout=15).json()
        content = response.get("content", "{}")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            analysis_dict = json.loads(json_match.group())
        else:
            analysis_dict = {}
            
    except Exception as e:
        print(f"Analysis error: {e}")
        analysis_dict = {}
    
    # Default values if analysis fails
    default_analysis = AnswerAnalysis(
        quality_score=5,
        relevance_score=5,
        completeness_score=5,
        technical_depth=3 if phase == InterviewPhase.TECHNICAL else 1,
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
        areas_to_probe=["General follow-up"]
    )
    
    # Merge with actual analysis
    if analysis_dict:
        try:
            return AnswerAnalysis(**analysis_dict)
        except:
            return default_analysis
    return default_analysis

def update_candidate_profile(analysis: AnswerAnalysis):
    """Update candidate profile based on analysis"""
    
    # Update skills and technologies
    if "skills" in analysis.extracted_info:
        for skill in analysis.extracted_info["skills"]:
            if skill not in interview_state.candidate_profile.skills:
                interview_state.candidate_profile.skills.append(skill)
    
    if "technologies" in analysis.extracted_info:
        for tech in analysis.extracted_info["technologies"]:
            if tech not in interview_state.candidate_profile.technologies:
                interview_state.candidate_profile.technologies.append(tech)
    
    # Update experience estimate
    if "experience_level" in analysis.extracted_info:
        level = analysis.extracted_info["experience_level"]
        if level == "senior" and (interview_state.candidate_profile.experience_years is None or interview_state.candidate_profile.experience_years < 5):
            interview_state.candidate_profile.experience_years = 8
        elif level == "mid" and (interview_state.candidate_profile.experience_years is None or interview_state.candidate_profile.experience_years < 3):
            interview_state.candidate_profile.experience_years = 4
    
    # Update communication style
    if analysis.communication_quality >= 7:
        interview_state.candidate_profile.communication_style = "clear and articulate"
    elif analysis.communication_quality <= 4:
        interview_state.candidate_profile.communication_style = "needs improvement"
    
    # Update confidence
    if "confidence_indicator" in analysis.extracted_info:
        conf = analysis.extracted_info["confidence_indicator"]
        if conf == "high":
            interview_state.candidate_profile.confidence_level = 5
        elif conf == "low":
            interview_state.candidate_profile.confidence_level = 2
    
    # Add red flags and positive signs
    interview_state.red_flags.extend(analysis.red_flags)
    interview_state.positive_signs.extend(analysis.positive_signs)
    
    # Adjust difficulty based on performance
    avg_score = (analysis.quality_score + analysis.relevance_score + analysis.completeness_score) / 3
    if avg_score >= 8:
        interview_state.difficulty_level = min(5, interview_state.difficulty_level + 1)
    elif avg_score <= 5:
        interview_state.difficulty_level = max(1, interview_state.difficulty_level - 1)

def generate_adaptive_question(last_answer_analysis: Optional[AnswerAnalysis] = None) -> str:
    """Generate next question based on conversation context and analysis"""
    
    # Build comprehensive prompt for adaptive questioning
    prompt = f"""You are Alex, a professional job interviewer. Your ONLY job is to ask interview questions.

CRITICAL RULES:
1. You are the INTERVIEWER, not the candidate
2. You ASK questions, you do NOT answer questions
3. If the candidate asks you a question, politely redirect and ask your own question
4. Never provide career advice, job search tips, or answer "when should I..." questions
5. Stay focused on evaluating the candidate for the {interview_state.job_role} position

CURRENT INTERVIEW PHASE: {interview_state.phase.value}

CANDIDATE PROFILE:
- Skills: {', '.join(interview_state.candidate_profile.skills[:3]) if interview_state.candidate_profile.skills else 'Unknown'}
- Technologies: {', '.join(interview_state.candidate_profile.technologies[:3]) if interview_state.candidate_profile.technologies else 'Unknown'}

YOUR TASK FOR THIS PHASE:
"""

    # Phase-specific instructions
    if interview_state.phase == InterviewPhase.GREETING:
        prompt += f"""
- Greet the candidate warmly
- Introduce yourself as Alex, the interviewer
- Ask them to introduce themselves and their background
- Keep it brief (1-2 sentences)

Example: "Hello! I'm Alex, and I'll be interviewing you today for the {interview_state.job_role} position. Could you please tell me about yourself?"

Your question:"""

    elif interview_state.phase == InterviewPhase.INTRODUCTION:
        prompt += f"""
- Ask about their background, experience, or motivation
- Focus on understanding their career journey
- Ask ONE clear question

Examples:
- "What drew you to apply for this {interview_state.job_role} role?"
- "Can you walk me through your relevant work experience?"
- "What are you looking for in your next position?"

Your question:"""

    elif interview_state.phase == InterviewPhase.TECHNICAL:
        tech_context = ""
        if interview_state.candidate_profile.technologies:
            tech_context = f"\nThey mentioned: {', '.join(interview_state.candidate_profile.technologies[:2])}"
        
        prompt += f"""
- Ask a technical question relevant to {interview_state.job_role}{tech_context}
- Test their knowledge and problem-solving ability
- Ask ONE specific technical question

Examples:
- "Can you explain how you would approach debugging a performance issue?"
- "Describe your experience with database optimization."
- "How do you ensure code quality in your projects?"

Your question:"""

    elif interview_state.phase == InterviewPhase.BEHAVIORAL:
        prompt += """
- Ask about past experiences, teamwork, or challenges
- Use behavioral interview techniques (STAR method)
- Ask ONE behavioral question

Examples:
- "Tell me about a time you faced a difficult deadline. How did you handle it?"
- "Describe a situation where you had to resolve a conflict with a teammate."
- "What's your approach to receiving critical feedback?"

Your question:"""

    elif interview_state.phase == InterviewPhase.SITUATIONAL:
        prompt += f"""
- Present a hypothetical scenario related to {interview_state.job_role}
- Ask how they would handle it
- Test their judgment and decision-making

Examples:
- "If you discovered a critical bug right before a release, what would you do?"
- "How would you prioritize multiple urgent tasks?"
- "A stakeholder requests a feature that conflicts with best practices. How do you respond?"

Your question:"""

    elif interview_state.phase == InterviewPhase.CLOSING:
        prompt += """
- Thank them for their time
- Ask if they have questions about the role or company
- Keep it professional and brief

Example: "Thank you for your time today. Do you have any questions for me about the role or the team?"

Your closing statement:"""

    else:
        return "Thank you for participating in this interview. We'll be in touch soon."

    # Add conversation context if available
    if interview_state.conversation_history:
        recent = interview_state.conversation_history[-2:]
        context_str = "\n".join([
            f"{'You' if h['role'] == 'interviewer' else 'Candidate'}: {h['content'][:100]}"
            for h in recent
        ])
        prompt += f"\n\nRECENT EXCHANGE:\n{context_str}\n\n"

    prompt += "\nREMEMBER: You are the interviewer. Ask a question, don't answer questions.\n\nYour next interview question:"

    try:
        payload = {
            "prompt": prompt,
            "n_predict": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "stop": ["\n\n", "Candidate:", "They said:", "Answer:", "Q:", "A:"],
        }
        
        response = requests.post(LLM_URL, json=payload, timeout=10).json()
        question = response.get("content", "").strip()
        
        # Clean the response
        question = clean_llm_response(question)
        
        # Validate it's actually a question
        if not question:
            question = get_fallback_question()
        elif "when should" in question.lower() or "how to" in question.lower() and "?" not in question[-50:]:
            # AI is trying to answer instead of ask
            question = get_fallback_question()
        
        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'
            
    except Exception as e:
        print(f"Question generation error: {e}")
        question = get_fallback_question()
    
    return question

def get_fallback_question() -> str:
    """Intelligent fallback questions based on phase and context"""
    
    phase = interview_state.phase
    role = interview_state.job_role
    
    fallbacks = {
        InterviewPhase.GREETING: [
            f"Hello! I'm Alex. Could you tell me about your background and what draws you to this {role} position?"
        ],
        InterviewPhase.INTRODUCTION: [
            "What aspects of your previous experience are most relevant to this role?",
            "Can you walk me through a project that showcases your skills?",
            "What are your career goals and how does this position fit into them?"
        ],
        InterviewPhase.TECHNICAL: {
            1: ["Can you explain a basic programming concept like inheritance or polymorphism?"],
            2: ["How would you approach debugging a performance issue in production?"],
            3: ["Can you describe a time you had to make a significant architectural decision?"],
            4: ["How do you ensure code quality and maintainability in large codebases?"],
            5: ["Describe the most complex technical challenge you've solved and your approach."]
        },
        InterviewPhase.BEHAVIORAL: [
            "Tell me about a time you faced a significant obstacle at work and how you overcame it.",
            "Describe a situation where you had to collaborate with a difficult team member.",
            "How do you handle receiving critical feedback on your work?"
        ],
        InterviewPhase.SITUATIONAL: [
            f"If you joined as a {role} and found a critical bug right before launch, what would you do?",
            "How would you prioritize multiple high-priority tasks with competing deadlines?",
            "Describe how you'd mentor a junior developer struggling with a task."
        ],
        InterviewPhase.CLOSING: [
            "We're almost out of time. Is there anything important about your experience we haven't covered?",
            "Do you have any questions for me about the role or the team?",
            "Thank you for your time today. What are your next steps or timeline?"
        ]
    }
    
    if phase == InterviewPhase.TECHNICAL:
        level_fallbacks = fallbacks[phase].get(interview_state.difficulty_level, fallbacks[phase][3])
        import random
        return random.choice(level_fallbacks)
    else:
        import random
        return random.choice(fallbacks.get(phase, ["Could you tell me more about that?"]))

def should_transition_phase() -> bool:
    """Determine if we should transition to next phase"""
    
    if interview_state.phase == InterviewPhase.ENDED:
        return False
    
    phase_config = PHASE_CONFIG.get(interview_state.phase, {})
    min_q = phase_config.get("min_questions", 2)
    max_q = phase_config.get("max_questions", 5)
    
    # Count questions in current phase
    phase_questions = sum(1 for q in interview_state.questions_asked 
                         if q.get("phase") == interview_state.phase)
    
    # Check if we've asked enough questions
    if phase_questions >= min_q:
        # Check time limit
        if interview_state.phase_start_time:
            time_elapsed = datetime.now() - interview_state.phase_start_time
            time_limit = phase_config.get("time_limit", timedelta(minutes=10))
            if time_elapsed >= time_limit:
                return True
        
        # Check if we've asked max questions
        if phase_questions >= max_q:
            return True
        
        # For technical phase, check if we've covered enough depth
        if interview_state.phase == InterviewPhase.TECHNICAL:
            if len(interview_state.candidate_profile.technologies) >= 3 and phase_questions >= 4:
                return True
    
    return False

def transition_to_next_phase():
    """Move to next interview phase"""
    
    phase_order = [
        InterviewPhase.GREETING,
        InterviewPhase.INTRODUCTION,
        InterviewPhase.TECHNICAL,
        InterviewPhase.BEHAVIORAL,
        InterviewPhase.SITUATIONAL,
        InterviewPhase.CLOSING
    ]
    
    current_idx = phase_order.index(interview_state.phase)
    
    if current_idx < len(phase_order) - 1:
        interview_state.phase = phase_order[current_idx + 1]
        interview_state.phase_start_time = datetime.now()
        interview_state.current_topic = None
        return True
    else:
        interview_state.phase = InterviewPhase.ENDED
        interview_state.end_time = datetime.now()
        return False

# ------------------------------------------------------
# API Endpoints
# ------------------------------------------------------
class StartInterviewRequest(BaseModel):
    job_role: str = "Software Engineer"
    focus_areas: Optional[List[str]] = None

@app.post("/start-interview")
async def start_interview(request: StartInterviewRequest):
    """Start a new interview session"""
    global interview_state
    interview_state = InterviewState()
    interview_state.job_role = request.job_role
    interview_state.interview_focus_areas = request.focus_areas or []
    interview_state.start_time = datetime.now()
    interview_state.phase_start_time = interview_state.start_time
    interview_state.phase = InterviewPhase.GREETING
    
    # Generate initial greeting
    greeting = generate_adaptive_question()
    
    interview_state.questions_asked.append({
        "question": greeting,
        "phase": interview_state.phase.value,
        "timestamp": datetime.now().isoformat()
    })
    
    interview_state.conversation_history.append({
        "role": "interviewer",
        "content": greeting,
        "phase": interview_state.phase.value,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "status": "Interview started",
        "job_role": interview_state.job_role,
        "phase": interview_state.phase.value,
        "interviewer_message": greeting,
        "session_id": id(interview_state)
    }

@app.post("/interview-response")
async def interview_response(file: UploadFile = File(...)):
    """Process candidate's voice response during interview"""
    
    if interview_state.phase == InterviewPhase.ENDED:
        return {
            "error": "Interview has ended. Please start a new interview or request the report.",
            "interviewer_message": "The interview has concluded. Thank you for your time."
        }
    
    # Validate file
    if not file.content_type or "audio" not in file.content_type and "video" not in file.content_type:
        raise HTTPException(status_code=400, detail="File must be audio or video")
    
    # Transcribe audio
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name
        
        segments, _ = whisper_model.transcribe(audio_path)
        user_text = " ".join([s.text for s in segments]).strip()
        
        if not user_text or len(user_text.split()) < 2:
            user_text = "I didn't catch that. Could you please repeat?"
    
    except Exception as e:
        print(f"Transcription error: {e}")
        user_text = "There was an issue processing your audio. Could you please try again?"
    
    # Get last question
    last_question = interview_state.questions_asked[-1]["question"] if interview_state.questions_asked else ""
    
    # Analyze the answer deeply
    analysis = analyze_candidate_answer(last_question, user_text, interview_state.phase)
    
    # Store answer with analysis
    answer_record = {
        "answer": user_text,
        "analysis": analysis.dict(),
        "phase": interview_state.phase.value,
        "timestamp": datetime.now().isoformat()
    }
    interview_state.answers_received.append(answer_record)
    
    # Update candidate profile based on analysis
    update_candidate_profile(analysis)
    
    # Add to conversation history
    interview_state.conversation_history.append({
        "role": "candidate",
        "content": user_text,
        "analysis_scores": {
            "quality": analysis.quality_score,
            "relevance": analysis.relevance_score,
            "completeness": analysis.completeness_score
        },
        "timestamp": datetime.now().isoformat()
    })
    
    # Check if we should transition phases
    if should_transition_phase():
        if not transition_to_next_phase():
            # Interview ended
            return {
                "transcript": user_text,
                "interviewer_message": "Thank you for your time. The interview is now complete.",
                "phase": interview_state.phase.value,
                "analysis_scores": {
                    "quality": analysis.quality_score,
                    "relevance": analysis.relevance_score,
                    "completeness": analysis.completeness_score
                },
                "interview_ended": True
            }
    
    # Generate adaptive next question
    next_question = generate_adaptive_question(analysis)
    
    # Store question
    interview_state.questions_asked.append({
        "question": next_question,
        "phase": interview_state.phase.value,
        "timestamp": datetime.now().isoformat()
    })
    
    # Add to conversation history
    interview_state.conversation_history.append({
        "role": "interviewer",
        "content": next_question,
        "phase": interview_state.phase.value,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "transcript": user_text,
        "interviewer_message": next_question,
        "phase": interview_state.phase.value,
        "analysis_scores": {
            "quality": analysis.quality_score,
            "relevance": analysis.relevance_score,
            "completeness": analysis.completeness_score,
            "technical_depth": analysis.technical_depth,
            "communication_quality": analysis.communication_quality
        },
        "candidate_profile_update": {
            "skills": interview_state.candidate_profile.skills[-3:],
            "technologies": interview_state.candidate_profile.technologies[-3:],
            "difficulty_level": interview_state.difficulty_level
        },
        "questions_asked": len(interview_state.questions_asked),
        "interview_ended": interview_state.phase == InterviewPhase.ENDED
    }

@app.post("/text-response")
async def text_interview_response(user_text: str):
    """Alternative endpoint for text-only responses (for testing)"""
    
    if interview_state.phase == InterviewPhase.ENDED:
        return {
            "error": "Interview has ended.",
            "interviewer_message": "The interview is complete. Thank you."
        }
    
    last_question = interview_state.questions_asked[-1]["question"] if interview_state.questions_asked else ""
    
    # Analyze the answer
    analysis = analyze_candidate_answer(last_question, user_text, interview_state.phase)
    
    # Store answer with analysis
    interview_state.answers_received.append({
        "answer": user_text,
        "analysis": analysis.dict(),
        "phase": interview_state.phase.value,
        "timestamp": datetime.now().isoformat()
    })
    
    # Update candidate profile
    update_candidate_profile(analysis)
    
    # Add to conversation history
    interview_state.conversation_history.append({
        "role": "candidate",
        "content": user_text,
        "analysis_scores": {
            "quality": analysis.quality_score,
            "relevance": analysis.relevance_score,
            "completeness": analysis.completeness_score
        },
        "timestamp": datetime.now().isoformat()
    })
    
    # Check phase transition
    if should_transition_phase():
        if not transition_to_next_phase():
            return {
                "interviewer_message": "Thank you. The interview is now complete.",
                "phase": "ended",
                "interview_ended": True
            }
    
    # Generate next question
    next_question = generate_adaptive_question(analysis)
    
    # Store question
    interview_state.questions_asked.append({
        "question": next_question,
        "phase": interview_state.phase.value,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "interviewer_message": next_question,
        "phase": interview_state.phase.value,
        "analysis_scores": {
            "quality": analysis.quality_score,
            "relevance": analysis.relevance_score,
            "completeness": analysis.completeness_score
        }
    }

@app.get("/interview-status")
async def get_interview_status():
    """Get current interview status and candidate profile"""
    
    phase_questions = sum(1 for q in interview_state.questions_asked 
                         if q.get("phase") == interview_state.phase.value)
    
    return {
        "phase": interview_state.phase.value,
        "job_role": interview_state.job_role,
        "questions_asked_total": len(interview_state.questions_asked),
        "phase_question_count": phase_questions,
        "phase_start_time": interview_state.phase_start_time.isoformat() if interview_state.phase_start_time else None,
        "difficulty_level": interview_state.difficulty_level,
        "candidate_profile": {
            "skills": interview_state.candidate_profile.skills,
            "technologies": interview_state.candidate_profile.technologies,
            "experience_years": interview_state.candidate_profile.experience_years,
            "confidence_level": interview_state.candidate_profile.confidence_level,
            "communication_style": interview_state.candidate_profile.communication_style
        },
        "conversation_summary": {
            "total_exchanges": len(interview_state.conversation_history),
            "last_topic": interview_state.current_topic
        }
    }

@app.post("/end-interview")
async def end_interview():
    """End the interview early"""
    interview_state.end_time = datetime.now()
    interview_state.phase = InterviewPhase.ENDED
    
    return {
        "status": "Interview ended",
        "message": "Interview terminated. You can request the report.",
        "duration": str(interview_state.end_time - interview_state.start_time) if interview_state.start_time else None,
        "total_questions": len(interview_state.questions_asked)
    }

@app.get("/interview-report")
async def get_interview_report():
    """Generate comprehensive interview report with AI analysis"""
    
    if not interview_state.answers_received:
        raise HTTPException(status_code=400, detail="No interview data available")
    
    # Calculate overall scores
    total_answers = len(interview_state.answers_received)
    avg_quality = sum(a["analysis"]["quality_score"] for a in interview_state.answers_received) / total_answers
    avg_relevance = sum(a["analysis"]["relevance_score"] for a in interview_state.answers_received) / total_answers
    avg_completeness = sum(a["analysis"]["completeness_score"] for a in interview_state.answers_received) / total_answers
    
    # Phase-wise analysis
    phase_scores = {}
    for answer in interview_state.answers_received:
        phase = answer["phase"]
        if phase not in phase_scores:
            phase_scores[phase] = []
        phase_scores[phase].append({
            "quality": answer["analysis"]["quality_score"],
            "relevance": answer["analysis"]["relevance_score"],
            "completeness": answer["analysis"]["completeness_score"]
        })
    
    # Generate AI-powered final assessment
    assessment_prompt = f"""
Based on this interview for {interview_state.job_role}, provide a final assessment:

Candidate Profile:
- Skills: {', '.join(interview_state.candidate_profile.skills)}
- Technologies: {', '.join(interview_state.candidate_profile.technologies)}
- Experience Level: {interview_state.candidate_profile.experience_years or 'Unknown'} years
- Communication Style: {interview_state.candidate_profile.communication_style}
- Average Scores: Quality={avg_quality:.1f}/10, Relevance={avg_relevance:.1f}/10, Completeness={avg_completeness:.1f}/10

Key Observations:
- Red Flags: {', '.join(interview_state.red_flags[:3]) if interview_state.red_flags else 'None noted'}
- Positive Signs: {', '.join(interview_state.positive_signs[:3]) if interview_state.positive_signs else 'Several positive indicators'}

Provide a concise assessment including:
1. Overall recommendation (Strong Hire/Hire/Maybe/No Hire)
2. Key strengths
3. Areas for improvement
4. Suggested next steps
5. Fit score (1-10)

Format as JSON:
{{
    "recommendation": "<string>",
    "strengths": ["<list>"],
    "improvement_areas": ["<list>"],
    "next_steps": ["<list>"],
    "fit_score": <int>,
    "summary": "<brief summary>"
}}
"""
    
    try:
        payload = {
            "prompt": assessment_prompt,
            "n_predict": 300,
            "temperature": 0.3,
            "stop": ["\n\n"],
        }
        
        response = requests.post(LLM_URL, json=payload, timeout=15).json()
        content = response.get("content", "{}")
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group())
        else:
            assessment = {
                "recommendation": "Hire",
                "strengths": ["Good communication", "Relevant skills"],
                "improvement_areas": ["Could benefit from more experience"],
                "next_steps": ["Technical assessment", "Team interview"],
                "fit_score": 7,
                "summary": "Competent candidate with room for growth"
            }
    except:
        assessment = {
            "recommendation": "Requires further evaluation",
            "fit_score": 5,
            "summary": "Interview completed, manual review recommended"
        }
    
    # Build comprehensive report
    duration = None
    if interview_state.start_time and interview_state.end_time:
        duration = str(interview_state.end_time - interview_state.start_time)
    elif interview_state.start_time:
        duration = str(datetime.now() - interview_state.start_time)
    
    report = {
        "interview_metadata": {
            "job_role": interview_state.job_role,
            "start_time": interview_state.start_time.isoformat() if interview_state.start_time else None,
            "end_time": interview_state.end_time.isoformat() if interview_state.end_time else datetime.now().isoformat(),
            "duration": duration,
            "total_questions": len(interview_state.questions_asked),
            "phases_covered": list(set(a["phase"] for a in interview_state.answers_received))
        },
        "candidate_assessment": assessment,
        "detailed_scores": {
            "overall": {
                "quality": round(avg_quality, 1),
                "relevance": round(avg_relevance, 1),
                "completeness": round(avg_completeness, 1)
            },
            "phase_breakdown": {
                phase: {
                    "avg_quality": round(sum(s["quality"] for s in scores) / len(scores), 1),
                    "avg_relevance": round(sum(s["relevance"] for s in scores) / len(scores), 1),
                    "avg_completeness": round(sum(s["completeness"] for s in scores) / len(scores), 1),
                    "question_count": len(scores)
                }
                for phase, scores in phase_scores.items()
            }
        },
        "candidate_profile": interview_state.candidate_profile.dict(),
        "qa_transcript": [
            {
                "phase": interview_state.questions_asked[i]["phase"],
                "question": interview_state.questions_asked[i]["question"],
                "answer": interview_state.answers_received[i]["answer"] if i < len(interview_state.answers_received) else None,
                "analysis": interview_state.answers_received[i]["analysis"] if i < len(interview_state.answers_received) else None
            }
            for i in range(len(interview_state.questions_asked))
        ],
        "red_flags": interview_state.red_flags,
        "positive_signs": interview_state.positive_signs,
        "difficulty_progression": interview_state.difficulty_level
    }
    
    return report

@app.post("/reset-interview")
async def reset_interview():
    """Reset interview state"""
    global interview_state
    interview_state = InterviewState()
    return {"status": "Interview reset successfully"}

@app.get("/debug-conversation")
async def debug_conversation():
    """Debug endpoint to see conversation history"""
    return {
        "conversation_history": interview_state.conversation_history,
        "state": {
            "phase": interview_state.phase.value,
            "difficulty": interview_state.difficulty_level,
            "current_topic": interview_state.current_topic
        }
    }

# ------------------------------------------------------
# Main execution
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)