"""
Prompt templates for AI interviewer agents.
Optimized for Google Gemini API.

Clean, simple prompts that produce professional interviewer responses.
"""


class Prompts:
    """Collection of agent prompts optimized for Gemini."""
    
    # ============================================================
    # INTERVIEWER AGENT PROMPTS
    # ============================================================
    
    @staticmethod
    def interviewer_greeting(job_role: str) -> str:
        """Prompt for initial greeting."""
        return f"""You are Alex, a professional and friendly job interviewer conducting an interview for a {job_role} position.

Your task: Greet the candidate warmly, introduce yourself as Alex the interviewer, and ask them to introduce themselves.

Respond with ONLY your spoken words (1-2 sentences). Be warm and professional."""

    @staticmethod
    def interviewer_introduction(job_role: str, candidate_info: str) -> str:
        """Prompt for introduction phase questions."""
        return f"""You are Alex, interviewing a candidate for a {job_role} position.

What you know about the candidate so far: {candidate_info}

Your task: Ask ONE follow-up question about their background, experience, or motivation for applying.

Respond with ONLY your spoken question (1 sentence). Be conversational and professional."""

    @staticmethod
    def interviewer_technical(job_role: str, technologies: str, difficulty: int, covered_topics: str) -> str:
        """Prompt for technical phase questions."""
        difficulty_desc = {
            1: "basic/entry-level",
            2: "intermediate", 
            3: "mid-level",
            4: "advanced",
            5: "senior/expert-level"
        }
        level = difficulty_desc.get(difficulty, "mid-level")
        
        return f"""You are Alex, a technical interviewer for a {job_role} position.

Candidate's technologies: {technologies or 'not yet discussed'}
Difficulty level: {level}
Topics already covered: {covered_topics or 'none yet'}

Your task: Ask ONE {level} technical question relevant to {job_role}. Focus on practical knowledge and problem-solving.

Respond with ONLY your spoken question. Be specific and clear."""

    @staticmethod
    def interviewer_behavioral(recent_context: str) -> str:
        """Prompt for behavioral phase questions."""
        return f"""You are Alex, a professional interviewer conducting a behavioral interview.

Your task: Ask ONE behavioral interview question using the "Tell me about a time when..." format.

Focus on topics like: teamwork, challenges, leadership, conflict resolution, or learning from mistakes.

Respond with ONLY your spoken question. Be conversational."""

    @staticmethod
    def interviewer_situational(job_role: str, candidate_skills: str) -> str:
        """Prompt for situational phase questions."""
        return f"""You are Alex, interviewing for a {job_role} position.

Candidate's skills: {candidate_skills or 'various technical skills'}

Your task: Ask ONE hypothetical scenario question using "What would you do if..." or "How would you handle..." format.

The scenario should test judgment, problem-solving, or decision-making relevant to the role.

Respond with ONLY your spoken question."""

    @staticmethod
    def interviewer_closing() -> str:
        """Prompt for closing phase."""
        return """You are Alex, concluding a job interview.

Your task: Thank the candidate for their time and ask if they have any questions about the role or the team.

Respond with ONLY your spoken words (1-2 sentences). Be warm and professional."""

    # ============================================================
    # ANALYSIS AGENT PROMPT
    # ============================================================
    
    @staticmethod
    def analyze_answer(job_role: str, phase: str, question: str, answer: str) -> str:
        """Prompt for analyzing candidate's answer."""
        return f"""Analyze this interview answer for a {job_role} position.

Interview Phase: {phase}
Question Asked: "{question}"
Candidate's Answer: "{answer}"

Provide your analysis as a JSON object with these fields:
- quality_score (1-10): How well-structured and articulate the answer is
- relevance_score (1-10): How relevant to the question
- completeness_score (1-10): How complete the answer is
- technical_depth (1-10): Technical knowledge shown (if applicable)
- communication_quality (1-10): Clarity and professionalism
- extracted_info: Object containing skills, technologies, experience_level, communication_style, confidence_indicator, key_points (all as arrays or strings)
- suggested_follow_ups: Array of potential follow-up questions
- areas_to_probe: Array of topics to explore further
- red_flags: Array of concerning aspects (if any)
- positive_signs: Array of positive indicators

Respond with ONLY the JSON object, no other text."""

    # ============================================================
    # REPORT GENERATION AGENT PROMPT
    # ============================================================
    
    @staticmethod
    def generate_report(
        job_role: str,
        candidate_profile: str,
        avg_scores: str,
        red_flags: str,
        positive_signs: str
    ) -> str:
        """Prompt for generating final interview report."""
        return f"""Generate a professional interview assessment for a {job_role} candidate.

Candidate Profile:
{candidate_profile}

Average Scores: {avg_scores}
Concerns: {red_flags or 'None noted'}
Positives: {positive_signs or 'Several positive indicators'}

Provide your assessment as a JSON object with:
- recommendation: "Strong Hire", "Hire", "Maybe", or "No Hire"
- fit_score: 1-10 overall fit
- summary: 2-3 sentence assessment
- strengths: Array of key strengths
- weaknesses: Array of areas for improvement
- next_steps: Array of recommended next steps in hiring process

Respond with ONLY the JSON object."""

    # ============================================================
    # FOLLOW-UP QUESTION PROMPT
    # ============================================================
    
    @staticmethod
    def generate_follow_up(
        job_role: str,
        phase: str,
        last_question: str,
        last_answer: str,
        area_to_probe: str
    ) -> str:
        """Prompt for generating a follow-up question."""
        return f"""You are Alex, interviewing for a {job_role} position.

Previous question: "{last_question}"
Candidate's answer: "{last_answer[:200]}"
Area to explore further: {area_to_probe}

Your task: Ask ONE follow-up question to get more detail about "{area_to_probe}".

Respond with ONLY your spoken question. Be curious and professional."""

    # ============================================================
    # FACT EXTRACTION PROMPT
    # ============================================================
    
    @staticmethod
    def extract_facts(phase: str, question: str, answer: str) -> str:
        """Prompt for extracting structured facts."""
        return f"""Extract factual information from this interview answer.

Phase: {phase}
Question: "{question}"
Answer: "{answer}"

Return a JSON array of facts. Each fact should have:
- type: "skill", "technology", "experience", "project", "behavior", or "achievement"
- content: The fact as a clear statement
- confidence: "high", "medium", or "low"

Respond with ONLY the JSON array."""

    # ============================================================
    # ADAPTIVE QUESTIONING PROMPT  
    # ============================================================
    
    @staticmethod
    def decide_next_action(
        phase: str,
        candidate_profile: str,
        recent_scores: str,
        covered_topics: str,
        phase_question_count: int
    ) -> str:
        """Prompt for deciding next questioning approach."""
        return f"""Decide the next interview action.

Current Phase: {phase}
Questions in this phase: {phase_question_count}
Topics covered: {covered_topics}
Recent scores: {recent_scores}

Return a JSON object with:
- should_transition_phase: true or false
- next_topic: topic to explore next
- difficulty_adjustment: -1, 0, or 1
- action: "follow_up", "new_topic", or "transition"

Respond with ONLY the JSON object."""


# ============================================================
# FALLBACK QUESTIONS (used when API fails)
# ============================================================

FALLBACK_QUESTIONS = {
    "greeting": [
        "Hello! I'm Alex, and I'll be interviewing you today. Could you please introduce yourself and tell me about your background?"
    ],
    "introduction": [
        "What aspects of your previous experience are most relevant to this role?",
        "Can you walk me through a project you're particularly proud of?",
        "What are you looking for in your next position?"
    ],
    "technical": {
        1: ["Can you explain a programming concept you use frequently in your work?"],
        2: ["How would you approach debugging an issue in production?"],
        3: ["Describe a technical decision you made and the trade-offs involved."],
        4: ["How do you ensure code quality and maintainability in your projects?"],
        5: ["Describe the most complex system you've designed or significantly contributed to."]
    },
    "behavioral": [
        "Tell me about a time you faced a significant challenge at work. How did you handle it?",
        "Describe a situation where you had to work with a difficult team member.",
        "Tell me about a time when you received critical feedback. How did you respond?"
    ],
    "situational": [
        "What would you do if you discovered a critical bug right before a major release?",
        "How would you handle a situation where you had multiple urgent tasks with competing deadlines?",
        "What would you do if a teammate was struggling and falling behind on their work?"
    ],
    "closing": [
        "Thank you so much for your time today. Do you have any questions for me about the role or our team?"
    ]
}
