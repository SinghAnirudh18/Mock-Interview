"""
Fact extraction utilities for interview answers.
Extracts skills, technologies, experience indicators, and other relevant facts.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFact:
    """Represents an extracted fact from an interview answer."""
    type: str  # skill, technology, experience, project, behavior
    content: str
    confidence: float  # 0.0 to 1.0
    source_phase: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FactExtractor:
    """
    Extracts structured facts from interview answers.
    """
    
    # Common technology keywords
    TECH_KEYWORDS = {
        'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'go', 'rust',
        'react', 'angular', 'vue', 'nodejs', 'node.js', 'express', 'django', 'flask', 'fastapi',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform',
        'git', 'linux', 'rest', 'graphql', 'api', 'microservices',
        'machine learning', 'ml', 'ai', 'deep learning', 'tensorflow', 'pytorch',
        'html', 'css', 'sass', 'webpack', 'npm', 'yarn'
    }
    
    # Skill indicators
    SKILL_PATTERNS = [
        r'(?:experienced|skilled|proficient|expert)\s+(?:in|with|at)\s+(\w+(?:\s+\w+)?)',
        r'(?:I|i)\s+(?:know|use|work with|specialize in)\s+(\w+(?:\s+\w+)?)',
        r'(?:strong|good)\s+(\w+)\s+skills',
    ]
    
    # Experience level indicators
    EXPERIENCE_INDICATORS = {
        'junior': ['learning', 'new to', 'recently started', 'intern', 'entry level'],
        'mid': ['a few years', 'some experience', 'familiar with', 'comfortable with'],
        'senior': ['many years', 'extensive experience', 'led', 'architected', 'mentored', 'expert']
    }
    
    def __init__(self):
        pass
    
    def extract_facts(
        self,
        phase: str,
        question: str,
        answer: str
    ) -> List[ExtractedFact]:
        """
        Extract facts from an interview answer.
        
        Args:
            phase: Current interview phase
            question: The question asked
            answer: The candidate's answer
            
        Returns:
            List of extracted facts
        """
        facts = []
        answer_lower = answer.lower()
        
        # Extract technologies
        tech_facts = self._extract_technologies(answer_lower, phase)
        facts.extend(tech_facts)
        
        # Extract skills
        skill_facts = self._extract_skills(answer, phase)
        facts.extend(skill_facts)
        
        # Extract experience indicators
        exp_facts = self._extract_experience(answer_lower, phase)
        facts.extend(exp_facts)
        
        # Extract behavioral indicators for behavioral phase
        if phase in ['behavioral', 'situational']:
            behavior_facts = self._extract_behaviors(answer, phase)
            facts.extend(behavior_facts)
        
        logger.info(f"Extracted {len(facts)} facts from answer")
        return facts
    
    def _extract_technologies(self, answer: str, phase: str) -> List[ExtractedFact]:
        """Extract mentioned technologies."""
        facts = []
        for tech in self.TECH_KEYWORDS:
            if tech in answer:
                facts.append(ExtractedFact(
                    type='technology',
                    content=tech,
                    confidence=0.9,
                    source_phase=phase
                ))
        return facts
    
    def _extract_skills(self, answer: str, phase: str) -> List[ExtractedFact]:
        """Extract mentioned skills."""
        facts = []
        for pattern in self.SKILL_PATTERNS:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for match in matches:
                if len(match) > 2:  # Filter out very short matches
                    facts.append(ExtractedFact(
                        type='skill',
                        content=match.strip(),
                        confidence=0.7,
                        source_phase=phase
                    ))
        return facts
    
    def _extract_experience(self, answer: str, phase: str) -> List[ExtractedFact]:
        """Extract experience level indicators."""
        facts = []
        for level, indicators in self.EXPERIENCE_INDICATORS.items():
            for indicator in indicators:
                if indicator in answer:
                    facts.append(ExtractedFact(
                        type='experience',
                        content=f"{level}_level_indicator:{indicator}",
                        confidence=0.6,
                        source_phase=phase
                    ))
                    break  # One indicator per level is enough
        return facts
    
    def _extract_behaviors(self, answer: str, phase: str) -> List[ExtractedFact]:
        """Extract behavioral indicators."""
        facts = []
        
        # Leadership indicators
        leadership_words = ['led', 'managed', 'coordinated', 'organized', 'mentored']
        for word in leadership_words:
            if word in answer.lower():
                facts.append(ExtractedFact(
                    type='behavior',
                    content=f"leadership_indicator:{word}",
                    confidence=0.7,
                    source_phase=phase
                ))
                break
        
        # Problem-solving indicators
        problem_words = ['solved', 'fixed', 'resolved', 'debugged', 'figured out', 'analyzed']
        for word in problem_words:
            if word in answer.lower():
                facts.append(ExtractedFact(
                    type='behavior',
                    content=f"problem_solving_indicator:{word}",
                    confidence=0.7,
                    source_phase=phase
                ))
                break
        
        # Teamwork indicators
        team_words = ['team', 'collaborated', 'together', 'group', 'colleagues']
        for word in team_words:
            if word in answer.lower():
                facts.append(ExtractedFact(
                    type='behavior',
                    content=f"teamwork_indicator:{word}",
                    confidence=0.7,
                    source_phase=phase
                ))
                break
        
        return facts
    
    def extract_from_analysis(self, analysis: Dict[str, Any]) -> List[ExtractedFact]:
        """
        Extract facts from an LLM analysis result.
        
        Args:
            analysis: Analysis dictionary from LLM
            
        Returns:
            List of extracted facts
        """
        facts = []
        extracted_info = analysis.get('extracted_info', {})
        
        # Extract skills from analysis
        for skill in extracted_info.get('skills', []):
            if skill and isinstance(skill, str):
                facts.append(ExtractedFact(
                    type='skill',
                    content=skill,
                    confidence=0.85,
                    source_phase='analysis'
                ))
        
        # Extract technologies from analysis
        for tech in extracted_info.get('technologies', []):
            if tech and isinstance(tech, str):
                facts.append(ExtractedFact(
                    type='technology',
                    content=tech,
                    confidence=0.85,
                    source_phase='analysis'
                ))
        
        # Extract key points
        for point in extracted_info.get('key_points', []):
            if point and isinstance(point, str):
                facts.append(ExtractedFact(
                    type='key_point',
                    content=point,
                    confidence=0.75,
                    source_phase='analysis'
                ))
        
        return facts


# Global instance
fact_extractor = FactExtractor()
