"""
Answer scoring and evaluation system.
Provides consistent scoring across all interview phases.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from models.schemas import AnswerAnalysis, InterviewPhase
from utils.config import config


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown for an answer."""
    quality: int
    relevance: int
    completeness: int
    technical_depth: int
    communication: int
    weighted_average: float
    interpretation: str


class AnswerScorer:
    """
    Scores and evaluates candidate answers.
    Uses weighted averages based on interview phase.
    """
    
    # Phase-specific score weights
    PHASE_WEIGHTS = {
        InterviewPhase.GREETING: {
            "quality": 0.1,
            "relevance": 0.2,
            "completeness": 0.2,
            "technical_depth": 0.0,  # Not relevant for greeting
            "communication": 0.5,
        },
        InterviewPhase.INTRODUCTION: {
            "quality": 0.2,
            "relevance": 0.25,
            "completeness": 0.25,
            "technical_depth": 0.05,
            "communication": 0.25,
        },
        InterviewPhase.TECHNICAL: {
            "quality": 0.15,
            "relevance": 0.2,
            "completeness": 0.2,
            "technical_depth": 0.3,
            "communication": 0.15,
        },
        InterviewPhase.BEHAVIORAL: {
            "quality": 0.2,
            "relevance": 0.25,
            "completeness": 0.25,
            "technical_depth": 0.0,
            "communication": 0.3,
        },
        InterviewPhase.SITUATIONAL: {
            "quality": 0.2,
            "relevance": 0.25,
            "completeness": 0.2,
            "technical_depth": 0.15,
            "communication": 0.2,
        },
        InterviewPhase.CLOSING: {
            "quality": 0.2,
            "relevance": 0.2,
            "completeness": 0.2,
            "technical_depth": 0.0,
            "communication": 0.4,
        },
    }
    
    # Score interpretation thresholds
    INTERPRETATIONS = {
        (0, 3): "Poor - Significant improvement needed",
        (3, 5): "Below Average - Some gaps identified",
        (5, 6.5): "Average - Meets basic expectations",
        (6.5, 8): "Good - Above average performance",
        (8, 9): "Very Good - Strong candidate",
        (9, 10.1): "Excellent - Outstanding performance",
    }
    
    @classmethod
    def calculate_weighted_score(
        cls,
        analysis: AnswerAnalysis,
        phase: InterviewPhase
    ) -> float:
        """
        Calculate weighted average score based on phase.
        
        Args:
            analysis: The answer analysis
            phase: Current interview phase
            
        Returns:
            Weighted score from 0-10
        """
        weights = cls.PHASE_WEIGHTS.get(phase, cls.PHASE_WEIGHTS[InterviewPhase.INTRODUCTION])
        
        score = (
            analysis.quality_score * weights["quality"] +
            analysis.relevance_score * weights["relevance"] +
            analysis.completeness_score * weights["completeness"] +
            analysis.technical_depth * weights["technical_depth"] +
            analysis.communication_quality * weights["communication"]
        )
        
        # Normalize to handle any weight sum variations
        weight_sum = sum(weights.values())
        return round(score / weight_sum * 10, 1) if weight_sum > 0 else 5.0
    
    @classmethod
    def get_score_interpretation(cls, score: float) -> str:
        """Get human-readable interpretation of a score."""
        for (low, high), interpretation in cls.INTERPRETATIONS.items():
            if low <= score < high:
                return interpretation
        return "Score out of range"
    
    @classmethod
    def get_score_breakdown(
        cls,
        analysis: AnswerAnalysis,
        phase: InterviewPhase
    ) -> ScoreBreakdown:
        """
        Get complete score breakdown for an answer.
        
        Args:
            analysis: The answer analysis
            phase: Current interview phase
            
        Returns:
            ScoreBreakdown with all details
        """
        weighted = cls.calculate_weighted_score(analysis, phase)
        
        return ScoreBreakdown(
            quality=analysis.quality_score,
            relevance=analysis.relevance_score,
            completeness=analysis.completeness_score,
            technical_depth=analysis.technical_depth,
            communication=analysis.communication_quality,
            weighted_average=weighted,
            interpretation=cls.get_score_interpretation(weighted)
        )
    
    @classmethod
    def aggregate_phase_scores(
        cls,
        analyses: List[AnswerAnalysis],
        phase: InterviewPhase
    ) -> Dict[str, Any]:
        """
        Calculate aggregate scores for a phase.
        
        Args:
            analyses: List of answer analyses from the phase
            phase: The interview phase
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not analyses:
            return {
                "avg_quality": 0,
                "avg_relevance": 0,
                "avg_completeness": 0,
                "avg_technical_depth": 0,
                "avg_communication": 0,
                "weighted_average": 0,
                "answer_count": 0,
            }
        
        n = len(analyses)
        
        return {
            "avg_quality": round(sum(a.quality_score for a in analyses) / n, 1),
            "avg_relevance": round(sum(a.relevance_score for a in analyses) / n, 1),
            "avg_completeness": round(sum(a.completeness_score for a in analyses) / n, 1),
            "avg_technical_depth": round(sum(a.technical_depth for a in analyses) / n, 1),
            "avg_communication": round(sum(a.communication_quality for a in analyses) / n, 1),
            "weighted_average": round(
                sum(cls.calculate_weighted_score(a, phase) for a in analyses) / n, 1
            ),
            "answer_count": n,
        }
    
    @classmethod
    def calculate_overall_scores(
        cls,
        all_analyses: List[AnswerAnalysis]
    ) -> Dict[str, float]:
        """
        Calculate overall scores across the entire interview.
        
        Args:
            all_analyses: All answer analyses from the interview
            
        Returns:
            Dictionary with overall averages
        """
        if not all_analyses:
            return {
                "quality": 0,
                "relevance": 0,
                "completeness": 0,
                "technical_depth": 0,
                "communication": 0,
            }
        
        n = len(all_analyses)
        
        return {
            "quality": round(sum(a.quality_score for a in all_analyses) / n, 1),
            "relevance": round(sum(a.relevance_score for a in all_analyses) / n, 1),
            "completeness": round(sum(a.completeness_score for a in all_analyses) / n, 1),
            "technical_depth": round(sum(a.technical_depth for a in all_analyses) / n, 1),
            "communication": round(sum(a.communication_quality for a in all_analyses) / n, 1),
        }
    
    @classmethod
    def get_recommendation(cls, overall_weighted: float) -> str:
        """
        Get hiring recommendation based on overall weighted score.
        
        Args:
            overall_weighted: The overall weighted score (0-10)
            
        Returns:
            Recommendation string
        """
        if overall_weighted >= 8.5:
            return "Strong Hire"
        elif overall_weighted >= 7.0:
            return "Hire"
        elif overall_weighted >= 5.5:
            return "Maybe"
        else:
            return "No Hire"
    
    @classmethod
    def validate_analysis(cls, analysis_dict: Dict[str, Any]) -> AnswerAnalysis:
        """
        Validate and normalize an analysis dictionary.
        Ensures all scores are within valid ranges.
        
        Args:
            analysis_dict: Raw analysis dictionary
            
        Returns:
            Validated AnswerAnalysis
        """
        def clamp(value, min_val=1, max_val=10):
            try:
                return max(min_val, min(max_val, int(value)))
            except (TypeError, ValueError):
                return 5  # Default to middle score
        
        return AnswerAnalysis(
            quality_score=clamp(analysis_dict.get("quality_score", 5)),
            relevance_score=clamp(analysis_dict.get("relevance_score", 5)),
            completeness_score=clamp(analysis_dict.get("completeness_score", 5)),
            technical_depth=clamp(analysis_dict.get("technical_depth", 3)),
            communication_quality=clamp(analysis_dict.get("communication_quality", 5)),
            extracted_info=analysis_dict.get("extracted_info", {}),
            suggested_follow_ups=analysis_dict.get("suggested_follow_ups", []),
            areas_to_probe=analysis_dict.get("areas_to_probe", []),
            red_flags=analysis_dict.get("red_flags", []),
            positive_signs=analysis_dict.get("positive_signs", []),
        )
