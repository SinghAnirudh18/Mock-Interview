"""
RAG (Retrieval-Augmented Generation) pipeline for the AI interviewer.
Stores and retrieves relevant context to improve question generation.
"""
import logging
from typing import List, Dict, Any, Optional

from .vector_db import memory_store
from .extractors import fact_extractor

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline for interview context management.
    Stores facts and retrieves relevant context for question generation.
    """
    
    def __init__(self):
        self.store = memory_store
        self.extractor = fact_extractor
    
    def store_answer_facts(
        self,
        session_id: str,
        facts: List[Dict[str, Any]],
        phase: str
    ) -> List[str]:
        """
        Store extracted facts from an answer in the vector database.
        
        Args:
            session_id: The interview session ID
            facts: List of fact dictionaries
            phase: Current interview phase (can be enum or string)
            
        Returns:
            List of stored fact IDs
        """
        # Handle phase enum
        phase_str = phase.value if hasattr(phase, 'value') else str(phase)
        
        if not facts:
            logger.debug("No facts to store")
            return []
        
        return self.store.store_facts(
            session_id=session_id,
            facts=facts,
            phase=phase_str
        )
    
    def get_relevant_context_for_question(
        self,
        session_id: str,
        current_phase: str,
        current_topic: Optional[str] = None,
        n_results: int = 3
    ) -> str:
        """
        Retrieve relevant context for generating the next question.
        
        Args:
            session_id: The interview session ID
            current_phase: Current interview phase (can be enum or string)
            current_topic: Optional current topic being explored
            n_results: Number of relevant facts to retrieve
            
        Returns:
            Formatted context string for the LLM
        """
        # Handle phase enum
        phase_str = current_phase.value if hasattr(current_phase, 'value') else str(current_phase)
        
        # Build query based on current phase and topic
        if current_topic:
            query = f"{phase_str} interview topic: {current_topic}"
        else:
            query = f"{phase_str} interview context"
        
        # Retrieve relevant facts
        relevant_facts = self.store.retrieve_relevant(
            session_id=session_id,
            query=query,
            n_results=n_results
        )
        
        if not relevant_facts:
            return ""
        
        # Format into context string
        context_parts = ["Relevant information from the interview so far:"]
        for fact in relevant_facts:
            content = fact.get('content', '')
            if content:
                context_parts.append(f"- {content}")
        
        return "\n".join(context_parts)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of all facts for a session.
        
        Args:
            session_id: The interview session ID
            
        Returns:
            Summary dictionary with categorized facts
        """
        # Retrieve all facts for the session
        all_facts = self.store.retrieve_relevant(
            session_id=session_id,
            query="interview summary",
            n_results=50
        )
        
        # Categorize facts
        summary = {
            "technologies": set(),
            "skills": set(),
            "experience_indicators": [],
            "behavioral_indicators": [],
            "key_points": []
        }
        
        for fact in all_facts:
            content = fact.get('content', '')
            metadata = fact.get('metadata', {})
            fact_type = metadata.get('fact_type', 'general')
            
            if fact_type == 'technology':
                summary["technologies"].add(content)
            elif fact_type == 'skill':
                summary["skills"].add(content)
            elif fact_type == 'experience':
                summary["experience_indicators"].append(content)
            elif fact_type == 'behavior':
                summary["behavioral_indicators"].append(content)
            else:
                summary["key_points"].append(content)
        
        # Convert sets to lists for JSON serialization
        summary["technologies"] = list(summary["technologies"])
        summary["skills"] = list(summary["skills"])
        
        return summary
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all stored facts for a session.
        
        Args:
            session_id: The interview session ID
            
        Returns:
            True if successful
        """
        return self.store.clear_session(session_id)


# Global instance
rag_pipeline = RAGPipeline()
