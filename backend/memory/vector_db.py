"""
Vector database interface for storing and retrieving interview facts.
Uses ChromaDB for vector storage and similarity search.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Vector database store for interview memory/facts.
    """
    
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialized = False
        
        if CHROMADB_AVAILABLE:
            try:
                self.client = chromadb.Client(Settings(
                    anonymized_telemetry=False
                ))
                self.collection = self.client.get_or_create_collection(
                    name="interview_facts",
                    metadata={"description": "Interview facts and extracted information"}
                )
                self._initialized = True
                logger.info("ChromaDB initialized successfully")
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")
        else:
            logger.warning("ChromaDB not available, memory store disabled")
    
    def store_facts(
        self,
        session_id: str,
        facts: List[Dict[str, Any]],
        phase: str
    ) -> List[str]:
        """
        Store facts in the vector database.
        
        Args:
            session_id: The interview session ID
            facts: List of fact dictionaries
            phase: Current interview phase
            
        Returns:
            List of stored fact IDs
        """
        if not self._initialized or not facts:
            return []
        
        try:
            ids = []
            documents = []
            metadatas = []
            
            for i, fact in enumerate(facts):
                fact_id = f"{session_id}_{phase}_{datetime.now().timestamp()}_{i}"
                ids.append(fact_id)
                
                # Create document text for embedding
                content = fact.get('content', str(fact))
                documents.append(content)
                
                # Metadata
                metadatas.append({
                    "session_id": session_id,
                    "phase": phase,
                    "fact_type": fact.get('type', 'general'),
                    "timestamp": datetime.now().isoformat()
                })
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(ids)} facts for session {session_id}")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to store facts: {e}")
            return []
    
    def retrieve_relevant(
        self,
        session_id: str,
        query: str,
        n_results: int = 5,
        phase_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant facts from the vector database.
        
        Args:
            session_id: The interview session ID
            query: Query text for similarity search
            n_results: Number of results to return
            phase_filter: Optional phase to filter by
            
        Returns:
            List of relevant fact dictionaries
        """
        if not self._initialized:
            return []
        
        try:
            where_filter = {"session_id": session_id}
            if phase_filter:
                where_filter["phase"] = phase_filter
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            facts = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    facts.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                    })
            
            return facts
            
        except Exception as e:
            logger.error(f"Failed to retrieve facts: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all facts for a session.
        
        Args:
            session_id: The interview session ID
            
        Returns:
            True if successful
        """
        if not self._initialized:
            return False
        
        try:
            # ChromaDB doesn't have a direct delete by metadata
            # Get all IDs for this session and delete them
            results = self.collection.get(
                where={"session_id": session_id}
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Cleared {len(results['ids'])} facts for session {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False


# Global instance
memory_store = MemoryStore()
