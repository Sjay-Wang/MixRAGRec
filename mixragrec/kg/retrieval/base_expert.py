"""
Base classes for KG retrieval experts.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """"""
    expert_id: int  # Expert ID (1-4)
    expert_name: str
    retrieved_knowledge: str
    metadata: Dict[str, Any]
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'expert_id': self.expert_id,
            'expert_name': self.expert_name,
            'retrieved_knowledge': self.retrieved_knowledge,
            'metadata': self.metadata,
            'confidence': self.confidence
        }


class BaseKGExpert(ABC):
    """"""
    
    def __init__(self, expert_id: int, expert_name: str, config: Dict[str, Any]):
        self.expert_id = expert_id  # Expert ID (1-4)
        self.expert_name = expert_name
        self.config = config
        
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> RetrievalResult:
        """
        Args:
        Returns:
        """
        pass
    
    @abstractmethod
    def initialize(self):
        """"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """"""
        return {
            'expert_id': self.expert_id,
            'expert_name': self.expert_name
        }
