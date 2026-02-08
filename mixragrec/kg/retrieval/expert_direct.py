"""
Expert 1: Direct Generator
Bypasses retrieval entirely, directly generates recommendations using LLM's internal knowledge.
"""

from typing import Dict, Any
from .base_expert import BaseKGExpert, RetrievalResult


class DirectGeneratorExpert(BaseKGExpert):
    """"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            expert_id=1,  # Expert ID: 1
            expert_name="DirectGenerator",
            config=config
        )
    
    def initialize(self):
        """"""
        print(f"✓ Expert 1 ({self.expert_name}) initialized")
    
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> RetrievalResult:
        """
        Args:
        Returns:
        """
        result = RetrievalResult(
            expert_id=self.expert_id,
            expert_name=self.expert_name,
            retrieved_knowledge="",
            metadata={
                'query': query,
                'mode': 'direct_generation',
                'retrieval_performed': False
            },
            confidence=1.0
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """"""
        stats = super().get_stats()
        stats.update({
            'retrieval_type': 'none',
            'description': 'Bypasses retrieval, uses LLM internal knowledge only'
        })
        return stats
