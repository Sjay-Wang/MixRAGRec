"""
MixRAGRec Recommendation Pipeline
Executes the complete flow: Expert Selection → KG Retrieval → Alignment → Recommendation

Expert IDs: 1-4
1: DirectGenerator - No retrieval, use LLM internal knowledge
2: TripleRetriever - Simple triple-based retrieval
3: SubgraphRetriever - 2-hop subgraph retrieval
4: ConnectedGraphRetriever - PageRank + MST based retrieval
"""

from typing import Dict, Any, List, Optional


class RecommendationPipeline:
    """
    Complete recommendation pipeline for MixRAGRec
    Coordinates the three agents to generate recommendations
    """
    
    def __init__(self, agent_coordinator):
        """
        Args:
            agent_coordinator: AgentCoordinator instance
        """
        self.agent_coordinator = agent_coordinator
    
    def execute(self,
                user_query: str,
                user_context: Optional[Dict[str, Any]] = None,
                conversation_history: Optional[List[Dict[str, Any]]] = None,
                expert_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute complete recommendation pipeline
        
        Args:
            user_query: User's query
            user_context: User context information
            conversation_history: Previous conversation
            expert_id: Specific KG expert to use (1-4, None for RL selection)
            
        Returns:
            Complete pipeline results including recommendations
        """
        return self.agent_coordinator.execute_pipeline(
            user_query=user_query,
            user_context=user_context or {},
            conversation_history=conversation_history or [],
            expert_id=expert_id
        )
    
    def batch_execute(self,
                     queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute pipeline for multiple queries
        
        Args:
            queries: List of query dictionaries
            
        Returns:
            List of results
        """
        results = []
        
        for query_item in queries:
            result = self.execute(
                user_query=query_item['query'],
                user_context=query_item.get('context'),
                conversation_history=query_item.get('history', [])
            )
            results.append(result)
        
        return results
