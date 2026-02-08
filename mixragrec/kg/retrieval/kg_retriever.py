"""
Unified KG Retriever that manages 4 retrieval experts.
Integrates with MixRAGRec framework as the retriever component.

Expert IDs (1-4):
- Expert 1: DirectGenerator - No retrieval, uses LLM internal knowledge
- Expert 2: TripleRetriever - Retrieves relevant factual triples
- Expert 3: SubgraphRetriever - Retrieves 2-hop subgraphs
- Expert 4: ConnectedGraphRetriever - Constructs connected subgraphs using PPR+MST
"""

from typing import Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_expert import BaseKGExpert, RetrievalResult
from .expert_direct import DirectGeneratorExpert
from .expert_triple import TripleRetrieverExpert
from .expert_subgraph import SubgraphRetrieverExpert
from .expert_connected import ConnectedGraphRetrieverExpert


class KGRetriever:
    """
    Expert IDs: 1-4 (not 0-3)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "KGRetriever"
        self.device = config.get('device', 'cpu')
        
        self.experts = {
            1: DirectGeneratorExpert(config),
            2: TripleRetrieverExpert(config),
            3: SubgraphRetrieverExpert(config),
            4: ConnectedGraphRetrieverExpert(config)
        }
        
        self.default_expert_id = config.get('knowledge_graph', {}).get('default_expert', 3)
        self.active_expert_id = self.default_expert_id
        
        self.retrieval_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    def initialize(self):
        """"""
        print("Initializing KG Retriever with 4 experts...")
        
        for expert_id, expert in self.experts.items():
            try:
                expert.initialize()
            except Exception as e:
                print(f"⚠ Warning: Expert {expert_id} initialization failed: {e}")
        
        print("✓ KG Retriever initialized")
    
    def set_active_retriever(self, expert_id: int):
        """"""
        if expert_id in self.experts:
            self.active_expert_id = expert_id
        else:
            raise ValueError(f"Expert ID {expert_id} not found. Must be 1-4.")
    
    def retrieve(self, query: str, top_k: int = 10, expert_id: int = None, **kwargs):
        """
        Args:
        Returns:
        """
        if expert_id is None:
            expert_id = self.active_expert_id
        
        if expert_id not in self.experts:
            raise ValueError(f"Invalid expert_id: {expert_id}. Must be 1-4.")
        
        self.retrieval_counts[expert_id] += 1
        
        expert = self.experts[expert_id]
        kg_result = expert.retrieve(query, top_k, **kwargs)
        
        documents = [kg_result.retrieved_knowledge] if kg_result.retrieved_knowledge else []
        scores = [kg_result.confidence]
        
        class SimpleResult:
            def __init__(self, docs, scrs, meta):
                self.documents = docs
                self.scores = scrs
                self.metadata = meta
        
        result = SimpleResult(
            docs=documents,
            scrs=scores,
            meta={
                'retriever_type': 'knowledge_graph',
                'expert_id': expert_id,
                'expert_name': kg_result.expert_name,
                'query': query,
                'kg_metadata': kg_result.metadata,
                'confidence': kg_result.confidence
            }
        )
        
        return result
    
    def retrieve_with_all_experts(self, query: str, top_k: int = 10) -> Dict[int, Any]:
        """
        Args:
        Returns:
        """
        results = {}
        
        for expert_id in [1, 2, 3, 4]:
            try:
                results[expert_id] = self.retrieve(query, top_k, expert_id=expert_id)
            except Exception as e:
                print(f"⚠ Expert {expert_id} failed: {e}")
                class SimpleResult:
                    def __init__(self):
                        self.documents = []
                        self.scores = []
                        self.metadata = {'error': str(e)}
                results[expert_id] = SimpleResult()
        
        return results
    
    def update_index(self, documents: List[str]):
        """
        Args:
        """
        print("⚠ KG Retriever does not support dynamic index updates.")
        print("  To update KG, rebuild the database and indices.")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """"""
        stats = {
            'name': self.name,
            'device': self.device,
            'default_expert_id': self.default_expert_id,
            'active_expert_id': self.active_expert_id
        }
        
        stats.update({
            'num_experts': len(self.experts),
            'retrieval_counts': self.retrieval_counts,
            'total_retrievals': sum(self.retrieval_counts.values()),
            'experts': {}
        })
        
        for expert_id, expert in self.experts.items():
            stats['experts'][expert_id] = expert.get_stats()
        
        return stats


def test_kg_retriever():
    """"""
    print("="*70)
    print("Testing KG Retriever with All Experts")
    print("="*70)
    
    config = {
        'device': 'cpu',
        'models': {
            'encoder': {'model_name': 'sentence-transformers/all-MiniLM-L6-v2'}
        },
        'knowledge_graph': {
            'kg_db_path': 'parsed_kg_from_dump.db',
            'kg_index_path': 'kg_indices',
            'default_expert': 3,  # Expert 3 (SubgraphRetriever)
            'max_neighbors_per_hop': 100,
            'max_subgraph_radius': 2
        }
    }
    
    retriever = KGRetriever(config)
    retriever.initialize()
    
    test_queries = [
        "animated movies with adventure",
        "science fiction films directed by famous directors",
        "movies about love and romance"
    ]
    
    for query in test_queries:
        print("\n" + "="*70)
        print(f"Query: {query}")
        print("="*70)
        
        results = retriever.retrieve_with_all_experts(query, top_k=5)
        
        for expert_id, result in results.items():
            print(f"\n--- Expert {expert_id}: {result.metadata.get('expert_name', 'Unknown')} ---")
            
            if result.documents:
                knowledge = result.documents[0]
                if len(knowledge) > 500:
                    print(knowledge[:500] + "...")
                else:
                    print(knowledge)
                
                print(f"\nConfidence: {result.metadata.get('confidence', 0):.3f}")
            else:
                print("No results")
    
    print("\n" + "="*70)
    print("Retrieval Statistics:")
    print("="*70)
    stats = retriever.get_retrieval_stats()
    import json
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    import json
    test_kg_retriever()
