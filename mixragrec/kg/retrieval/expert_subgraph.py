"""
Expert 3: Subgraph Retriever
Retrieves k-hop neighborhood subgraphs around top-k relevant entities.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_expert import BaseKGExpert, RetrievalResult
from ..database.kg_database import KGDatabase
from ..indexing.text_generator import SubgraphTextGenerator
from ..indexing.text_generator import TripleTextGenerator


class SubgraphRetrieverExpert(BaseKGExpert):
    """"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            expert_id=3,  # Expert ID: 3
            expert_name="SubgraphRetriever",
            config=config
        )
        
        self.db_path = config.get('knowledge_graph', {}).get('kg_db_path', 'data/parsed_kg_from_dump.db')
        self.index_dir = Path(config.get('knowledge_graph', {}).get('kg_indices_path', 'data/kg_indices'))
        
        model_name = config.get('models', {}).get('encoder', {}).get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        device = config.get('device', 'cpu')
        
        self.model = SentenceTransformer(model_name, device=device)
        
        self.entity_embeddings = None
        self.entity_ids = None
        self.entity_meta = None
        
        self.subgraph_text_gen = SubgraphTextGenerator()
        self.triple_text_gen = TripleTextGenerator()
        
        kg_config = config.get('knowledge_graph', {})
        self.max_neighbors_per_hop = kg_config.get('max_neighbors_per_hop', 100)
        self.subgraph_radius = kg_config.get('max_subgraph_radius', 2)
    
    def initialize(self):
        """"""
        print(f"Loading entity index from {self.index_dir}...")
        
        entity_index_path = self.index_dir / "entity_index.npy"
        self.entity_embeddings = np.load(entity_index_path)
        
        entity_ids_path = self.index_dir / "entity_ids.json"
        with open(entity_ids_path, 'r', encoding='utf-8') as f:
            self.entity_ids = json.load(f)
        
        entity_meta_path = self.index_dir / "entity_meta.json"
        with open(entity_meta_path, 'r', encoding='utf-8') as f:
            self.entity_meta = json.load(f)
        
        print(f"✓ Expert 3 ({self.expert_name}) initialized")
        print(f"  Loaded {len(self.entity_ids):,} entities, embedding dim: {self.entity_embeddings.shape[1]}")
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """
        Args:
        Returns:
        """
        query_embedding = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        
        similarities = np.dot(self.entity_embeddings, query_embedding)
        
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        merged_entities = {}
        merged_relations = []
        retrieved_entity_info = []
        
        with KGDatabase(self.db_path) as db:
            for idx in top_k_indices:
                entity_id = self.entity_ids[idx]
                entity_meta = self.entity_meta[idx]
                score = float(similarities[idx])
                
                subgraph = db.get_2hop_subgraph(entity_id, max_neighbors_per_hop=self.max_neighbors_per_hop)
                
                for eid, entity in subgraph['entities'].items():
                    if eid not in merged_entities:
                        merged_entities[eid] = entity
                
                for relation in subgraph['relations']:
                    rel_key = f"{relation['head']}_{relation['relation']}_{relation['tail']}"
                    if not any(f"{r['head']}_{r['relation']}_{r['tail']}" == rel_key for r in merged_relations):
                        merged_relations.append(relation)
                
                retrieved_entity_info.append({
                    'entity_id': entity_id,
                    'entity_name': entity_meta['name'],
                    'entity_type': entity_meta['entity_type'],
                    'score': score,
                    'subgraph_size': {
                        'entities': len(subgraph['entities']),
                        'relations': len(subgraph['relations'])
                    }
                })
        
        merged_subgraph = {
            'entities': merged_entities,
            'relations': merged_relations
        }
        
        knowledge_texts = []
        knowledge_texts.append(f"Retrieved {top_k} relevant entities and their 2-hop subgraphs:")
        knowledge_texts.append("")
        
        for i, entity_info in enumerate(retrieved_entity_info, 1):
            knowledge_texts.append(
                f"{i}. {entity_info['entity_name']} ({entity_info['entity_type']}) "
                f"[score: {entity_info['score']:.3f}, "
                f"subgraph: {entity_info['subgraph_size']['entities']} entities, "
                f"{entity_info['subgraph_size']['relations']} relations]"
            )
        
        knowledge_texts.append("")
        knowledge_texts.append(f"Merged subgraph contains:")
        knowledge_texts.append(f"  Total entities: {len(merged_entities)}")
        knowledge_texts.append(f"  Total relations: {len(merged_relations)}")
        knowledge_texts.append("")
        
        knowledge_texts.append("Key relationships:")
        for i, relation in enumerate(merged_relations[:20], 1):
            head_entity = merged_entities.get(relation['head'])
            tail_entity = merged_entities.get(relation['tail'])
            
            if head_entity and tail_entity:
                triple_text = self.triple_text_gen.triple_to_text(
                    head_entity, relation['relation'], tail_entity
                )
                knowledge_texts.append(f"  • {triple_text}")
        
        if len(merged_relations) > 20:
            knowledge_texts.append(f"  ... and {len(merged_relations) - 20} more relationships")
        
        knowledge_text = "\n".join(knowledge_texts)
        
        result = RetrievalResult(
            expert_id=self.expert_id,
            expert_name=self.expert_name,
            retrieved_knowledge=knowledge_text,
            metadata={
                'query': query,
                'top_k': top_k,
                'retrieved_entities': retrieved_entity_info,
                'merged_subgraph': {
                    'num_entities': len(merged_entities),
                    'num_relations': len(merged_relations)
                },
                'avg_score': float(np.mean([e['score'] for e in retrieved_entity_info]))
            },
            confidence=float(np.mean([e['score'] for e in retrieved_entity_info]))
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """"""
        stats = super().get_stats()
        stats.update({
            'retrieval_type': 'subgraph',
            'total_entities': len(self.entity_ids) if self.entity_ids else 0,
            'subgraph_radius': self.subgraph_radius,
            'description': 'Retrieves k-hop neighborhood subgraphs around top entities'
        })
        return stats
