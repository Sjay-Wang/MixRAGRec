"""
Expert 2: Triple Retriever
Retrieves the most relevant factual triples from the KG.
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
from ..indexing.text_generator import TripleTextGenerator


class TripleRetrieverExpert(BaseKGExpert):
    """"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            expert_id=2,  # Expert ID: 2
            expert_name="TripleRetriever",
            config=config
        )
        
        self.db_path = config.get('knowledge_graph', {}).get('kg_db_path', 'data/parsed_kg_from_dump.db')
        self.index_dir = Path(config.get('knowledge_graph', {}).get('kg_indices_path', 'data/kg_indices'))
        
        model_name = config.get('models', {}).get('encoder', {}).get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        device = config.get('device', 'cpu')
        
        self.model = SentenceTransformer(model_name, device=device)
        
        self.triple_embeddings = None
        self.triple_ids = None
        self.triple_meta = None
        
        self.text_gen = TripleTextGenerator()
        
    def initialize(self):
        """"""
        print(f"Loading triple index from {self.index_dir}...")
        
        triple_index_path = self.index_dir / "triple_index.npy"
        self.triple_embeddings = np.load(triple_index_path)
        
        triple_ids_path = self.index_dir / "triple_ids.json"
        with open(triple_ids_path, 'r', encoding='utf-8') as f:
            self.triple_ids = json.load(f)
        
        triple_meta_path = self.index_dir / "triple_meta.json"
        with open(triple_meta_path, 'r', encoding='utf-8') as f:
            self.triple_meta = json.load(f)
        
        print(f"✓ Expert 2 ({self.expert_name}) initialized")
        print(f"  Loaded {len(self.triple_ids):,} triples, embedding dim: {self.triple_embeddings.shape[1]}")
    
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> RetrievalResult:
        """
        Args:
        Returns:
        """
        query_embedding = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        
        similarities = np.dot(self.triple_embeddings, query_embedding)
        
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieved_triples = []
        retrieved_texts = []
        
        for idx in top_k_indices:
            triple_meta = self.triple_meta[idx]
            score = float(similarities[idx])
            
            triple_text = f"{triple_meta['head_entity_name']} {self.text_gen.relation_to_natural_language(triple_meta['relation_type'])} {triple_meta['tail_entity_name']}"
            
            retrieved_triples.append({
                'head': triple_meta['head_entity_name'],
                'relation': triple_meta['relation_type'],
                'tail': triple_meta['tail_entity_name'],
                'score': score,
                'text': triple_text
            })
            
            retrieved_texts.append(f"• {triple_text} (score: {score:.3f})")
        
        knowledge_text = "Retrieved factual triples:\n" + "\n".join(retrieved_texts)
        
        result = RetrievalResult(
            expert_id=self.expert_id,
            expert_name=self.expert_name,
            retrieved_knowledge=knowledge_text,
            metadata={
                'query': query,
                'top_k': top_k,
                'num_retrieved': len(retrieved_triples),
                'triples': retrieved_triples,
                'avg_score': float(np.mean([t['score'] for t in retrieved_triples]))
            },
            confidence=float(np.mean([t['score'] for t in retrieved_triples]))
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """"""
        stats = super().get_stats()
        stats.update({
            'retrieval_type': 'triple',
            'total_triples': len(self.triple_ids) if self.triple_ids else 0,
            'description': 'Retrieves most relevant factual triples'
        })
        return stats
