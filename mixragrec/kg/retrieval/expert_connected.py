"""
Expert 4: Connected Graph Retriever
Constructs connected subgraphs using Personalized PageRank and Kruskal's MST.

Algorithm:
1. Encode query with SentenceTransformer
2. Retrieve top-k seed entities by semantic similarity
3. Build 2-hop graph from seed entities
4. Run Personalized PageRank (unweighted) to find structurally important nodes
5. Use Kruskal's MST with semantic edge weights to connect important nodes
   - Edge weight = 1 - cosine_similarity(query, relation_type)
   - Lower cost = higher semantic relevance = preferred edge
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
import networkx as nx

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_expert import BaseKGExpert, RetrievalResult
from ..database.kg_database import KGDatabase
from ..indexing.text_generator import TripleTextGenerator


class ConnectedGraphRetrieverExpert(BaseKGExpert):
    """"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            expert_id=4,  # Expert ID: 4
            expert_name="ConnectedGraphRetriever",
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
        
        self.text_gen = TripleTextGenerator()
        
        self.relation_embedding_cache: Dict[str, np.ndarray] = {}
        
        self.alpha = 0.85
        self.pagerank_iterations = 100
    
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
        
        print(f"✓ Expert 4 ({self.expert_name}) initialized")
        print(f"  Loaded {len(self.entity_ids):,} entities, embedding dim: {self.entity_embeddings.shape[1]}")
    
    def _get_relation_embedding(self, relation_type: str) -> np.ndarray:
        """
        Args:
        Returns:
        """
        if relation_type not in self.relation_embedding_cache:
            relation_text = self.text_gen.relation_to_natural_language(relation_type)
            embedding = self.model.encode(relation_text, convert_to_numpy=True, normalize_embeddings=True)
            self.relation_embedding_cache[relation_type] = embedding
        
        return self.relation_embedding_cache[relation_type]
    
    def build_graph_from_entities(self, entity_ids: List[str], db: KGDatabase, 
                                  max_radius: int = 2) -> nx.Graph:
        """
        Args:
        Returns:
        """
        G = nx.Graph()
        
        all_entities = set(entity_ids)
        all_relations = []
        
        current_entities = set(entity_ids)
        for _ in range(max_radius):
            next_entities = set()
            
            for entity_id in current_entities:
                neighbors = db.get_neighbors(entity_id)
                
                for neighbor in neighbors:
                    neighbor_id = neighbor['entity_id']
                    
                    all_relations.append({
                        'head': entity_id,
                        'tail': neighbor_id,
                        'relation': neighbor['relation_type']
                    })
                    
                    if neighbor_id not in all_entities:
                        all_entities.add(neighbor_id)
                        next_entities.add(neighbor_id)
            
            current_entities = next_entities
            if not current_entities:
                break
        
        for entity_id in all_entities:
            entity = db.get_entity(entity_id)
            if entity:
                G.add_node(entity_id, **entity)
        
        for relation in all_relations:
            G.add_edge(
                relation['head'],
                relation['tail'],
                relation=relation['relation']
            )
        
        return G
    
    def personalized_pagerank(self, G: nx.Graph, seed_entities: List[str]) -> Dict[str, float]:
        """
        Args:
        Returns:
        """
        personalization = {node: 0.0 for node in G.nodes()}
        for entity_id in seed_entities:
            if entity_id in personalization:
                personalization[entity_id] = 1.0 / len(seed_entities)
        
        try:
            pagerank_scores = nx.pagerank(
                G,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.pagerank_iterations
            )
        except:
            pagerank_scores = {node: 1.0 / len(G.nodes()) for node in G.nodes()}
        
        return pagerank_scores
    
    def extract_mst(self, G: nx.Graph, important_nodes: Set[str], 
                    query_embedding: np.ndarray) -> nx.Graph:
        """
        Args:
        Returns:
        """
        if len(G.nodes()) <= 50:
            return G
        
        if len(important_nodes) < 2:
            return G.subgraph(important_nodes).copy()
        
        try:
            for u, v, data in G.edges(data=True):
                relation_type = data.get('relation', 'related_to')
                
                relation_embedding = self._get_relation_embedding(relation_type)
                
                similarity = float(np.dot(query_embedding, relation_embedding))
                
                data['semantic_cost'] = 1.0 - similarity
            
            existing_important_nodes = important_nodes & set(G.nodes())
            
            if len(existing_important_nodes) < 2:
                return G.subgraph(existing_important_nodes).copy()
            
            
            try:
                mst = nx.minimum_spanning_tree(G, weight='semantic_cost', algorithm='kruskal')
                
                mst_nodes = set(mst.nodes())
                if existing_important_nodes.issubset(mst_nodes):
                    steiner_nodes = set()
                    important_list = list(existing_important_nodes)
                    
                    for i in range(len(important_list)):
                        for j in range(i + 1, len(important_list)):
                            try:
                                path = nx.shortest_path(mst, important_list[i], important_list[j])
                                steiner_nodes.update(path)
                            except nx.NetworkXNoPath:
                                continue
                    
                    if steiner_nodes:
                        return mst.subgraph(steiner_nodes).copy()
                    
            except Exception:
                pass
            
            paths_nodes = set()
            important_list = list(existing_important_nodes)
            
            for i in range(len(important_list)):
                for j in range(i + 1, len(important_list)):
                    try:
                        path = nx.shortest_path(G, important_list[i], important_list[j], 
                                               weight='semantic_cost')
                        paths_nodes.update(path)
                    except nx.NetworkXNoPath:
                        continue
            
            paths_nodes.update(existing_important_nodes)
            
            if paths_nodes:
                subgraph = G.subgraph(paths_nodes).copy()
                return subgraph
            
            return G.subgraph(existing_important_nodes).copy()
            
        except Exception as e:
            return G.subgraph(important_nodes & set(G.nodes())).copy()
    
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> RetrievalResult:
        """
        Pipeline:
        Args:
        Returns:
        """
        query_embedding = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        
        similarities = np.dot(self.entity_embeddings, query_embedding)
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        seed_entity_ids = [self.entity_ids[idx] for idx in top_k_indices]
        seed_scores = {self.entity_ids[idx]: float(similarities[idx]) for idx in top_k_indices}
        
        with KGDatabase(self.db_path) as db:
            G = self.build_graph_from_entities(seed_entity_ids, db, max_radius=2)
            
            pagerank_scores = self.personalized_pagerank(G, seed_entity_ids)
            
            sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            important_nodes = set([node for node, score in sorted_nodes[:min(100, len(sorted_nodes))]])
            
            important_nodes.update(seed_entity_ids)
            
            connected_subgraph = self.extract_mst(G, important_nodes, query_embedding)
            
            knowledge_texts = []
            knowledge_texts.append(f"Connected subgraph analysis:")
            knowledge_texts.append("")
            knowledge_texts.append(f"Seed entities ({len(seed_entity_ids)}):")
            
            for i, entity_id in enumerate(seed_entity_ids, 1):
                entity = db.get_entity(entity_id)
                if entity:
                    knowledge_texts.append(
                        f"  {i}. {entity['name']} ({entity['entity_type']}) "
                        f"[relevance: {seed_scores[entity_id]:.3f}, "
                        f"importance: {pagerank_scores.get(entity_id, 0):.4f}]"
                    )
            
            knowledge_texts.append("")
            knowledge_texts.append(f"Connected graph statistics:")
            knowledge_texts.append(f"  Nodes: {connected_subgraph.number_of_nodes()}")
            knowledge_texts.append(f"  Edges: {connected_subgraph.number_of_edges()}")
            knowledge_texts.append("")
            
            knowledge_texts.append("Important relationships in connected graph:")
            
            edges_with_cost = []
            for u, v, data in connected_subgraph.edges(data=True):
                u_entity = db.get_entity(u)
                v_entity = db.get_entity(v)
                relation = data.get('relation', 'related_to')
                semantic_cost = data.get('semantic_cost', 0.5)
                
                if u_entity and v_entity:
                    edges_with_cost.append({
                        'u_name': u_entity['name'],
                        'v_name': v_entity['name'],
                        'relation': relation,
                        'cost': semantic_cost,
                        'similarity': 1.0 - semantic_cost
                    })
            
            edges_with_cost.sort(key=lambda x: x['similarity'], reverse=True)
            
            for i, edge in enumerate(edges_with_cost[:30]):
                triple_text = f"{edge['u_name']} {self.text_gen.relation_to_natural_language(edge['relation'])} {edge['v_name']}"
                knowledge_texts.append(f"  • {triple_text} [relevance: {edge['similarity']:.3f}]")
            
            if len(edges_with_cost) > 30:
                knowledge_texts.append(f"  ... and {len(edges_with_cost) - 30} more relationships")
        
        knowledge_text = "\n".join(knowledge_texts)
        
        result = RetrievalResult(
            expert_id=self.expert_id,
            expert_name=self.expert_name,
            retrieved_knowledge=knowledge_text,
            metadata={
                'query': query,
                'top_k': top_k,
                'seed_entities': seed_entity_ids,
                'graph_stats': {
                    'nodes': connected_subgraph.number_of_nodes(),
                    'edges': connected_subgraph.number_of_edges()
                },
                'avg_seed_score': float(np.mean(list(seed_scores.values()))),
                'algorithm': 'PPR (unweighted) + Kruskal MST (semantic weight)'
            },
            confidence=float(np.mean(list(seed_scores.values())))
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """"""
        stats = super().get_stats()
        stats.update({
            'retrieval_type': 'connected_graph',
            'total_entities': len(self.entity_ids) if self.entity_ids else 0,
            'algorithm': 'PPR (unweighted) + Kruskal MST (semantic similarity)',
            'description': 'Constructs connected subgraphs using structural importance (PPR) and semantic relevance (MST)',
            'relation_cache_size': len(self.relation_embedding_cache)
        })
        return stats
