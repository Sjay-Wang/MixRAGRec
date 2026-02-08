"""
Knowledge Graph Indexer using SentenceBert.
Creates vector indices for entities and triples.
"""

import numpy as np
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..database.kg_database import KGDatabase
from .text_generator import EntityTextGenerator, TripleTextGenerator


class KGIndexer:
    """"""
    
    def __init__(self,
                 db_path: str = "parsed_kg_from_dump.db",
                 index_dir: str = "kg_indices",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = None):
        """
        Args:
        """
        self.db_path = db_path
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading SentenceBert model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"✓ Model loaded on {self.device}")
        
        self.entity_text_gen = EntityTextGenerator()
        self.triple_text_gen = TripleTextGenerator()
        
    def build_entity_index(self, batch_size: int = 128) -> Dict[str, Any]:
        """
        Args:
        Returns:
        """
        print("\n" + "="*70)
        print("Building Entity Index")
        print("="*70)
        
        with KGDatabase(self.db_path) as db:
            db.cursor.execute('SELECT entity_id, entity_type, name, attributes FROM entities')
            rows = db.cursor.fetchall()
            
            print(f"Total entities: {len(rows):,}")
            
            entity_ids = []
            entity_texts = []
            entity_meta = []
            
            print("Generating entity texts...")
            for row in tqdm(rows, desc="Processing entities"):
                entity_id, entity_type, name, attributes_json = row
                
                entity = {
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'name': name,
                    'attributes': json.loads(attributes_json) if attributes_json else {}
                }
                
                text = self.entity_text_gen.entity_to_text(entity)
                
                entity_ids.append(entity_id)
                entity_texts.append(text)
                entity_meta.append({
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'name': name
                })
            
            print(f"\nEncoding {len(entity_texts):,} entities with SentenceBert...")
            embeddings = self.model.encode(
                entity_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            print(f"✓ Generated embeddings: shape {embeddings.shape}")
            
            entity_index_path = self.index_dir / "entity_index.npy"
            entity_ids_path = self.index_dir / "entity_ids.json"
            entity_meta_path = self.index_dir / "entity_meta.json"
            
            np.save(entity_index_path, embeddings)
            
            with open(entity_ids_path, 'w', encoding='utf-8') as f:
                json.dump(entity_ids, f, ensure_ascii=False, indent=2)
            
            with open(entity_meta_path, 'w', encoding='utf-8') as f:
                json.dump(entity_meta, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Entity index saved to {entity_index_path}")
            print(f"✓ Entity IDs saved to {entity_ids_path}")
            print(f"✓ Entity metadata saved to {entity_meta_path}")
            
            stats = {
                'total_entities': len(entity_ids),
                'embedding_dim': embeddings.shape[1],
                'index_file': str(entity_index_path),
                'model_name': self.model_name
            }
            
            return stats
    
    def build_triple_index(self, batch_size: int = 128) -> Dict[str, Any]:
        """
        Args:
        Returns:
        """
        print("\n" + "="*70)
        print("Building Triple Index")
        print("="*70)
        
        with KGDatabase(self.db_path) as db:
            db.cursor.execute('''
                SELECT relation_id, relation_type, head_entity_id, tail_entity_id
                FROM relations
            ''')
            rows = db.cursor.fetchall()
            print(f"Total triples: {len(rows):,}")
            triple_ids = []
            triple_texts = []
            triple_meta = []
            print("Generating triple texts...")
            for row in tqdm(rows, desc="Processing triples"):
                relation_id, relation_type, head_id, tail_id = row
                head_entity = db.get_entity(head_id)
                tail_entity = db.get_entity(tail_id)
                if not head_entity or not tail_entity:
                    continue
                text = self.triple_text_gen.triple_to_text(
                    head_entity, relation_type, tail_entity
                )
                triple_ids.append(relation_id)
                triple_texts.append(text)
                triple_meta.append({
                    'relation_id': relation_id,
                    'relation_type': relation_type,
                    'head_entity_id': head_id,
                    'head_entity_name': head_entity['name'],
                    'tail_entity_id': tail_id,
                    'tail_entity_name': tail_entity['name']
                })
            print(f"\nEncoding {len(triple_texts):,} triples with SentenceBert...")
            embeddings = self.model.encode(
                triple_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            print(f"✓ Generated embeddings: shape {embeddings.shape}")
            triple_index_path = self.index_dir / "triple_index.npy"
            triple_ids_path = self.index_dir / "triple_ids.json"
            triple_meta_path = self.index_dir / "triple_meta.json"
            np.save(triple_index_path, embeddings)
            with open(triple_ids_path, 'w', encoding='utf-8') as f:
                json.dump(triple_ids, f, ensure_ascii=False, indent=2)
            with open(triple_meta_path, 'w', encoding='utf-8') as f:
                json.dump(triple_meta, f, ensure_ascii=False, indent=2)
            print(f"✓ Triple index saved to {triple_index_path}")
            print(f"✓ Triple IDs saved to {triple_ids_path}")
            print(f"✓ Triple metadata saved to {triple_meta_path}")
            stats = {
                'total_triples': len(triple_ids),
                'embedding_dim': embeddings.shape[1],
                'index_file': str(triple_index_path),
                'model_name': self.model_name
            }
            return stats
    def build_all_indices(self, batch_size: int = 128):
        print("="*70)
        print("KG Indexer - Building All Indices")
        print("="*70)
        print(f"Database: {self.db_path}")
        print(f"Output directory: {self.index_dir}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print("="*70)
        entity_stats = self.build_entity_index(batch_size)
        triple_stats = self.build_triple_index(batch_size)
        stats = {
            'entity_index': entity_stats,
            'triple_index': triple_stats,
            'model_name': self.model_name,
            'device': self.device
        }
        stats_path = self.index_dir / "index_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print("\n" + "="*70)
        print("✓ All Indices Built Successfully!")
        print("="*70)
        print(f"Entity index: {entity_stats['total_entities']:,} entities")
        print(f"Triple index: {triple_stats['total_triples']:,} triples")
        print(f"Embedding dimension: {entity_stats['embedding_dim']}")
        print(f"Statistics saved to: {stats_path}")
        print("="*70)
        return stats
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build KG vector indices")
    parser.add_argument('--db-path', type=str, default='parsed_kg_from_dump.db',
                       help='KG database path')
    parser.add_argument('--index-dir', type=str, default='kg_indices',
                       help='Index output directory')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='SentenceBert model name')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for encoding')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    args = parser.parse_args()
    indexer = KGIndexer(
        db_path=args.db_path,
        index_dir=args.index_dir,
        model_name=args.model,
        device=args.device
    )
    stats = indexer.build_all_indices(batch_size=args.batch_size)
    print("\n✓ Indexing complete!")
    print(f"\nNext steps:")
    print("  1. Test retrieval: python test_kg_retrieval.py")
    print("  2. Integrate experts into MixRAGRec framework")
if __name__ == "__main__":
    main()
