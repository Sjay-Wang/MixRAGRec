"""
Knowledge Graph database operations using SQLite.
Stores entities and relations in a local database.
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict


class KGDatabase:
    """"""
    
    def __init__(self, db_path: str = "parsed_kg_from_dump.db"):
        """
        Args:
        """
        self.db_path = Path(db_path)
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        
    def close(self):
        """"""
        if self.conn:
            self.conn.commit()
            self.conn.close()
            
    def __enter__(self):
        """"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """"""
        self.close()
    
    def initialize_schema(self):
        """"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                attributes TEXT,
                aliases TEXT
            )
        ''''''
            CREATE TABLE IF NOT EXISTS relations (
                relation_id TEXT PRIMARY KEY,
                relation_type TEXT NOT NULL,
                head_entity_id TEXT NOT NULL,
                tail_entity_id TEXT NOT NULL,
                attributes TEXT,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (head_entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY (tail_entity_id) REFERENCES entities(entity_id)
            )
        ''''''
            CREATE INDEX IF NOT EXISTS idx_relation_head 
            ON relations(head_entity_id)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_relation_tail 
            ON relations(tail_entity_id)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_relation_type 
            ON relations(relation_type)
        ''')
        self.conn.commit()
        print("✓ Database schema initialized")
    def clear_all(self):
        self.cursor.execute('DELETE FROM relations')
        self.cursor.execute('DELETE FROM entities')
        self.conn.commit()
        print("✓ Database cleared")
    def insert_entity(self, 
                     entity_id: str, 
                     entity_type: str = 'Resource',
                     name: str = '',
                     attributes: Optional[Dict[str, Any]] = None,
                     aliases: Optional[List[str]] = None):
        """
        Args:
        """
        if not name:
            name = entity_id
        attributes_json = json.dumps(attributes or {})
        aliases_json = json.dumps(aliases or [])
        self.cursor.execute('''
            INSERT OR REPLACE INTO entities 
            (entity_id, entity_type, name, attributes, aliases)
            VALUES (?, ?, ?, ?, ?)
        ''', (entity_id, entity_type, name, attributes_json, aliases_json))
    def insert_relation(self,
                       head_entity_id: str,
                       relation_type: str,
                       tail_entity_id: str,
                       attributes: Optional[Dict[str, Any]] = None,
                       confidence: float = 1.0):
        """
        Args:
        """
        relation_id = f"{head_entity_id}_{relation_type}_{tail_entity_id}"
        attributes_json = json.dumps(attributes or {})
        self.cursor.execute('''
            INSERT OR REPLACE INTO relations
            (relation_id, relation_type, head_entity_id, tail_entity_id, attributes, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (relation_id, relation_type, head_entity_id, tail_entity_id, attributes_json, confidence))
    def batch_insert_entities(self, entities: List[Tuple]):
        """
        Args:
        """
        self.cursor.executemany('''
            INSERT OR REPLACE INTO entities 
            (entity_id, entity_type, name, attributes, aliases)
            VALUES (?, ?, ?, ?, ?)
        ''', entities)
    def batch_insert_relations(self, relations: List[Tuple]):
        """
        Args:
        """
        self.cursor.executemany('''
            INSERT OR REPLACE INTO relations
            (relation_id, relation_type, head_entity_id, tail_entity_id, attributes, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', relations)
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute('''
            SELECT entity_id, entity_type, name, attributes, aliases
            FROM entities WHERE entity_id = ?
        ''', (entity_id,))
        row = self.cursor.fetchone()
        if row:
            return {
                'entity_id': row[0],
                'entity_type': row[1],
                'name': row[2],
                'attributes': json.loads(row[3]) if row[3] else {},
                'aliases': json.loads(row[4]) if row[4] else []
            }
        return None
    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Args:
        Returns:
        """
        if relation_type:
            self.cursor.execute('''
                SELECT tail_entity_id, relation_type, confidence
                FROM relations 
                WHERE head_entity_id = ? AND relation_type = ?
            ''', (entity_id, relation_type))
        else:
            self.cursor.execute('''
                SELECT tail_entity_id, relation_type, confidence
                FROM relations 
                WHERE head_entity_id = ?
            ''', (entity_id,))
        neighbors = []
        for row in self.cursor.fetchall():
            neighbors.append({
                'entity_id': row[0],
                'relation_type': row[1],
                'confidence': row[2]
            })
        return neighbors
    def get_2hop_subgraph(self, entity_id: str, max_neighbors_per_hop: int = 100) -> Dict[str, Any]:
        """
        Args:
        Returns:
        """
        entities_dict = {}
        relations_list = []
        center = self.get_entity(entity_id)
        if not center:
            return {'entities': {}, 'relations': []}
        entities_dict[entity_id] = center
        hop1_neighbors = self.get_neighbors(entity_id)
        hop1_neighbors = hop1_neighbors[:max_neighbors_per_hop]
        for neighbor in hop1_neighbors:
            neighbor_id = neighbor['entity_id']
            relations_list.append({
                'head': entity_id,
                'relation': neighbor['relation_type'],
                'tail': neighbor_id,
                'confidence': neighbor['confidence']
            })
            neighbor_entity = self.get_entity(neighbor_id)
            if neighbor_entity:
                entities_dict[neighbor_id] = neighbor_entity
        for neighbor in hop1_neighbors:
            neighbor_id = neighbor['entity_id']
            hop2_neighbors = self.get_neighbors(neighbor_id)
            hop2_neighbors = hop2_neighbors[:max_neighbors_per_hop]
            for hop2_neighbor in hop2_neighbors:
                hop2_id = hop2_neighbor['entity_id']
                relations_list.append({
                    'head': neighbor_id,
                    'relation': hop2_neighbor['relation_type'],
                    'tail': hop2_id,
                    'confidence': hop2_neighbor['confidence']
                })
                if hop2_id not in entities_dict:
                    hop2_entity = self.get_entity(hop2_id)
                    if hop2_entity:
                        entities_dict[hop2_id] = hop2_entity
        return {
            'entities': entities_dict,
            'relations': relations_list
        }
    def get_statistics(self) -> Dict[str, Any]:
        stats = {}
        self.cursor.execute('SELECT COUNT(*) FROM entities')
        stats['total_entities'] = self.cursor.fetchone()[0]
        self.cursor.execute('SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type')
        stats['entities_by_type'] = dict(self.cursor.fetchall())
        self.cursor.execute('SELECT COUNT(*) FROM relations')
        stats['total_relations'] = self.cursor.fetchone()[0]
        self.cursor.execute('SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type ORDER BY COUNT(*) DESC LIMIT 20')
        stats['top_relation_types'] = dict(self.cursor.fetchall())
        return stats
    def print_statistics(self):
        stats = self.get_statistics()
        print("\nKnowledge Graph Statistics")
        print("=" * 60)
        print(f"Total Entities: {stats['total_entities']:,}")
        print(f"Total Relations: {stats['total_relations']:,}")
        print("\nEntities by Type:")
        for entity_type, count in stats['entities_by_type'].items():
            print(f"  • {entity_type}: {count:,}")
        print("\nTop Relation Types:")
        for relation_type, count in list(stats['top_relation_types'].items())[:10]:
            print(f"  • {relation_type}: {count:,}")
        print("=" * 60)
def main():
    with KGDatabase() as db:
        db.initialize_schema()
        db.insert_entity('Toy_Story', 'Movie', 'Toy Story')
        db.insert_entity('John_Lasseter', 'Person', 'John Lasseter')
        db.insert_relation('Toy_Story', 'director', 'John_Lasseter')
        db.conn.commit()
        entity = db.get_entity('Toy_Story')
        print("Entity:", entity)
        neighbors = db.get_neighbors('Toy_Story')
        print("Neighbors:", neighbors)
        db.print_statistics()
if __name__ == "__main__":
    main()
