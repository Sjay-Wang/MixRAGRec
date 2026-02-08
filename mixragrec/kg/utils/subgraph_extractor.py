"""
Subgraph extractor for building 2-hop movie knowledge graphs from DBpedia.
"""

import csv
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json

from ..database.rdf_parser import RDFParser
from ..database.kg_database import KGDatabase


class SubgraphExtractor:
    """"""
    
    IMPORTANT_RELATIONS = {
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'http://www.w3.org/2000/01/rdf-schema#label',
        'http://dbpedia.org/ontology/abstract',
        
        'http://dbpedia.org/ontology/director',
        'http://dbpedia.org/ontology/starring',
        'http://dbpedia.org/ontology/producer',
        'http://dbpedia.org/ontology/writer',
        'http://dbpedia.org/ontology/musicComposer',
        'http://dbpedia.org/ontology/cinematography',
        'http://dbpedia.org/ontology/editing',
        
        'http://dbpedia.org/ontology/genre',
        'http://dbpedia.org/ontology/distributor',
        'http://dbpedia.org/ontology/productionCompany',
        'http://dbpedia.org/property/studio',
        
        'http://dbpedia.org/ontology/releaseDate',
        'http://dbpedia.org/ontology/country',
        'http://dbpedia.org/ontology/language',
        
        'http://dbpedia.org/ontology/sequel',
        'http://dbpedia.org/ontology/prequel',
        'http://dbpedia.org/property/basedOn',
        
        'http://dbpedia.org/ontology/budget',
        'http://dbpedia.org/ontology/gross',
        'http://dbpedia.org/ontology/runtime',
    }
    
    def __init__(self, 
                 dump_dir: str = "dataset/dbpedia_dumps",
                 db_path: str = "parsed_kg_from_dump.db"):
        """
        Args:
        """
        self.dump_dir = Path(dump_dir)
        self.db_path = db_path
        self.parser = RDFParser()
        
        self.movie_uris = set()
        self.movie_id_map = {}   # ML1M ID -> DBpedia URI
        self.entities = {}       # entity_id -> entity_data
        self.relations = []      # list of (head, relation, tail)
        
    def load_movie_mappings(self, map_file: str = "dataset/ML1M/map.csv", limit: int = None) -> Set[str]:
        """
        Args:
        Returns:
        """
        movie_uris = set()
        
        print(f"Loading mappings from {map_file}")
        
        delimiter = '\t' if map_file.endswith('.tsv') else ','
        
        with open(map_file, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            headers = header_line.split(delimiter)
            
            # - item_id::string, URI::string (MovieLens CSV)
            # - artist_id, dbpedia_url (Last.FM TSV)
            id_col = None
            uri_col = None
            
            for i, h in enumerate(headers):
                h_lower = h.lower().strip()
                if 'id' in h_lower and id_col is None:
                    id_col = i
                if 'uri' in h_lower or 'url' in h_lower or 'dbpedia' in h_lower:
                    uri_col = i
            
            if id_col is None or uri_col is None:
                raise ValueError(f"Cannot detect ID and URI columns in {map_file}. Headers: {headers}")
            
            print(f"  Detected format: delimiter='{delimiter}', id_col={headers[id_col]}, uri_col={headers[uri_col]}")
            
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(delimiter)
                if len(parts) <= max(id_col, uri_col):
                    continue
                
                item_id = parts[id_col].strip()
                uri = parts[uri_col].strip()
                
                if uri and uri.startswith('http'):
                    movie_uris.add(uri)
                    self.movie_id_map[item_id] = uri
        
        self.movie_uris = movie_uris
        entity_type = 'artists' if 'lfm' in map_file.lower() or 'lastfm' in map_file.lower() else 'movies'
        print(f"✓ Loaded {len(movie_uris)} {entity_type}")
        
        return movie_uris
    
    def extract_0hop_entities(self, dump_files: Dict[str, Path]):
        """
        Args:
        """
        print("\n" + "="*60)
        print("Phase 1: Extracting 0-hop entities (movies)")
        print("="*60)
        
        for dataset_name, file_path in dump_files.items():
            if not file_path.exists():
                print(f"⚠ File not found: {file_path}")
                continue
            
            print(f"\nProcessing {dataset_name}...")
            
            for subject, predicate, obj, obj_type in self.parser.parse_file(file_path, self.movie_uris):
                entity_id = self.parser.extract_entity_id(subject)
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = {
                        'entity_id': entity_id,
                        'entity_type': 'Movie',
                        'name': entity_id.replace('_', ' '),
                        'attributes': {},
                        'aliases': []
                    }
                
                prop_name = self.parser.get_property_name(predicate)
                
                if 'label' in predicate.lower() and obj_type == 'literal':
                    self.entities[entity_id]['name'] = obj
                
                elif 'abstract' in predicate.lower() and obj_type == 'literal':
                    self.entities[entity_id]['attributes']['abstract'] = obj[:500]
                
                elif obj_type == 'literal':
                    self.entities[entity_id]['attributes'][prop_name] = obj
                
                elif obj_type == 'uri' and self.parser.is_dbpedia_resource(obj):
                    tail_id = self.parser.extract_entity_id(obj)
                    self.relations.append({
                        'head': entity_id,
                        'relation': prop_name,
                        'tail': tail_id,
                        'obj_type': obj_type
                    })
        
        print(f"\n✓ Extracted {len(self.entities)} movie entities")
        print(f"✓ Extracted {len(self.relations)} relations from movies")
    
    def extract_1hop_neighbors(self, dump_files: Dict[str, Path]):
        """
        Args:
        """
        print("\n" + "="*60)
        print("Phase 2: Extracting 1-hop neighbors")
        print("="*60)
        
        hop1_uris = set()
        for relation in self.relations:
            hop1_uris.add(relation['tail'])
        
        hop1_uri_full = set()
        for uri in hop1_uris:
            hop1_uri_full.add(f"http://dbpedia.org/resource/{uri}")
        
        print(f"Found {len(hop1_uris)} unique 1-hop entities")
        
        for dataset_name, file_path in dump_files.items():
            if not file_path.exists():
                continue
            
            print(f"\nProcessing {dataset_name} for 1-hop entities...")
            
            for subject, predicate, obj, obj_type in self.parser.parse_file(file_path, hop1_uri_full):
                entity_id = self.parser.extract_entity_id(subject)
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = {
                        'entity_id': entity_id,
                        'entity_type': 'Resource',
                        'name': entity_id.replace('_', ' '),
                        'attributes': {},
                        'aliases': []
                    }
                
                prop_name = self.parser.get_property_name(predicate)
                
                if 'type' in predicate.lower() and obj_type == 'uri':
                    type_name = self.parser.extract_entity_id(obj)
                    if 'Person' in type_name or 'Artist' in type_name:
                        self.entities[entity_id]['entity_type'] = 'Person'
                    elif 'Genre' in type_name:
                        self.entities[entity_id]['entity_type'] = 'Genre'
                    elif 'Company' in type_name or 'Organisation' in type_name:
                        self.entities[entity_id]['entity_type'] = 'Organization'
                
                if 'label' in predicate.lower() and obj_type == 'literal':
                    self.entities[entity_id]['name'] = obj
                
                elif 'abstract' in predicate.lower() and obj_type == 'literal':
                    self.entities[entity_id]['attributes']['abstract'] = obj[:500]
                
                elif obj_type == 'literal':
                    self.entities[entity_id]['attributes'][prop_name] = obj
                
                elif obj_type == 'uri' and self.parser.is_dbpedia_resource(obj):
                    tail_id = self.parser.extract_entity_id(obj)
                    self.relations.append({
                        'head': entity_id,
                        'relation': prop_name,
                        'tail': tail_id,
                        'obj_type': obj_type
                    })
        
        print(f"\n✓ Total entities: {len(self.entities)}")
        print(f"✓ Total relations: {len(self.relations)}")
    
    def extract_2hop_neighbors(self, dump_files: Dict[str, Path], max_2hop: int = 10000):
        """
        Args:
        """
        print("\n" + "="*60)
        print("Phase 3: Extracting 2-hop neighbors")
        print("="*60)
        
        hop2_uris = set()
        for relation in self.relations:
            tail_id = relation['tail']
            if tail_id not in self.entities:
                hop2_uris.add(tail_id)
        
        hop2_uris = set(list(hop2_uris)[:max_2hop])
        
        hop2_uri_full = set()
        for uri in hop2_uris:
            hop2_uri_full.add(f"http://dbpedia.org/resource/{uri}")
        
        print(f"Extracting up to {len(hop2_uris)} 2-hop entities")
        
        for dataset_name, file_path in [('labels', dump_files.get('labels')), 
                                        ('instance_types', dump_files.get('instance_types'))]:
            if not file_path or not file_path.exists():
                continue
            
            print(f"\nProcessing {dataset_name} for 2-hop entities...")
            
            for subject, predicate, obj, obj_type in self.parser.parse_file(file_path, hop2_uri_full):
                entity_id = self.parser.extract_entity_id(subject)
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = {
                        'entity_id': entity_id,
                        'entity_type': 'Resource',
                        'name': entity_id.replace('_', ' '),
                        'attributes': {},
                        'aliases': []
                    }
                
                if 'label' in predicate.lower() and obj_type == 'literal':
                    self.entities[entity_id]['name'] = obj
                elif 'type' in predicate.lower() and obj_type == 'uri':
                    type_name = self.parser.extract_entity_id(obj)
                    if 'Person' in type_name:
                        self.entities[entity_id]['entity_type'] = 'Person'
                    elif 'Place' in type_name or 'Location' in type_name:
                        self.entities[entity_id]['entity_type'] = 'Place'
        
        print(f"\n✓ Final entity count: {len(self.entities)}")
        print(f"✓ Final relation count: {len(self.relations)}")
    
    def save_to_database(self):
        """"""
        print("\n" + "="*60)
        print("Saving to database")
        print("="*60)
        
        with KGDatabase(self.db_path) as db:
            db.initialize_schema()
            db.clear_all()
            
            print("Inserting entities...")
            entity_tuples = []
            for entity_id, entity_data in tqdm(self.entities.items(), desc="Preparing entities"):
                entity_tuples.append((
                    entity_id,
                    entity_data['entity_type'],
                    entity_data['name'],
                    json.dumps(entity_data['attributes']),
                    json.dumps(entity_data.get('aliases', []))
                ))
            
            db.batch_insert_entities(entity_tuples)
            db.conn.commit()
            print(f"✓ Inserted {len(entity_tuples)} entities")
            
            print("Inserting relations...")
            relation_tuples = []
            for relation in tqdm(self.relations, desc="Preparing relations"):
                head = relation['head']
                rel_type = relation['relation']
                tail = relation['tail']
                
                if head in self.entities and tail in self.entities:
                    relation_id = f"{head}_{rel_type}_{tail}"
                    relation_tuples.append((
                        relation_id,
                        rel_type,
                        head,
                        tail,
                        json.dumps({}),
                        1.0
                    ))
            
            db.batch_insert_relations(relation_tuples)
            db.conn.commit()
            print(f"✓ Inserted {len(relation_tuples)} relations")
            
            db.print_statistics()
    
    def extract_subgraph(self, 
                        movie_limit: int = None,
                        save_to_db: bool = True,
                        map_file: str = "dataset/ML1M/map.csv") -> Dict[str, Any]:
        """
        Args:
        Returns:
        """
        self.load_movie_mappings(map_file=map_file, limit=movie_limit)
        
        dump_files = {}
        for dataset_name, filename in [
            ('mappingbased_objects', 'mappingbased_objects_en.ttl'),
            ('instance_types', 'instance_types_en.ttl'),
            ('labels', 'labels_en.ttl'),
            ('short_abstracts', 'short_abstracts_en.ttl'),
        ]:
            file_path = self.dump_dir / filename
            if file_path.exists():
                dump_files[dataset_name] = file_path
            else:
                print(f"⚠ Warning: {filename} not found")
        
        if not dump_files:
            print("✗ Error: No dump files found!")
            return {}
        
        self.extract_0hop_entities(dump_files)
        
        self.extract_1hop_neighbors(dump_files)
        
        self.extract_2hop_neighbors(dump_files)
        
        if save_to_db:
            self.save_to_database()
        
        return {
            'entities': self.entities,
            'relations': self.relations,
            'movie_count': len(self.movie_uris)
        }


def main():
    """"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract DBpedia subgraph for MovieLens")
    parser.add_argument('--limit', type=int, default=10, help='Number of movies to process (default: 10 for testing)')
    parser.add_argument('--full', action='store_true', help='Process all movies')
    parser.add_argument('--dump-dir', type=str, default='dataset/dbpedia_dumps', help='DBpedia dumps directory')
    parser.add_argument('--db-path', type=str, default='parsed_kg_from_dump.db', help='Output database path')
    
    args = parser.parse_args()
    
    movie_limit = None if args.full else args.limit
    
    print("="*60)
    print("DBpedia Subgraph Extractor for MovieLens 1M")
    print("="*60)
    print(f"Mode: {'Full dataset' if args.full else f'Test with {args.limit} movies'}")
    print(f"Dump directory: {args.dump_dir}")
    print(f"Database: {args.db_path}")
    print("="*60)
    
    extractor = SubgraphExtractor(dump_dir=args.dump_dir, db_path=args.db_path)
    result = extractor.extract_subgraph(movie_limit=movie_limit)
    
    print("\n" + "="*60)
    print("Extraction Complete!")
    print("="*60)
    print(f"Movies processed: {result.get('movie_count', 0)}")
    print(f"Total entities: {len(result.get('entities', {}))}")
    print(f"Total relations: {len(result.get('relations', []))}")
    print("="*60)


if __name__ == "__main__":
    main()
