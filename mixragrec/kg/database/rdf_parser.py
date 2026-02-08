"""
RDF/Turtle format parser for DBpedia dumps.
Parses triple statements (subject, predicate, object).
"""

import re
from typing import Tuple, Optional, Iterator, Set
from pathlib import Path
from tqdm import tqdm


class RDFParser:
    """"""
    
    TRIPLE_PATTERN = re.compile(
        r'<([^>]+)>\s+<([^>]+)>\s+(.+?)\s+\.',
        re.UNICODE
    )
    
    URI_PATTERN = re.compile(r'<([^>]+)>')
    
    LITERAL_PATTERN = re.compile(r'"(.+?)"(?:@([a-z]{2})|\\^\\^<([^>]+)>)?')
    
    def __init__(self):
        self.parsed_count = 0
        self.error_count = 0
        
    def parse_object(self, obj_str: str) -> Tuple[str, str]:
        """
        Args:
        Returns:
        """
        obj_str = obj_str.strip()
        
        uri_match = self.URI_PATTERN.match(obj_str)
        if uri_match:
            return uri_match.group(1), 'uri'
        
        literal_match = self.LITERAL_PATTERN.match(obj_str)
        if literal_match:
            value = literal_match.group(1)
            value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
            return value, 'literal'
        
        return obj_str, 'literal'
    
    def parse_line(self, line: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Args:
        Returns:
        """
        line = line.strip()
        
        if not line or line.startswith('#') or line.startswith('@'):
            return None
        
        match = self.TRIPLE_PATTERN.match(line)
        if not match:
            self.error_count += 1
            return None
        
        subject = match.group(1)
        predicate = match.group(2)
        object_str = match.group(3)
        
        obj_value, obj_type = self.parse_object(object_str)
        
        self.parsed_count += 1
        
        return subject, predicate, obj_value, obj_type
    
    def parse_file(self, file_path: Path, target_subjects: Optional[Set[str]] = None) -> Iterator[Tuple[str, str, str, str]]:
        """
        Args:
        Yields:
            (subject, predicate, object, object_type)
        """
        print(f"Parsing {file_path.name}...")
        
        file_size = file_path.stat().st_size
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f, \
             tqdm(total=file_size, unit='iB', unit_scale=True, desc="Parsing") as pbar:
            
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                
                parsed = self.parse_line(line)
                if parsed is None:
                    continue
                
                subject, predicate, obj_value, obj_type = parsed
                
                if target_subjects is None or subject in target_subjects:
                    yield subject, predicate, obj_value, obj_type
        
        print(f"✓ Parsed {self.parsed_count:,} triples ({self.error_count:,} errors)")
    
    def extract_entity_id(self, uri: str) -> str:
        """
        Args:
            uri: DBpedia URI (e.g., http://dbpedia.org/resource/Toy_Story)
        Returns:
        """
        if uri.startswith('http://dbpedia.org/resource/'):
            return uri.replace('http://dbpedia.org/resource/', '')
        elif uri.startswith('http://dbpedia.org/property/'):
            return uri.replace('http://dbpedia.org/property/', '')
        elif uri.startswith('http://dbpedia.org/ontology/'):
            return uri.replace('http://dbpedia.org/ontology/', '')
        else:
            return uri.split('/')[-1]
    
    def is_dbpedia_resource(self, uri: str) -> bool:
        """"""
        return uri.startswith('http://dbpedia.org/resource/')
    
    def get_property_name(self, predicate_uri: str) -> str:
        """
        Args:
        Returns:
        """
        if predicate_uri.startswith('http://dbpedia.org/ontology/'):
            return predicate_uri.replace('http://dbpedia.org/ontology/', 'dbo:')
        elif predicate_uri.startswith('http://dbpedia.org/property/'):
            return predicate_uri.replace('http://dbpedia.org/property/', 'dbp:')
        elif predicate_uri.startswith('http://www.w3.org/1999/02/22-rdf-syntax-ns#'):
            return predicate_uri.replace('http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'rdf:')
        elif predicate_uri.startswith('http://www.w3.org/2000/01/rdf-schema#'):
            return predicate_uri.replace('http://www.w3.org/2000/01/rdf-schema#', 'rdfs:')
        else:
            return predicate_uri.split('/')[-1]


def test_parser():
    """"""
    parser = RDFParser()
    
    test_lines = [
        '<http://dbpedia.org/resource/Toy_Story> <http://dbpedia.org/ontology/director> <http://dbpedia.org/resource/John_Lasseter> .',
        '<http://dbpedia.org/resource/Toy_Story> <http://www.w3.org/2000/01/rdf-schema#label> "Toy Story"@en .',
        '<http://dbpedia.org/resource/Toy_Story> <http://dbpedia.org/property/budget> "30000000"^^<http://www.w3.org/2001/XMLSchema#integer> .',
    ]
    
    print("Testing RDF Parser:")
    print("-" * 60)
    
    for line in test_lines:
        result = parser.parse_line(line)
        if result:
            subject, predicate, obj, obj_type = result
            print(f"Subject: {parser.extract_entity_id(subject)}")
            print(f"Predicate: {parser.get_property_name(predicate)}")
            print(f"Object: {obj} (type: {obj_type})")
            print()


if __name__ == "__main__":
    test_parser()
