"""
Text generator for entities and triples.
Converts KG elements into natural language text for indexing and retrieval.
"""

from typing import Dict, Any, List, Tuple


class EntityTextGenerator:
    """"""
    
    @staticmethod
    def entity_to_text(entity: Dict[str, Any]) -> str:
        """
        Args:
        Returns:
        """
        entity_id = entity['entity_id']
        entity_type = entity['entity_type']
        name = entity['name']
        attributes = entity.get('attributes', {})
        
        text_parts = [f"{entity_type}: {name}"]
        
        if 'rdfs:comment' in attributes:
            comment = attributes['rdfs:comment']
            if len(comment) > 200:
                comment = comment[:200] + "..."
            text_parts.append(f"Description: {comment}")
        
        important_attrs = ['dbo:abstract', 'abstract', 'description']
        for attr in important_attrs:
            if attr in attributes and attr != 'rdfs:comment':
                value = str(attributes[attr])
                if len(value) > 150:
                    value = value[:150] + "..."
                text_parts.append(f"{attr}: {value}")
                break
        
        text = ". ".join(text_parts)
        
        return text
    
    @staticmethod
    def entity_to_short_text(entity: Dict[str, Any]) -> str:
        """
        Args:
        Returns:
        """
        entity_type = entity['entity_type']
        name = entity['name']
        return f"{entity_type}: {name}"


class TripleTextGenerator:
    """"""
    
    RELATION_TEMPLATES = {
        'dbo:director': '{head} is directed by {tail}',
        'dbo:starring': '{head} stars {tail}',
        'dbo:producer': '{head} is produced by {tail}',
        'dbo:writer': '{head} is written by {tail}',
        'dbo:musicComposer': '{head} has music composed by {tail}',
        'dbo:cinematography': '{head} has cinematography by {tail}',
        'dbo:editing': '{head} is edited by {tail}',
        'dbo:genre': '{head} belongs to genre {tail}',
        'dbo:distributor': '{head} is distributed by {tail}',
        'dbo:productionCompany': '{head} is produced by company {tail}',
        'dbo:country': '{head} is from country {tail}',
        'dbo:language': '{head} is in language {tail}',
        'dbo:releaseDate': '{head} was released on {tail}',
        'dbo:budget': '{head} has budget {tail}',
        'dbo:gross': '{head} grossed {tail}',
        'dbo:runtime': '{head} has runtime {tail}',
        'dbo:birthPlace': '{head} was born in {tail}',
        'dbo:occupation': '{head} works as {tail}',
        'dbo:spouse': '{head} is married to {tail}',
        'dbo:award': '{head} won award {tail}',
    }
    
    @staticmethod
    def relation_to_natural_language(relation_type: str) -> str:
        """
        Args:
        Returns:
        """
        if ':' in relation_type:
            rel = relation_type.split(':')[1]
        else:
            rel = relation_type
        
        import re
        natural = re.sub(r'([A-Z])', r' \1', rel).strip().lower()
        
        return natural
    
    @staticmethod
    def triple_to_text(head_entity: Dict[str, Any], 
                      relation_type: str,
                      tail_entity: Dict[str, Any]) -> str:
        """
        Args:
        Returns:
        """
        head_name = head_entity['name']
        tail_name = tail_entity['name']
        
        if relation_type in TripleTextGenerator.RELATION_TEMPLATES:
            template = TripleTextGenerator.RELATION_TEMPLATES[relation_type]
            text = template.format(head=head_name, tail=tail_name)
        else:
            natural_rel = TripleTextGenerator.relation_to_natural_language(relation_type)
            text = f"{head_name} {natural_rel} {tail_name}"
        
        # text += f" [Head: {head_entity['entity_type']}, Tail: {tail_entity['entity_type']}]"
        
        return text
    
    @staticmethod
    def triple_to_structured_text(head_entity: Dict[str, Any],
                                  relation_type: str,
                                  tail_entity: Dict[str, Any]) -> str:
        """
        Args:
        Returns:
        """
        basic_text = TripleTextGenerator.triple_to_text(head_entity, relation_type, tail_entity)
        
        type_info = f"[{head_entity['entity_type']} -> {tail_entity['entity_type']}]"
        
        context = ""
        if 'rdfs:comment' in head_entity.get('attributes', {}):
            head_desc = head_entity['attributes']['rdfs:comment'][:100]
            context = f" Context: {head_desc}..."
        
        return f"{basic_text} {type_info}{context}"


class SubgraphTextGenerator:
    """"""
    
    @staticmethod
    def subgraph_to_text(subgraph: Dict[str, Any], 
                        center_entity_id: str = None,
                        include_attributes: bool = True) -> str:
        """
        Args:
        Returns:
        """
        entities = subgraph.get('entities', {})
        relations = subgraph.get('relations', [])
        
        text_parts = []
        
        if center_entity_id and center_entity_id in entities:
            center = entities[center_entity_id]
            text_parts.append(f"Main entity: {EntityTextGenerator.entity_to_text(center)}")
        
        text_parts.append(f"\nRelationships ({len(relations)} total):")
        
        relations_by_type = {}
        for rel in relations:
            rel_type = rel['relation']
            if rel_type not in relations_by_type:
                relations_by_type[rel_type] = []
            relations_by_type[rel_type].append(rel)
        
        for rel_type, rels in sorted(relations_by_type.items())[:10]:
            text_parts.append(f"\n{rel_type}:")
            for rel in rels[:5]:
                head = entities.get(rel['head'])
                tail = entities.get(rel['tail'])
                if head and tail:
                    triple_text = TripleTextGenerator.triple_to_text(head, rel_type, tail)
                    text_parts.append(f"  - {triple_text}")
        
        if include_attributes:
            text_parts.append(f"\nEntities in subgraph ({len(entities)} total):")
            
            entities_by_type = {}
            for entity in entities.values():
                etype = entity['entity_type']
                if etype not in entities_by_type:
                    entities_by_type[etype] = []
                entities_by_type[etype].append(entity)
            
            for etype, ents in sorted(entities_by_type.items()):
                names = [e['name'] for e in ents[:10]]
                text_parts.append(f"  {etype}: {', '.join(names)}")
                if len(ents) > 10:
                    text_parts.append(f"    ... and {len(ents) - 10} more")
        
        return "\n".join(text_parts)


def test_text_generation():
    """"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from database.kg_database import KGDatabase
    
    print("="*70)
    print("Testing Text Generation")
    print("="*70)
    
    with KGDatabase() as db:
        entity = db.get_entity('Toy_Story')
        if entity:
            print("\n1. Entity to Text:")
            print("-" * 70)
            text = EntityTextGenerator.entity_to_text(entity)
            print(text)
            
            short_text = EntityTextGenerator.entity_to_short_text(entity)
            print(f"\nShort version: {short_text}")
        
        print("\n\n2. Triple to Text:")
        print("-" * 70)
        
        neighbors = db.get_neighbors('Toy_Story')
        if neighbors:
            neighbor = neighbors[0]
            head_entity = db.get_entity('Toy_Story')
            tail_entity = db.get_entity(neighbor['entity_id'])
            relation_type = neighbor['relation_type']
            
            if head_entity and tail_entity:
                triple_text = TripleTextGenerator.triple_to_text(
                    head_entity, relation_type, tail_entity
                )
                print(f"Basic: {triple_text}")
                
                structured_text = TripleTextGenerator.triple_to_structured_text(
                    head_entity, relation_type, tail_entity
                )
                print(f"\nStructured: {structured_text}")
        
        print("\n\n3. Subgraph to Text:")
        print("-" * 70)
        
        subgraph = db.get_2hop_subgraph('Toy_Story', max_neighbors_per_hop=50)
        subgraph_text = SubgraphTextGenerator.subgraph_to_text(
            subgraph, center_entity_id='Toy_Story'
        )
        print(subgraph_text[:1000])
        if len(subgraph_text) > 1000:
            print(f"\n... (total length: {len(subgraph_text)} characters)")
    
    print("\n" + "="*70)
    print("✓ Text generation test complete!")
    print("="*70)


if __name__ == "__main__":
    test_text_generation()
