from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def initialize_encoder():
    """Initialize the sentence transformer model"""
    try:
        # Try with smaller model first
        return SentenceTransformer('paraphrase-MiniLM-L3-v2')
    except:
        # Fallback to even smaller model
        return SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, entities, relations, encoder):
    """Perform semantic search on entities and relations"""
    try:
        # Encode query
        query_embedding = encoder.encode([query])
        
        # Prepare texts for comparison
        entity_texts = [f"{e.name} ({e.type})" for e in entities]
        relation_texts = []
        relation_objects = []
        
        for r in relations:
            entity1 = next((e for e in entities if e.id == r.entity1_id), None)
            entity2 = next((e for e in entities if e.id == r.entity2_id), None)
            if entity1 and entity2:
                text = f"{entity1.name} {r.relation_type} {entity2.name}"
                relation_texts.append(text)
                relation_objects.append(r)
        
        # Encode all texts
        all_texts = entity_texts + relation_texts
        if not all_texts:
            return {'entities': [], 'relations': []}
        
        embeddings = encoder.encode(all_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top results
        entity_count = len(entities)
        entity_indices = similarities[:entity_count]
        relation_indices = similarities[entity_count:]
        
        # Sort results
        entity_results = []
        for i, score in enumerate(entity_indices):
            if score > 0.3:  # Threshold
                entity_results.append({
                    'entity': {
                        'id': entities[i].id,
                        'name': entities[i].name,
                        'type': entities[i].type
                    },
                    'score': float(score)
                })
        
        relation_results = []
        for i, score in enumerate(relation_indices):
            if score > 0.3 and i < len(relation_objects):
                relation_results.append({
                    'relation': {
                        'id': relation_objects[i].id,
                        'type': relation_objects[i].relation_type,
                        'entity1': next((e.name for e in entities if e.id == relation_objects[i].entity1_id), ''),
                        'entity2': next((e.name for e in entities if e.id == relation_objects[i].entity2_id), '')
                    },
                    'score': float(score)
                })
        
        # Sort by score
        entity_results.sort(key=lambda x: x['score'], reverse=True)
        relation_results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'entities': entity_results[:5],
            'relations': relation_results[:5]
        }
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return {'entities': [], 'relations': []}