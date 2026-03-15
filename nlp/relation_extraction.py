import spacy

def extract_relations(text, entities, nlp_model):
    """Extract relationships between entities"""
    doc = nlp_model(text)
    relations = []
    
    # Simple pattern-based relation extraction
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass') and token.head.pos_ == 'VERB':
            subject = token.text
            verb = token.head.text
            for child in token.head.children:
                if child.dep_ in ('dobj', 'attr', 'prep'):
                    object_text = child.text
                    
                    # Find corresponding entity objects
                    entity1 = find_entity(subject, entities)
                    entity2 = find_entity(object_text, entities)
                    
                    if entity1 and entity2:
                        relations.append({
                            'entity1': entity1,
                            'entity2': entity2,
                            'type': verb,
                            'confidence': 0.85
                        })
    
    return relations

def find_entity(text, entities):
    """Find entity object by text"""
    for entity in entities:
        if entity.name.lower() in text.lower() or text.lower() in entity.name.lower():
            return entity
    return None