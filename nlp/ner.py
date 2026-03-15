import spacy

def extract_entities(text, nlp_model):
    """Extract named entities from text"""
    doc = nlp_model(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'confidence': 0.95  # Placeholder confidence
        })
    
    return entities