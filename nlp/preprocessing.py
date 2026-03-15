import re
import spacy

def preprocess_text(text):
    """Clean and preprocess text"""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text.strip()