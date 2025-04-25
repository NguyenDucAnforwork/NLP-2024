"""
Text cleaning utilities for Vietnamese legal documents.
"""
import re

def normalize(text: str) -> str:
    """
    Normalize Vietnamese legal text.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize Vietnamese characters
    mapping = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
        'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
        'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ',
    }
    
    for old, new in mapping.items():
        text = text.replace(old, new)
    
    # Replace special quotes with standard quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Replace various dashes with standard dash
    text = text.replace('–', '-').replace('—', '-')
    
    # Normalize spaces after punctuation
    text = re.sub(r'([.,;:!?)])(\S)', r'\1 \2', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text