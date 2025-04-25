"""
Tools for working with text segments in Vietnamese legal documents.
"""
import re
from typing import List, Dict, Any, Tuple, Optional

def split_text_into_segments(text: str, max_segment_length: int = 1000) -> List[str]:
    """
    Split text into segments at logical boundaries, keeping segments under max_segment_length.
    
    Args:
        text: Text to split
        max_segment_length: Maximum length for each segment
        
    Returns:
        List of text segments
    """
    if len(text) <= max_segment_length:
        return [text]
    
    # Try to split at section boundaries first
    segments = []
    
    # First, try to split at double newlines (paragraph boundaries)
    pattern = r'\n\n+'
    parts = re.split(pattern, text)
    
    current_segment = ""
    
    for part in parts:
        if len(current_segment + part) + 2 <= max_segment_length:  # +2 for the newlines
            if current_segment:
                current_segment += "\n\n" + part
            else:
                current_segment = part
        else:
            # If current segment is non-empty, add it to segments
            if current_segment:
                segments.append(current_segment)
            
            # If this part is shorter than max_segment_length, use it as the start of the next segment
            if len(part) <= max_segment_length:
                current_segment = part
            else:
                # If part is too long, split it at sentence boundaries
                sentences = split_into_sentences(part)
                sub_segments = split_sentences_into_segments(sentences, max_segment_length)
                segments.extend(sub_segments)
                current_segment = ""
    
    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)
    
    return segments

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Basic sentence splitting for Vietnamese text
    sentence_pattern = r'(?<=[.!?;])\s+'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def split_sentences_into_segments(sentences: List[str], max_segment_length: int) -> List[str]:
    """
    Group sentences into segments not exceeding max_segment_length.
    
    Args:
        sentences: List of sentences to group
        max_segment_length: Maximum length for each segment
        
    Returns:
        List of text segments
    """
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        if len(current_segment + sentence) + 1 <= max_segment_length:  # +1 for the space
            if current_segment:
                current_segment += " " + sentence
            else:
                current_segment = sentence
        else:
            # If current segment is non-empty, add it to segments
            if current_segment:
                segments.append(current_segment)
            
            # If this sentence is shorter than max_segment_length, use it as the start of the next segment
            if len(sentence) <= max_segment_length:
                current_segment = sentence
            else:
                # If sentence is too long, split it by words
                words = sentence.split()
                current_part = ""
                
                for word in words:
                    if len(current_part + word) + 1 <= max_segment_length:  # +1 for the space
                        if current_part:
                            current_part += " " + word
                        else:
                            current_part = word
                    else:
                        segments.append(current_part)
                        current_part = word
                
                if current_part:
                    current_segment = current_part
                else:
                    current_segment = ""
    
    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)
    
    return segments

def count_and_limit_tokens(text: str, max_tokens: int = 1000) -> Tuple[int, str]:
    """
    Rough estimate of token count and truncate text to max tokens if needed.
    
    Args:
        text: Text to process
        max_tokens: Maximum number of tokens allowed
        
    Returns:
        Tuple of (token_count, truncated_text)
    """
    # Very rough approximation: 1 token ~= 4 characters for Vietnamese
    token_count = len(text) // 4
    
    if token_count <= max_tokens:
        return token_count, text
    
    # Truncate text to approximate max_tokens
    truncated_text = text[:max_tokens * 4]
    
    # Try to find a sentence boundary to make the truncation cleaner
    last_period = truncated_text.rfind('.')
    last_question = truncated_text.rfind('?')
    last_exclamation = truncated_text.rfind('!')
    
    last_boundary = max(last_period, last_question, last_exclamation)
    
    if last_boundary > 0:
        truncated_text = truncated_text[:last_boundary + 1]
    
    return max_tokens, truncated_text