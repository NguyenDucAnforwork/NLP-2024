"""
Utility scripts for the law document scraper system.
"""
from .clean import normalize
from .parse_text_to_structure import parse_text_to_structure
from .work_segment import split_text_into_segments, split_into_sentences, count_and_limit_tokens

__all__ = [
    "normalize",
    "parse_text_to_structure",
    "split_text_into_segments",
    "split_into_sentences",
    "count_and_limit_tokens"
]