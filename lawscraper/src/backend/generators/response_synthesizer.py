"""
Response synthesizer for generating coherent responses.
"""
from typing import Any, Optional

from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

def create_response_synthesizer(
    llm: Any, 
    response_mode: str = "compact",
    text_qa_template: Optional[PromptTemplate] = None
) -> Any:
    """
    Create a response synthesizer with the given parameters.
    
    Args:
        llm: Language model to use
        response_mode: Mode for response synthesis
        text_qa_template: Template for text QA
        
    Returns:
        Response synthesizer
    """
    return get_response_synthesizer(
        response_mode=response_mode,
        llm=llm,
        text_qa_template=text_qa_template
    )