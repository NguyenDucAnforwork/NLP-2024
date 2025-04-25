"""
LLM-based generator implementation for the law document system.
"""
import os
from typing import List, Dict, Any, Optional, Tuple

from llama_index.llms.gemini import Gemini
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

from .base_generator import BaseGenerator
from .response_synthesizer import create_response_synthesizer

class LLMGenerator(BaseGenerator):
    """
    Generator that uses a language model to generate responses.
    """
    
    def __init__(
        self,
        model_name: str = "models/gemini-2.0-flash",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        response_mode: str = "compact"
    ):
        """
        Initialize the LLM generator.
        
        Args:
            model_name: Name of the language model to use
            temperature: Temperature for generation
            api_key: API key for the language model (defaults to environment variable)
            response_mode: Mode for response synthesis
        """
        super().__init__(temperature=temperature)
        
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Initialize LLM
        self.llm = Gemini(api_key=self.api_key, model=model_name, temperature=temperature)
        self.logger.info(f"Initialized Gemini LLM with model {model_name}")
        
        # Response mode
        self.response_mode = response_mode
    
    def generate_response(self, query_str: str, nodes: List[NodeWithScore]) -> str:
        """
        Generate a response based on the query and retrieved nodes.
        
        Args:
            query_str: Query string
            nodes: List of retrieved nodes
            
        Returns:
            Generated response
        """
        self.logger.info(f"Generating response for query: {query_str}")
        
        # Apply similarity postprocessing to get most relevant results
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        filtered_nodes = postprocessor.postprocess_nodes(nodes)
        self.logger.info(f"After postprocessing: {len(filtered_nodes)} nodes")
        
        # Create response synthesizer
        response_template = PromptTemplate("""
            Use only following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know and don't
            try to make up an answer.

            Context:
            {context_str}

            Question: {query_str}

            Answer in Vietnamese language:
        """)
        
        response_synthesizer = create_response_synthesizer(
            llm=self.llm,
            response_mode=self.response_mode,
            text_qa_template=response_template
        )
        
        # Generate response
        response = response_synthesizer.synthesize(query_str, filtered_nodes)
        
        return response