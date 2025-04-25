"""
Base generator implementation for the law document system.
"""
import abc
import logging
from typing import List, Dict, Any, Optional, Tuple

from llama_index.core.schema import NodeWithScore

class BaseGenerator(abc.ABC):
    """
    Abstract base class for all generators in the system.
    """
    
    def __init__(self, temperature: float = 0.7):
        """
        Initialize the base generator.
        
        Args:
            temperature: Temperature for generation
        """
        self.temperature = temperature
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    def generate_response(self, query_str: str, nodes: List[NodeWithScore]) -> str:
        """
        Generate a response based on the query and retrieved nodes.
        
        Args:
            query_str: Query string
            nodes: List of retrieved nodes
            
        Returns:
            Generated response
        """
        pass
    
    def filter_nodes(self, nodes: List[NodeWithScore], similarity_cutoff: float = 0.7) -> List[NodeWithScore]:
        """
        Filter nodes based on similarity score.
        
        Args:
            nodes: List of nodes to filter
            similarity_cutoff: Minimum similarity score to keep
            
        Returns:
            Filtered list of nodes
        """
        filtered_nodes = [node for node in nodes if node.score >= similarity_cutoff]
        
        self.logger.info(f"Filtered {len(nodes)} nodes to {len(filtered_nodes)} nodes")
        
        return filtered_nodes
    
    def rerank_nodes(self, query_str: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Rerank nodes based on relevance to the query.
        
        Args:
            query_str: Query string
            nodes: List of nodes to rerank
            
        Returns:
            Reranked list of nodes
        """
        # Default implementation just sorts by score
        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        return sorted_nodes