"""
Base retriever implementation for the system.
"""
import abc
import logging
from typing import List, Dict, Any, Optional

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever

class BaseRetriever(LlamaBaseRetriever):
    """
    Abstract base class for all retrievers in the system.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the base retriever.
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes given query.
        
        Args:
            query_bundle: Query bundle containing the query string
            
        Returns:
            List of retrieved nodes with scores
        """
        pass