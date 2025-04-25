"""
BM25 retriever implementation for sparse retrieval.
"""
from typing import List, Dict, Any, Optional

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever

from .base_retriever import BaseRetriever

class BM25Retriever(BaseRetriever):
    """
    Wrapper for LlamaIndex's BM25Retriever.
    """
    
    def __init__(self, docstore, similarity_top_k: int = 5, **kwargs):
        """
        Initialize the BM25 retriever.
        
        Args:
            docstore: Document store containing the documents
            similarity_top_k: Number of top results to return
        """
        super().__init__(**kwargs)
        self._bm25_retriever = LlamaBM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=similarity_top_k
        )
        self._similarity_top_k = similarity_top_k
        
        self.logger.info(f"Created BM25 retriever with top_k={similarity_top_k}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using BM25 algorithm.
        
        Args:
            query_bundle: The query bundle containing the query string
            
        Returns:
            List of nodes with similarity scores
        """
        self.logger.info(f"BM25 retrieval for query: {query_bundle.query_str}")
        
        try:
            nodes = self._bm25_retriever.retrieve(query_bundle.query_str)
            self.logger.info(f"Retrieved {len(nodes)} nodes with BM25")
            return nodes
        except Exception as e:
            self.logger.error(f"Error in BM25 retrieval: {e}")
            return []