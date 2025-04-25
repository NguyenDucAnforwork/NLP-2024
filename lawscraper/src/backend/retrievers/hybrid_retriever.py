"""
Hybrid retriever implementation that combines vector and BM25 retrieval.
"""
from typing import List, Dict, Any, Optional

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import BaseDocumentStore

from .base_retriever import BaseRetriever

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines vector similarity with other relevance signals.
    """
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        docstore: BaseDocumentStore,
        similarity_top_k: int = 2,
        out_top_k: Optional[int] = None,
        alpha: float = 0.5,
        **kwargs
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_index: The vector index to use for similarity search
            docstore: The document store containing the documents
            similarity_top_k: Number of top results to consider from vector search
            out_top_k: Number of final results to return (defaults to similarity_top_k)
            alpha: Weight between node similarity (alpha) and doc similarity (1-alpha)
        """
        super().__init__(**kwargs)
        self._vector_index = vector_index
        self._embed_model = vector_index._embed_model
        self._retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
        self._out_top_k = out_top_k or similarity_top_k
        self._docstore = docstore
        self._alpha = alpha
        
        self.logger.info(f"Created hybrid retriever with alpha={alpha}, top_k={similarity_top_k}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes given query by combining vector search with document similarity.
        
        Args:
            query_bundle: The query bundle containing the query string
            
        Returns:
            List of nodes with hybrid similarity scores
        """
        self.logger.info(f"Hybrid retrieval for query: {query_bundle.query_str}")
        
        # First retrieve chunks using vector similarity
        nodes = self._retriever.retrieve(query_bundle.query_str)
        self.logger.info(f"Retrieved {len(nodes)} nodes from vector search")
        
        # Get documents and embedding similarity between query and documents
        try:
            # Get doc embeddings
            docs = [self._docstore.get_document(n.node.index_id) for n in nodes]
            doc_embeddings = [d.embedding for d in docs if d.embedding is not None]
            
            if not doc_embeddings:
                self.logger.warning("No document embeddings found, falling back to vector search only")
                return nodes[:self._out_top_k]
            
            query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
            
            # Compute doc similarities
            doc_similarities, doc_idxs = get_top_k_embeddings(
                query_embedding, doc_embeddings, similarity_top_k=len(doc_embeddings)
            )
            
            # Compute final similarity with doc similarities and original node similarity
            result_tups = []
            for doc_idx, doc_similarity in zip(doc_idxs, doc_similarities):
                node = nodes[doc_idx]
                # Weight alpha * node similarity + (1-alpha) * doc similarity
                full_similarity = (self._alpha * node.score) + ((1 - self._alpha) * doc_similarity)
                
                self.logger.debug(
                    f"Doc {doc_idx} (node score: {node.score}, doc similarity: {doc_similarity}, "
                    f"full similarity: {full_similarity})"
                )
                result_tups.append((full_similarity, node))
            
            result_tups = sorted(result_tups, key=lambda x: x[0], reverse=True)
            
            # Update scores
            for full_score, node in result_tups:
                node.score = full_score
            
            return [n for _, n in result_tups][:self._out_top_k]
            
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {e}")
            self.logger.warning("Falling back to vector search only")
            return nodes[:self._out_top_k]