"""
Parent index implementation for hierarchical indexing.
"""
from typing import List, Dict, Tuple, Any, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import TextNode

from .base_index import BaseVectorIndex

class ParentVectorIndex(BaseVectorIndex):
    """
    Parent index implementation for the parent-child indexing strategy.
    """
    
    def __init__(
        self,
        collection_name: str = "vietnamese_legal_parent",
        embed_model_path: str = "./data/embeddings/bge-m3",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize the parent vector index.
        
        Args:
            collection_name: Name of the collection in Qdrant
            embed_model_path: Path to the embedding model
            qdrant_url: URL for Qdrant server
            qdrant_api_key: API key for Qdrant server
        """
        super().__init__(
            collection_name=collection_name,
            embed_model_path=embed_model_path,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
    
    def create_parent_index(
        self,
        documents: List[Document],
        chunk_size: int = 2048,
        chunk_overlap: int = 200
    ) -> Tuple[VectorStoreIndex, List[TextNode]]:
        """
        Create a parent index from documents.
        
        Args:
            documents: List of documents to index
            chunk_size: Size of parent chunks
            chunk_overlap: Overlap between parent chunks
            
        Returns:
            Tuple of (parent_index, parent_nodes)
        """
        # Create hierarchical node parser
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[chunk_size],
            chunk_overlaps=[chunk_overlap]
        )
        
        # Parse nodes
        all_nodes = node_parser.get_nodes_from_documents(documents)
        
        # Filter parent nodes (level 0)
        parent_nodes = [n for n in all_nodes if n.level == 0]
        
        self.logger.info(f"Created {len(parent_nodes)} parent nodes with chunk size {chunk_size}")
        
        # Create index
        parent_index = self.create_index_from_documents(parent_nodes)
        
        return parent_index, parent_nodes