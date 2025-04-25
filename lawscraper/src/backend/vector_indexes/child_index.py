"""
Child index implementation for hierarchical indexing.
"""
from typing import List, Dict, Tuple, Any, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import TextNode

from .base_index import BaseVectorIndex

class ChildVectorIndex(BaseVectorIndex):
    """
    Child index implementation for the parent-child indexing strategy.
    """
    
    def __init__(
        self,
        collection_name: str = "vietnamese_legal_child",
        embed_model_path: str = "./data/embeddings/bge-m3",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize the child vector index.
        
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
    
    def create_child_index(
        self,
        documents: List[Document],
        parent_nodes: List[TextNode],
        chunk_size: int = 512,
        chunk_overlap: int = 100
    ) -> Tuple[VectorStoreIndex, List[TextNode]]:
        """
        Create a child index from documents.
        
        Args:
            documents: List of documents to index
            parent_nodes: List of parent nodes (for reference)
            chunk_size: Size of child chunks
            chunk_overlap: Overlap between child chunks
            
        Returns:
            Tuple of (child_index, child_nodes)
        """
        # Create hierarchical node parser
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, chunk_size],  # First level is parent size
            chunk_overlaps=[200, chunk_overlap]  # First level is parent overlap
        )
        
        # Parse nodes
        all_nodes = node_parser.get_nodes_from_documents(documents)
        
        # Filter child nodes (level 1)
        child_nodes = [n for n in all_nodes if n.level == 1]
        
        self.logger.info(f"Created {len(child_nodes)} child nodes with chunk size {chunk_size}")
        
        # Establish parent-child relationships
        parent_map = {node.node_id: node for node in parent_nodes}
        
        for child in child_nodes:
            # Find parent based on text content overlap
            for parent_id, parent in parent_map.items():
                if child.text in parent.text:
                    child.parent_node = {"node_id": parent_id}
                    break
        
        # Create index
        child_index = self.create_index_from_documents(child_nodes)
        
        return child_index, child_nodes