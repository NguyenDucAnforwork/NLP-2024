"""
Base class for vector index implementations.
"""
import logging
import os
from typing import List, Dict, Tuple, Any, Optional

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client

class BaseVectorIndex:
    """
    Base class for vector index implementations.
    """
    
    def __init__(
        self,
        collection_name: str,
        embed_model_path: str = "./data/embeddings/bge-m3",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize the base vector index.
        
        Args:
            collection_name: Name of the collection in Qdrant
            embed_model_path: Path to the embedding model
            qdrant_url: URL for Qdrant server (defaults to environment variable)
            qdrant_api_key: API key for Qdrant server (defaults to environment variable)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_path)
        self.logger.info(f"Initialized embedding model from {embed_model_path}")
        
        # Get Qdrant credentials
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")
        
        # Initialize Qdrant client
        self.client = qdrant_client.QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
    def create_index_from_documents(
        self,
        documents: List[Document],
        chunk_size: int = 512,
        chunk_overlap: int = 100
    ) -> VectorStoreIndex:
        """
        Create a vector index from documents.
        
        Args:
            documents: List of documents to index
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            VectorStoreIndex created from the documents
        """
        # Create vector store
        vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        index = VectorStoreIndex(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        self.logger.info(f"Created index with {len(documents)} documents in collection {self.collection_name}")
        return index
    
    def load_index_from_cloud(self) -> Tuple[VectorStoreIndex, List[TextNode]]:
        """
        Load index from Qdrant Cloud.
        
        Returns:
            Tuple of (index, nodes)
        """
        # Get points from Qdrant
        points = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,
            with_payload=True,
            with_vectors=True
        )[0]
        
        # Convert points to nodes
        nodes = [
            TextNode(
                text=point.payload.get('content', ''),
                id_=str(point.id),
                embedding=point.vector,
                metadata=point.payload
            )
            for point in points
        ]
        
        self.logger.info(f"Loaded {len(nodes)} nodes from collection {self.collection_name}")
        
        # Create index from nodes
        index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        
        return index, nodes
        
    def delete_collection(self):
        """
        Delete the collection from Qdrant.
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error deleting collection {self.collection_name}: {e}")