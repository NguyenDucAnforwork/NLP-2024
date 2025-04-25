"""
Document processing functionality for the vector index system.
"""
import os
import json
from typing import List, Dict, Tuple, Any, Optional

from llama_index.core import Document

# Import the scripts
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../scripts'))
from clean import normalize
from parse_text_to_structure import parse_text_to_structure

class LegalDocumentProcessor:
    """
    Processor for Vietnamese legal documents.
    """
    
    @staticmethod
    def prepare_documents(
        legal_texts: List[str],
        document_ids: Optional[List[str]] = None,
        min_length: int = 400,
        max_length: int = 2000
    ) -> List[Document]:
        """
        Prepare LlamaIndex Document objects from Vietnamese legal texts.
        
        Args:
            legal_texts: List of raw legal document texts
            document_ids: List of document identifiers
            min_length: Minimum length for merging segments
            max_length: Maximum length for merging segments
            
        Returns:
            List of Document objects with metadata
        """
        documents = []
        
        for i, text in enumerate(legal_texts):
            doc_id = document_ids[i] if document_ids and i < len(document_ids) else f"doc_{i}"
            
            # Normalize the text
            normalized_text = normalize(text)
            
            # Parse the structure with custom parser
            structured_data = parse_text_to_structure(normalized_text, min_length, max_length)
            
            # Process each section
            for key, content in structured_data.items():
                if key.startswith("Chương"):
                    # This is a chapter
                    chapter_title = content.get('title', '')
                    
                    # Process each article in this chapter
                    for article_key, article_content in content.items():
                        if article_key == 'title':
                            continue
                            
                        if isinstance(article_content, dict) and 'content' in article_content:
                            # Create document for this article
                            doc_text = article_content['content']
                            metadata = {
                                'document_id': doc_id,
                                'chapter': key,
                                'chapter_title': chapter_title,
                                'article': article_key
                            }
                            
                            # Add parent links if available
                            if 'parent_links' in article_content:
                                metadata['parent_links'] = article_content['parent_links']
                                metadata['is_merged'] = True
                            
                            documents.append(Document(text=doc_text, metadata=metadata))
                
                elif key.startswith("Điều") or "+" in key:
                    # This is a direct article or merged articles
                    if isinstance(content, dict) and 'content' in content:
                        doc_text = content['content']
                        metadata = {
                            'document_id': doc_id,
                            'article': key
                        }
                        
                        # Add parent links if available
                        if 'parent_links' in content:
                            metadata['parent_links'] = content['parent_links']
                            metadata['is_merged'] = True
                        
                        documents.append(Document(text=doc_text, metadata=metadata))
        
        return documents
    
    @staticmethod
    def load_from_json(filepath: str) -> Tuple[List[str], List[str]]:
        """
        Load legal texts and document IDs from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Tuple of (legal_texts, document_ids)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract texts and document IDs
            legal_texts = [item.get('noi_dung', '') for item in data]
            document_ids = [item.get('so_hieu', f"doc_{i}") for i, item in enumerate(data)]
            
            return legal_texts, document_ids
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return [], []