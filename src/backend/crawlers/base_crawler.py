"""
Base crawler implementation for the law document crawler system.
Provides core functionality that specific crawlers can build upon.
"""
import abc
import logging
import os
from typing import List, Dict, Any, Optional

class BaseCrawler(abc.ABC):
    """
    Abstract base class for all crawlers in the system.
    Defines the interface that all crawler implementations must follow.
    """
    
    def __init__(self, 
                 output_path: str = "data/crawled_data",
                 max_pages: int = 10,
                 timeout: int = 30,
                 delay: int = 2):
        """
        Initialize the base crawler with common parameters.
        
        Args:
            output_path: Directory to save crawled data
            max_pages: Maximum number of pages to crawl
            timeout: Request timeout in seconds
            delay: Delay between requests in seconds
        """
        self.output_path = output_path
        self.max_pages = max_pages
        self.timeout = timeout
        self.delay = delay
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
    
    @abc.abstractmethod
    def start_crawl(self, start_url: str) -> None:
        """
        Start the crawling process from the given URL.
        
        Args:
            start_url: URL to start crawling from
        """
        pass
    
    @abc.abstractmethod
    def parse_item(self, response: Any) -> Dict[str, Any]:
        """
        Parse an item from the response.
        
        Args:
            response: Response object from the crawler
            
        Returns:
            Dictionary with parsed data
        """
        pass
    
    def save_items(self, items: List[Dict[str, Any]], filename: str) -> None:
        """
        Save crawled items to a file.
        
        Args:
            items: List of items to save
            filename: Name of the file to save to
        """
        import json
        
        output_file = os.path.join(self.output_path, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(items)} items to {output_file}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        import re
        text = re.sub(r'[\r\n\t]+', ' ', text)
        return re.sub(r'\s{2,}', ' ', text).strip()