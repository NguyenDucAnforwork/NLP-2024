"""
Playwright-based crawler implementation for the law document crawler system.
"""
import asyncio
import re
import time
from typing import List, Dict, Any, Optional

from playwright.async_api import async_playwright, Page, Response
from .base_crawler import BaseCrawler

class PlaywrightCrawler(BaseCrawler):
    """
    Crawler implementation using Playwright for browser automation.
    """
    
    def __init__(self, 
                 output_path: str = "data/crawled_data",
                 max_pages: int = 10,
                 timeout: int = 30,
                 delay: int = 2,
                 headless: bool = True):
        """
        Initialize the Playwright crawler.
        
        Args:
            output_path: Directory to save crawled data
            max_pages: Maximum number of pages to crawl
            timeout: Request timeout in seconds
            delay: Delay between requests in seconds
            headless: Whether to run the browser in headless mode
        """
        super().__init__(output_path, max_pages, timeout, delay)
        self.headless = headless
        self.browser = None
        self.context = None
    
    def start_crawl(self, start_url: str) -> None:
        """
        Start the crawling process from the given URL.
        
        Args:
            start_url: URL to start crawling from
        """
        asyncio.run(self._start_crawl_async(start_url))
    
    async def _start_crawl_async(self, start_url: str) -> None:
        """
        Asynchronous implementation of the crawling process.
        
        Args:
            start_url: URL to start crawling from
        """
        items = []
        
        async with async_playwright() as playwright:
            self.browser = await playwright.chromium.launch(headless=self.headless)
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            page = await self.context.new_page()
            await page.goto(start_url, timeout=self.timeout * 1000)
            
            # Process the first page
            page_num = 1
            
            while page_num <= self.max_pages:
                self.logger.info(f"Processing page {page_num} of {self.max_pages}")
                
                # Get all links on the page
                links = await page.query_selector_all('a.item-link')
                
                for link in links:
                    href = await link.get_attribute('href')
                    if href:
                        full_url = start_url.split('/')[0] + '//' + start_url.split('/')[2] + href
                        
                        # Open the link in a new tab
                        item_page = await self.context.new_page()
                        await item_page.goto(full_url, timeout=self.timeout * 1000)
                        
                        # Parse the item
                        item = await self._parse_item_async(item_page)
                        if item:
                            items.append(item)
                        
                        # Close the tab
                        await item_page.close()
                        
                        # Respect the delay
                        await asyncio.sleep(self.delay)
                
                # Go to the next page if available
                next_button = await page.query_selector('a.next-page')
                if next_button:
                    await next_button.click()
                    await page.wait_for_load_state('networkidle')
                    page_num += 1
                else:
                    break
            
            # Save the results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.save_items(items, f"law_documents_{timestamp}.json")
    
    async def _parse_item_async(self, page: Page) -> Optional[Dict[str, Any]]:
        """
        Parse an item from the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary with parsed data or None if parsing failed
        """
        try:
            # Wait for content to load
            await page.wait_for_selector('.content', timeout=self.timeout * 1000)
            
            # Extract the text content
            content = await page.inner_text('.content')
            
            # Extract metadata
            metadata_rows = await page.query_selector_all('.metadata tr')
            metadata = {}
            
            for row in metadata_rows:
                cells = await row.query_selector_all('td')
                if len(cells) >= 2:
                    key = await cells[0].inner_text()
                    value = await cells[1].inner_text()
                    metadata[key.strip()] = value.strip()
            
            # Clean the content
            cleaned_content = self.clean_text(content)
            
            return {
                'noi_dung': cleaned_content,
                'so_hieu': metadata.get('Số hiệu', ''),
                'noi_ban_hanh': metadata.get('Nơi ban hành', ''),
                'ngay_ban_hanh': metadata.get('Ngày ban hành', ''),
                'url': page.url
            }
        except Exception as e:
            self.logger.error(f"Error parsing item: {e}")
            return None
    
    def parse_item(self, response: Any) -> Dict[str, Any]:
        """
        Parse an item from the response (implements abstract method).
        
        Args:
            response: Response object from the crawler
            
        Returns:
            Dictionary with parsed data
        """
        # This is just a placeholder as we're using the async version
        return {}