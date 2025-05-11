"""
Playwright-based crawler implementation for the law document crawler system.
"""
import asyncio
import re
import time
import os
import sys
import urllib.parse
from typing import List, Dict, Any, Optional

# Simplify imports to avoid path manipulation issues
try:
    # When imported as a module
    from .base_crawler import BaseCrawler
except (ImportError, ValueError):
    # When run directly as a script
    from base_crawler import BaseCrawler

# Import playwright only if available
try:
    from playwright.async_api import async_playwright, Page, Browser, Route, Request
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Playwright is not installed. Please install it using:")
    print("pip install playwright")
    print("Then install browser binaries with:")
    print("playwright install")
    if __name__ == "__main__":
        sys.exit(1)

class PlaywrightCrawler(BaseCrawler):
    """
    Crawler implementation using Playwright for browser automation.
    """
    
    def __init__(self, 
                 output_path: str = "data/test",
                 max_pages: int = 3,
                 timeout: int = 15,
                 delay: int = 0.5,  # Reduced for concurrent processing
                 headless: bool = True,
                 max_concurrent: int = 4):  # Added concurrency control
        """
        Initialize the Playwright crawler.
        
        Args:
            output_path: Directory to save crawled data
            max_pages: Maximum number of pages to crawl
            timeout: Request timeout in seconds
            delay: Delay between requests in seconds
            headless: Whether to run the browser in headless mode
            max_concurrent: Maximum number of concurrent tabs
        """
        super().__init__(output_path, max_pages, timeout, delay)
        self.headless = headless
        self.browser = None
        self.context = None
        self.max_concurrent = max_concurrent
    
    def start_crawl(self, start_url: str) -> None:
        """
        Start the crawling process from the given URL.
        
        Args:
            start_url: URL to start crawling from
        """
        asyncio.run(self._start_crawl_async(start_url))
    
    async def _start_crawl_async(self, start_url: str) -> None:
        """
        Asynchronous implementation of the crawling process using concurrent tabs.
        
        Args:
            start_url: URL to start crawling from
        """
        all_items = []
        
        async with async_playwright() as playwright:
            self.browser = await playwright.chromium.launch(headless=self.headless)
            
            # Main page for crawling search results
            page = await self.browser.new_page()
            
            # Block unnecessary resources for performance
            await page.route("**/*", lambda route, request: asyncio.create_task(
                route.abort() if request.resource_type in ["image", "stylesheet", "font", "script"] 
                else route.continue_()
            ))
            
            # Semaphore to limit concurrent page processing
            sem = asyncio.Semaphore(self.max_concurrent)
            start_num = 8  # Start from page 4 as per the original code

            # Process pages
            for page_num in range(start_num, self.max_pages + start_num):
                # Construct the URL with page number
                if "page=" in start_url:
                    url = re.sub(r'page=\d+', f'page={page_num}', start_url)
                else:
                    url = f"{start_url}&page={page_num}" if "?" in start_url else f"{start_url}?page={page_num}"
                
                print(f"\n📄 Processing page {page_num}/{self.max_pages + start_num - 1}: {url}")
                
                try:
                    await page.goto(url, timeout=self.timeout * 1000)
                    
                    # Wait for the search results to load
                    await page.wait_for_selector("div.Content-SearchLegal", timeout=self.timeout * 1000)
                except Exception as e:
                    print(f"⚠️ Failed to load page {url}: {e}")
                    continue
                
                # Extract links to law documents
                # Using the selector from the example code
                link_elements = page.locator('div.nq p.nqTitle a')
                
                # Alternative selectors to try if the first one doesn't work
                if await link_elements.count() == 0:
                    link_elements = page.locator('a.vb-title')
                
                # Collect all links
                links = []
                for i in range(await link_elements.count()):
                    href = await link_elements.nth(i).get_attribute("href")
                    if href:
                        full_url = urllib.parse.urljoin(page.url, href)
                        links.append(full_url)
                
                print(f"🔗 Found {len(links)} law document links")
                
                if not links:
                    print("No links found. Trying alternative selector...")
                    # Try a broader selector
                    alt_links = await page.query_selector_all('a[href*="van-ban"]')
                    for link in alt_links:
                        href = await link.get_attribute('href')
                        if href and '/van-ban/' in href:
                            links.append(href if href.startswith('http') else urllib.parse.urljoin(page.url, href))
                    print(f"Found {len(links)} links with alternative selector")
                
                # Process links concurrently with limited tabs
                tasks = [self._process_law_page(link, sem) for link in links]
                items = await asyncio.gather(*tasks)
                items = [item for item in items if item]  # Filter out None results
                
                all_items.extend(items)
                print(f"Collected {len(items)} items from page {page_num}")
            
            # Save all collected items
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"law_documents_{timestamp}.json"
            print(f"\n✅ Crawling complete. Collected {len(all_items)} items.")
            self.save_items(all_items, output_file)
            
            await page.close()
    
    async def _process_law_page(self, url: str, sem: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
        """
        Process a single law document page with concurrency control.
        
        Args:
            url: URL of the law document page
            sem: Semaphore for concurrency control
            
        Returns:
            Dictionary with parsed data or None if parsing failed
        """
        async with sem:  # Limit concurrent tabs
            try:
                # Create a new page
                page = await self.browser.new_page()
                
                # Block unnecessary resources
                await page.route("**/*", lambda route, request: asyncio.create_task(
                    route.abort() if request.resource_type in ["image", "stylesheet", "font", "script"] 
                    else route.continue_()
                ))
                
                await page.goto(url, timeout=self.timeout * 1000)
                
                # Extract content using the example code's approach
                try:
                    # Wait for content to load using the selector from the example
                    await page.wait_for_selector(
                        "#tblcontainer > tbody > tr > td > table > tbody > tr:nth-child(1) > td", 
                        timeout=self.timeout * 1000
                    )
                    
                    # Extract content and metadata
                    sentences = await page.locator(".content1").all_text_contents()
                    
                    metadata_elements = page.locator("xpath=//div[@id='divThuocTinh']//tr/td[2]")
                    metadata = []
                    
                    for i in range(await metadata_elements.count()):
                        text = await metadata_elements.nth(i).inner_text()
                        metadata.append(text.strip())
                    
                    # Create item if we have sufficient data
                    if sentences and len(metadata) >= 3:
                        item = {
                            'url': url,
                            'noi_dung': self.clean_text(sentences),
                            'so_hieu': self.clean_text([metadata[0]]),
                            'noi_ban_hanh': self.clean_text([metadata[1]]),
                            'ngay_ban_hanh': self.clean_text([metadata[2]])
                        }
                        
                        # Filter by year
                        try:
                            year = int(item['ngay_ban_hanh'].split('/')[-1])
                            if year >= 2020:
                                print(f"✅ Successfully processed: {item['so_hieu']} ({year})")
                                await page.close()
                                return item
                            else:
                                print(f"⏭️ Skipping document from {year}: {item['so_hieu']}")
                        except ValueError:
                            print(f"⚠️ Invalid date format: {item['ngay_ban_hanh']}")
                    else:
                        print(f"❌ Insufficient data: content={bool(sentences)}, metadata={len(metadata)}")
                        
                except Exception as e:
                    print(f"❌ Error parsing page content: {e}")
                
                await page.close()
                
            except Exception as e:
                print(f"❌ Error processing {url}: {e}")
                try:
                    await page.close()
                except:
                    pass
                
            # Add a small delay to avoid hammering the server
            await asyncio.sleep(self.delay)
            
        return None
    
    def clean_text(self, texts):
        """
        Clean text by removing extra whitespace, newlines, etc.
        
        Args:
            texts: List of text strings
        """
        txt = ' '.join(texts)
        txt = re.sub(r'[\r\n\t]+', ' ', txt)
        return re.sub(r'\s{2,}', ' ', txt).strip()
    
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
    
    def save_items(self, items: List[Dict[str, Any]], filename: str) -> None:
        """
        Save the crawled items to a file.
        
        Args:
            items: List of items to save
            filename: Name of the file to save to
        """
        if not items:
            print("No items to save!")
            return
            
        try:
            import json
            os.makedirs(self.output_path, exist_ok=True)
            output_file = os.path.join(self.output_path, filename)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False)
                
            print(f"Saved {len(items)} items to {output_file}")
        except Exception as e:
            print(f"Error saving items to {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    def parse_single_document(self, document_url: str) -> None:
        """
        Parse a single document URL directly.
        
        Args:
            document_url: URL of the document to parse
        """
        asyncio.run(self._parse_single_document_async(document_url))
    
    async def _parse_single_document_async(self, document_url: str) -> None:
        """
        Asynchronous implementation for parsing a single document.
        
        Args:
            document_url: URL of the document to parse
        """
        async with async_playwright() as playwright:
            self.browser = await playwright.chromium.launch(headless=self.headless)
            
            # Create a semaphore for consistency with the main method
            sem = asyncio.Semaphore(1)
            
            print(f"Processing single document: {document_url}")
            item = await self._process_law_page(document_url, sem)
            
            # Save the item
            if item:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = f"single_document_{timestamp}.json"
                self.save_items([item], output_file)
            else:
                print("Failed to parse the document.")

if __name__ == "__main__":
    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright is not available. Please install it properly.")
        sys.exit(1)
    
    print("Playwright is available. Starting crawler...")
    try:
        # Set up output path
        current_dir = os.getcwd()
        output_path = os.path.join(current_dir, "data", "test")
        os.makedirs(output_path, exist_ok=True)
        print(f"Output will be saved to: {output_path}")
        
        # Create crawler instance
        crawler = PlaywrightCrawler(
            output_path=output_path,
            max_pages=1,
            timeout=10,
            delay=0.5,
            headless=True,  # Set to False to see the browser
            max_concurrent=4
        )
        
        # Start crawling
        crawler.start_crawl("https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&type=0&status=0&signer=0&sort=1&lan=1&scan=0&org=0&fields=&page=1")
    except Exception as e:
        print(f"Error running crawler: {e}")
        import traceback
        traceback.print_exc()