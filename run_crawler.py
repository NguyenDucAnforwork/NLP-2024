"""
Simple script to run the PlaywrightCrawler directly.
"""
import os
import sys
from src.backend.crawlers.playwright_crawler import PlaywrightCrawler

if __name__ == "__main__":
    print("🚀 Starting law document crawler...")
    
    # Set up output directory
    output_path = os.path.join(os.getcwd(), "data", "test")
    os.makedirs(output_path, exist_ok=True)
    print(f"📁 Output will be saved to: {output_path}")
    
    # Create crawler with optimized settings
    crawler = PlaywrightCrawler(
        output_path=output_path,
        max_pages=200,
        timeout=10,
        delay=0.5,
        headless=True,  # Set to False to see the browser for debugging
        max_concurrent=4
    )
    
    # Default search URL or specific document URL from command line
    default_url = "https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&type=0&status=0&signer=0&sort=1&lan=1&scan=0&org=0&fields=&page=1"
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
        if "/van-ban/" in url and not "/tim-van-ban" in url:
            print(f"🔍 Processing single document: {url}")
            crawler.parse_single_document(url)
        else:
            print(f"🔍 Crawling search results from: {url}")
            crawler.start_crawl(url)
    else:
        print(f"🔍 Using default search URL: {default_url}")
        crawler.start_crawl(default_url)
    
    print("✅ Crawler finished")
