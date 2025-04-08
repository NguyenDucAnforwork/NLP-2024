#!/usr/bin/env python
import os
import sys
from datetime import datetime

# Add the project path to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lawscraper'))

# Import required components
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from lawscraper.spiders.lawspider import LawspiderSpider

if __name__ == '__main__':
    # Get timestamp for output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"laws_{timestamp}.json"
    
    # Get project settings
    settings = get_project_settings()
    
    # Optional overrides
    settings.set('HTTPCACHE_ENABLED', False)  # Disable cache for this run
    
    # Create crawler process with project settings
    process = CrawlerProcess(settings)
    
    # Start the spider
    process.crawl(LawspiderSpider)
    
    # Set output 
    process.settings.set('FEEDS', {output_file: {'format': 'json', 'encoding': 'utf8'}})
    
    print(f"Starting spider, output will be saved to {output_file}")
    # Start the crawl process
    process.start()
