"""
Specialized Scrapy spider for crawling Vietnamese legal documents.
"""
import re
import scrapy
from scrapy_playwright.page import PageMethod

from .playwright_crawler import PlaywrightCrawler
from typing import Any, Dict

class LawSpider(scrapy.Spider):
    """
    Scrapy spider for crawling Vietnamese legal documents from thuvienphapluat.vn
    """
    name = 'lawspider'
    allowed_domains = ['thuvienphapluat.vn']
    start_urls = ['https://thuvienphapluat.vn/page/tim-van-ban.aspx?...&page=1']
    page_num = 1
    max_pages = 3

    def start_requests(self):
        yield scrapy.Request(
            self.start_urls[0],
            callback=self.parse,
            meta={
                'playwright': True,
                'playwright_page_methods': [
                    PageMethod('wait_for_load_state', 'load', timeout=5000)
                ],
            },
        )

    async def parse(self, response):
        response = response.replace(encoding='utf-8')
        page = response.meta['playwright_page']

        for link in response.css('a.vb-title::attr(href)').getall():
            url = response.urljoin(link)
            yield scrapy.Request(
                url, callback=self.parse_law_page,
                meta={
                    'playwright': True,
                    'playwright_page_methods': [
                        PageMethod('wait_for_load_state', 'load', timeout=5000)
                    ],
                },
            )

        await page.close()

        # pagination
        self.page_num += 1
        if self.page_num <= self.max_pages:
            next_page = re.sub(r'page=\d+', f'page={self.page_num}', response.url)
            yield scrapy.Request(
                next_page, callback=self.parse,
                meta={
                    'playwright': True,
                    'playwright_page_methods': [
                        PageMethod('wait_for_load_state', 'load', timeout=5000)
                    ],
                },
            )

    async def parse_law_page(self, response):
        response = response.replace(encoding='utf-8')
        page = response.meta['playwright_page']

        sentences = response.css('.content1 ::text').getall()
        metadata = response.xpath('//div[@id="divThuocTinh"]//tr/td[2]/text()').getall()
        if sentences and len(metadata) >= 3:
            item = {
                'noi_dung': self.clean_text(sentences),
                'so_hieu': self.clean_text([metadata[0]]),
                'noi_ban_hanh': self.clean_text([metadata[1]]),
                'ngay_ban_hanh': self.clean_text([metadata[2]])
            }
            try:
                year = int(item['ngay_ban_hanh'].split('/')[-1])
                if year >= 2020:
                    yield item
            except ValueError:
                pass

        await page.close()

    def clean_text(self, texts):
        txt = ' '.join(texts)
        txt = re.sub(r'[\r\n\t]+', ' ', txt)
        return re.sub(r'\s{2,}', ' ', txt).strip()


class ScrapyLawCrawler(PlaywrightCrawler):
    """
    Adapter class to run the Scrapy spider using the crawler interface.
    """
    
    def start_crawl(self, start_url: str) -> None:
        """
        Start crawling using the Scrapy spider.
        
        Args:
            start_url: URL to start crawling from
        """
        from scrapy.crawler import CrawlerProcess
        from scrapy.utils.project import get_project_settings
        import os
        
        # Update start URL if provided
        if start_url:
            LawSpider.start_urls = [start_url]
        
        # Configure settings
        settings = get_project_settings()
        settings.set('FEEDS', {
            os.path.join(self.output_path, f'law_documents_{self._get_timestamp()}.json'): {
                'format': 'json',
                'encoding': 'utf8'
            }
        })
        
        # Start crawler
        process = CrawlerProcess(settings)
        process.crawl(LawSpider)
        process.start()
        
    def _get_timestamp(self):
        import time
        return time.strftime("%Y%m%d_%H%M%S")
    
    def parse_item(self, response: Any) -> Dict[str, Any]:
        """
        Parse an item from the response.
        
        Args:
            response: Response object from the crawler
            
        Returns:
            Dictionary with parsed data
        """
        # This method is not used directly as Scrapy handles parsing
        return {}