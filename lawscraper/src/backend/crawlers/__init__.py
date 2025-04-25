"""
Crawlers module for the law document system.
"""
from .base_crawler import BaseCrawler
from .playwright_crawler import PlaywrightCrawler
from .lawspider import LawSpider, ScrapyLawCrawler

__all__ = [
    "BaseCrawler",
    "PlaywrightCrawler",
    "LawSpider",
    "ScrapyLawCrawler"
]