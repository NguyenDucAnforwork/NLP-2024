# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

## middlewares.py

import random
import logging
import requests
from scrapy import signals
from scrapy.downloadermiddlewares.useragent import UserAgentMiddleware
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.utils.response import response_status_message
import time
from scrapy.exceptions import IgnoreRequest
from urllib.parse import urlencode
from random import randint
from scrapy.http import Headers

class ScrapeOpsFakeBrowserHeaderAgentMiddleware:

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def __init__(self, settings):
        self.scrapeops_api_key = settings.get('SCRAPEOPS_API_KEY')
        self.scrapeops_endpoint = settings.get('SCRAPEOPS_FAKE_BROWSER_HEADER_ENDPOINT', 'http://headers.scrapeops.io/v1/browser-headers?') 
        self.scrapeops_fake_browser_headers_active = settings.get('SCRAPEOPS_FAKE_BROWSER_HEADER_ENABLED', False)
        self.scrapeops_num_results = settings.get('SCRAPEOPS_NUM_RESULTS')
        self.headers_list = []
        self._get_headers_list()
        self._scrapeops_fake_browser_headers_enabled()

    def _get_headers_list(self):
        payload = {'api_key': self.scrapeops_api_key}
        if self.scrapeops_num_results is not None:
            payload['num_results'] = self.scrapeops_num_results
        response = requests.get(self.scrapeops_endpoint, params=urlencode(payload))
        json_response = response.json()
        self.headers_list = json_response.get('result', [])

    def _get_random_browser_header(self):
        random_index = randint(0, len(self.headers_list) - 1)
        return self.headers_list[random_index]

    def _scrapeops_fake_browser_headers_enabled(self):
        if self.scrapeops_api_key is None or self.scrapeops_api_key == '' or self.scrapeops_fake_browser_headers_active == False:
            self.scrapeops_fake_browser_headers_active = False
        else:
            self.scrapeops_fake_browser_headers_active = True
    
    def process_request(self, request, spider):        
        random_browser_header = self._get_random_browser_header()
        request.headers = Headers(random_browser_header)

class ScrapeOpsFakeUserAgentMiddleware:
    """Middleware to rotate user agents using ScrapeOps API or fallback to random"""
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
        
    def __init__(self, settings):
        self.scrapeops_api_key = settings.get('SCRAPEOPS_API_KEY')
        self.scrapeops_endpoint = settings.get('SCRAPEOPS_FAKE_USER_AGENT_ENDPOINT', 'http://headers.scrapeops.io/v1/user-agents') 
        self.scrapeops_fake_user_agents_active = settings.get('SCRAPEOPS_FAKE_USER_AGENT_ENABLED', False)
        self.scrapeops_num_results = settings.get('SCRAPEOPS_NUM_RESULTS', 50)
        self.headers_list = []
        self._get_user_agents_list()
        self.logger = logging.getLogger(__name__)
        
        # Fallback user agents if API doesn't work
        self.fallback_user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        ]
        
    def _get_user_agents_list(self):
        """Get a list of user agents from ScrapeOps API"""
        if self.scrapeops_fake_user_agents_active and self.scrapeops_api_key:
            try:
                payload = {'api_key': self.scrapeops_api_key}
                if self.scrapeops_num_results:
                    payload['num_results'] = self.scrapeops_num_results
                response = requests.get(
                    self.scrapeops_endpoint,
                    params=payload,
                    timeout=10
                )
                json_response = response.json()
                self.user_agents_list = json_response.get('result', [])
            except Exception as e:
                logging.error(f"Error fetching user agents from ScrapeOps API: {e}")
                # Fall back to predefined list
                self.user_agents_list = self.fallback_user_agents
        else:
            self.user_agents_list = self.fallback_user_agents
            
    def process_request(self, request, spider):
        """Set a random user agent for each request"""
        if not self.user_agents_list:
            self._get_user_agents_list()
            
        random_user_agent = random.choice(self.user_agents_list)
        request.headers['User-Agent'] = random_user_agent
        spider.logger.debug(f"Using User-Agent: {random_user_agent}")

class CustomRetryMiddleware(RetryMiddleware):
    """Custom retry middleware with exponential backoff"""
    
    def __init__(self, settings):
        super().__init__(settings)
        self.logger = logging.getLogger(__name__)
        # Add Cloudflare specific error codes
        self.retry_http_codes = set(list(self.retry_http_codes) + [520, 521, 522, 523, 524, 525])
        
    def process_response(self, request, response, spider):
        if request.meta.get('dont_retry', False):
            return response
            
        # Handle Cloudflare specific errors
        if response.status in [520, 521, 522, 523, 524, 525]:
            self.logger.warning(f"Cloudflare protection error {response.status} detected. Waiting longer before retry.")
            wait_time = 180 + 60 * (2 ** (request.meta.get('retry_times', 0)))  # Longer wait for Cloudflare errors
            wait_time = min(wait_time, 900)  # Max 15 minutes
            
            self.logger.info(f"Throttling Cloudflare response {response.status} for {wait_time} seconds")
            time.sleep(wait_time)
            reason = f"Cloudflare error: {response_status_message(response.status)}"
            
            # Modify headers to look more like a browser
            request.headers['User-Agent'] = random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            ])
            request.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
            request.headers['Accept-Language'] = 'en-US,en;q=0.5'
            request.headers['Accept-Encoding'] = 'gzip, deflate, br'
            request.headers['Referer'] = 'https://thuvienphapluat.vn/'
            request.headers['Sec-Fetch-Dest'] = 'document'
            request.headers['Sec-Fetch-Mode'] = 'navigate'
            request.headers['Sec-Fetch-Site'] = 'same-origin'
            
            return self._retry(request, reason, spider) or response
        
        # Handle other retry codes
        if response.status in self.retry_http_codes:
            wait_time = 60 * (2 ** (request.meta.get('retry_times', 0)))  # Exponential backoff
            wait_time = min(wait_time, 600)  # Max 10 minutes
            
            self.logger.info(f"Throttling response {response.status} for {wait_time} seconds")
            time.sleep(wait_time)
            reason = response_status_message(response.status)
            return self._retry(request, reason, spider) or response
        
        # Check for hidden captchas or blocking
        if response.status == 200:
            body_text = response.text.lower()
            if any(term in body_text for term in ['captcha', 'blocked', 'security check', 'cloudflare', 'error code', 'rate limit']):
                self.logger.warning("Detected potential captcha or blocking page!")
                wait_time = 180  # 3 minutes
                self.logger.info(f"Waiting for {wait_time} seconds before retrying")
                time.sleep(wait_time)
                reason = "Detected captcha or blocking"
                return self._retry(request, reason, spider) or response
            
        return response
        
    def process_exception(self, request, exception, spider):
        # Add custom handling for network errors
        wait_time = 60 * (2 ** (request.meta.get('retry_times', 0)))
        wait_time = min(wait_time, 600)  # Max 10 minutes
        self.logger.info(f"Exception {exception} occurred. Waiting for {wait_time} seconds")
        time.sleep(wait_time)
        return super().process_exception(request, exception, spider)

