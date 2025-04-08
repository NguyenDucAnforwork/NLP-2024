import logging
import random
from urllib.parse import urlencode

class RotatingProxyMiddleware:
    """Middleware to use rotating proxies"""
    
    def __init__(self, settings):
        self.logger = logging.getLogger(__name__)
        
        # If you have proxies, add them here
        self.proxies = settings.getlist('ROTATING_PROXIES', [])
        
        # Example proxies (replace with your own)
        # self.proxies = [
        #     'http://username:password@proxy1.example.com:8080',
        #     'http://username:password@proxy2.example.com:8080',
        # ]
        
        self.proxy_enabled = settings.getbool('PROXY_ENABLED', False)
        if not self.proxies:
            self.proxy_enabled = False
            
        if self.proxy_enabled:
            self.logger.info(f"Proxy middleware enabled with {len(self.proxies)} proxies")
        else:
            self.logger.info("Proxy middleware disabled or no proxies available")
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
        
    def process_request(self, request, spider):
        if not self.proxy_enabled or not self.proxies:
            return
            
        # Don't override existing proxy
        if 'proxy' in request.meta:
            return
            
        # Choose a random proxy
        proxy = random.choice(self.proxies)
        request.meta['proxy'] = proxy
        
        # Add authentication if needed
        # auth = basic_auth_header('username', 'password')
        # request.headers['Proxy-Authorization'] = auth
        
        # Log the proxy being used (for debugging)
        masked_proxy = proxy.replace('http://', '').replace('https://', '')
        if '@' in masked_proxy:
            masked_proxy = masked_proxy.split('@')[1]  # Hide credentials in logs
        
        spider.logger.debug(f"Using proxy: {masked_proxy}")
        
    def process_exception(self, request, exception, spider):
        # Handle proxy-related exceptions
        if 'proxy' in request.meta:
            proxy = request.meta['proxy']
            spider.logger.warning(f"Proxy {proxy} failed: {exception}")
            
            # Remove the failing proxy
            if proxy in self.proxies:
                self.proxies.remove(proxy)
                spider.logger.info(f"Removed failed proxy. {len(self.proxies)} proxies left")
                
            # Try another proxy
            if self.proxies:
                new_proxy = random.choice(self.proxies)
                request.meta['proxy'] = new_proxy
                
                # Don't filter URL
                request.meta['dont_filter'] = True
                return request
