# Scrapy settings for lawscraper project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "lawscraper"

SPIDER_MODULES = ["lawscraper.spiders"]
NEWSPIDER_MODULE = "lawscraper.spiders"

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Obey robots.txt rules - Setting to False to bypass any blocks
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
# Further reducing concurrent requests to avoid overwhelming the server
CONCURRENT_REQUESTS = 4

# Configure a delay for requests for the same website (default: 0)
DOWNLOAD_DELAY = 4  # Increased delay
RANDOMIZE_DOWNLOAD_DELAY = True
# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 1  # Reduced to 1 to avoid triggering rate limits
CONCURRENT_REQUESTS_PER_IP = 1

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers to make requests more browser-like
DEFAULT_REQUEST_HEADERS = {
   "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
   "Accept-Language": "en-US,en;q=0.9",
   "Accept-Encoding": "gzip, deflate, br",
   "Connection": "keep-alive",
   "Upgrade-Insecure-Requests": "1",
   "Sec-Fetch-Dest": "document",
   "Sec-Fetch-Mode": "navigate",
   "Sec-Fetch-Site": "same-origin",
   "Sec-Fetch-User": "?1"
}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    "lawscraper.middlewares.LawscraperSpiderMiddleware": 543,
#}

# Enable or disable downloader middlewares
SCRAPEOPS_API_KEY = 'ae0f7ae8-cd6d-40cf-bcd4-e8e17dcfb195'
SCRAPEOPS_FAKE_USER_AGENT_ENDPOINT = 'http://headers.scrapeops.io/v1/user-agents'
SCRAPEOPS_FAKE_BROWSER_HEADER_ENABLED = True
SCRAPEOPS_FAKE_USER_AGENT_ENABLED = True
SCRAPEOPS_NUM_RESULTS = 50

# Enable proxy middleware if needed (disabled by default)
PROXY_ENABLED = False
ROTATING_PROXIES = []  # Add your proxies here if you have them

DOWNLOADER_MIDDLEWARES = {
   "lawscraper.middlewares.ScrapeOpsFakeUserAgentMiddleware": 400,
   "lawscraper.middlewares.CustomRetryMiddleware": 500,
   "lawscraper.proxymiddleware.RotatingProxyMiddleware": 600,
   "scrapy.downloadermiddlewares.retry.RetryMiddleware": None,  # Disable default retry middleware
   "scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware": 750,
}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
   "lawscraper.pipelines.LawscraperPipeline": 300,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
# The initial download delay
AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
# Enable showing throttling stats for every response received:
AUTOTHROTTLE_DEBUG = True

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 24 hours
HTTPCACHE_DIR = "httpcache"
HTTPCACHE_IGNORE_HTTP_CODES = [503, 504, 400, 401, 403, 404, 408, 429]
HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

# Retry settings for failed requests
RETRY_ENABLED = True
RETRY_TIMES = 10  # Increased from 7
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 403, 404, 522, 524, 520, 521, 525, 530]
RETRY_PRIORITY_ADJUST = -1

# Longer timeout for requests
DOWNLOAD_TIMEOUT = 90

# Log settings
LOG_LEVEL = 'INFO'  # Changed from WARNING to INFO for better visibility

# Additional settings for robustness
HTTPERROR_ALLOWED_CODES = [403, 404, 429]  # Allow these errors to be processed by spider

# Add a new setting to avoid being blocked
DOWNLOAD_FAIL_ON_DATALOSS = False

# Additional settings for avoiding detection
DEPTH_PRIORITY = 1
SCHEDULER_DISK_QUEUE = 'scrapy.squeues.PickleFifoDiskQueue'
SCHEDULER_MEMORY_QUEUE = 'scrapy.squeues.FifoMemoryQueue'

# Add Referrer policy to appear more browser-like
REFERRER_POLICY = 'same-origin'