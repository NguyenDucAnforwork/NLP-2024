# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class LawscraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    noi_dung = scrapy.Field()
    so_hieu = scrapy.Field()
    noi_ban_hanh = scrapy.Field()
    ngay_ban_hanh = scrapy.Field()
    
