import scrapy
import re
from lawscraper.items import LawscraperItem  # giữ nguyên nếu bạn đã định nghĩa Item

class LawspiderSpider(scrapy.Spider):
    name = 'lawspider'
    allowed_domains = ['thuvienphapluat.vn']
    start_urls = [
        'https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&match=True&type=0&status=0&signer=0&sort=1&lan=1&scan=0&org=0&fields=&page=1'
    ]
    page_num = 1
    max_pages = 100  # chỉnh tùy ý số trang cần crawl

    def parse(self, response):
        # lấy link các văn bản pháp luật trên trang hiện tại
        links = response.css('a.vb-title::attr(href)').getall()
        for link in links:
            if not link.startswith('http'):
                link = 'https://thuvienphapluat.vn' + link
            yield scrapy.Request(link, callback=self.parse_law_page)

        # sang trang tiếp theo
        self.page_num += 1
        if self.page_num <= self.max_pages:
            next_page = f'https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&match=True&type=0&status=0&signer=0&sort=1&lan=1&scan=0&org=0&fields=&page={self.page_num}'
            yield scrapy.Request(next_page, callback=self.parse)

    def parse_law_page(self, response):
        law_item = LawscraperItem()
        
        sentences = response.css('.content1 ::text').getall()
        metadata = response.xpath('//div[@id="divThuocTinh"]//table//tr/td[2]/text()').getall()

        if not sentences or len(metadata) < 3:
            return

        law_item['noi_dung'] = self.clean_text(sentences)
        law_item['so_hieu'] = self.clean_text([metadata[0]])
        law_item['noi_ban_hanh'] = self.clean_text([metadata[1]])
        law_item['ngay_ban_hanh'] = self.clean_text([metadata[2]])

        # bỏ các văn bản trước năm 2020
        try:
            year = int(law_item['ngay_ban_hanh'].split('/')[-1])
            if year < 2020:
                return
        except:
            return

        yield law_item

    def clean_text(self, text_list):
        text = ' '.join(text_list)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
