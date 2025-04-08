# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from bs4 import BeautifulSoup
import re
from llama_index.core.schema import Document

class LawscraperPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        # Get the content, default to an empty string if None
        value = adapter.get('content', '')
        if value is None:
            value = ''

        # Use BeautifulSoup to clean up the HTML
        soup = BeautifulSoup(value, 'html.parser')
        cleaned_html = ' '.join(soup.stripped_strings)

        # Remove extra spaces and newlines
        cleaned_html = re.sub(r'\s+', ' ', cleaned_html).strip()
        cleaned_html = re.sub(r'[\n\r]', '', cleaned_html)

        # adapter['content'] = Document(text=cleaned_html)
        return item

