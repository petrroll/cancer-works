# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import scrapy
import functions
import unicodedata

class DamaCzSpider(scrapy.Spider):
    name = 'dama_cz'
    curr_page = 0
    domain = 'diskuse.dama.cz'
    start_url = ''

    def __init__(self, maxpages=0, start_url="", delay=1.0, *args, **kwargs):
        super(DamaCzSpider, self).__init__(*args, **kwargs)
        self.start_url = start_url
        self.maxpages = int(maxpages)

    def start_requests(self):
        # paging is 30 submissions per page, instead of of doing actuall paging we jump 30 submissions
        for i in range(self.maxpages):
            print "------ SCRAPING: %s&f=%d" % (self.start_url, i*30)
            yield scrapy.Request("%s&f=%d" % (self.start_url, i*30), callback=self.parse)

    def parse(self, response):
        texts = response.xpath('//p[@class="na_text"]').extract()
        name_dates = response.xpath('//p[starts-with(@class,"na_jmeno")]').extract()
        # workaround for some shitty formatting on behalf of damacz
        padding = len(texts) - len(name_dates)

        for i in range(len(texts)-padding):
            text = texts[i+padding].encode('utf8')
            text = functions.strip_accents(text)
            text = functions.clean_text(text)
            name_date = functions.strip_accents(name_dates[i])
            name_date = functions.simple_clean_text(name_date)
            name = name_date.split()[-6:-5][0]      # lets count from the back, its more reliable
            date = name_date.split()[-5:-4][0]      # -||-
            date = functions.process_date_dama(date)

            # add to db
            yield {
            'domain': self.domain,
            'url': self.start_url,
            'text': text,
            'date': date,
            'name': name
            }
