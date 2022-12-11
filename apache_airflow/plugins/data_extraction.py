# -*- coding: utf-8 -*-
import requests
import pandas as pd
from bs4 import BeautifulSoup
from shutil import ignore_patterns

class DataExtract:
    def __init__(self, traffyURL):
        self.traffyURL = traffyURL

    def web_scraping(self):
        url = self.traffyURL

        res = requests.get(url)
        res.status_code

        soup = BeautifulSoup(res.text,"html.parser")

        block = soup.find("div", {"class" : 'wp-block-cover aligncenter is-light'})
        inner_block = block.find('div', {'class':'wp-block-cover__inner-container'})
        data_url = inner_block.find('a').find('span').text
        print("start extract data")
        return data_url
        
    """# Extract Data"""

    def api_caller(self, data_url, offset):
        url = data_url + f'?offset={offset}'
        res = requests.get(url)
        data = pd.DataFrame(res.json()['results'])
        print("start call api data")
        return data

    def exportData(self):
        data_url = self.web_scraping()
        offset_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        # First DF
        df = self.api_caller(data_url, 0)
        for offset in offset_list:
            new_df = self.api_caller(data_url, offset)
            df = pd.concat([df, new_df], ignore_index = True)
        keep_col_list = ['type','type_id','comment','coords','district', 'subdistrict', 'province', 'timestamp']
        delete_col_list = []
        for col in df:
            if col not in keep_col_list:
                delete_col_list.append(col)
        df = df.drop(delete_col_list, axis = 1)
        df.to_csv('./data/traffy_fondue_data.csv', index= False, encoding="UTF-8")
        # print("get all 10K data")