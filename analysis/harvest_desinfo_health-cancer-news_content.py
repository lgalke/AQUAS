#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "harvest data from https://www.naturalnews.com/ and adding to data directory"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "

import csv
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup


df = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/desinfos_health-news_urls.csv', sep=',')
df.drop_duplicates(inplace=True)
infos_df = pd.DataFrame(columns=['category-id', 'text-id', 'text'])

for index, row in df.iterrows():

    row = str(row[0])
    print(row)
    if row.startswith('https://'):
        url = ''.join(row)
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # get article text
            try:
                text = soup.get_text()
                text = text.replace('\n', '').replace('\r', '').replace('Your NameYour emailMessage or CancelSCIENCEFOODHEALTHMEDICINEPOLLUTIONCANCERCLIMATE', '')
                clean_text = text.split('RECENT NEWS & ARTICLES', 1)
                c_text = clean_text[0]
                cleaner_text = c_text.split('Sources include:', 1)
                cleaned_text = cleaner_text[0]
                info = pd.DataFrame({'category-id': 3, 'text-id': [url], 'text': [cleaned_text]})
                print(info)
                infos_df = pd.concat([infos_df, info], ignore_index=True)
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
    else:
        continue
infos_df.to_csv('data/desinfo_healthcancer-news_text-2.csv', index=False)