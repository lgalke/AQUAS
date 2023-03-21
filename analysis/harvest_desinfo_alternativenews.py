#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "transfer url data from desinfo webpage and adding to data directory"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import sys

df = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/desinfo_alternativenews-health-news_url.csv', sep=',')
df.drop_duplicates(inplace=True)
infos_df = pd.DataFrame(columns=['category-id', 'text-id', 'text'])
hdr = {'User-Agent': 'Mozilla/5.0'}


for index, row in df.iterrows():
    row = str(row[0])
    if row.startswith('https:'):
        url = ''.join(row)
        try:
            response = requests.get(url, headers=hdr)
            soup = BeautifulSoup(response.text, 'html.parser', )

            # get article text
            try:
                text = soup.get_text()
                text = text.replace('\n\n', '').replace('\r', '').replace('Your Name\nYour email\nMessage\n or CancelSCIENCE\nFOOD\nHEALTH\nMEDICINE\nPOLLUTION\nCANCER\nCLIMATE', '')
                clean_text = text.split("Sources include:", 1)
                if "Sources for this article include:" in clean_text:
                    clean_text = clean_text.split("Sources include:", 1)
                    clean_text = clean_text[0]
                else: continue
                clean_text = clean_text[0]
                info = pd.DataFrame({'category-id': 3, 'text-id': [url], 'text': [clean_text]})
                print(info)
                infos_df = pd.concat([infos_df, info], ignore_index=True)
                #sys.exit()
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
    else:
        continue
infos_df.to_csv('data/desinfo_alternativenews_text.csv', index=False)
#https://health-news.asp//health.news/2019-03-14-prebiotics-fight-rotovirus.html