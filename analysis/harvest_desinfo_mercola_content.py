#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "transfer url data from desinfo webpage and adding to data directory: https://www.mercola.com/ e.g.: https://takecontrol.substack.com/p/what-happens-during-menopause"

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

df = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/desinfo_mercola_urls.csv', sep=',')
df.drop_duplicates(inplace=True)
infos_df = pd.DataFrame(columns=['category-id', 'text-id', 'text'])

for index, row in df.iterrows():
    row = str(row[0])
    if row.startswith('https:'):
        url = ''.join(row)
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # get article text
            try:
                text = soup.get_text()
                text = text.replace('\n\n', '').replace('\r', '')
                clean_text = text.split("Sources include:", 1)
                info = pd.DataFrame({'category-id': 3, 'text-id': [url], 'text': [text]})
                print(info)
                infos_df = pd.concat([infos_df, info], ignore_index=True)
                #sys.exit()
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
    else:
        continue
infos_df.to_csv('data/desinfo_mercola_text.csv', index=False)