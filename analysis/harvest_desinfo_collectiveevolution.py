#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "transfer content data from desinfo url.csv; content from https://www.collective-evolution.com/ more specifically: https://cevo.mykajabi.com/blog"
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

df = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/desinfo_collectiveevolution_urls.csv', sep=',')
df.drop_duplicates(inplace=True)
infos_df = pd.DataFrame(columns=['category-id', 'text-id', 'text'])

for index, row in df.iterrows():
    row = str(row[0])
    if row.startswith('https://cevo.mykajabi'):
        url = ''.join(row)
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # get article text
            try:
                text = soup.get_text()
                text = text.replace('\n\n', '').replace('\r', '').replace('Podcast\nArticles\nMembership\nAbout\nContact', '')
                cleaned = text.split('We hate SPAM. We will never sell your information, for any reason.', 1)
                clean_text = cleaned[0]
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
infos_df.to_csv('data/desinfo_collectiveevolution_text.csv', index=False)