#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "aggregate data from all categories to one set"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "

import csv
import sys
import pandas as pd

cols = ['category-id', 'text']
science_df_1 = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/PMC-doi_text-2022-12-19.csv', sep=',', usecols=cols)
science_df_2 = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/PMC-doi_text-2023-02-26.csv', sep=',', usecols=cols)
science_df = pd.concat([science_df_1, science_df_2], ignore_index=True)
#print(science_df.head())
print('number of science paper:', len(science_df))


popscience_df_1 = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/popscience_wikipedia-text.csv', sep=',', usecols=cols)
popscience_df_2 = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/popscience_medlineplus_text.csv', sep=',', usecols=cols)
popscience_df = pd.concat([popscience_df_1, popscience_df_2], ignore_index= True)
print('number of popscience paper:', len(popscience_df))

desinfo_df_1 = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/desinfo_alternativenews_text-2.csv', sep=',', usecols=cols)
desinfo_df_2 = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/desinfo_collectiveevolution_text.csv', sep=',', usecols=cols)
desinfo_df_3 = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/desinfo_healthcancer_text.csv', sep=',', usecols=cols)
desinfo_df_4 = pd.read_csv('/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/desinfo_alternativenews_text-2.csv', sep=',', usecols=cols)
desinfo_df = pd.concat([desinfo_df_1, desinfo_df_2], ignore_index = True)
print(desinfo_df.head())
desinfo_df = pd.concat([desinfo_df, desinfo_df_3], ignore_index= True)
desinfo_df = pd.concat([desinfo_df, desinfo_df_4], ignore_index=True)
print('number of popscience paper:', len(desinfo_df))

number_desinfo = len(desinfo_df)
ready_dataset = pd.concat([science_df.head(number_desinfo), popscience_df.head(number_desinfo)], ignore_index=True)
#print(ready_dataset.head())
ready_dataset = pd.concat([ready_dataset, desinfo_df], ignore_index=True)
print('number of final data set', len(ready_dataset))

#ready_dataset.to_csv('data/ready_dataset.csv', index=False)
