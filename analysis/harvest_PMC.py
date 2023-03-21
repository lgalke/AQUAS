#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = """ A script for fast harvesting of PMC tar.gz 
                    for authors, title, journal, year, abstract, text, """
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "


import argparse
import tarfile
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


def get_doi_text(tree):
    try:
        doi = tree.find('.//article-id[@pub-id-type="doi"]')
        tex = tree.findall('.//sec/p')
        txt = ''
        for elem in tex:
            txt += elem.text.strip()
        row = pd.DataFrame({'category-id': 1,'text-id':[doi.text],'text':[txt]})
        return row
    except Exception as e:
        print('Exception', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tarfile')
    args = parser.parse_args()
    df = pd.DataFrame(columns=['category-id', 'text-id','text'])
    with tarfile.open(args.tarfile) as tf:
        for member in tf:
            xf = tf.extractfile(member)
            tree = ET.parse(xf)
            data = get_doi_text(tree)
            try:
                df = pd.concat([df, data], ignore_index=True)
            except Exception as e:
                print(e)
    df[df['text'].str.strip().astype(bool)]
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    df.to_csv('data/PMC-doi_text-2023-02-28.csv', index=False)
    print('done')

if __name__ == '__main__':
    main()
