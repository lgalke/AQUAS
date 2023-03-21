import scrapy
from scrapy.linkextractors import LinkExtractor
import requests
from bs4 import BeautifulSoup
import argparse
import csv

argparser = argparse.ArgumentParser()
argparser.add_argument("url")
argparser.add_argument("output")
args = argparser.parse_args()


with open(f'data/{args.output}', 'a') as file:

    url= args.url
    #url = 'https://medlineplus.gov/healthtopics.html'
    #url = "http://www.naturalnews.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href")
        try:
            if href.startswith("http"):
                print(href)
                file.write(href)
                file.write('\n')
            elif href.startswith("/"):
                print(url + href)
                file.write(url + href)
                file.write('\n')
        except Exception as e:
            print(e)



