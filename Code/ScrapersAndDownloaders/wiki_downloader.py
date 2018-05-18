#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 12:36:05 2018

@author: rita
"""

import requests 
import os
from bs4 import BeautifulSoup as bs

os.chdir('/Users/jose/Google Drive/Kaggle/DSR Project/Code/')
import wiki_scraper as wscrap

path = '/Users/jose/Google Drive/Kaggle/DSR Project/Dataset/Wikipedia/'
numArticles = 5

r = requests.get('https://en.wikipedia.org/wiki/Wikipedia:Unusual_articles')

soup = bs(r.text, 'html.parser')

allLinks = soup.findAll('a')
allLinks = [tag.get('href') for tag in allLinks]

allLinks = [link for link in allLinks if (str(link).startswith(u"/wiki/"))]
allLinks = [link for link in allLinks if not (u"Category" in str(link))]
allLinks = [link for link in allLinks if not (u"Special" in str(link))]
allLinks = [link for link in allLinks if not (u"File" in str(link))]
allLinks = [link for link in allLinks if not (u"Wikipedia" in str(link))]
allLinks = [link for link in allLinks if not (u"Portal" in str(link))]
allLinks = [link for link in allLinks if not (u"Talk" in str(link))]
allLinks = [link for link in allLinks if not (u"Help" in str(link))]
allLinks = [link for link in allLinks if not (u"Template" in str(link))]

for nextURL in allLinks:
    nextURL = 'https://en.wikipedia.org'+nextURL
    for i in range(numArticles):
        print(nextURL)
        nextURL = wscrap.wiki_down(path,nextURL)
    