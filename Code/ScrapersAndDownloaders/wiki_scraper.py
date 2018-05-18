#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:56:11 2018

@author: rita
"""
import requests 
import os
import re
from bs4 import BeautifulSoup as bs
import random


def wiki_down(path, url):

    ''' Function to download Wikipedia articles from url, clean them and save them in my Wikipedia folder. '''

    r = requests.get(url)
    
    soup = bs(r.text, 'html.parser')
    
    file_name = re.sub(r'\-', r'', re.sub(r'\s', r'_', str(soup.title.string).lower())) + '.txt'
    
    file_name = re.sub(r'\/',r'_',file_name)
    
    article = soup.get_text()
    
    #find the beginning of the article, remove everything before it:
    article = article.split('From Wikipedia, the free encyclopedia')[1]
    #find the References[edit] part and cut before it
    article = article.split('References[edit]')[0]
    
    #remove all the line breaks ('\n'), references to [edit] and [X] where X is a number
    article = re.sub(r'\[\d*\]','', article)
    article = re.sub(r'\[(edit)\]','', article)
    article = re.sub(u'\n',' ', article)
    article = re.sub(r'\s+', ' ', article)
    
    with open(os.path.join(path, file_name), 'w') as f:
        f.write(article)

    return getNextLink(soup)


def getNextLink(soup):
    allLinks = soup.findAll('a')
    allLinks = [tag.get('href') for tag in allLinks]
    
    allLinks = [link for link in allLinks if (str(link).startswith(u"/wiki/"))]
    allLinks = [link for link in allLinks if not (u"Category" in str(link))]
    allLinks = [link for link in allLinks if not (u"Special" in str(link))]
    allLinks = [link for link in allLinks if not (u"File" in str(link))]
    allLinks = [link for link in allLinks if not (u"Wikipedia" in str(link))]
    allLinks = [link for link in allLinks if not (u"Portal" in str(link))]
    allLinks = [link for link in allLinks if not (u"Talk:" in str(link))]
    allLinks = [link for link in allLinks if not (u"Help:" in str(link))]
    allLinks = [link for link in allLinks if not (u"Template" in str(link))]

    
    return 'https://en.wikipedia.org'+random.choice(allLinks)
    
    # and not str(link).find(u'Category'))]
