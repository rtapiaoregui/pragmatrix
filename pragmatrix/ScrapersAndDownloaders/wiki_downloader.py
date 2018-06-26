#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 12:36:05 2018

@author: rita
"""

import requests 
import os
import re
#import random
from bs4 import BeautifulSoup as bs



def wiki_down(url = 'https://en.wikipedia.org/wiki/Special:Random', wiki_files = paths.get('wiki_path')):

    """
    Function to download Wikipedia articles, clean them superficially 
    and save them in a folder stored in my computer.
    
    If the url is not specified, it chooses a random article, thanks to Wikipedia's API
    """

    r = requests.get(url)    
    soup = bs(r.text, 'html.parser')
    file_name = re.sub(r'\-', r'', re.sub(r'\s', r'_', str(soup.title.string).lower())) + '.txt'    
    file_name = re.sub(r'\/', r'_', file_name)
    
    article = soup.get_text()
    
    # Finds where the actual article starts and removes anything being displayed before that:
    article = article.split('From Wikipedia, the free encyclopedia')[1]
    # Finds where the actual article ends and removes anything being displayed after that:
    article = article.split('References[edit]')[0]
    
    article = re.sub(r'\[\d*\]','', article)
    article = re.sub(r'\[(edit)\]','', article)
    article = re.sub(u'\n', ' ', article)
    article = re.sub(r'\s+', ' ', article)
    
    with open(os.path.join(wiki_files, file_name), 'w') as f:
        f.write(article)


        
num_articles = 10000            
for i in tqdm(range(num_articles)):
    wiki_down()
