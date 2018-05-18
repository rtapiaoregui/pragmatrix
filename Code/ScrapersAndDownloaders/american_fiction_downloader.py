#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:38:04 2018

@author: rita

Downloads short stories from the American Short Fiction online magazine.
"""

import os

import requests
from bs4 import BeautifulSoup as bs
import re
from tqdm import tqdm


os.chdir('/Users/rita/Google Drive/DSR/DSR Project/Dataset/american_short_fiction')


# First, I want to get all the unique links leading to posts hosting short stories.
links = []
for page in range(1,13):      
    web_page = "http://americanshortfiction.org/category/web-exclusives/page/" + str(page) + "/"
    original_content = requests.get(web_page).content
    soup = bs(original_content, "lxml")
    for link in soup.find_all('a'):
        if re.search(r'/201\d/', str(link.get('href'))):
            links.append(str(link.get('href')))
            
links = list(set(links))

# Secondly, I download and save all the short stories in files.
num = 0
for link in tqdm(links):
    content1 = requests.get(link).content
    soup1 = bs(content1, "lxml")
    strings = []
    for link0 in soup1.find_all('div', {'class':'entry-content'}):
        for link1 in link0.find_all('p'):
            for i in link1.contents:
                if not re.match(r'\<img', str(i.string)):
                    strings.append(str(i.string))
    strings = ' '.join(strings)
    num += 1
    file = open(os.path.join(os.getcwd(), str(num) + '.txt'), 'w')
    file.write(strings)
    file.close()
        
                            
                        