#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 13:35:48 2018

@author: rita

Extracts articles posted on the Oxford Dictionaries Blog and 
the Collins Dictionary Word Lovers' Blog
and saves them as files in the oxbord_blog folder

"""

import requests 
import os
import re
from bs4 import BeautifulSoup as bs
from tqdm import tqdm

import prepro_funcs as prep

 
def oxford_down(paths): 
      
    oxford_corpus = []
    for num in range(1, 242):
        if num == 1:
            site = 'https://blog.oxforddictionaries.com'
        else :
            site = 'https://blog.oxforddictionaries.com/page/' + str(num) + '/' 

        r_obj = prep.get_proxies(site)
        soup = bs(r_obj.text, 'lxml')
            
        tags = soup.find_all('a')
        tags = [tag.get('href') for tag in tags if re.search(r'(https://blog.oxforddictionaries.com/)(\d+/\d+/\d+/\w+)', str(tag))]
    
        for tag in list(set(tags)):
            print(tag)
            r = requests.get(tag).text  
            soup1 = bs(r, 'lxml')
            
            corpus = []
            for link0 in soup1.find_all('div', {'class':'gen-post-content add-post-content'}):
                for link1 in link0.find_all('p'):
                    for i in link1.contents:
                        if not re.match(r'\<img', str(i.string)):
                            corpus.append(str(i.string))
                            
            corpus = ' '.join(corpus)
            
            if re.search(r'/', soup1.title.string):
                soup1.title.string = re.sub(r'/', '#', soup1.title.string)
                
            with open(os.path.join(paths['oxford_path'], ('_'.join(soup1.title.string.split(' -')[0].lower().split(' ')) + '.txt')), 'w') as f:
                f.write(corpus)
                
            oxford_corpus.append(corpus)
                
    return oxford_corpus



def collins_down(paths):
    
    site = "https://www.collinsdictionary.com/word-lovers-blog/new/?pageNo="
    root = "https://www.collinsdictionary.com"

    links = []
    page = []
    for i in tqdm(range(66)):
        site1 = site + str(i)
        r_obj = prep.get_proxies(site1)
        soup = bs(r_obj.content, 'lxml')
        for i in soup.find_all('a'):
            tag = i.get('href')
            page.append(tag)
    
    links = list(set(page))
    
    collins_corpus = []
    for link in tqdm(links):
        link = str(link)
        if re.match('/', link) and re.search(r'\?pageNo', link) == None:
            link = root + str(link)
            print(link)
        try:
            r_obj1 = prep.get_proxies(link)
            soup1 = bs(r_obj1.content, "lxml")
            strings = []
            corpus = []
            for link1 in soup1.find_all('p'):
                for i in link1.contents:
                    strings.append(str(i.string))
            corpus = ''.join(str(strings))
            file = open(os.path.join(paths['collins_path'], link.split('/')[-1].replace('.html', '') + '.txt'), 'w')
            file.write(corpus)
            file.close()
            
            collins_corpus.append(corpus)
        except:
            continue
        
    return collins_corpus
        
oxford_corpus = oxford_down(paths)  
collins_corpus = collins_down(paths)