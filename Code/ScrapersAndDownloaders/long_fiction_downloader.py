#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 21:00:30 2018

@author: rita

Classical literature downloader: 
    
Novels from the online library of the University of Adelaide

"""


import os
import re
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup as bs
import string

os.chdir(paths['code_path'])
import prepro_funcs as prep


def adelaide_down(adelaide_files = paths.get('adelaide_path')):    
    
    adelaide_path = "https://ebooks.adelaide.edu.au/meta/titles/"
    root = "https://ebooks.adelaide.edu.au"
    letters = list(string.ascii_uppercase)
    
    pages = []
    for let in letters:
        pages.append(str(adelaide_path + str(let) + ".html"))

    adelaide_corpus = []
    links = []
    for url in tqdm(pages):
        original_content = prep.get_proxies(url)
        soup = bs(original_content.content, "lxml")
        for link in soup.find_all('a'):
            if re.match(r'/\w/\w+', str(link.get('href'))):
                links.append(str(link.get('href')))
                   
    links = list(set(links))

    for i in tqdm(links):
        name = i
        i = str(root + i)
        try:
            r_obj = prep.get_proxies(os.path.join(i, 'complete.html'))
            soup1 = bs(r_obj.content, "lxml")
            strings = []
            for link1 in soup1.find_all('p'):
                for a in link1.contents:
                    strings.append(str(a.string))
            adelaide_corpus.append(''.join(str(strings)))
            
        except:
            continue

        for elem in adelaide_corpus:
            with open(os.path.join(adelaide_files,
                                   str(name).replace("/", '_') + '.txt'
                                   ), 'w') as file:
                file.write(elem)
                       
    return adelaide_corpus



def bookshelf_lit():

    bookshelf_path = 'http://www.publicbookshelf.com/fiction/affair-styles/'
    bookshelf_files = "/Users/rita/Google Drive/DSR/DSR Project/Dataset/long_fiction/bookshelf/agatha_christie"

    found = []
    url_text = requests.get(bookshelf_path).text
    soup = bs(url_text, 'lxml')
    for line in soup.find_all('ul', {'class': 'arrowsList'}):
        for href in line.find_all('a'):
            if re.finditer(r'http\:\/\/www\.publicbookshelf\.com\/fiction\/.*(?=")', str(href)):
                found.append(str(href))
                
    clean = []
    book = []
    for link in range(len(found[:-1])):
        url = found[link].split('"')[1]
        print(url)
        for num in range(1, 35):
            if num == 1:
                url_text = requests.get(str(url)).text
            else:
                try:
                    url_text = requests.get(str(url) + '-' + str(num)).text
                    
                except:
                    continue
                
            soup = bs(url_text, 'lxml')
            for line in soup.find_all('section', {'id':'text', 'class': 'mobileChapterContent'}):
                for i in line.find_all('p'):
                    for a in i.contents:
                        clean.append(str(a.string))
  
        book = ''.join(str(clean))
        
    with open(os.path.join(bookshelf_files, 'affair-styles.txt'), 'w') as file:
        file.write(book)
       
    return book



#book = bookshelf_lit()   
adelaide_corpus = adelaide_down()
  