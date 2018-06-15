#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:14:38 2018

@author: rita
"""

import numpy as np
import re
from collections import Iterable
from lxml.html import fromstring
import requests
import random
from nltk.tokenize import sent_tokenize



# Functions to preprocess the data for feature extraction
def get_proxies(url):
    
    try:
        r_obj = requests.get(url)
    except:
        proxy = 0
        while not type(proxy) == str:
            
            try:
                url = 'https://free-proxy-list.net/'
                response = requests.get(url)
                parser = fromstring(response.text)
                ip = random.choice(parser.xpath('//tbody/tr'))
                if ip.xpath('.//td[7][contains(text(),"yes")]'):
                    # Grabbing IP and corresponding PORT
                    proxy = ":".join([ip.xpath('.//td[1]/text()')[0], ip.xpath('.//td[2]/text()')[0]])
                    proxies = {"http": proxy, "https": proxy}
                    
            except:
               continue
   
        r_obj = requests.get(url, proxies)
               
    return r_obj
       


def flatten(l):
    
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, str):
            for sub in flatten(el):
                yield sub
        else:
            yield el
 
    

def docs_normalizer(piece):
    
    if len(piece) < 3000:
        print("Sorry, the text couldn't be used, because it is too short.")
        return None
    
    else:
        sent = sent_tokenize(piece)
        length = 0
        batches = []
        pieces = []
        for s in range(len(sent)):
            try:
                if (((s < len(sent)-1) and (re.search(re.compile(sent[s]), sent[s+1])) and
                    (re.search(re.compile(sent[s]), sent[s+2]))) or (len(sent[s])>=3000)):
                    print("""
                          Sorry, the text couldn't be used, 
                          because the sentences it is comprised of 
                          are probably not meant to be read sequentially.
                          """)
                    return None
                # Adding the sentences' lengths together until the sum reaches 3k characters
                # and storing the sentences in batches
                else:
                    length += len(sent[s])
                    batches.append(sent[s])
                    if (length > 3000):
                        batch = ' '.join(batches)
                        pieces.append(batch)
                        length = 0
                        batches = []
            except:
                # Found a sentence with miss-matching parenthesis
                continue
                    
        # Including only up to 5 excerpts per text
        if len(pieces) > 5:
            idx = np.random.choice(len(pieces), 5, replace = False)
            pieces = [pieces[i] for i in idx]
            
        return pieces





          