#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:49:09 2018

@author: rita
"""

import requests 
from bs4 import BeautifulSoup as bs
import re
import os
import pickle
import time
import pandas as pd
    
os.chdir('/Users/rita/Google Drive/DSR/DSR Project/Code/ScrapersAndDownloaders')
import collocations_dict_scraper as col

tools_path = '/Users/rita/Google Drive/DSR/DSR Project/Dataset/tools/'


def verbal_abuse(tools_path = tools_path):
    
    path = '/Users/rita/Google Drive/DSR/DSR Project/Dataset/tools/'
    
    irreg_verbs = pd.read_csv(os.path.join(path, 'irregular_verbs.csv'), sep = ';')
    irreg_verbs = irreg_verbs.applymap(lambda x: x.lower().replace('/', ' '))
    irreg_verbs.columns = 'a', 'b', 'c', 'd', 'e'
    irreg_verbs = pd.melt(irreg_verbs.T)
    irreg_verbs = irreg_verbs.groupby('variable')['value'].apply(lambda x: ' '.join(x))
    irreg_verbs = irreg_verbs.map(lambda x: x.split(' '))
    irr_verbs = irreg_verbs.values.tolist()
    
    irr_vbs = {}
    for i in range(len(irr_verbs)):
        irr_vbs[irr_verbs[i][0]] = '(' + '|'.join(irr_verbs[i]) + ')'
        
    with open(os.path.join(path, 'irr_verbs_dict.pickle'), 'wb') as file:
        pickle.dump(irr_vbs, file)
    
    return irr_vbs


def download_collocations(tools_path = tools_path):

#    proxies = cl.get_proxies()
    text = requests.get('https://www.wordfrequency.info/free.asp?s=y').text
    soup = bs(text, "lxml")
    
    ## Scraping the website containing the most frequent AEnglish words
    words = []
    for word in soup.find_all('td'):
        for w in word.contents:
            if re.match(r'[a-z]{3,}', str(w.string)):
                words.append(w.string)
    
    words = list(set(words))
    
    
    ## Storing the words from my list of most frequent words 
    ## that appear in the English Collocations Dictionary and the Oxford Dictionaries
    my_coll_dict = {}
    expls = {}
    for word in words:
        print(word)
        my_coll_dict[word], expls[word] = col.collocate(word)
        my_coll_dict.setdefault(word, []).extend(col.phrase(word))
        
    
    my_coll_dict2 = {}
    for key, value in my_coll_dict.items():
        if not re.match(r'\[\]', str(value)):
            my_coll_dict2[key] = value
        
    
    # Unit testing
    my_coll_dict3 = [len(list(my_coll_dict2[i])) for i in my_coll_dict2.keys()]
    my_coll_dict5 = [len(list(my_coll_dict[i])) for i in my_coll_dict.keys()]
    
    assert not 0 in my_coll_dict3
    assert 0 in my_coll_dict5
    
    my_coll_dict4 = [len(list(my_coll_dict2)[i]) > 1 for i in range(len(list(my_coll_dict2)))]
    
    assert not False in my_coll_dict4
    
    # Short version:
    coll_dict = {str(word): my_coll_dict2[word] for word in list(my_coll_dict2.keys())}
    coll_expls = {str(ex): expls[ex] for ex in list(expls.keys())}
    
    irr_vbs = verbal_abuse()
    verbs = list(irr_vbs.keys())
    for key, value in coll_dict.items():
        words = [v.split(' ') for v in value]
        new_ws = []
        for w in words:
            new_as = []
            for a in w:
                if a in verbs:
                    a = irr_vbs.get(a)
                new_as.append(a)
            new_ws.append(' '.join(new_as))
        coll_dict[key] = new_ws
            
    with open(os.path.join(tools_path, 'dict_colls.pickle'), 'wb') as file:
        pickle.dump(coll_dict, file, protocol = pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(tools_path, 'expls_dicts.pickle'), 'wb') as file:
        pickle.dump(coll_expls, file, protocol = pickle.HIGHEST_PROTOCOL)   

    return coll_dict, coll_expls




if not ('colls_dict_loaded' and 'colls_expls_loaded') in locals():
    print('\nTrying to compile the dictionaries of collocations and collocations examples.\n')
    
    try:
        with open(os.path.join(tools_path, 'dict_colls.pickle'), 'rb') as file:
            colls_dict_loaded = pickle.load(file)

        with open(os.path.join(tools_path, 'expls_dicts.pickle'), 'rb') as file:
            colls_expls_loaded = pickle.load(file)
            
        print('\nCompillation was succesfull.')
            
    except:
        print("Compiling dictionaries. Go do laundry, because I'm gonna be busy for a while.")
        start = time.time()
        colls_dict_loaded, colls_expls_loaded = download_collocations()
        end = time.time()
        print('more precisely, {} seconds. '.format(end - start))
        
else:
    print('There was no need to import the dictionaries, because you already had them loaded in your workspace.')


