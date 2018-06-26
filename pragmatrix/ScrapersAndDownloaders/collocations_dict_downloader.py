#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:49:09 2018

@author: rita

Script to build a collocation's dictionary 
with the most frequent words from the English vocabulary

"""

import requests 
from bs4 import BeautifulSoup as bs
import re
import os
import csv
import pickle
import time
import pandas as pd
import numpy as np
    
os.chdir('/Users/rita/Google Drive/DSR/DSR Project/Code/ScrapersAndDownloaders')
import collocations_dict_scraper as col

tools_path = '/Users/rita/Google Drive/DSR/DSR Project/Data/tools/'
redownload_common_words = False


def download_collocations(paths, redownload_common_words = False, join_files_only = False):

    if redownload_common_words:
        text = requests.get('https://www.wordfrequency.info/free.asp?s=y').text
        soup = bs(text, "lxml")
        
        # Scraping the website containing the most frequent English words
        words = []
        for word in soup.find_all('td'):
            for w in word.contents:
                if re.match(r'[a-z]{3,}', str(w.string)):
                    words.append(w.string)
    
        words = list(set(words))
        wd_df = pd.DataFrame(words)
        wd_df.to_csv(paths['common_wd_path'])
    
    else:        
        with open(paths['common_wd_path'], 'r') as f:
            reader = csv.reader(f)
            words = list(reader)
            words = [w[0] for w in words]
        
    # Storing the words from my list of most frequent words 
    # that appear in the English Collocations Dictionary and the Oxford Dictionaries
    words.remove('special')
    words.remove('invent')
    words.remove('anticipate')
    words.remove('major')
    words.remove('captain')

    whole_dict = {}
    
    if (join_files_only):
        # Joining pre-downloaded files together
        for batch_index in range(200):
            print("Now doing batch number {}/200".format(batch_index))
            with open(os.path.join(tools_path, 'dict_colls_' + str(batch_index) + '.pickle'), 'rb') as file:
                coll_dict = pickle.load(file) 
            
            whole_dict.update(coll_dict)        
                
    else:
        # Downloading the dictionary and saving it in batches
        for batch_index in range(200):
            print("Now doing batch number {}/200".format(batch_index))
                      
            begin = int(np.round(batch_index*np.floor(len(words)/200), 0))
            end = int(np.round(min(len(words), (batch_index+1)*np.floor(len(words)/200)-1), 0))
            
            my_coll_dict = {}
            for i, word in enumerate(words[begin:end]):
                word = word.split(' (PL)')[0]
                print('{} - {}'.format(word, i))           
                collocations, word_type, trans_words = col.collocate(word)
                # Making sure the values of the collocations' dictionary I am trying to compile
                # can be converted into regular expressions, which is what I try to do 
                # in my collocater function (to be found in my classy_n_funky script).
                [re.compile(c) for c in collocations]
                cats = []
                metamorphii = []
                if len(word_type) > 1:
                    for wdt in word_type:
                        cat = wdt.i.string
                        cats.append(cat)
                else:
                    cats = word_type
                
                for wd_type in cats:
                    # Checking the word's morphology to build its derivative forms, 
                    # which are to be included as keys in the collocations' dictionary being compiled. 
                    # 'Nou' stands for 'noun'. See: 
                    # [<p class="word"> <b>basketball </b><i>nou</i>n  </p>]
                    if re.search(r'nou', str(wd_type)): 
                        word1 = word + 's'
                        word2 = word + 'es'
                        if re.search(r'[^aeiou]y$', word):
                            derivate = word.rstrip('y') + 'ies'
                        elif re.search(r'([^aeiou]|[^aeiou][aeiou])f$', word):
                            derivate = word.rstrip('f') + 'ves'
                        elif re.search(r'([^aeiou]|[^aeiou][aeiou])fe$', word):
                            derivate = word.rstrip('fe') + 'ves'
                        elif re.search(r'man$', word):
                            derivate = word.rstrip('man') + 'men'
                        elif re.search(r'child', word):
                            derivate = word + 'ren'
                        elif re.search(r'foot', word):
                            derivate = 'feet'
                        else:
                            derivate = ''
                            metamorphii = [word, word1, word2]
                            
                        if derivate:
                            metamorphii = [word, word1, word2, derivate]
                    # 'ver' stands for 'verb'. See: 
                    # [<p class="word"> <b>assault </b><i>ver</i>b  </p>]
                    elif re.search(r'ver', str(wd_type)):
                        irr_vbs = col.verbal_abuse()
                        if irr_vbs.get(word):
                            trans_word = irr_vbs.get(word)
                            trans_word = trans_word.lstrip("(").rstrip(")") 
                            metamorphii = [trans for trans in trans_word.split('|')]
                        else:
                            word1 = word + 's'
                            word2 = word + 'ing'
                            word3 = word + 'ed'
                            word4 = word + word[-1] + 'ing'
                            word5 = word + word[-1] + 'ed'
                            metamorphii = [word, word1, word2, word3, word4, word5]
                            conjugated = []
                            if re.search(r'e$', word):
                                conju1 = word.rstrip('e') + 'ing'
                                conju2 = word + 'd'
                                conjugated.extend([conju1, conju2])
                            elif re.search(r'ie$', word):
                                conjugated = word.rstrip('ie') + 'ying'
                            elif re.search(r'y$', word):
                                conju1 = word.rstrip('y') + 'ied'
                                conju2 = word.rstrip('y') + 'ies'
                                conjugated.extend([conju1, conju2])
                            else:
                                conjugated = ''
                                
                            if conjugated:
                                metamorphii.extend(conjugated)
                            
                    elif re.search('adj', str(wd_type)):
                        word1 = word + 'er'
                        word2 = word + 'est'
                        metamorphii = [word, word1, word2]
                        declined = []
                        if re.search(r'y$', word):
                            declined1 = word.rstrip('y') + 'iest'
                            declined2 = word.rstrip('y') + 'ier'
                            declined.extend([declined1, declined2])
                        elif re.search(r'e$', word):
                            declined1 = word + 'r'
                            declined2 = word + 'st'
                            declined.extend([declined1, declined2])
                        else:
                            declined = ''
                            
                        if declined:
                            metamorphii.extend(declined)
                    else:
                        metamorphii = word
                
                if metamorphii:    
                    for term in metamorphii:
                        my_coll_dict[term] = collocations
                        my_coll_dict[term] = my_coll_dict.setdefault(term, set()).union(set(col.phrase(word)))                            
            
                my_coll_dict2 = {}
                for key, value in my_coll_dict.items():
                    if not re.match(r'\[\]', str(value)):
                        my_coll_dict2[key] = value
                              
            # Unit testing
            my_coll_dict3 = [len(list(my_coll_dict2[i])) for i in my_coll_dict2.keys()]
            my_coll_dict5 = [len(list(my_coll_dict[i])) for i in my_coll_dict.keys()]
            
            assert not 0 in my_coll_dict3
            assert not 0 in my_coll_dict5
            
            my_coll_dict4 = [len(list(my_coll_dict2)[i]) > 1 for i in range(len(list(my_coll_dict2)))]
            
            assert not False in my_coll_dict4
            
            coll_dict = {str(word): my_coll_dict2[word] for word in list(my_coll_dict2.keys())}
    
            whole_dict.update(coll_dict)        
                
            with open(os.path.join(tools_path, 'dict_colls_' + str(batch_index) + '.pickle'), 'wb') as file:
                pickle.dump(coll_dict, file, protocol = pickle.HIGHEST_PROTOCOL) 

    #store the fully joined dictionary
    with open(paths['colls_dict_path'], 'wb') as file:
        pickle.dump(whole_dict, file, protocol = pickle.HIGHEST_PROTOCOL)
        
    return whole_dict



def check_colls_down_success(paths):

    if not 'colls_dict_loaded' in locals():
        print('\nTrying to compile the dictionaries of collocations and collocations examples.\n')
        
        try:
            with open(paths['colls_dict_path'], 'rb') as file:
                colls_dict_loaded = pickle.load(file)
    
            print('\nCompilation was succesfull.')
                
        except:
            print("Compiling dictionaries. Go do laundry, because I'm gonna be busy for a while.")
            start = time.time()
            colls_dict_loaded = download_collocations(paths, join_files_only = False)
            end = time.time()
            print('more precisely, {} seconds. '.format(end - start))
            
    else:
        print('There was no need to import the dictionaries, because you already had them loaded in your workspace.')
    
    
    return colls_dict_loaded


colls_dict_loaded = check_colls_down_success(paths)

    