#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:41:45 2018

@author: rita

Building the word-collocations dictionary from 
the Online Oxford Collocation Dictionary and the phrases 
from the English Oxford Dictionaries

"""

import requests
import re
from bs4 import BeautifulSoup as bs
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import pickle
from itertools import chain


os.chdir('/Users/rita/Google Drive/DSR/DSR Project/Code')
import prepro_funcs as prep


def verbal_abuse():
    
    """
    With this function I want to create a regular expression for
    the different conjugated forms the most common irregular verbs can adopt.
    """
    irr_verbs_path = '/Users/rita/Google Drive/DSR/DSR Project/Data/tools/irregular_verbs.csv'
    
    irreg_verbs = pd.read_csv(irr_verbs_path, sep = ';')
    irreg_verbs = irreg_verbs.applymap(lambda x: x.lower().replace('/', ' '))
    irreg_verbs.columns = 'a', 'b', 'c', 'd', 'e'
    irreg_verbs = pd.melt(irreg_verbs.T)
    irreg_verbs = irreg_verbs.groupby('variable')['value'].apply(lambda x: ' '.join(x))
    irreg_verbs = irreg_verbs.map(lambda x: x.split(' '))
    irr_verbs = irreg_verbs.values.tolist()
    
    irr_vbs = {}
    for i in range(len(irr_verbs)):
        irr_vbs[irr_verbs[i][0]] = '(' + '|'.join(list(set(irr_verbs[i]))) + ')'

    return irr_vbs



def word_identify(word, morphos, word_type):

    """
    With this function I want to make sure that the words 
    included in the collocations match the different derived forms 
    the words looked up in the collocations' dictionary can adopt
    """
 
    if re.search('noun', str(word_type)):
        if re.search(r'[^aeiou]y$', word):
            trans_word = '(the|a|an)?\s*' + word.rstrip('y') + '(y|ies)'
        elif re.search(r'[^aeiou]f$', word):
            if re.search(r'ff$', word):
                trans_word = '(the|a|an)?\s*' + word.rstrip('f') + '(ff|ves)'
            else:
                trans_word = '(the|a|an)?\s*' + word.rstrip('f') + '(f|ves)'
        elif re.search(r'[^aeiou]fe$', word):
            trans_word = '(the|a|an)?\s*' + word.rstrip('fe') + '(fe|ves)'
        elif re.search(r'[^aeiou][aeiou]f$', word):
            trans_word = '(the|a|an)?\s*' + word.rstrip('f') + '(f|ves)'
        elif re.search(r'[^aeiou][aeiou]fe$', word):
            trans_word = '(the|a|an)?\s*' + word.rstrip('fe') + '(fe|ves)'
        elif re.search(r'man$', word):
            trans_word = '(the|a|an)?\s*' + word.rstrip('man') + '(man|men)'
        elif re.search(r'child', word):
            trans_word = '(the|a|an)?\s*' + '(child|children)'
        elif re.search(r'foot', word):
            trans_word = '(the|a|an)?\s*' + '(foot|feet)'
        else:
            trans_word = '(the|a|an)?\s*' + word + '(s|es)?'
            
    elif re.search('verb', str(word_type)):
        irr_vbs = verbal_abuse()
        if irr_vbs.get(word):
            trans_word = irr_vbs.get(word)
        else:
            if re.search(r'e$', word):
                trans_word = word.rstrip('e') + '(e|ed|ing|es)'
            elif re.search(r'ie$', word):
                trans_word = word.rstrip('ie') + '(ie|ied|ying|ies)'
            elif re.search(r'y$', word):
                trans_word = word.rstrip('y') + '(y|ied|ying|ies)'
            else:
                trans_word = word + word[-1] + '?' + '(ed|ing|s)?'    
            
    elif re.search('adj', str(word_type)):
        if re.search(r'y$', word):
            trans_word = word.rstrip('y') + '(y|iest|ier)'
        elif len(word) < 5:
            trans_word = word + '(st|r|er|est)?'            
        else:
            trans_word = word
    else:
        trans_word = word
              

    irr_vbs = verbal_abuse()
    verbs = list(irr_vbs.keys())
    
    new_values = []
    for key, value in morphos.items():
        before = []
        after = []
        other = []
        if (re.match(re.compile(word), key) or (re.search(r'verb', str(word_type)) and re.search(r'(prep|adv)', key))):
            for vals in value:
                if vals:
                    vals = vals.strip()
                    if len(re.findall(r'\s', vals)) > 0:
                        vals = vals.split(' ')
                        temp = []
                        for v in vals:
                            if v in verbs:
                                v = irr_vbs.get(v)
                            temp.append(v)
                        vals = ' '.join(temp)
                    else:
                        if vals in verbs:
                            vals = irr_vbs.get(vals)
                    before.append(trans_word + ' ' + vals)
                
        elif re.search(re.compile(word), key) or (re.search(r'adj', key) or (re.search(r'adv', key) and re.search(r'adj', str(word_type)))):
            for vals in value:
                if vals:
                    vals = vals.strip()
                    if len(re.findall(r'\s', vals)) > 0:
                        vals = vals.split(' ')
                        temp = []
                        for v in range(len(vals)):
                            if vals[v] in verbs:
                                vals[v] = irr_vbs.get(vals[v])
                            else:
                                if (v == 0 and re.search('to', vals[v+1])):
                                    if re.search(r'e$', vals[v]):
                                        vals[v] = vals[v].rstrip('e') + '(e|ed|ing|es)'
                                    elif re.search(r'ie$', vals[v]):
                                        vals[v] = vals[v].rstrip('ie') + '(ie|ied|ying|ies)'
                                    elif re.search(r'y$', vals[v]):
                                        vals[v] = vals[v].rstrip('y') + '(y|ied|ying|ies)'
                                    else:
                                        vals[v] = vals[v] + vals[v][-1] + '?' + '(ed|ing|s)?' 
                            temp.append(vals[v])
                        vals = ' '.join(temp)
                    else:
                        if vals in verbs:
                            vals = irr_vbs.get(vals)
                    after.append(vals + ' ' + trans_word)

        else:
            for vals in value:
                vals = vals.strip()
                if vals in verbs:
                    if vals:
                        vals = vals.strip()
                        if len(re.findall(r'\s', vals)) > 0:
                            vals = vals.split(' ')
                            temp = []
                            for v in vals:
                                if v in verbs:
                                    v = irr_vbs.get(v)
                                temp.append(v)
                            vals = ' '.join(temp)
                        else:
                            if vals in verbs:
                                vals = irr_vbs.get(vals)
                        if re.search(re.compile('~'), vals):
                            vals = vals.replace('~', trans_word)
                        else:
                            vals = vals + ' ' + trans_word
                        other.append(vals)

        new_values.append([before, after, other])
        
    new_values = sum(new_values, [])
    new_values = sum([val for val in new_values if not (len(val) == 0 or val == trans_word)], [])

    return new_values, trans_word
            


def colls_extractor(word, several):
    
    """
    Scrapes the Online Oxford Collocation Dictionary to extract collocations  
    """

    r_obj = prep.get_proxies('http://oxforddictionary.so8848.com/search?word={}'.format(word))
    core_soup = bs(r_obj.text, 'lxml')
    
    if several[0] == True:
        soup = core_soup.find_all('div', {'class':'item'})
    else:
        soup = core_soup.find('div', {'class':'item'})
        
    if several[1] == True:
        word_type = core_soup.find_all('p', {'class':'word'})
    else:
        word_type = core_soup.find('p', {'class':'word'})

    new_values = []
    trans_words = []
    for idx in range(len(word_type)):
        wt = word_type[idx].i.string.strip()
        temp_dict1 = {}
        temp_dict2 = {}
        morphos = {}
        for a in soup[idx].find_all('p'):
            morf = str(a.find('u'))
            morf = morf.lower().lstrip('<u> ').rstrip(' </u>')
            bolds = [str(bold.string).replace('|', ',').replace(', etc. ', ' ').strip(' ').split(',') for bold in a.find_all('b')]
            bolds = sum(bolds, [])
            temp_dict1.setdefault(morf, []).append(bolds)
            for b in a.contents:
                impossibles = []
                if not ((re.match(r'<', str(b))) or (re.search(r'=', str(b))) or (b in bolds)):
                    not_bold = b.string.replace('|', ',').replace(', etc. ', ' ').strip(' ').split(',')
                    impossibles.append(not_bold)
                temp_dict2.setdefault(morf, []).append(impossibles)
                temp_dict2 = {key: list(prep.flatten(value)) for key, value in temp_dict2.items()}
        for k, v in chain(temp_dict1.items(), temp_dict2.items()):
            morphos.setdefault(k, []).append(v)
        morphos = {key: list(prep.flatten(value)) for key, value in morphos.items()}
        values, trans = word_identify(word, morphos, str(wt))
        new_values.append(values)
        trans_words.append(trans)
    
    new_values = list(set(sum(new_values, [])))

    return new_values, trans_words



def collocate(word):
    
    # This tries to get the content of the website, 
    # and, if it doesn't work, it tries with another proxy.
    ox_coll = prep.get_proxies('http://oxforddictionary.so8848.com/search?word={}'.format(word))
    core_soup = bs(ox_coll.text, 'lxml')        

    several = []
    try:
        soup = core_soup.find_all('div', {'class':'item'})
        try:
            word_type = core_soup.find_all('p', {'class':'word'})
            sev_items = True
            sev_words = True
            several.extend([sev_items, sev_words])
            new_values, trans_words = colls_extractor(word, several)
        except:
            word_type = core_soup.find('p', {'class':'word'})
            sev_items = True
            sev_words = False
            several.extend([sev_items, sev_words])
            new_values, trans_words = colls_extractor(word, several)
            
    except:
        soup = core_soup.find('div', {'class':'item'})
        try:
            word_type = core_soup.find_all('p', {'class':'word'})
            sev_items = False
            sev_words = True
            several.extend([sev_items, sev_words])
            new_values, trans_words = colls_extractor(word, several)
        except:
            word_type = core_soup.find('p', {'class':'word'})
            sev_items = False
            sev_words = False
            several.extend([sev_items, sev_words])
            new_values, trans_words = colls_extractor(word, several)

    elems_down = []
    for nv in new_values:
        if re.search(r'\)[\?\w\s\.,;:\+\*\\]*\)', nv) \
            or re.match(r'[\?\w\s\.,;:\+\*\\]*\)', nv) \
            or re.search(r'\([\?\w\s\.,;:\+\*\\]*$', nv) \
            or re.search(r'\([\?\w\s\.,;:\+\*\\]*\(', nv) \
            or re.match(r'\.', nv) \
            or re.search(r'\s\.$', nv):
                #print(nv)
                elems_down.append(nv)
    
    if elems_down:
        temp_set = set(new_values)
        new_values = temp_set - set(elems_down)
        new_values = list(new_values)    
    
    new_values = [re.sub(r'\ssb\/sth', ' \w+', ph) for ph in new_values]    
    new_values = [re.sub(r'\s?(the)?\/?(its|your)*\/your\s\(the\|a\|an\)\?', " (the|your|its|his|her|their|our|\w+'s)?\s?", ph) for ph in new_values]
    new_values = [re.sub(r'\s?(the)?\/?(its|your)\s?', " (the|your|its|his|her|their|our|\w+'s)?\s?", ph) for ph in new_values]
    new_values = [re.sub(r'\s?(a|an)\/the', ' (the|a|an)?', ph) for ph in new_values]
    new_values = [re.sub(r'\s?the\s\(the\|a\|an\)\?', ' the ', ph) for ph in new_values]
    new_values = [re.sub(r"\/", '|', ph) for ph in new_values]
    parent_wrapper0 = re.compile(r'\s([a-z]+\|[a-z]+)\s')
    parent_wrapper1 = re.compile(r'\s([a-z]+\|[a-z]+\|[a-z]+)\s')
    parent_wrapper2 = re.compile(r'\s([a-z]+\|[a-z]+\|[a-z]+\|[a-z]+)\s')
    parent_wrapper3 = re.compile(r'\s([a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+)\s')
    parent_wrapper4 = re.compile(r'\s([a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+)\s')
    parent_wrapper5 = re.compile(r'\s([a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+)\s')
    parent_wrapper6 = re.compile(r'\s([a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+)\s')
    parent_wrapper7 = re.compile(r'\s([a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+)\s')
    parent_wrapper8 = re.compile(r'\s([a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+\|[a-z]+)\s')
        
    for pw in [parent_wrapper0, parent_wrapper1, parent_wrapper2, 
               parent_wrapper3, parent_wrapper4, parent_wrapper5,
               parent_wrapper6, parent_wrapper7, parent_wrapper8]:
        new_values = [pw.sub(r' (\1) ' , str(' ' + ph + ' ')).strip() for ph in new_values]
    
    modi = [re.sub(r"\ssb$", " \w+", ph).strip() for ph in new_values]
    modi = [re.sub(r"\ssb\s", " \w+ ", ph).strip() for ph in new_values]
    modi = [re.sub(r"\s\\\w\+'s\s(\(the\|a\|an\)\?)?\s?", " (the|your|its|his|her|their|our|\w+'s)\s?", ph).strip() for ph in modi]
    modi = [re.sub(r"\ssth$", " \w+", ph).strip() for ph in modi]
    modi = [re.sub(r"\ssth\s", " \w+ ", ph).strip() for ph in modi]   
    modi = [re.sub(r'\s\\w\+$', '', ph) for ph in modi]
    modi = [re.sub(r"\(\\w\+\)\*", "\w+", ph).strip() for ph in modi]
    modi = [re.sub(r"…", "\w+", ph) for ph in modi]
    modi = [re.sub(r"=\s*", "", ph) for ph in modi]   
    modi = [re.sub(r"\s{2,}", " ", ph).strip() for ph in modi]
    modi = [re.sub(r"figurative\s", "", ph).strip() for ph in modi]
    
    collocations = set(modi)
    collocations.difference_update(set(trans_words + [word]))
    
#    examples = [ejem.string.strip() for ejem in soup.find_all('i') if re.findall(r'\s\w', ejem.string.strip())]
#    examples = [re.sub(r'\(informal\)\s', '', ex).strip() for ex in examples]
#    examples = [re.sub(r"\(figurative\)\s", "", ex).strip() for ex in examples]
#    examples = [re.sub(r"\(all\sfigurative\)\s", "", ex).strip() for ex in examples]
#    
    return collocations, word_type, trans_words#, examples



def phrase(word):
    
    """
    I also want to include the idioms and phrases a given word can appear in 
    according to the English Oxford Dictionaries.
    """
    
    r_obj = prep.get_proxies('https://en.oxforddictionaries.com/definition/' + word)
    soup = bs(r_obj.content, 'lxml')
    phrases = []
    for a in soup.find_all('strong', {'class':'phrase'}):
        phrase = re.sub(r'\(or.*\)\s*', '', a.string)
        phrase = re.sub(r'\(', '', phrase)
        phrase = re.sub(r'\)', '', phrase)
        phrase = re.sub(r'someone/something', "\w+", phrase)
        phrase = re.sub(r'something', "\w+", phrase)
        phrase = re.sub(r'someone', "\w+", phrase)
        phrase = re.sub(r'\sone', "\s\w+", phrase)
        phrase = re.sub(r'—', "\w+", phrase)
        phrase = phrase.strip()
        if re.search('/', phrase) and (len(re.findall('/', phrase)) == 1):
            first, last = phrase.split('/')
            phrase0 = ' '.join(first.split(' ')[:-1]) + ' ' + last
            phrase1 = first + ' ' + ' '.join(first.split(' ')[1:])
            phrases.append(phrase0.strip())
            phrases.append(phrase1.strip())
        elif re.search('/', phrase) and (len(re.findall('/', phrase)) == 2):
            first, second, last = phrase.split('/')
            phrase0 = first + ' ' + ' '.join(second.split(' ')[1:]) + ' ' + ' '.join(last.split(' ')[1:])
            phrase1 = ' '.join(first.split(' ')[:-1]) + ' ' + ' '.join(second.split(' ')[:-1]) + ' ' + last
            phrases.append(phrase0.strip())
            phrases.append(phrase1.strip())
            
        else:
            phrases.append(phrase)
    
    phrases = list(set(phrases))
  
    return phrases