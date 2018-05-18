#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:41:45 2018

@author: rita
"""

import requests
import re
from bs4 import BeautifulSoup as bs
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def collocate(word):
    
    ox_coll = requests.get('http://oxforddictionary.so8848.com/search?word={}'.format(word))
    soup = bs(ox_coll.text, 'lxml')
    
    colls1 = [str(bold.string).replace('|', ',').replace(', etc. ', ' ').strip(' ').split(',') for bold in soup.find_all('b')]
    
    impossibles = []
    for i in soup.find_all('div', {'class':'item'}):
        for a in i.find_all('p'):
            for b in a.contents:
                if not re.match(r'\<', str(b)) and not re.search(r'\(\=\s[\w+\s]+\)', str(b)):
                    not_bold = b.string.replace('|', ',').replace(', etc. ', ' ').strip(' ').split(',')
                    impossibles.append(not_bold)
                    
    colls1.extend([sum(impossibles, [])])                
    colls2 = [c.strip() for cl in colls1 for c in cl if (not c == word and not c == '')]
    
    phrase = [elem for elem in colls2 if re.search(r'\s', elem)]
    parentheses = [elem for elem in phrase if re.search(r'\(', elem)]
    opt1 = [elem.replace('(', '').replace(')', '') for elem in parentheses]
    opt2 = [re.sub(r'(\s\(\w+\))', '', elem) for elem in parentheses]
    phrase.extend(opt1 + opt2)
    
    alt_phrases = [re.sub(r'\s(sb/sth)(\W\w\s\W)?', '', ph) for ph in phrase if re.findall(r'\/', ph)]
    alts = [re.findall(r"(.*\s)?([A-Za-z']+)\/([A-Za-z']+)\/?([A-Za-z']+)?(.*)?", ph) for ph in alt_phrases if re.search(r'\/', ph)]
    alts = sum(alts, [])
    
    alt0a = [elem[0] + elem[1] for elem in alts if (len(elem[0].split())>=4 and len(elem[4].split())>=4)]
    alt0b = [elem[2] + elem[4] for elem in alts if (len(elem[0].split())>=4 and len(elem[4].split())>=4)]
    
    alt1 = [elem[0] + elem[1] + elem[4] for elem in alts]
    alt2 = [elem[0] + elem[2] + elem[4] for elem in alts]
    alt3 = [elem[0] + elem[3] + elem[4] for elem in alts]
    
    all_alts = alt0a + alt0b + alt1 + alt2 + alt3
    all_alts = [elem for elem in all_alts if not(re.search(r'\s{2}', elem) or (re.match(r'\s', elem) or re.search(r'\s$', elem)))]
    
    double_alts = [re.findall(r"(.*\s)?([A-Za-z']+)\/([A-Za-z']+)\/?([A-Za-z']+)?(.*)?", ph) for ph in all_alts if re.search(r'\/', ph)]
    double_alts = sum(double_alts, [])
    
    alt1 = [elem[0] + elem[1] + elem[4] for elem in double_alts]
    alt2 = [elem[0] + elem[2] + elem[4] for elem in double_alts]
    alt3 = [elem[0] + elem[3] + elem[4] for elem in double_alts]
    
    all_double_alts = alt1 + alt2 + alt3
    all_double_alts = [elem for elem in all_double_alts if not(re.search(r'\s{2}', elem) or (re.match(r'\s', elem) or re.search(r'\s$', elem)))]
    
    phrase.extend(all_alts)
    phrase.extend(all_double_alts)
    colls2.extend(phrase)
    
    phrases = [elem for elem in colls2 if not (re.search(r'\(', elem) or re.search(r'\)', elem) or re.search(r'\/', elem))]
    phrases = [elem for elem in phrases if len(elem)>1]
    
    modi = [ph.lower().replace('~', word) for ph in phrases]
    modi = [ph.replace("your", "\w+") for ph in modi]
    modi = [re.sub(r"sth$", "", ph).strip() for ph in modi]
    modi = [re.sub(r"…", "\w+", ph).strip() for ph in modi]
    modi = [re.sub(r"\s{2,}", " ", ph).strip() for ph in modi]
    modi = [re.sub(r"figurative\s", "", ph).strip() for ph in modi]
    
    
    phrases = list(set(modi))
    
    examples = [ejem.string.strip() for ejem in soup.find_all('i') if re.findall(r'\s\w', ejem.string.strip())]
    examples = [re.sub(r'\(informal\)\s', '', ex).strip() for ex in examples]
    examples = [re.sub(r"\(figurative\)\s", "", ex).strip() for ex in examples]
    examples = [re.sub(r"\(all\sfigurative\)\s", "", ex).strip() for ex in examples]
    
    return phrases, examples



def phrase(word):
    
#    proxies = cl.get_proxies()
    or_content = requests.get('https://en.oxforddictionaries.com/definition/' + word).content
    soup = bs(or_content, 'lxml')
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