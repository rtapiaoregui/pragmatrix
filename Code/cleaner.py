#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:09:12 2018

@author: rita

Functions to preprocess the downloaded corpora 
and turn them into the observations of my primary dataset. 
"""


import os

import re
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import pandas as pd
from langdetect import detect

# My functions and classes
# os.chdir(paths['code_path'])
from prepro_funcs import docs_normalizer as norm


# Functions to clean the raw text files downloaded 
# from both literary and non-literary sources.
def general_cleaner(doc):
    
    doc = str(doc)
    
    for i in range(1,8):
        replacement = '\\x9' + str(i)
        doc = doc.replace(replacement, "'")
        
    # This, to several other non typografic characters.
    doc = doc.replace('\\x85', '').replace('\\x9', '')
    doc = re.sub(r'\\x97\w+(?=.{1,250}x97)', '(', doc)
    doc = re.sub(r'\\x97.{250,}\\x97', ')', doc)    
    doc = re.sub(r'\/FOOT', '', doc)
    doc = re.sub(r'\s?(\\+n|\\+r|\\+s|\\+t)+', ' ', doc)
    doc = re.sub(r'\\*xa0', ' ', doc)
    doc = re.sub(r'_', '', doc)
    doc = re.sub(r'\:{2,}', '', doc)
    doc = re.sub(r'\<.*\>', '', doc)
    doc = re.sub(r'\[.*\]\s', '', doc)
    doc = re.sub(r'\(.*\{.*\}.*\)', '', doc)
    doc = re.sub(r'(?<=\()\s', '', doc)
    doc = re.sub(r'\s(?=\))', '', doc)
    doc = re.sub(r'\s\(\)\s(?=[A-Z])', '. ', doc)
    doc = re.sub(r'\s\(\)\s', ' ', doc)
    doc = re.sub(r'\{.*\}\s', '', doc)
    doc = re.sub(r'\[', '', doc)
    doc = re.sub(r'\]', '', doc)
    doc = re.sub(r'■', '', doc)
    doc = re.sub(r'(`|")', "'", doc)
    doc = re.sub(r"\\*\'\,\s\\*\'*", ' ', doc)
    doc = re.sub(r'\.\s\'\.\s', '. ', doc)
    doc = re.sub(r'\\', '', doc)
    doc = re.sub(u'¬ ', '', doc)
    doc = re.sub(r'\^', '', doc)
    doc = re.sub(r'[0-9 .]*[A-Z ]{6,}\d*\s*', '', doc)
    doc = re.sub(r'\d+\s*(?=[A-Z])', '', doc)
    doc = re.sub(r'\s+(?=[!?,.;:])', '', doc)
    doc = re.sub(r'\s+', ' ', doc)
    doc = re.sub(r'\.{2}\s', '. ', doc)
    doc = doc.replace('-', '-')
    doc = doc.replace('- - ', '--')
    doc = doc.replace('- ', '')
    doc = re.sub(r'(?<=[^\s])\-\s(?=\w)', '', doc)
    doc = re.sub(r"\s(?=\'[sntmrdel]{1,3}\s)", '', doc)
    doc = re.sub(r"(?<=\w\')\s(?=[sntmrdel]{1,3}\s)", '', doc)
    doc = re.sub(r"\'{2,}", "' '", doc)
    doc = re.sub(r'\,(?=\s[A-Z])', '.', doc)
    doc = re.sub(r'[!?,.;:-]\.', '.', doc)
    doc = re.sub(r'Note\s\d+\.?', '', doc)
    doc = re.sub(r'(?<=[!?,;.:])(?=[^\s])', ' ', doc)
        
    return doc


## LITERATURE   

# Short fiction:
def eeuu_cleaner(paths):
    
    os.chdir(paths['eeuu_path'])
    dirty = []
    for i in os.listdir():
        file = open(i, 'r')
        s = file.read()
        file.close()
        dirty.append(s)
    
    corpus = []
    for piece in dirty:
        piece = re.sub(r'None', '', str(piece))
        sents = sent_tokenize(piece)
        sents = sents[:-3]
        piece = ' '.join(sents)
        corpus.append(piece)
    
    eeuu_corpus = []    
    for piece in corpus:
        piece = general_cleaner(piece)
        eeuu_corpus.append(piece)
        
    eeuu_corpus = list(set(eeuu_corpus))
        
    return eeuu_corpus



def short_fic_cleaner(paths, specific_path):
 
    os.chdir(paths.get(specific_path))

    dirty = []
    for i in os.listdir():
        if re.search(u'.txt', i):
            with open(i,'r') as f:
                dirty.append(f.read())
                
    corpus = []        
    for piece in dirty:
        piece = general_cleaner(piece)
        corpus.append(piece)
               
    corpus = list(set(corpus))
        
    return corpus


    
def short_fic_cleaner2(paths, specific_path):
 
    os.chdir(paths.get(specific_path))
    
    dirty = []
    for i in os.listdir():
        if re.search(u'.txt', i):
            with open(i, 'r') as f:
                dirty.append(f.read())
                
    corpus = []      
    for piece in dirty:
        piece = re.sub(r'None', '', str(piece))
        piece = general_cleaner(piece)
        corpus.append(piece)
        
    corpus = list(set(corpus))
                
    return corpus



def waccamaw_cleaner(paths):
    
    os.chdir(paths['arab_path'])
    
    dirty = []
    for i in os.listdir():
        if re.search(u'.txt', i):
            with open(i, 'r') as file:
                dirty.append(file.read())
    
    corpus = []
    for piece in dirty:
        piece = re.sub(r'None', '', str(piece))
        sents = sent_tokenize(piece)
        sents = sents[1:-1]
        corpus.append(' '.join(sents))
    
    waccamaw_corpus = []    
    for piece in corpus:
        piece = general_cleaner(piece)
        waccamaw_corpus.append(piece)
        
    waccamaw_corpus = list(set(waccamaw_corpus))

    return waccamaw_corpus



def arab_cleaner(paths):
    
    os.chdir(paths['arab_path'])
    
    dirty = []
    for i in os.listdir():
        if re.search(u'.txt', i):
            with open(i, 'r') as file:
                dirty.append(file.read())
    
    corpus = []
    for c in dirty:
        sents = sent_tokenize(c)
        sents = sents[1:-1]
        corpus.append(' '.join(sents))
    
    arab_corpus = []    
    for piece in corpus:
        piece = general_cleaner(piece)
        arab_corpus.append(piece)
        
    arab_corpus = list(set(arab_corpus))

    return arab_corpus



# Long fiction
def adelaide_cleaner(paths):
    
    os.chdir(paths['adelaide_path'])

    dirty = []
    for i in os.listdir():
        if re.search(u'.txt', i):
            with open(i,'r') as f:
                dirty.append(f.read())
                
    corpus = []      
    for piece in dirty:
        piece = str(piece)
        cond_eng1 = re.search(r'and', piece)
        cond_eng2 = re.search(r'with', piece)
        cond_eng3 = re.search(r'that', piece)
        cond_eng4 = re.search(r'of', piece)
        cond_eng5 = re.search(r'to', piece)
        cond_junk0 = len(re.findall(r'\d', piece)) > 15
        cond_junk1 = len(re.findall('[A-Z]\w+\.\,', piece)) > 2
        if (cond_eng1 and cond_eng2 and cond_eng3 and cond_eng4 and
            cond_eng5 and cond_junk0 and cond_junk1):
            sents = sent_tokenize(piece)[6:-1]
            corpus.append(' '.join(sents))
                
    adelaide_corpus = []
    for piece in corpus:
        piece = re.sub(r'None', '', str(piece))
        piece = general_cleaner(piece)
        adelaide_corpus.append(piece)
        
    adelaide_corpus = list(set(adelaide_corpus))            
    
    return adelaide_corpus
    



# NOT LITERATURE
def ling_blogs_cleaner(paths, specific_path):
 
    os.chdir(paths.get(specific_path))
    
    dirty = []
    for i in os.listdir():
        if re.search(u'.txt', i):
            with open(i, 'r') as file:
                dirty.append(file.read())
    
    corpus = []
    for piece in dirty:
        piece = re.sub(r'None', '', str(piece))
        sents = sent_tokenize(piece)
        sents = sents[2:]
        corpus.append(' '.join(sents))
       
    ling_blogs_corpus = []
    for piece in corpus:
        piece = general_cleaner(piece)
        ling_blogs_corpus.append(piece)
        
    ling_blogs_corpus = list(set(ling_blogs_corpus))
                
    return ling_blogs_corpus



def wiki_cleaner(paths):
    
    os.chdir(paths['wiki_path'])
    
    corpus = []
    for i in os.listdir()[:6000]:
        if re.search(u'.txt', i):
            with open(i, 'r') as file:
                corpus.append(file.read())
          
    # Defining the variables        
    batch = []
    wiki_corpus = []
    k = 10
    batch_len = round(len(corpus) // k)
    
    for i in tqdm(range(k)):
        last_sent = []
        for c in corpus[i * batch_len:(i + 1) * batch_len]:
            sents = sent_tokenize(c)
            sents = sents[6:-7]
            sentences = []
            for s in sents:
                s = str(s)
                # Defining my if-statement conditions:
                cond0 = len(re.findall(r'\d', s)) <= 5
                cond1 = re.search(r'\d\s\[A-Z]', s) == None
                cond2 = len(re.findall(r'[^\w\s]', s)) <= len(s)*0.17
                cond3 = len(re.findall(r'\s\w\s', s)) <= len(s)*0.20
                cond4 = len(re.findall('[A-Z]\w+', s)) <= len(s)*0.35
                cond5 = re.search(r'(www\.|\^|\+|\*|\@|\|)', s) == None
                cond6 = re.search(r'\.[a-z]', s) == None
                cond7 = re.search(r'None', s) == None
                cond8 = len(s) < 1000
                if (cond0 and cond1 and cond2 and cond3 and
                    cond4 and cond5 and cond6 and cond7 and cond8):
                    sentences.append(s)
            last_sent.append(''.join(str(sentences)))
        batch.append(last_sent)
    
    for corpus in tqdm(batch):
        for piece in corpus:
            piece = general_cleaner(piece)
            piece = re.sub(r'\s\.{3,}', ' ', piece)
            wiki_corpus.append(piece)
    
    wiki_corpus = list(set(wiki_corpus))
    
    return wiki_corpus



# News:
def news_cleaner(paths, specific_path):
 
    os.chdir(paths.get(specific_path))

    dirty = []
    for i in os.listdir():
        if re.search(u'.txt', i):
            with open(i,'r') as f:
                dirty.append(f.read()) 
                
    corpus = []
    for c in dirty:
        c = str(c)
        cond_eng1 = re.search(r'and', c)
        cond_eng2 = re.search(r'with', c)
        cond_eng3 = re.search(r'that', c)
        cond_eng4 = re.search(r'of', c)
        cond_eng5 = re.search(r'to', c)
        if cond_eng1 and cond_eng2 and cond_eng3 and cond_eng4 and cond_eng5:
            sents = sent_tokenize(c)
            sents = sents[3:-2]
            sentences = []
            for s in sents:
                s = str(s)
                # Defining my if-statement conditions:
                cond0 = len(re.findall(r'\d', s)) <= 5
                cond1 = re.search(r'\d\s\[A-Z]', s) == None
                cond2 = len(re.findall(r'[^\w\s]', s)) <= len(s)*0.17
                cond3 = len(re.findall(r'\s\w\s', s)) <= len(s)*0.20
                cond4 = len(re.findall('[A-Z]\w+', s)) <= len(s)*0.35
                cond5 = re.search(r'(www|\^|\+|\*|\@|\|)', s) == None
                cond6 = re.search(r'\.[a-z]', s) == None
                cond7 = re.search(r'(newsletter|email|suscribe)', s) == None, 
                cond8 = re.search(r'None', s) == None
                if (cond0 and cond1 and cond2 and cond3 and cond4 and cond5 
                    and cond6 and cond7 and cond8):
                    sentences.append(s)
            corpus.append(''.join(str(sentences)))
            

    news_corpus = []
    for piece in corpus:            
        piece = general_cleaner(piece)
        news_corpus.append(piece)
           
    news_corpus = list(set(news_corpus))
        
    return news_corpus
         


# Functions to clean, cut in 3000 to 6000 characters, 
# and convert the downloaded text files into dataframe observations.
    
# To clean and cut:

    
## LITERATURE
    
# Short fiction:
def resource_main(paths):
    resource_corpora = short_fic_cleaner(paths, 'resource_path')
    resource_corpora = [norm(i) for i in resource_corpora if not norm(i) == None]
    resource_corpora = sum(resource_corpora, [])
    return resource_corpora


def electric_main(paths):
    electric_corpora = short_fic_cleaner(paths, 'electric_path')
    electric_corpora = [norm(i) for i in electric_corpora if not norm(i) == None]
    electric_corpora = sum(electric_corpora, [])
    return electric_corpora


def usa_main(paths):
    usa_corpora = short_fic_cleaner2(paths, 'usa_path')
    usa_corpora = [norm(i) for i in usa_corpora if not norm(i) == None]
    usa_corpora = sum(usa_corpora, [])
    return usa_corpora


def ny_fiction_main(paths):
    ny_fiction_corpora = short_fic_cleaner2(paths, 'ny_fiction_path')
    ny_fiction_corpora = [norm(i) for i in ny_fiction_corpora if not norm(i) == None]
    ny_fiction_corpora = sum(ny_fiction_corpora, [])
    return ny_fiction_corpora


def eeuu_main(paths):
    eeuu_corpora = eeuu_cleaner(paths)
    eeuu_corpora = [norm(i) for i in eeuu_corpora if not norm(i) == None]
    eeuu_corpora = sum(eeuu_corpora, [])
    return eeuu_corpora


def waccamaw_main(paths):
    waccamaw_corpora = waccamaw_cleaner(paths)
    waccamaw_corpora = [norm(i) for i in waccamaw_corpora if not norm(i) == None]
    waccamaw_corpora = sum(waccamaw_corpora, [])
    return waccamaw_corpora


def arab_st_main(paths):
    arab_corpora = arab_cleaner(paths)
    arab_corpora = [norm(i) for i in arab_corpora if not norm(i) == None]
    arab_corpora = sum(arab_corpora, [])
    return arab_corpora


# Long fiction:
def adelaide_main(paths):
    adelaide_corpus = adelaide_cleaner(paths)
    adelaide_corpus = [norm(i) for i in adelaide_corpus if not norm(i) == None]
    adelaide_corpus = sum(adelaide_corpus, [])
    return adelaide_corpus



## NOT LITERATURE
    
# Linguistic blogs:
def oxford_main(paths):
    oxford_corpora = ling_blogs_cleaner(paths, 'oxford_path')
    oxford_corpora = [norm(i) for i in oxford_corpora if not norm(i) == None]
    oxford_corpora = sum(oxford_corpora, [])
    return oxford_corpora


def collins_main(paths):
    collins_corpora = ling_blogs_cleaner(paths, 'collins_path')
    collins_corpora = [norm(i) for i in collins_corpora if not norm(i) == None]
    collins_corpora = sum(collins_corpora, [])
    return collins_corpora


# Wikipedia:
def wiki_main(paths):
    wiki_corpora = wiki_cleaner(paths)
    wiki_corpora = [norm(i) for i in wiki_corpora if not norm(i) == None]
    wiki_corpora = sum(wiki_corpora, [])
    return wiki_corpora


# News:
def nytimes_main(paths):
    nytimes_corpora = news_cleaner(paths, 'nytimes_path')
    nytimes_corpora = [norm(i) for i in nytimes_corpora if not norm(i) == None]
    nytimes_corpora = sum(nytimes_corpora, [])
    return nytimes_corpora


def washington_main(paths):
    washington_corpora = news_cleaner(paths, 'washington_path')
    washington_corpora = [norm(i) for i in washington_corpora if not norm(i) == None]
    washington_corpora = sum(washington_corpora, [])
    return washington_corpora


def independent_main(paths):
    independent_corpora = news_cleaner(paths, 'independent_path')
    independent_corpora = [norm(i) for i in independent_corpora if not norm(i) == None]
    independent_corpora = sum(independent_corpora, [])
    return independent_corpora


def bbc_main(paths):
    bbc_corpora = news_cleaner(paths, 'bbc_path')
    bbc_corpora = [norm(i) for i in bbc_corpora if not norm(i) == None]
    bbc_corpora = sum(bbc_corpora, [])
    return bbc_corpora


def guardian_main(paths):
    guardian_corpora = news_cleaner(paths, 'guardian_path')
    guardian_corpora = [norm(i) for i in guardian_corpora if not norm(i) == None]
    guardian_corpora = sum(guardian_corpora, [])
    return guardian_corpora


def latimes_main(paths):
    latimes_corpora = news_cleaner(paths, 'latimes_path')
    latimes_corpora = [norm(i) for i in latimes_corpora if not norm(i) == None]
    latimes_corpora = sum(latimes_corpora, [])
    return latimes_corpora


def daily_main(paths):
    daily_corpora = news_cleaner(paths, 'daily_path')
    daily_corpora = [norm(i) for i in daily_corpora if not norm(i) == None]
    daily_corpora = sum(daily_corpora, [])
    return daily_corpora


def sfchronicle_main(paths):
    sfchronicle_corpora = news_cleaner(paths, 'sfchronicle_path')
    sfchronicle_corpora = [norm(i) for i in sfchronicle_corpora if not norm(i) == None]
    sfchronicle_corpora = sum(sfchronicle_corpora, [])
    return sfchronicle_corpora


def houston_main(paths):
    houston_corpora = news_cleaner(paths, 'houston_path')
    houston_corpora = [norm(i) for i in houston_corpora if not norm(i) == None]
    houston_corpora = sum(houston_corpora, [])
    return houston_corpora


def india_main(paths):
    india_corpora = news_cleaner(paths, 'india_path')
    india_corpora = [norm(i) for i in india_corpora if not norm(i) == None]
    india_corpora = sum(india_corpora, [])
    return india_corpora


# My test set:
    
# Yelp reviews:    
def yelp_main(paths):
    
    yelp_path = paths.get('yelp_path')

    yelp_reviews = []
    usefulness = []
    comicality = []
    ice_ice_baby = []
    amount = 0
    with open(yelp_path, encoding='utf_8') as f:
        for line in f:
            review = line.split('text":"')[1].split('","useful')[0]
            if (len(review) >= 3000 and len(review) <= 6000 and detect(review) == 'en'):
                amount += 1
                useful = int(line.split('useful":')[1].split(',"funny')[0])
                funny = int(line.split('funny":')[1].split(',"cool')[0])
                cool = int(line.split('cool":')[1].split('}')[0])
                yelp_reviews.append(general_cleaner(review))
                usefulness.append(useful)
                comicality.append(funny)
                ice_ice_baby.append(cool)
                if (amount == 10000):
                    break             
                
    return yelp_reviews, usefulness, comicality, ice_ice_baby


# Amazon reviews:    
def amazon_main(paths):
    
    amazon_path = paths.get('amazon_path')

    amazon_reviews = []
    helpful_ratio = []
    amount = 0
    with open(amazon_path, encoding='utf_8') as f:
        for line in f:
            review = line.split('reviewText": "')[1].split('", "overall')[0]
            rating = tuple(line.split('"helpful": [')[1].split('], "')[0].replace("'", '').split(", "))
            if (len(review) >= 3000 and len(review) <= 6000 and detect(review) == 'en') and int(rating[1]) >= 25:
                amount += 1
                helpful = int(rating[0])/int(rating[1])
                amazon_reviews.append(general_cleaner(review))
                helpful_ratio.append(helpful)
                if (amount == 10000):
                    break             
                
    return amazon_reviews, helpful_ratio


# To convert list of texts into the column of a dataframe 
# including the source the texts belong to:
def dataframer(paths):

    # Literature
    arab_corpora = arab_st_main(paths)
    eeuu_corpora = eeuu_main(paths)
    resource_corpora = resource_main(paths)
    electric_corpora = electric_main(paths)
    usa_corpora = usa_main(paths)
    ny_fiction_corpora = ny_fiction_main(paths)
    waccamaw_corpora = waccamaw_main(paths)
    
    adelaide_corpora = adelaide_main(paths)

    
    df_arab = pd.DataFrame(pd.Series(arab_corpora))
    df_arab = df_arab.assign(source = 'arabian_stories')
    df_arab = df_arab.assign(source_cat = 'short_fiction')
    
    df_eeuu = pd.DataFrame(pd.Series(eeuu_corpora))
    df_eeuu = df_eeuu.assign(source = 'ee_uu')
    df_eeuu = df_eeuu.assign(source_cat = 'short_fiction')
    
    df_resource = pd.DataFrame(pd.Series(resource_corpora))
    df_resource = df_resource.assign(source = 'resource')
    df_resource = df_resource.assign(source_cat = 'short_fiction')
    
    df_electric = pd.DataFrame(pd.Series(electric_corpora))
    df_electric = df_electric.assign(source = 'electric')
    df_electric = df_electric.assign(source_cat = 'short_fiction')
        
    df_usa = pd.DataFrame(pd.Series(usa_corpora))
    df_usa = df_usa.assign(source = 'usa')
    df_usa = df_usa.assign(source_cat = 'short_fiction')
            
    df_ny_fic = pd.DataFrame(pd.Series(ny_fiction_corpora))
    df_ny_fic = df_ny_fic.assign(source = 'ny_fiction')
    df_ny_fic = df_ny_fic.assign(source_cat = 'short_fiction')
    
    df_waccamaw = pd.DataFrame(pd.Series(waccamaw_corpora))
    df_waccamaw = df_waccamaw.assign(source = 'waccamaw')
    df_waccamaw = df_waccamaw.assign(source_cat = 'short_fiction')


    df_adelaide = pd.DataFrame(pd.Series(adelaide_corpora))
    df_adelaide = df_adelaide.assign(source = 'adelaide')
    df_adelaide = df_adelaide.assign(source_cat = 'long_fiction')

    
    # Joining them together:
    df_lit = pd.concat([
            df_arab, df_eeuu, 
            df_resource, df_electric, 
            df_usa, df_ny_fic, 
            df_waccamaw, df_adelaide
            ]) 
    df_lit = df_lit.assign(literariness = 1)
    df_lit = df_lit.rename(columns={0:'text'})   
    
    
    # Not literature
    oxford_corpora = oxford_main(paths)
    collins_corpora = collins_main(paths)
    
    wiki_corpora = wiki_main(paths)
    
    nytimes_corpora = nytimes_main(paths)
    washington_corpora = washington_main(paths)
    independent_corpora = independent_main(paths)
    bbc_corpora = bbc_main(paths)
    guardian_corpora = guardian_main(paths)
    latimes_corpora = latimes_main(paths)
    daily_corpora = daily_main(paths)
    sfchronicle_corpora = sfchronicle_main(paths)
    india_corpora = india_main(paths)
    houston_corpora = houston_main(paths)
    
    yelp_reviews, usefulness, comicality, ice_ice_baby = yelp_main(paths)
    amazon_reviews, helpful_ratio = amazon_main(paths)

    
    df_oxford = pd.DataFrame(pd.Series(oxford_corpora))
    df_oxford = df_oxford.assign(source = 'oxford_blog_entries')
    df_oxford = df_oxford.assign(source_cat = 'linguistic_blogs')
    
    df_collins = pd.DataFrame(pd.Series(collins_corpora))
    df_collins = df_collins.assign(source = 'collins_blog_entries')
    df_collins = df_collins.assign(source_cat = 'linguistic_blogs')
    
    
    df_wiki = pd.DataFrame(pd.Series(wiki_corpora))
    df_wiki = df_wiki.assign(source = 'wikipedia')
    df_wiki = df_wiki.assign(source_cat = 'wikipedia_cat')
    
    
    df_nytimes = pd.DataFrame(pd.Series(nytimes_corpora))
    df_nytimes = df_nytimes.assign(source = 'ny_times')
    df_nytimes = df_nytimes.assign(source_cat = 'news')
    
    df_washington = pd.DataFrame(pd.Series(washington_corpora))
    df_washington = df_washington.assign(source = 'washington_post')
    df_washington = df_washington.assign(source_cat = 'news')
    
    df_independent = pd.DataFrame(pd.Series(independent_corpora))
    df_independent = df_independent.assign(source = 'independent')
    df_independent = df_independent.assign(source_cat = 'news')
    
    df_bbc = pd.DataFrame(pd.Series(bbc_corpora))
    df_bbc = df_bbc.assign(source = 'bbc')
    df_bbc = df_bbc.assign(source_cat = 'news')
    
    df_guardian = pd.DataFrame(pd.Series(guardian_corpora))
    df_guardian = df_guardian.assign(source = 'guardian')
    df_guardian = df_guardian.assign(source_cat = 'news')
        
    df_latimes = pd.DataFrame(pd.Series(latimes_corpora))
    df_latimes = df_latimes.assign(source = 'la_times')
    df_latimes = df_latimes.assign(source_cat = 'news')

    df_daily = pd.DataFrame(pd.Series(daily_corpora))
    df_daily = df_daily.assign(source = 'daily')
    df_daily = df_daily.assign(source_cat = 'news')
    
    df_sfchronicle = pd.DataFrame(pd.Series(sfchronicle_corpora))
    df_sfchronicle = df_sfchronicle.assign(source = 'sfchronicle')
    df_sfchronicle = df_sfchronicle.assign(source_cat = 'news') 
    
    df_india = pd.DataFrame(pd.Series(india_corpora))
    df_india = df_india.assign(source = 'india')
    df_india = df_india.assign(source_cat = 'news')
      
    df_houston = pd.DataFrame(pd.Series(houston_corpora))
    df_houston = df_houston.assign(source = 'houston')
    df_houston = df_houston.assign(source_cat = 'news')


    df_yelp = pd.DataFrame(pd.Series(yelp_reviews))
    df_yelp = df_yelp.assign(source = 'yelp_reviews')
    df_yelp = df_yelp.assign(source_cat = 'reviews')
    df_yelp = df_yelp.assign(usefulness = pd.Series(usefulness))
    df_yelp = df_yelp.assign(comicality = pd.Series(comicality))
    df_yelp = df_yelp.assign(ice_ice_baby = pd.Series(ice_ice_baby))
    
    df_amazon = pd.DataFrame(pd.Series(amazon_reviews))
    df_amazon = df_amazon.assign(source = 'amazon_reviews')
    df_amazon = df_amazon.assign(source_cat = 'reviews')
    df_amazon = df_amazon.assign(helpfulness = pd.Series(helpful_ratio))

    df_reviews = df_yelp.merge(df_amazon, how = 'outer')

    df_non_lit = pd.concat([
            df_oxford, df_collins, 
            df_wiki, 
            df_nytimes, df_washington, 
            df_independent, df_bbc, 
            df_guardian, df_latimes, 
            df_daily, df_sfchronicle, 
            df_india, df_houston
            ]) 
    
    df_non_lit = df_non_lit.merge(df_reviews, how = 'outer')
    df_non_lit = df_non_lit.assign(literariness = 0)
    df_non_lit = df_non_lit.rename(columns={0:'text'})
    

    # Creating and saving the chief dataset:
    df = df_lit.append(df_non_lit)
    df.to_csv(paths['chief_df_path'])
    
    return df



#df = dataframer(paths)


