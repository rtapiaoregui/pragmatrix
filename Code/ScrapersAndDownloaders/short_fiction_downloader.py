#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:21:30 2018

@author: rita

Downloader of contemporary literature: short stories from 
- Every Writers Resource
- the American Short Fiction online magazine
- the American Literature online magazine
- Arabian Stories
- Electric Literature
- the New Yorker Fiction Magazine
- the Waccamaw journal of contemporary literature

"""

import requests
from bs4 import BeautifulSoup as bs
import os
import re
from tqdm import tqdm


# Functions and classes
classes_path = '/Users/rita/Google Drive/DSR/DSR Project/Code/'
os.chdir(classes_path)
import prepro_funcs as prep



def resource_down(resource_files):
    
    resource_website = 'https://www.everywritersresource.com/shortstories/page/'
        
    allUrls = []                
    for num in tqdm(range(1, 53)):
        site = resource_website + str(num) + '/'
        r_obj = prep.get_proxies(site)
        soup = bs(r_obj.text, 'lxml')
        tags = soup.find_all('a')
        urls = list(set([tag.get('href') for tag in tags if ('title' in tag.attrs and ('rel' not in tag.attrs or tag.get('rel')==['bookmark']))]))
        allUrls.append(urls)
    
    allUrls = list(set([url for lurl in allUrls for url in lurl]))
        
    resource_corpus = []
    names = []
    for url in tqdm(allUrls):
        content = requests.get(url).content
        soup = bs(content, 'lxml')
        strings = []
        try:
            for link0 in soup.find_all('div', {'class':'post-content'}):   
                for link1 in link0.find_all('p'):
                    for i in link1.contents:
                        strings.append(str(i.string))
            resource_corpus.append(''.join(str(strings)))
            names.append(url)
            
        except:
            continue
                
    for elem in range(len(resource_corpus)):
        file = open(os.path.join(resource_files, names[elem].split('/')[-2] + '.txt'), 'w')
        file.write(resource_corpus[elem])
        file.close()
            
    return resource_corpus



def eeuu_down(eeuu_files):
    
    eeuu_page = "http://americanshortfiction.org/category/web-exclusives/page/"
    
    # First, I want to get all the unique links leading to posts hosting short stories.
    links = []
    for page in range(1, 13):      
        original_page = eeuu_page + str(page) + "/"
        original_content = requests.get(original_page).content
        soup = bs(original_content, "lxml")
        for link in soup.find_all('a'):
            if re.search(r'/201\d/', str(link.get('href'))):
                links.append(str(link.get('href')))
                
    links = list(set(links))
    
    # Secondly, I download and save all the short stories in files.
    eeuu_corpus = []
    for link in tqdm(links):
        content1 = requests.get(link).content
        soup1 = bs(content1, "lxml")
        strings = []
        for link0 in soup1.find_all('div', {'class':'entry-content'}):
            for link1 in link0.find_all('p'):
                for i in link1.contents:
                    if not re.match(r'\<img', str(i.string)):
                        strings.append(str(i.string))
        eeuu_corpus.append(''.join(str(strings)))
        
    for elem in range(len(eeuu_corpus)):
        file = open(os.path.join(eeuu_files, str(links[elem].split('/')[-2] + '.txt')), 'w')
        file.write(eeuu_corpus[elem])
        file.close()
    
    return eeuu_corpus



def usa_down(usa_files):
       
    usa_path = 'https://americanliterature.com/100-great-short-stories'
    root = 'https://americanliterature.com'

    original_content = prep.get_proxies(usa_path)
    soup = bs(original_content.content, "lxml")
    links = []   
    for url in soup.find_all('span'):
        for a in url.find_all('a', href = True):
            links.append(a['href'])
    
    corpus = []
    for i in tqdm(links):
        more_content = prep.get_proxies(root + str(i))
        soup = bs(more_content.content, "lxml")
        sents = []
        for parag in soup.find_all('p'):
            sents.append(str(parag.string))
        corpus.append(''.join(str(sents)))
    
    for elem in range(len(corpus)):  
        with open(os.path.join(usa_files,
                               links[elem].split('/author/')[1].replace('/short-story/','__').replace('-', '_') + '.txt'
                               ), 'w') as file:
            file.write(str(corpus[elem]))

    return corpus
          

       
def arabian_stories_down(arab_st_files): 

    download_anyway = True
        
    mine = ['https://arabianstories.com/project-description/',
     'https://arabianstories.com/faq/',
     'https://arabianstories.com/translators-note/',
     'https://arabianstories.com/team/',
     'https://arabianstories.com/contact-us/',
     'https://arabianstories.com/literary-contests/',
     'https://arabianstories.com/cassiopeia/',
     'https://arabianstories.com/and-the-champs-are/',
     'https://arabianstories.com/two-thousand-nights-awakening/',
     'https://arabianstories.com/kickstarter-campaign/',
     'https://arabianstories.com/two-thousand-nights-awakening/when-a-tree-is-shaken/',
     'https://arabianstories.com/two-thousand-nights-awakening/final-results/',
     'https://arabianstories.com/map/',
     'https://arabianstories.com/blog/',
     'https://arabianstories.com/shop/'] 

    
    if download_anyway == True:
        print("Downloading corpus...")
        
        arabian_stories = []
        
        for num in range(1, 17):
            
            site = 'https://arabianstories.com/category/stories/page/' + str(num) + '/' 
            text = requests.get(site).text
            soup = bs(text, 'lxml')
                
            tags = soup.find_all('a')
            tags = [tag.get('href') for tag in tags if len(tag.attrs) == 1]
            tags = [tag for tag in tags if re.match(r'(https://arabianstories.com/)', 
                                                    str(tag)) and (re.search(r'(category)', 
                                                        str(tag))==None and re.search(r'(author)', 
                                                           str(tag))==None)]
            tags = [tag for tag in tags if not tag in mine]
            
            for tag in list(set(tags)):
                print(tag)
                r = requests.get(tag).text  
                soup = bs(r, 'lxml')
                corpus = soup.get_text().split('Rita Tapia Oregui')[2]
                corpus = corpus.split('Tweet')[0]
                internal_path = os.path.join(arab_st_files, ('_'.join(soup.title.string.split(' -')[0].lower().split(' ')) + '.txt'))

                with open(internal_path, 'w') as wr:
                    wr.write(corpus)
                                
                arabian_stories.append(corpus)
                    
        return arabian_stories
                    


def electric_down(electric_files):
       
    electric_path = 'https://electricliterature.com/recommended-reading-archives-7eb326fa8cf4'
    
    original_content = prep.get_proxies(electric_path)
    soup = bs(original_content.content, "lxml")
    links = []
    for a in soup.find_all('a', href=True):
        links.append(a['href'])
    
    links = list(set(links))
    
    corpus = []
    urls = []
    for url in tqdm(links):
        try:
            new_content = prep.get_proxies(url)
            soup = bs(new_content.content, "lxml")
            sents = []
            for parag in soup.find_all('p'):
                sents.append(str(parag.string))
            corpus.append(''.join(str(sents)))
            urls.append(url)
            
        except:
            continue
        
    for elem in range(len(corpus)):
        urls[elem] = re.sub(r'\-\d+\.*', '', urls[elem])
        with open(os.path.join(electric_files,
                               urls[elem].split('/')[-1].replace('-','_') + '.txt'
                               ), 'w') as file:
            file.write(str(corpus[elem])) 
    
    return corpus

                

def ny_fiction_down(ny_fiction_files):
   
    ny_fiction_path = 'https://www.newyorker.com/magazine/fiction'
    root = "https://www.newyorker.com"
    
    links = []
    for i in range(1, 95):
        if i == 1:
            path = ny_fiction_path
        else:
            path = ny_fiction_path + '/page/' + str(i)

        original_content = prep.get_proxies(path)
        soup = bs(original_content.content, "lxml")
        for a in soup.find_all('a', href=True):
            href = str(a['href'])
            if re.search(r'magazine/20\d{2}/\d{2}/\d{2}/\w+', href):
                links.append(href)

    links = list(set(links))
    
    corpus = []
    names = []
    for url in tqdm(links):
        try:
            url = root + url
            new_content = prep.get_proxies(url)
            soup = bs(new_content.content, "lxml")
            sents = []
            for parag in soup.find_all('p'):
                sents.append(str(parag.string))
            corpus.append(''.join(str(sents)))
            names.append(url)
            
        except:
            continue
        
    for elem in tqdm(range(len(corpus))):
        names[elem] = re.sub(r'\-\d+\.*', '', names[elem])
        with open(os.path.join(ny_fiction_files,
                               names[elem].split('/')[-1] + '.txt'
                               ), 'w') as file:
            file.write(str(corpus[elem]))    
    
    return corpus



def waccamaw_down(waccamaw_files):
    
    waccamaw_path = 'http://waccamawjournal.com/category/fiction/'
    
    links = []
    for i in range(2):
        if i == 0:
            path = waccamaw_path
        else:
            path = waccamaw_path + '/page/2/'
            
        original_content = prep.get_proxies(path)
        soup = bs(original_content.content, "lxml")
        for a in soup.find_all('a', href=True):
            href = str(a['href'])
            if re.search(r'waccamawjournal.com/fiction/[a-z\-]+', href):
                links.append(href)
                
    corpus = []
    names = []
    for url in tqdm(links):
        new_content = prep.get_proxies(url)
        soup = bs(new_content.content, "lxml")
        sents = []
        for parag in soup.find_all('p'):
            sents.append(str(parag.string))
        corpus.append(''.join(str(sents)))
        names.append(url)
        
    for elem in tqdm(range(len(corpus))):
        names[elem] = names[elem].split('fiction/')[1].replace('-', '_')
        with open(os.path.join(waccamaw_files,
                               names[elem].split('/')[0] + '.txt'
                               ), 'w') as file:
            file.write(str(corpus[elem]))
            
    return corpus
    


def short_fiction_downloads(paths):
    
    arabian_stories = arabian_stories_down(paths.get('arab_path'))              
    eeuu_corpus = eeuu_down(paths.get('eeuu_path'))
    resource_corpus = resource_down(paths.get('resource_path'))
    usa_corpus = usa_down(paths.get('usa_path'))
    electric_corpus = electric_down(paths.get('electric_path'))
    ny_fiction_corpus = ny_fiction_down(paths.get('ny_fiction_path'))
    waccamaw_corpus = waccamaw_down(paths.get('waccamaw_path'))
    
    return (arabian_stories, eeuu_corpus, resource_corpus, usa_corpus, 
            electric_corpus, ny_fiction_corpus, waccamaw_corpus)


miscellanea = short_fiction_downloads(paths)
