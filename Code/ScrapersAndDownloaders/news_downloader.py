#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:45:13 2018

@author: rita

Downloads news articles from the New York Times, the Washington Post, 
the Independent, the BBC, the Guardian, the Los Angeles Times, 
the NY Daily News, the San Francisco Chronicle, the Houston Chronicle,
and the Times of India.

"""


import os
from bs4 import BeautifulSoup as bs
import re
from tqdm import tqdm

classes_path = '/Users/rita/Google Drive/DSR/DSR Project/Code/'
os.chdir(classes_path)
import prepro_funcs as prep


# Variables:

# External paths:
nytimes_expath = "https://www.nytimes.com/section/"
washington_expath = "https://www.washingtonpost.com/"
independent_expath = "https://www.independent.co.uk/"
bbc_expath = "http://www.bbc.com/"
guardian_expath = "https://www.theguardian.com/"
latimes_expath = "http://www.latimes.com/"
daily_expath = "http://www.nydailynews.com/"
sfchronicle_expath = "https://www.sfchronicle.com/"
houston_expath = "https://www.houstonchronicle.com/"
india_expath = "https://timesofindia.indiatimes.com/"

# sections:
nytimes_sections = """
world, opinion, us, health, 
sports, arts, books, fashion, 
food, travel, magazine, 
obituaries, reader-center, 
insider, politics, nyregion, 
business, technology, science
""".replace('\n', '').strip().split(',')

washington_sections =  """
politics, opinions, sports, local,
national, world, business, tech,
lifestyle, entertainment
""".replace('\n', '').strip().split(',')

independent_sections =  """
extras, news/uk, news/world, news/science,
news/health, topic/brexit, news/media, news/long_reads,
infact, voices, life-style/fashion, travel, 
life-style/gadgets-and-tech, life-style/food-and-drink/recipes, 
news/business/indyventure, arts-entertainment/tv, 
arts-entertainment/films, arts-entertainment/music, 
arts-entertainment/books, arts-entertainment/art, 
arts-entertainment/theatre-dance
""".replace('\n', '').strip().split(',')

bbc_sections =  """
news, news/world, news/uk, news/business, news/newsbeat
news/technology, news/science_and_environment, news/stories, 
news/entertainment_and_arts, news/health, earth, 
earth/columns/discoveries, culture, travel,
travel/columns/adventure-experience, travel/columns/culture-identity 
""".replace('\n', '').strip().split(',')

guardian_sections =  """
international, uk/lifeandstyle, world, uk-news, science, 
music/classical-music-and-opera, cities, global-development, 
football, music, books, artanddesign, games, stage, fashion, 
uk/technology, uk/business, uk/environment, uk/tv-and-radio, 
uk/film, uk/travel, uk/money, observer, 
lifeandstyle/love-and-sex, lifeandstyle/food-and-drink, 
lifeandstyle/home-and-garden, lifeandstyle/health-and-wellbeing, 
lifeandstyle/women, lifeandstyle/family, tone/recipes, 
tone/obituaries, tone/editorials, index/contributors,
sport/rugby-union, sport/cycling
""".replace('\n', '').strip().split(',')

latimes_sections =  """
local, politics, business, nation, entertainment, 
opinion, food, sports, business/technology
world, local/obituaries, business/realestate, 
style, science, travel
""".replace('\n', '').strip().split(',')

daily_sections =  """
new-york, sports, news/crime, entertainment/gossip, 
news/national, news/politics, news/world,
entertainment/movies, entertainment/tv, 
entertainment/music, entertainment/theater-arts, 
life-style/health, life-style/eats, opinion
""".replace('\n', '').strip().split(',')

sfchronicle_sections =  """
local, local/sanfrancisco/, local/bayarea, 
elections, us-world, us-world/science, opinion, 
sports, business, entertainment, food, 
entertainment/arts-theater, entertainment/books,
lifestyle/travel, lifestyle, style, 
foodandhome/home-design, investigations, 
2018/in-depth-projects, 2017/in-depth-projects,
2016/in-depth-projects, 2015/in-depth-projects,
""".replace('\n', '').strip().split(',')

houston_sections =  """
local, local/gray-matters, politics, 
us-world, us-world/space, us-world/world, us-world/us, 
us-world/science-environment, us-world/politics-policy, 
sports, business, business/energy, business/real-estate, 
business/real-estate/deal-of-the-week, business/bizfeed, 
business/businesspeople, business/medical, business/retail, 
business/retail, business/personal-finance, business/top-workplaces, 
business/chron-100, business/bloomberg, opinion, 
opinion/editorials, opinion/outlook, opinion/columnists, 
author/brian-t-smith, author/chris-tomlinson, author/lisa-falkenberg, 
author/jerome-solomon, author/erica-grieder, author/john-mcclain, 
entertainment, entertainment/books, entertainment/movies-and-tv, 
entertainment/music-theater-and-arts, entertainment/rodeo, 
entertainment/restaurants-bars, flavor, lifestyle, 
lifestyle/society, lifestyle/style, lifestyle/escapes, 
techburger, local/specialsections
""".replace('\n', '').strip().split(',')

india_sections =  """
city, life-style, india, 
life-style/events, life-style/fashion, life-style/home-garden,
life-style/books, life-style/spotlight, life-style/beauty, 
life-style/health-fitness, life-style/relationships, 
sports, business, world, world/mad-mad-world, 
world/middle-east, world/china, world/europe, world/us, 
elections/assembly-elections/karnataka, 
?utm_source=TOInewHP_TILwidget&utm_medium=NavLi&utm_campaign=TOInewHP
""".replace('\n', '').strip().split(',')



def news_down(external_path, internal_path, sections):
   
    themes = []
    for i in sections:
        themes.append(str(external_path + str(i).replace(" ",'')))
    
    links = []
    for section in tqdm(themes):
        original_content = prep.get_proxies(section)
        soup = bs(original_content.content, "lxml")
        for link in soup.find_all('a'):
            if (re.search(r'\d+', str(link.get('href'))) or 
                re.search(r'\.html', str(link.get('href'))) or
                re.search(r'\\', str(link.get('href')))):
                links.append(str(link.get('href')))
                
    links = list(set(links))
    
    for link in tqdm(links):
        if re.match('/', link):
            link = external_path + str(link)
        try:
            content1 = prep.get_proxies(link)
            soup1 = bs(content1.content, "lxml")
            strings = []
            corpus = []
            for link1 in soup1.find_all('p'):
                for i in link1.contents:
                    strings.append(str(i.string))
            corpus = ' '.join(strings)
            file = open(os.path.join(internal_path, link.split('/')[-1].replace('.html', '') + '.txt'), 'w')
            file.write(corpus)
            file.close()
        except:
            continue
        
    return corpus

                           

nytimes_corpus = news_down(nytimes_expath, paths.get('nytimes_path'), nytimes_sections)
washington_corpus = news_down(washington_expath, paths.get('washington_path'), washington_sections)
independent_corpus = news_down(independent_expath, paths.get('independent_path'), independent_sections)
bbc_corpus = news_down(bbc_expath, paths.get('bbc_path'), bbc_sections)
guardian_corpus = news_down(guardian_expath, paths.get('guardian_path'), guardian_sections)
latimes_corpus = news_down(latimes_expath, paths.get('latimes_path'), latimes_sections)
daily_corpus = news_down(daily_expath, paths.get('daily_path'), daily_sections)
sfchronicle_corpus = news_down(sfchronicle_expath, paths.get('sfchronicle_path'), sfchronicle_sections)
houston_corpus = news_down(houston_expath, paths.get('houston_path'), houston_sections)
india_corpus = news_down(india_expath, paths.get('india_path'), india_sections)

