#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:52:38 2018

@author: rita

Pragmatrix Main Script

"""

# Imports
import os

import requests
import re
from collections import OrderedDict
#import gensim
#import spacy
import pickle
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelBinarizer


# Functions and classes
classes_path = '/Users/rita/Google Drive/DSR/DSR Project/Code/'
os.chdir(classes_path)
import classy_n_funky as cl
import feature_trove as feat

# Variables:
spacy_model = 'en_core_web_lg'
 
dfs_path = '/Users/rita/Google Drive/DSR/DSR Project/Dataset/datasets'
paths = {
        'chief_df_path': os.path.join(dfs_path, 'primary_dataset.csv'), 
        'spacy_df_path': os.path.join(dfs_path, 'spacy_dataset.csv'), 
        'bows_df_path': os.path.join(dfs_path, 'bows_dataset.csv'), 
        'colls_df_path': os.path.join(dfs_path, 'colls_dataset.csv'), 
        'tfidf_df_path': os.path.join(dfs_path, 'tfidfs_dataset.csv'),
        'vecs_df_path': os.path.join(dfs_path, 'doc2vec_dist_dataset.csv'), 
        'feature_dict_path': os.path.join(dfs_path, 'feature_dict.pkl')
        }

common_path = '/Users/rita/Google Drive/DSR/DSR Project/Dataset'
cleaner_paths = {
        'eeuu_path' : os.path.join(common_path, 'short_fiction', 'eeuu'),
        'resource_path' : os.path.join(common_path, 'short_fiction', 'eyewriters'),
        'usa_path' : os.path.join(common_path, 'short_fiction', 'usa'),
        'arab_path' : os.path.join(common_path, 'short_fiction', 'arabian_stories'),
        'electric_path' : os.path.join(common_path, 'short_fiction', 'electric'),
        'ny_fiction_path' : os.path.join(common_path, 'short_fiction', 'ny_fiction'),        
        'adelaide_path' : os.path.join(common_path, 'long_fiction', 'adelaide'),
        'bookshelf_path' : os.path.join(common_path, 'long_fiction', 'bookshelf', 'agatha_christie'),        
        'oxford_path' : os.path.join(common_path, 'linguistic_blogs', 'oxford_blog'),
        'collins_path' : os.path.join(common_path, 'linguistic_blogs', 'collins_blog'),        
        'wiki_path' : os.path.join(common_path, 'wikipedia'),        
        'nytimes_path' : os.path.join(common_path, 'news', 'nytimes'),
        'washington_path' : os.path.join(common_path, 'news', 'washington'),
        'independent_path' : os.path.join(common_path, 'news', 'independent'),
        'bbc_path' : os.path.join(common_path, 'news', 'bbc'),
        'guardian_path' : os.path.join(common_path, 'news', 'guardian'),
        'latimes_path' : os.path.join(common_path, 'news', 'latimes'),
        'daily_path' : os.path.join(common_path, 'news', 'daily'),
        'sfchronicle_path' : os.path.join(common_path, 'news', 'sfchronicle'),
        'india_path' : os.path.join(common_path, 'news', 'india'),
        'houston_path' : os.path.join(common_path, 'news', 'houston')
        }
    

columns_to_idf = ['text', 'POS_tag', 'syntactic_dependency']

cols_to_remove = ['text', 'source', 'source_cat', 'literariness', 
                  'length_text', 'n_words', 'n_sentences', 
                  'POS_tag', 'syntactic_dependency']


os.chdir(common_path) 
# Loading  the lastly-modified df and a dictionary with the different feature columns 
# grouped by how they have been extracted:
chief_df, rich_df, feature_dict = feat.dfs_initializer(paths, cleaner_paths, columns_to_idf)
    
#rich_df = rich_df.sample(250)
    
## Modelling options:

options = input("""
                Choose one of the available options, which can be found specified below 
                (must be a number between 1 and 4):
                    
                Option 1:
                    model = logistic regession
                    features = basic, colls, tfidfs
                    y = literariness
                    
                Option 2:
                     model = doc2vec
                     features = the original text observations and Spacy's tags
                     y = literariness, source, source_cat
                     
                 Option 3:
                     model = xgboost
                     features = all the features of rich_df but the bows
                     y = source_cat
                    
                Option 4: 
                     model = LSTM 
                     features = only the original text observations
                     y = source_cat
                    
                """)

options = int(options)


if options == 1:  
    
    print("""
          You chose to train a logistic regression 
          to predict the degree of literariness of each text observation.
          
          """)

    linear_feat = [
            'x', 
            'targets', 
            'basics', 
            'collocations', 
            'spacy', 
            'bow_1gram', 
            'bow_2gram',  
            ]
    
    columns_to_train = [i for a in linear_feat for i in feature_dict[a]]
    log_reg_df = rich_df.loc[:, rich_df.columns.isin(columns_to_train)][:]
    y_log_reg = log_reg_df.pop('literariness')
    
    log_reg_model, log_reg_score = cl.log_reg(log_reg_df, y_log_reg, cols_to_remove)
 

  
if options == 2:
    
    print("""
          You chose to train a doc2vec.
          
          """)

    wecker_feat = [
            'x',
            'spacy', 
            'targets', 
            ]
    
    columns_to_train = [i for a in wecker_feat for i in feature_dict[a]]
    wecker_df = rich_df.loc[:, rich_df.columns.isin(columns_to_train)][:]

    y_cols = ['literariness', 'source', 'source_cat']
    y_wecker = wecker_df[y_cols]
    wecker_df.drop(y_cols, axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(wecker_df, y_wecker, test_size = 0.3, random_state = 123, shuffle = True)
    
    df_vecs, feat_wecker, preds_dict = cl.Doc2wecker().fit(X_train, y_train).transform(X_test)
        
    source_cat_preds = np.array([value for key, value in OrderedDict(sorted(preds_dict.items())).items() if re.search('source_cat', key)]).T
    source_preds = np.array([
            value for key, value in OrderedDict(sorted(preds_dict.items())).items() if re.search('source', key) and not (re.search('source_cat', key))
            ]).T
    
    lb_source_cat = LabelBinarizer()
    y_test_sc_trans = lb_source_cat.fit(y_train['source_cat']).transform(y_test['source_cat'])
        
    lb_source = LabelBinarizer() 
    y_test_s_trans = lb_source.fit(y_train['source']).transform(y_test['source'])
    
    score_lit = roc_auc_score(np.array(y_test['literariness']), np.array(preds_dict.get('literariness_1')))
    score_source_cat = log_loss(y_test_sc_trans, source_cat_preds)
    score_source = log_loss(y_test_s_trans, source_preds)
    print("""
          The scores for literariness: {};
          source_cat: {}; 
          and source: {}.
          """.format(score_lit, score_source_cat, score_source))
 

if options == 3:
        
    print("""
          You chose to train a xgboost model 
          to predict the categories the observations' sources belong to.
          
          """)

    non_xgb_feat = [
            'bow_1gram', 
            'bow_2gram',
            ]
    
    xgb_feat = list(set(list(feature_dict.keys())) - set(non_xgb_feat))
    columns_to_train = [i for a in xgb_feat for i in feature_dict[a]]
    xgb_df = rich_df.loc[:, rich_df.columns.isin(columns_to_train)][:]
    y_tree = xgb_df.pop('source_cat')
    
    xgboost_model, xgboost_score = cl.multi_class_xgboost(xgb_df, y_tree, cols_to_remove)
    xgb.plot_importance(xgboost_model.best_estimator_.named_steps['clf'], max_num_features=20, importance_type='gain')
    


if options == 4:
    
    print("""
      You chose to train a LSTM neural network 
      to predict the sites the observations were extracted from.
      
      """)

    lstm_feat = chief_df.text
    y_lstm = chief_df.pop('source_cat')
    
    lstm_model, accuracy = cl.multi_class_LSTM(lstm_feat, y_lstm)
    
    history = lstm_model.summary()
    print(history)
    print(accuracy*100)