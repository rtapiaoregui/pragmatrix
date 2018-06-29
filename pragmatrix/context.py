#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:05:34 2018

@author: rita
"""
import os
import pickle
from keras.models import load_model


def contextualizer():

    # Paths:
    common_path = '/Users/rita/Google Drive/DSR/DSR Project'
    code_path = os.path.join(common_path, 'pragmatrix')
    data_path = os.path.join(common_path, 'Data')
    corpora_path = os.path.join(data_path, 'corpus')
    sets_path = os.path.join(data_path, 'datasets')
    
       
    paths = {
            'code_path' : code_path,
            'irr_verbs_path' : os.path.join(data_path, 'tools', 'irregular_verbs.csv'),
            'common_wd_path' : os.path.join(data_path, 'tools', 'common_words.csv'),
            'colls_dict_path' : os.path.join(data_path, 'tools', 'collocations_dict.pickle'),
            'glove_path' : os.path.join(data_path, 'tools', 'glove.840B.300d.txt'),
            'embedd_path' : os.path.join(data_path, 'tools', 'embeddings.pickle'),
            'nli_tokenizer_path' : os.path.join(data_path, 'models', 'nli_tokenizer.pickle'),
            'lstm_tokenizer_path' : os.path.join(data_path, 'models', 'lstm_tokenizer.pickle'),
            'nli_model_path' : os.path.join(data_path, 'models', 'NLI_nn.h5'),
            'terminator_path' : os.path.join(data_path, 'models', 'terminator.pkl'),
            'lstm_weights_path' : os.path.join(data_path, 'models', 'lstm_weights.hdf5'),
            'tree_plots_path' : os.path.join(data_path, 'plots', 'trees'), 
            'multi_nli_paths' : os.path.join(sets_path, 'multinli_1.0', 'multinli_1.0_train.jsonl'),
            'snli_paths' : os.path.join(sets_path, 'snli_1.0', 'snli_1.0_train.jsonl'),
            'chief_df_path' : os.path.join(sets_path, 'primary_dataset.csv'),
            'train_df_path' : os.path.join(sets_path, 'train_set.csv'),
            'test_df_path' : os.path.join(sets_path, 'test_set.csv'), 
            'rich_df_path' : os.path.join(sets_path, 'feature_enriched_dataset.csv'), 
            'feature_dict_path' : os.path.join(sets_path, 'feature_dict.pkl'),
            'df_colls_path' : os.path.join(sets_path, 'df_colls.pkl'), 
            'df_tfidf_path' : os.path.join(sets_path, 'df_tfidf.pkl'), 
            'df_doc2vec_path' : os.path.join(sets_path, 'df_doc2vec.pkl'), 
            'eeuu_path' : os.path.join(corpora_path, 'short_fiction', 'eeuu'),
            'resource_path' : os.path.join(corpora_path, 'short_fiction', 'eyewriters'),
            'usa_path' : os.path.join(corpora_path, 'short_fiction', 'usa'),
            'arab_path' : os.path.join(corpora_path, 'short_fiction', 'arabian_stories'),
            'electric_path' : os.path.join(corpora_path, 'short_fiction', 'electric'),
            'ny_fiction_path' : os.path.join(corpora_path, 'short_fiction', 'ny_fiction'),
            'waccamaw_path' : os.path.join(corpora_path, 'short_fiction', 'waccamaw'),
            'adelaide_path' : os.path.join(corpora_path, 'long_fiction', 'adelaide'),
            'bookshelf_path' : os.path.join(corpora_path, 'long_fiction', 'bookshelf', 'agatha_christie'),        
            'oxford_path' : os.path.join(corpora_path, 'linguistic_blogs', 'oxford_blog'),
            'collins_path' : os.path.join(corpora_path, 'linguistic_blogs', 'collins_blog'),        
            'wiki_path' : os.path.join(corpora_path, 'wikipedia'),        
            'nytimes_path' : os.path.join(corpora_path, 'news', 'nytimes'),
            'washington_path' : os.path.join(corpora_path, 'news', 'washington'),
            'independent_path' : os.path.join(corpora_path, 'news', 'independent'),
            'bbc_path' : os.path.join(corpora_path, 'news', 'bbc'),
            'guardian_path' : os.path.join(corpora_path, 'news', 'guardian'),
            'latimes_path' : os.path.join(corpora_path, 'news', 'latimes'),
            'daily_path' : os.path.join(corpora_path, 'news', 'daily'),
            'sfchronicle_path' : os.path.join(corpora_path, 'news', 'sfchronicle'),
            'india_path' : os.path.join(corpora_path, 'news', 'india'),
            'houston_path' : os.path.join(corpora_path, 'news', 'houston'),
            'yelp_path' : os.path.join(corpora_path, 'yelp', 'review.json'), 
            'amazon_path' : os.path.join(corpora_path, 'amazon', 'movies_and_TV_reviews.json')
            }
     
        
    # Input variables:
    spacy_model = 'en_core_web_lg'
    
    columns_to_idf = ['text', 'POS_tag', 'syntactic_dependency']
    
    cols_to_remove = ['text', 'source', 'source_cat', 'literariness',
                      'comicality', 'helpfulness', 'ice_ice_baby', 
                      'usefulness', 'POS_tag', 'syntactic_dependency']
    
    with open(paths['colls_dict_path'], 'rb') as file:
        dict_colls = pickle.load(file)
            
    nli_model = load_model(paths['nli_model_path'])
    with open(paths['nli_tokenizer_path'], 'rb') as handle:
        nli_tokenizer = pickle.load(handle)

    return (paths, 
            spacy_model, 
            columns_to_idf, 
            cols_to_remove, 
            dict_colls, 
            nli_tokenizer, 
            nli_model)
