#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:25:16 2018

@author: rita

Feature enriching script
"""
# Imports
import os
import pandas as pd
import pickle
from operator import itemgetter

# Functions and classes
# os.chdir(paths['code_path'])
from pragmatrix import classy_n_funky as cl
from pragmatrix import cleaner

from sklearn.base import BaseEstimator, TransformerMixin


    
class add_features(BaseEstimator, TransformerMixin):
      
    def __init__(self, columns_to_idf, dict_colls, nli_model, nli_tokenizer, paths,
                 pth_subset = ['df_colls_path', 'df_tfidf_path', 'df_doc2vec_path']):
        
        self.columns_to_idf = columns_to_idf
        self.dict_colls = dict_colls
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.paths = itemgetter(*pth_subset)(paths)
        
        self.colls_ = cl.collocater(self.dict_colls)
        self.tfidfer_ = cl.TFIDFer(self.columns_to_idf)
        self.doc2wecker_ = cl.Doc2wecker()
        
        attributes = [attr for attr in dir(self) 
        if not callable(getattr(self, attr)) and
        attr.endswith('_') and not attr.startswith("__")]
                
        self.attributes = attributes
        
    
    def fit(self, X, y = None):
        
        X['text'] = X['text'].map(lambda x: str(x).lower())
        self.colls_.fit(X)
        df_spacy, feat_spacy = cl.spacyer(X)
        print("Data set's size after increase with the addition of spacy's tags: \n{}\n".format(df_spacy.shape))
        self.tfidfer_.fit(df_spacy)
        self.doc2wecker_.fit(df_spacy)
        
        return self


    def transform(self, X):  
        
        X['text'] = X['text'].map(lambda x: str(x).lower())

        # Plus basic features:
        X, feat_basic = cl.bassicker(X)
        print("""
              Names of the data set's columns and data set's size 
              after increase with the addition of the basic features:
                  {}
                  {}
                  """.format(X.shape, X.columns))

        # Plus colls:
        X, feat_colls = self.colls_.transform(X)
        print("\nData set's size after increase with the addition of df_colls: \n{}\n".format(X.shape))

         # Plus spacy tags:
        X, feat_spacy = cl.spacyer(X)
        print("\nData set's size after increase with the addition of spacy's tags: \n{}\n".format(X.shape))

        # Plus Tfidfs:
        X, feat_tfidf = self.tfidfer_.transform(X)
        print("\nData set's size after increase with the addition of tfidfs: \n{}\n".format(X.shape))   

        # Plus doc2vec distances to categories and raw vectors:
        X, feat_wecker, _ = self.doc2wecker_.transform(X)
        print("\nData set's size after increase with the addition of document vectors' distances to category vectors' distances: \n{}\n".format(X.shape)) 

        X, feat_nli = cl.NLIyer(X, self.nli_model, self.nli_tokenizer)
        print("\nData set's size after increase with the addition of the nli-NN predictions on contradiction, concordance and neutrality: \n{}\n".format(X.shape)) 

        feature_dict = {**feat_basic, **feat_spacy, **feat_colls, **feat_tfidf, **feat_wecker, **feat_nli}

        return X, feature_dict
    
    
    def save(self):

        for at in range(len(self.attributes)):   
            with open(self.paths[at], 'wb') as f:
                pickle.dump(getattr(self, self.attributes[at]), f)

        
    def load(self):
        
        for at in range(len(self.attributes)):   
            with open(self.paths[at], 'rb') as f:
                temp_atr = pickle.load(f)
            setattr(self, self.attributes[at], temp_atr)



def dfs_initializer(paths, 
                    columns_to_idf, 
                    dict_colls, 
                    nli_model,
                    nli_tokenizer,
                    fit_featurizer = True,
                    sample_ratio = 1):
        
    # Loading and transforming the data set compiled last:
    feat_dict_in_wd = ('feature_dict' in globals()) or ('feature_dict' in locals())
    dfs_postfit_in_wd = os.path.exists(paths['df_colls_path'])
    chief_df_in_wd = ('df' in globals()) or ('df' in locals())
    fdict_file_yes = os.path.exists(paths['feature_dict_path'])
    chdf_file_yes = os.path.exists(paths['chief_df_path'])
  
    if not chdf_file_yes:
        print('\nBuilding the chief data set.\n')
        df = cleaner.dataframer(paths)
        print("""
              You are now in possession of the freshly-created chief or primary data set.
              The chief data set's info:
                  
              {}
                  
              Small sample:
                  
              {}
                      
              Percentage of literature: {}
              
              Amount of observations in each 'source' category:
                  
              {}
                  
              """.format(
              df.info(), 
              df.sample(15), 
              (df[(df.literariness == 1)].shape[0]/df.shape[0])*100, 
              df['source'].value_counts()[::-1]
              ))
        
        df.to_csv(paths['chief_df_path'])
            
    else:
        if not chief_df_in_wd:
            print('\nLoading the chief data set.\n')
            df = pd.read_csv(paths['chief_df_path'])
            df = df.rename(columns = {'Unnamed: 0': 'index'})
            df = df.set_index('index', drop = True)
            df = df.reset_index(drop = True)
           
        else:
            print('\nYou already had the chief data set in your workspace.\n')
        
    # Shuffling before splitting the chief_df into train and test set:
    df = df.sample(frac = sample_ratio)
    test_set = df.loc[df['source_cat'] == "reviews"][:]
    test_set.to_csv(paths['test_df_path'])
    train_set = df.loc[df['source_cat'] != "reviews"][:]
    train_set.to_csv(paths['train_df_path'])    
        
    if not fdict_file_yes:
        print('\nBuilding the feature-enriched data set.\n')
        if not (dfs_postfit_in_wd):
            featurizer = add_features(columns_to_idf, dict_colls, nli_model, nli_tokenizer, paths)
            featurizer.fit(train_set)
            featurizer.save()
        else:
            featurizer = add_features(columns_to_idf, dict_colls, nli_model, nli_tokenizer, paths)
            featurizer.load()
            
        rich_df, feature_dict = featurizer.transform(df)
        rich_df.to_csv(paths['rich_df_path'])
        with open(paths['feature_dict_path'], 'wb') as f:
            pickle.dump(feature_dict, f)
                
        print("""
              You are now in possession of the freshly-created feature-enriched data set. 
              
              These are the added features:
                  
                  {}
                  
                  """.format([feature_dict.keys()]))
                
    else:
        if not (feat_dict_in_wd and rich_df):
            print('\nLoading the feature-enriched data set, plus the feature dictionary.\n')
            
            with open(paths['feature_dict_path'], 'rb') as f:
                feature_dict = pickle.load(f)
                
            rich_df = pd.read_csv(paths['rich_df_path'])
            rich_df = rich_df.rename(columns = {'Unnamed: 0': 'index'})
            rich_df = rich_df.set_index('index', drop = True)
            rich_df = rich_df.reset_index(drop = True)
            rich_df = rich_df.sample(frac = sample_ratio)
            print("""
                  You are now in possession of the feature-enriched data set you had saved previously. 
                  
                  These are the added features:
                      
                      {}
                      
                      """.format(list(feature_dict.keys())))
        else:
            print("""
                  You already had the feature-enriched data set loaded on your working directory. 
                  It's name, shape and some of the observations it comprises read as follows:
                      
                      df_tfidf:
                          
                      {}
                      
                      {}
                      """.format(rich_df.shape, rich_df.sample(10)))
                    
    return df, rich_df, feature_dict
