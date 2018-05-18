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

# Functions and classes
classes_path = '/Users/rita/Google Drive/DSR/DSR Project/Code/'
os.chdir(classes_path)
import classy_n_funky as cl
import cleaner


    
def add_features(df, paths, columns_to_idf):
      
    # Plus basic features:
    df_basic, feat_basic = cl.basicker(df)
    print("""
          Names of the data set's columns and data set's size 
          after increase with the addition of the basic features:
              {}
              {}
              """.format(df_basic.shape, df_basic.columns))
    
    # Plus spacy tags:
    df_spacy, feat_spacy = cl.spacyer(df_basic)
    print("Data set's size after increase with the addition of spacy's tags: \n{}\n".format(df_spacy.shape))
    df_spacy.to_csv(paths['spacy_df_path'])
            
    # Plus colls:
    df_colls, feat_colls = cl.collocater(df_spacy)
    print("Data set's size after increase with the addition of df_colls: \n{}\n".format(df_colls.shape))
    df_colls.to_csv(paths['colls_df_path'])
    
    # Plus BOWer:
    df_bows, feat_bow = cl.BOWer().fit_transform(df_colls)
    print("Data set's size after increase with the addition of bags of words: \n{}\n".format(df_bows.shape))   
    df_bows.to_csv(paths['bows_df_path'])    

    # Plus Tfidfs:
    df_tfidf, feat_tfidf = cl.TFIDFer(columns_to_idf).fit_transform(df_bows)
    print("Data set's size after increase with the addition of tfidfs: \n{}\n".format(df_tfidf.shape))   
    df_tfidf.to_csv(paths['tfidf_df_path'])
    
    # Plus doc2vec distances to categories and raw vectors:
    df_vecs, feat_wecker, _ = cl.Doc2wecker().fit_transform(df_tfidf)
    print("Data set's size after increase with the addition of document vectors' distances to category vectors' distances: \n{}\n".format(df_vecs.shape))   
    df_vecs.to_csv(paths['vecs_df_path'])
    
    feature_dict = {**feat_basic, **feat_spacy, **feat_colls, **feat_bow, **feat_tfidf, **feat_wecker}
    with open(paths['feature_dict_path'], 'wb') as f:
        pickle.dump(feature_dict, f)
    
    
    return df_vecs, feature_dict



def dfs_initializer(paths, cleaner_paths, columns_to_idf):
        
    # Loading and transforming the data set compiled last:
    feat_dict_in_wd = ('feature_dict' in globals()) or ('feature_dict' in locals())
    last_df_in_wd = ('rich_df' in globals()) or ('rich_df' in locals())
    chief_df_in_wd = ('df' in globals()) or ('df' in locals())
    fdict_file_yes = os.path.exists(paths['feature_dict_path'])
    chdf_file_yes = os.path.exists(paths['chief_df_path'])
  
    if not chdf_file_yes:
        print('\nBuilding the chief data set.\n')
        df = cleaner.dataframer(cleaner_paths, paths)
        print("""
              You are now in possession of the freshly-create chief or primary data set.
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
            
    else:
        if not chief_df_in_wd:
            print('\nLoading the chief data set.\n')
            df = pd.read_csv(paths['chief_df_path'])
            df = df.rename(columns = {'Unnamed: 0': 'index'})
            df = df.set_index('index', drop = True)
            df = df.reset_index(drop = True)
                        
            # Shuffling to obtain the rich_df:
            df = df.sample(frac=1)
        else:
            print('\nYou already had the chief data set in your workspace.\n')
            df = df.sample(frac=1)
        
    if not fdict_file_yes:
        print('\nBuilding the feature-enriched data set.\n')
        rich_df, feature_dict = add_features(df, paths, columns_to_idf)
        print("""
              You are now in possession of the freshly-created feature-enriched data set. 
              
              These are the added features:
                  
                  {}
                  
                  """.format([feature_dict.keys()]))
                
    else:
        if not (feat_dict_in_wd and last_df_in_wd):
            print('\nLoading the first and last data sets, plus the feature dictionary.\n')
            
            with open(paths['feature_dict_path'], 'rb') as f:
                feature_dict = pickle.load(f)
                
            rich_df = pd.read_csv(paths['vecs_df_path'])
            rich_df = rich_df.rename(columns = {'Unnamed: 0': 'index'})
            rich_df = rich_df.set_index('index', drop = True)
            rich_df = rich_df.reset_index(drop = True)
            print("""
                  You are now in possession of the feature-enriched data set you had saved previously. 
                  
                  These are the added features:
                      
                      {}
                      
                      """.format([feature_dict.keys()]))
        else:
            print("""
                  You already had the feature-enriched data set loaded on your working directory. 
                  It's name, shape and some of the observations it comprises read as follows:
                      
                      df_tfidf:
                          
                      {}
                      
                      {}
                      """.format(rich_df.shape, rich_df.sample(10)))
        
            
    return df, rich_df, feature_dict
