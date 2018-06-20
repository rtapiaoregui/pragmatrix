#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:52:38 2018

@author: rita

Pragmatrix Main Script

"""

# Imports
import os

import re
from collections import OrderedDict
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model

import xgboost as xgb
from xgboost import XGBRegressor


# Paths:
common_path = '/Users/rita/Google Drive/DSR/DSR Project'
code_path = os.path.join(common_path, 'Code')
data_path = os.path.join(common_path, 'Data')
corpora_path = os.path.join(data_path, 'corpus')
sets_path = os.path.join(data_path, 'datasets_small')

   
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
 

os.chdir(paths['code_path'])
import classy_n_funky as cl
import feature_trove as feat
import classifiers as cla

    
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


# Loading  the lastly-modified df and a dictionary with the different feature columns 
# grouped by how they have been extracted:
chief_df, rich_df, feature_dict = feat.dfs_initializer(paths, 
                                                       columns_to_idf, 
                                                       dict_colls,
                                                       nli_model,
                                                       nli_tokenizer,
                                                       sample_ratio = 0.1)

# Splitting the feature-enriched data set into training and test set:    
train_set = rich_df.loc[rich_df.source_cat != 'reviews']
test_set = rich_df.loc[rich_df.source_cat == 'reviews']
 
   
## Modelling options:
options = input("""
                Choose one of the available options, which can be found specified below 
                (must be a number between 1 and 5):
                    
                Option 1:
                    model: logistic regression
                    features: basic, colls
                    y: literariness
                    
                Option 2:
                     model: tree
                     features: you will be prompted to choose further
                     y: all training set labels (literariness, source_cat, source)
                    
                Option 3:
                     model: doc2vec
                     features: the original text observations and Spacy's tags
                     y: all training set labels (literariness, source_cat, source)
                     
                 Option 4:
                     model: xgboost
                     
                     features:
                     
                         You can either choose to train the xgboost model with all 
                     but the doc2vec distances between the observations' and the 
                     labels' vectors, or the classifier to extract the predictions
                     that will be used to establish whether there is a correlation
                     between how useful reviews seem and how literary they are.
                     This last model, called 'the terminator', employs the ensembling
                     technique known as stacking, because it includes the predictions 
                     from the doc2vec classifier.
                     
                         There is also a third option available, that entails training a 
                     regression model to see whether the Amazon reviews' ratings on 
                     'helpfulness' can be predicted taking only the features that 
                     highlight the importance of the textual form, 
                     over that of the content, into account.
                     
                     y: you will be prompted to choose further
                    
                Option 5: 
                     model: LSTM 
                     features: only the original text observations
                     y: all training set labels (literariness, source_cat, source) 
                     
                """)

options = int(options)



if options == 1:  
    
    print("""
          You chose to train a logistic regression 
          to predict the degree of literariness of each text observation.
          
          """)

    linear_feat = [
            'x', 
            'targets_train', 
            'basics', 
            'collocations',  
            ]
    
    columns_to_train = [i for a in linear_feat for i in feature_dict[a]]
    log_reg_df = train_set.loc[:, train_set.columns.isin(columns_to_train)][:]
    y_log_reg = log_reg_df.pop('literariness')
    
    log_reg_model, log_reg_score = cla.log_reg(log_reg_df, y_log_reg, cols_to_remove)

 

if options == 2:
    
    alternative = input("""
                        You chose to train a simple tree to see the extent to which the added
                          a) basic features
                          b) collocations features
                          c) nli features
                          d) spacy features
                          e) tfidf features
                          f) doc2vec features
                         help predict
                          1) the observations' degree of literariness.
                          2) the categories the observations' sources fall into.
                          3) the observations' sources.
                        
                        Insert your choice as a letter directly followed by a number.
                        Don't wrap your input in quotation marks.
                        
                        """)

    if re.match('a', alternative):
        tree_feat = ['x', 'targets_train', 'basics']
        plot_name = 'basic'
    elif re.match('b', alternative):
        tree_feat = ['x', 'targets_train', 'collocations']
        plot_name = 'colls'
    elif re.match('c', alternative):
        tree_feat = ['x', 'targets_train', 'nli']
        plot_name = 'nli'
    elif re.match('d', alternative):
        tree_feat = ['x', 'targets_train', 'spacy']
        plot_name = 'spacy'
    elif re.match('e', alternative):
        tree_feat = ['x', 'targets_train', 
                     'text_1gram', 'text_2gram',
                     'pos_tags_1gram', 'pos_tags_2gram',
                     'syn_dep_1gram', 'syn_dep_2gram']
        plot_name = 'tfidf'
    else:
        tree_feat = ['x', 'targets_train', 
                     'doc2vec_vecs_text',
                     'soft_doc2vec_dists_text', 
                     'doc2vec_vecs_pos_tags',
                     'soft_doc2vec_dists_pos_tags',
                     'doc2vec_vecs_syn_deps',
                     'soft_doc2vec_dists_syn_deps']
        plot_name = 'doc2vec'
        
    columns_to_train = [i for a in tree_feat for i in feature_dict[a]]
    tree_df = train_set.loc[:, train_set.columns.isin(columns_to_train)][:]
    assert len(list(set(list(tree_df)) - set(columns_to_train)))==0
    assert len(list(set(columns_to_train) - set(list(tree_df))))==0
    print(columns_to_train)
        
    if re.search('2', alternative):
        y_tree = tree_df.pop('source_cat')
        plot_name = plot_name + '_source_cat_tree'
    elif re.search('3', alternative):
        y_tree = tree_df.pop('source')
        plot_name = plot_name + '_source_tree'
    else:
        y_tree = tree_df.pop('literariness')
        plot_name = plot_name + '_liter_tree'        

    
    tree_model, tree_score = cla.rooted(paths, tree_df, y_tree, cols_to_remove, plot_name)
    print('\nThe out of sample accuracy score for this tree is: {}\n'.format(tree_score))
 
    
    
if options == 3:
    
    print("""
          You chose to train a doc2vec.
          
          """)

    wecker_feat = [
            'x',
            'spacy', 
            'targets_train', 
            ]
    
    columns_to_train = [i for a in wecker_feat for i in feature_dict[a]]
    wecker_df = train_set.loc[:, train_set.columns.isin(columns_to_train)][:]

    y_cols = ['literariness', 'source', 'source_cat']
    y_wecker = wecker_df[y_cols]
    wecker_df.drop(y_cols, axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(wecker_df, y_wecker, test_size = 0.3, random_state = 123, shuffle = True)
    
    df_vecs, feat_wecker, preds_dict = cl.Doc2wecker().fit(X_train, y_train).transform(X_test)
        
    source_cat_preds = np.array([value for key, value in OrderedDict(sorted(preds_dict.items())).items() if re.search('source_cat', key)]).T
    source_preds = np.array([
            value for key, value in OrderedDict(
                    sorted(preds_dict.items())).items() if re.search('source', key) and not (re.search('source_cat', key))
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
 


if options == 4:
        
    alternative = input(
            """
            You chose to train a xgboost model.
            
            What woud you like to predict?
            a) the observations' degree of literariness
            b) the categories the observations' sources belong to
            c) the observations' sources
            d) the terminator 
            (the classifier to predict the observations' degree 
            of literariness with all features)
            e) Amazon reviews' ratings on helpfulness
            
            """)
            

    if (alternative == 'a') or (alternative == 'b') or (alternative == 'c'):
        
        non_xgb_feat = [
        'soft_doc2vec_dists_text',
        'soft_doc2vec_dists_pos_tags',
        'soft_doc2vec_dists_syn_deps'
        ]
        
        xgb_feat = list(set(list(feature_dict.keys())) - set(non_xgb_feat))
        columns_to_train = [i for a in xgb_feat for i in feature_dict[a]]
        xgb_df = train_set.loc[:, train_set.columns.isin(columns_to_train)][:]
        assert len(list(set(list(xgb_df)) - set(columns_to_train)))==0
        assert len(list(set(columns_to_train) - set(list(xgb_df))))==0
        
        metric = 'neg_log_loss'
    
        if (alternative == 'c'):
            y_xgb = xgb_df.pop('source')
        elif (alternative == 'b'):
            y_xgb = xgb_df.pop('source_cat')
        else:
            metric = 'roc_auc'
            y_xgb = xgb_df.pop('literariness')
            
        xgboost_model, xgboost_score = cla.xgbooster(xgb_df, y_xgb, cols_to_remove, metric)
        xgb.plot_importance(xgboost_model.best_estimator_.named_steps['clf'], max_num_features = 20, importance_type = 'gain')
        print("The model's score: {}".format(xgboost_score))
            
    elif (alternative == 'd'):
        xgb_df = train_set.loc[:, ~train_set.columns.isin(['d2v_dist_text_literariness_1',
                                                       'd2v_dist_text_literariness_0',
                                                       'd2v_dist_pos_tags_literariness_1',
                                                       'd2v_dist_pos_tags_literariness_0', 
                                                       'd2v_dist_syn_deps_literariness_1',
                                                       'd2v_dist_syn_deps_literariness_0'])]
        metric = 'roc_auc'
        y_xgb = xgb_df.pop('literariness')

        xgboost_model, xgboost_score = cla.xgbooster(xgb_df, y_xgb, cols_to_remove, metric)
        xgb.plot_importance(xgboost_model.best_estimator_.named_steps['clf'], max_num_features = 20, importance_type = 'gain')
        print("The model's score: {}".format(xgboost_score))
        
        xgb_test = test_set.loc[:, ~test_set.columns.isin(['d2v_dist_text_literariness_1',
                                                       'd2v_dist_text_literariness_0',
                                                       'd2v_dist_pos_tags_literariness_1',
                                                       'd2v_dist_pos_tags_literariness_0', 
                                                       'd2v_dist_syn_deps_literariness_1',
                                                       'd2v_dist_syn_deps_literariness_0'])]
            
        xgb_test = cla.Dropping(cols_to_remove).fit_transform(xgb_test)
        predicted_probas = xgboost_model.predict_proba(xgb_test)[:, 1]

        rev_ratings = test_set.loc[:, test_set.columns.isin([
                'comicality', 'helpfulness', 'ice_ice_baby', 'usefulness'
                ])]
            
        rev_ratings.reset_index(drop = True, inplace = True)
        selector = test_set.source
        selector.reset_index(drop = True, inplace = True)
        
        to_viz = pd.concat([selector, pd.Series(predicted_probas), rev_ratings], axis = 1)
        to_viz.rename(columns = {0: 'predicted_probas'}, inplace = True)
        
        amazon_preds = to_viz.loc[to_viz.source == 'amazon_reviews']
        yelp_preds = to_viz.loc[to_viz.source != 'amazon_reviews']
        
        print("The correlation between the ratings on usefulness of business reviews on Yelp and the reviews' degree of literariness: {}".format(
                yelp_preds.predicted_probas.corr(yelp_preds.usefulness)))
        print()
        print("The correlation between the ratings on helpfulness of movie and TV reviews on Amazon and the reviews' degree of literariness: {}".format(
                amazon_preds.predicted_probas.corr(amazon_preds.helpfulness)))
        
    else:
        to_train = test_set.loc[test_set.source == 'amazon_reviews']
        non_xgbreg_feat = set(['text_1gram', 'text_2gram', 'doc2vec_vecs_text', 'soft_doc2vec_dists_text', 'nli'])
        all_feat = set(list(feature_dict.keys()))
        xgbreg_feat = all_feat.difference(non_xgbreg_feat)
        columns_to_train = [i for a in xgbreg_feat for i in feature_dict[a]]
        
        xgb_df = to_train.loc[:, to_train.columns.isin(columns_to_train)]
        metric = 'neg_mean_squared_error'
        y_xgb = xgb_df.pop('helpfulness')
        xgboost_model, xgboost_score = cla.xgbooster(xgb_df, y_xgb, cols_to_remove, metric, clf_type = XGBRegressor())
        xgb.plot_importance(xgboost_model.best_estimator_.named_steps['clf'], max_num_features = 20, importance_type = 'gain')
        print("The model's score: {}".format(xgboost_score))



if options == 5:
    
    alternative = input("""
                        You chose to train a LSTM neural network.
                        
                        What woud you like to predict?
                        a) the observations' degree of literariness
                        b) the categories the observations' sources belong to
                        c) the observations' sources
                        
                        """)
    
    lstm_feat = train_set.text
    my_loss = 'categorical_crossentropy'
    
    if (alternative == 'c'):
        cm_plot_labels = list(train_set.source.value_counts())
        y_lstm = train_set.pop('source')
        n_output = len(y_lstm.value_counts())
    elif (alternative == 'b'):
        cm_plot_labels = list(train_set.source_cat.value_counts())
        y_lstm = train_set.pop('source_cat')
        n_output = len(y_lstm.value_counts())
    else:
        cm_plot_labels = ['non-literary observations', 'literary observations']
        my_loss = 'binary_crossentropy'
        y_lstm = train_set.pop('literariness')
        n_output = len(y_lstm.value_counts())
    
    lstm_model = cla.LSTMer(paths,
                            lstm_feat,
                            y_lstm,
                            n_output,
                            my_loss,
                            cm_plot_labels)
        
    history = lstm_model.summary()
    print(history)
    
 