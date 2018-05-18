#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:34:28 2018

@author: rita

Classes to extract features from my main dataset and create a new dataset with them
"""

import os
import pickle
import requests
import random
import re
import pandas as pd
import numpy as np
from scipy import spatial
from tqdm import tqdm
from lxml.html import fromstring
from itertools import cycle

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from gensim.sklearn_api.d2vmodel import D2VTransformer
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec

from sklearn.linear_model import LogisticRegression as lreg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer
from keras.models import Sequential
from keras import layers as lay



spacy_model = 'en_core_web_lg'


# Feature classes
class BOWer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.model1 = CountVectorizer(stop_words = 'english', max_features = 600)
        self.model2 = CountVectorizer(ngram_range = (2,4), max_features = 400)
        
    
    def fit(self, X, y = None):
        
        self.model1.fit(X['text'])
        self.model2.fit(X['text'])
        
        return self
    
        
    def transform(self, X):
        
        tf1 = self.model1.transform(X['text'])
        tf2 = self.model2.transform(X['text'])
        
        df_1gram = pd.DataFrame(tf1.toarray(), columns = self.model1.get_feature_names())
        df_2gram = pd.DataFrame(tf2.toarray(), columns = self.model2.get_feature_names())
    
        X.reset_index(inplace = True, drop = True)
        X = pd.merge(X, df_1gram, left_index=True, right_index=True, how='outer')
        X = pd.merge(X, df_2gram, left_index=True, right_index=True, how='outer')
        
        feat_bow = {
                'bow_1gram' : df_1gram.columns,
                'bow_2gram' : df_1gram.columns
                }

        return X, feat_bow
    
    
    
class TFIDFer(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns_to_idf):
        
        for col in columns_to_idf:
            if col == 'text':
                self.model_tx1 = TfidfVectorizer(stop_words = 'english', max_features = 600)
                self.model_tx2 = TfidfVectorizer(ngram_range = (2,4), max_features = 400)
                self.text = 'text'
            elif col == 'POS_tag':
                self.model_pt1 = TfidfVectorizer(stop_words = 'english', max_features = 300)
                self.model_pt2 = TfidfVectorizer(ngram_range = (2,4), max_features = 150)
                self.POS_tag = 'POS_tag'
            else:
                self.model_sd1 = TfidfVectorizer(stop_words = 'english', max_features = 300)
                self.model_sd2 = TfidfVectorizer(ngram_range = (2,4), max_features = 150)
                self.syn_dep = 'syntactic_dependency'
                
    
    def fit(self, X, y = None):
        
        self.model_tx1.fit(X[self.text])
        self.model_tx2.fit(X[self.text])

        self.model_pt1.fit(X[self.POS_tag])
        self.model_pt2.fit(X[self.POS_tag])

        self.model_sd1.fit(X[self.syn_dep])
        self.model_sd2.fit(X[self.syn_dep])
        
        return self
    
        
    def transform(self, X):
        
        tf_tx1 = self.model_tx1.transform(X[self.text])
        tf_tx2 = self.model_tx2.transform(X[self.text])

        tf_pt1 = self.model_pt1.transform(X[self.POS_tag])
        tf_pt2 = self.model_pt2.transform(X[self.POS_tag])

        tf_sd1 = self.model_sd1.transform(X[self.syn_dep])
        tf_sd2 = self.model_sd2.transform(X[self.syn_dep])
        
        df_tx1 = pd.DataFrame(tf_tx1.toarray(), columns = self.model_tx1.get_feature_names())
        df_tx2 = pd.DataFrame(tf_tx2.toarray(), columns = self.model_tx2.get_feature_names())
        
        df_ps1 = pd.DataFrame(tf_pt1.toarray(), columns = self.model_pt1.get_feature_names())
        df_ps2 = pd.DataFrame(tf_pt2.toarray(), columns = self.model_pt2.get_feature_names())
    
        df_sd1 = pd.DataFrame(tf_sd1.toarray(), columns = self.model_sd1.get_feature_names())
        df_sd2 = pd.DataFrame(tf_sd2.toarray(), columns = self.model_sd2.get_feature_names())
    
        X.reset_index(inplace = True, drop = True)
        X = X.merge(df_tx1, left_index=True, right_index=True).\
        merge(df_tx2, left_index=True, right_index=True).\
        merge(df_ps1, left_index=True, right_index=True).\
        merge(df_ps2, left_index=True, right_index=True).\
        merge(df_sd1, left_index=True, right_index=True).\
        merge(df_sd2, left_index=True, right_index=True)
        
        feat_tfidf = {
                'text_1gram' : df_tx1.columns,
                'text_2gram' : df_tx2.columns,
                'pos_tags_1gram' : df_ps1.columns,
                'pos_tags_2gram' : df_ps2.columns,
                'syn_dep_1gram' : df_sd1.columns,
                'syn_dep_2gram' : df_sd2.columns
                }
        
        return X, feat_tfidf


class Doc2wecker(BaseEstimator, TransformerMixin):
    
    def __init__(self, window = 5, min_count = 5, epochs = 100, vector_size = 200):
        # Initialize configuration of Doc2Vec model
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.vector_size = vector_size
        

    def fit(self, X, y = None): 
        
        if y is not None:
            X = X.merge(y, left_index=True, right_index=True)
            
        doc2vec_objs = doc2vec_prep_for_fit(X)
        self.model = {}
        for obj in range(len(doc2vec_objs)):
        # Doc2Vec both creates the object and trains at the same time
            self.model[obj] = Doc2Vec(documents = doc2vec_objs[obj],
                                 window = self.window, 
                                 min_count = self.min_count, 
                                 workers = 4, 
                                 epochs = self.epochs, 
                                 vector_size = self.vector_size)
            

        return self
        
    def transform(self, X): 

        raw_text = [word_tokenize(str(i)) for i in X.text]  
        raw_tags = [word_tokenize(str(i)) for i in X.POS_tag] 
        raw_syn_deps = [word_tokenize(str(i)) for i in X.syntactic_dependency] 
        
        raws = [raw_text, raw_tags, raw_syn_deps]       
        
        #Extract which category tags were used to train the model
        category_tags = [tag for tag in list(self.model[0].docvecs.doctags.keys()) if not 'obs' in str(tag)]
        
        # Comparing how close the vector from this document is to each of
        # the category tags we used to train the model
        dists_model = {}
        vectors = {}
        for idx in range(len(self.model)):
            dist_categories = {}
            vectors[idx] = []
            for i, cat in enumerate(category_tags):
                category_vector = self.model[idx].docvecs[cat]
                dist_per_doc = []
                for raw in raws[idx]:
                    # For each text in the data frame, apply model.infer_vector,
                    # which generates the vectors corresponding to that text
                    this_vector = self.model[idx].infer_vector(raw)
                    distance = spatial.distance.cosine(this_vector, category_vector)
                    dist_per_doc.append(distance)
                    # I only want to append each document vector once.
                    if i == 0:
                         vectors[idx].append(this_vector)
                dist_categories[i] = dist_per_doc
            dists_model[idx] = dist_categories
                           

        # Add both the raw vectors and the distances to each of the category labels
        # to the original data frame
        df_list = []
        names = ['text', 'pos_tags', 'syn_deps']
        feat_wecker = {}
        for key in vectors.keys():
            # Building the different data frames for each combination of vectors 
            # per document representing the features taken into consideration 
            # and their respective distances to the category vectors.
            df_vecs = pd.DataFrame(vectors.get(key))        
            df_vecs.columns = ['d2v_vec_' + names[key] + '_' + str(i) for i in range(df_vecs.shape[1])]
            feat_wecker['doc2vec_vecs_' + names[key]] = df_vecs.columns
            df_list.append(df_vecs)

            dist_categories = pd.DataFrame(dists_model.get(key))
            dist_categories.columns = ['d2v_dist_' + names[key] + '_' + cat for cat in category_tags]
            
            # Applying softmax to determine to which category' vector the document vector is closest
            lit_cols = [i for i in dist_categories.columns if re.search(r'literariness', str(i))]
            source_cols = [i for i in dist_categories.columns if (re.search(r'source', str(i)) and not re.search(r'source_cat', str(i)))]
            scat_cols = [i for i in dist_categories.columns if re.search(r'source_cat', str(i))]
            
            soft_lit = dist_categories[lit_cols].apply(inverse_softmax, axis = 1)
            soft_source = dist_categories[source_cols].apply(inverse_softmax, axis = 1)
            soft_scat = dist_categories[scat_cols].apply(inverse_softmax, axis = 1)
            
            df_dists = pd.concat([soft_lit, soft_source, soft_scat], axis = 1)
            feat_wecker['soft_doc2vec_dists_' + names[key]] = df_dists.columns
            df_list.append(df_dists)      
        
        X.reset_index(inplace = True, drop = True)
        new_df = pd.concat(df_list, axis = 1)
        X = X.merge(new_df, left_index=True, right_index=True)

        preds_dict = {}
        for cat in category_tags:
            preds_dict[cat] = new_df[[i for i in new_df.columns if re.search(str(cat), i)]].mean(axis = 1)
        
 
        return X, feat_wecker, preds_dict



class Dropping(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols_to_remove):
        self.cols_to_remove = cols_to_remove
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.drop([c for c in self.cols_to_remove if c in X.columns], axis = 1)      
        return X
    
    
    
# Functions to preprocess the data for feature extraction
def get_proxies():
    
    proxy = 0
    while not type(proxy) == str:
        
        try:
            url = 'https://free-proxy-list.net/'
            response = requests.get(url)
            parser = fromstring(response.text)
            ip = random.choice(parser.xpath('//tbody/tr'))
            if ip.xpath('.//td[7][contains(text(),"yes")]'):
                # Grabbing IP and corresponding PORT
                proxy = ":".join([ip.xpath('.//td[1]/text()')[0], ip.xpath('.//td[2]/text()')[0]])
                proxies = {"http": proxy, "https": proxy}
                
        except:
           continue
               
    return proxies
       


def basicker(X):
       
    sentences = [sent_tokenize(obs) for obs in X['text']]
    words = [word_tokenize(obs) for obs in X['text']]  
    
    # Here I want to compute the average distance for each document between 
    # the lengths of its sentences, as well as the standard deviation 
    # of the lengths of each of the documents' sentences.
    n_sentences = []
    len_sents = []
    dists = []
    
    std_len_sents = []
    mean_dists = []
    
    for doc in sentences:
        n_sentences.append(len(doc))
        for i in range(0, len(doc)):
            if i == 0:
                length_s = len(doc[i])
                len_sents.append(length_s)
            else:
                length_s = len(doc[i])
                len_sents.append(length_s)
                dist = abs(len(doc[i]) - len(doc[i-1])) / max(len(doc[i]), len(doc[i-1]))
                dists.append(dist)
        std_len_sents.append(np.std(len_sents))
        mean_dists.append(np.mean(dists))
        
    
    X = X.assign(length_text = [len(t) for t in X['text']]) 
    X = X.assign(n_words = [len(words) for words in words])  
    X = X.assign(n_sentences = n_sentences)
    X = X.assign(sent_length = X['n_words']/X['n_sentences'])
    X = X.assign(std_len_sents = std_len_sents)
    X = X.assign(mean_dists = mean_dists)
        
    X['source'] = X['source'].astype('category')
    X['source_cat'] = X['source_cat'].astype('category')
    
    # Info:
    print("The longest text in each 'source' category: \n{}\n".format(X.groupby('source')['length_text'].max().sort_values()[::-1]))
    print("Data set's size after increase with the addition of basic features: \n{}\n".format(X.shape))

    feat_basic = {
            'x' : ['text'], 
            'targets' : ['source', 'source_cat', 'literariness'], 
            'basics' : ['length_text', 'n_words', 'n_sentences', 
                        'sent_length', 'std_len_sents', 'mean_dists']
            }

    return X, feat_basic

    

def spacy_parser(X, text_column = 'text', spacy_model = 'en_core_web_lg'):
    
    nlp = spacy.load(spacy_model)
    parsed_text = [nlp(tx) for tx in X[text_column]]
    
    return parsed_text



def spacyer(X, text_column = 'text', spacy_model = 'en_core_web_lg'):
    
    parsed_text = spacy_parser(X)
        
    doc_is_digit = []
    doc_is_punct = []
    doc_log_probability = []
    doc_out_of_vocabulary = []
    doc_pos_tag = []
    doc_syntactic_dependency = []
    verbal_count = []
    for doc in tqdm(parsed_text):
        is_digit = []
        is_punct = []
        log_probability = []
        is_out_of_vocabulary = []
        pos_tag = []
        syntactic_dependency = []
        count_verbs = 0
        for word in doc:
            is_digit.append(word.is_digit)
            is_punct.append(word.is_punct)
            log_probability.append(word.prob)
            is_out_of_vocabulary.append(word.is_oov)
            pos_tag.append(word.tag_)
            syntactic_dependency.append(word.dep_)
        verbal_count.append(count_verbs)
        doc_is_digit.append(np.mean(is_digit))
        doc_is_punct.append(np.mean(is_punct))
        doc_log_probability.append(np.mean(log_probability))
        doc_out_of_vocabulary.append(np.mean(is_out_of_vocabulary))
        doc_pos_tag.append(' '.join(pos_tag))
        doc_syntactic_dependency.append(' '.join(syntactic_dependency))

        
    spacy_df = pd.DataFrame({
            'mean_log_probability' : doc_log_probability,
            'mean_out_of_vocab' : doc_out_of_vocabulary,
            'is_digit' : doc_is_digit,
            'is_punct' : doc_is_punct, 
            'syntactic_dependency' : doc_syntactic_dependency,
            'POS_tag' : doc_pos_tag
            })
    
    X.reset_index(inplace = True, drop = True)
    X = pd.merge(X, spacy_df, left_index=True, right_index=True, how='outer')

    X[text_column] = X[text_column].map(lambda x: str(x).lower())

    feat_spacy = {
            'spacy' : spacy_df.columns
            }
    
    return X, feat_spacy



def collocater(X):

    colls_path = '/Users/rita/Google Drive/DSR/DSR Project/Dataset/tools'
    
    with open(os.path.join(colls_path, 'dict_colls.pickle'), 'rb') as file:
        dict_colls = pickle.load(file)


    eligible_ratio_total = []
    matches_per_doc = []
    for i in range(X.shape[0]):
        text = str(X.iloc[i]['text'])
        words = word_tokenize(str(text))
        eligible = [p for p in words if p in dict_colls.keys()]

        matches_per_el = []
        for w in eligible:
            pos_colls = [i.split(' ') for i in dict_colls[w]]
            w_idx = words.index(w)
            match_per_w = 0
            for col in pos_colls:
                if (len(col) > 1) and not (len(col) == 2 and re.search(r'(a|that|of|the)', str(col))):
                    col = re.compile(' '.join(col))
                    low = w_idx - 5
                    high = w_idx + 6
                    if low < 0:
                        low = 0
                    if high > len(words):
                        high = len(words)
                    segment = ' '.join(words[low:high])
                    if re.search(col, segment):
                        match_per_w += 1
                        print('{} is a collocation of {}'.format(col, w))
                else:
                    col = re.compile(''.join(col))
                    if (re.search(col, words[(w_idx + 1)]) or re.search(col, words[(w_idx + 1)])):
                        match_per_w += 1
            if match_per_w > 1:
                match_per_w = 1
                
            matches_per_el.append(match_per_w)        

        if len(eligible) > 0:
            matches_per_doc.append(np.sum(matches_per_el))
        else:
            matches_per_doc.append(0)
            
        eligible_ratio_total.append(len(eligible))        

    X = X.assign(matches_per_doc = matches_per_doc)
    X = X.assign(eligible_ratio_total = eligible_ratio_total/np.asarray(X.n_words))
    X = X.assign(colls_per_eligible_wds = np.asarray(X.matches_per_doc)/eligible_ratio_total)
    
    print("The sources ranked according to the collocations ratio of their text observations: \n{}\n".format(
            X.groupby('source')['colls_per_eligible_wds'].mean().sort_values()
            ))
    
    feat_colls = {
            'collocations' : ['colls_ratio_abs', 'eligible_ratio_total', 'colls_ratio']
            }
    
    return X, feat_colls



def doc2vec_prep_for_fit(X):
    
    texts = [word_tokenize(str(i)) for i in X.text]
    tags = [word_tokenize(str(i)) for i in X.POS_tag]
    syn_deps = [word_tokenize(str(i)) for i in X.syntactic_dependency]

    con_obs = [str('obs_' + str(i)) for i in range(X.shape[0])]
    con_source = [str('source_' + str(X.iloc[i].source)) for i in range(X.shape[0])]
    con_source_cat = [str('source_cat_' + str(X.iloc[i].source_cat)) for i in range(X.shape[0])]
    con_literariness = [str('literariness_'+ str(X.iloc[i].literariness)) for i in range(X.shape[0])]
    
    contexts = [con_obs, con_source, con_source_cat, con_literariness]

    # This is equivalent to transposing "contexts"
    context_per_text = []    
    for i in range(X.shape[0]):
        species = []
        for con in contexts:    
            species.append(con[i])
        context_per_text.append(species)
        
    # Create one TaggedDocument per text in the original df
    # with the first element being the list of words
    # and the second element the list of tags that correspond to that document
    doc2vec_text = []
    doc2vec_tags = []
    doc2vec_syn_deps = []
    for i in range(X.shape[0]):
        doc2vec_text.append(models.doc2vec.TaggedDocument(texts[i], context_per_text[i]))
        doc2vec_tags.append(models.doc2vec.TaggedDocument(tags[i], context_per_text[i]))
        doc2vec_syn_deps.append(models.doc2vec.TaggedDocument(syn_deps[i], context_per_text[i]))
        
    doc2vec_objs = [doc2vec_text, doc2vec_tags, doc2vec_syn_deps]
        
    return doc2vec_objs


def inverse_softmax(x):
    # Since we are measuring distances, we need to use the inverse values
    x = x*(-1)
    # Compute softmax values for all elements of an array x (grouping by rows)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



# Models
def log_reg(X, y, cols_to_remove):
       
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123, shuffle = True)
    
    steps = [('colls_dropper', Dropping(cols_to_remove)),
             ('clf', lreg())]
    
    pipe = Pipeline(steps)
    
    param_grid = {'clf__C': np.linspace(5000, 11000, 1000)}
 
    model = GridSearchCV(estimator = pipe, param_grid = param_grid,
                       cv = 5, scoring = "precision", refit = True)    

    model.fit(X_train, y_train)
    scoring = model.score(X_test, y_test)
    
    print('The best score: ', model.best_score_)
    print()
    print('The best parameters: ', model.best_params_)
    
    return model, scoring



def multi_class_xgboost(X, y, cols_to_remove):
   
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123, shuffle = True)
      
    # Building the feature extraction pipeline
    steps = [('colls_dropper', Dropping(cols_to_remove)),
             ('clf', XGBClassifier())]
    
    pipe = Pipeline(steps)
       
    param_grid = {'clf__n_estimators': np.arange(50, 200),
                  'clf__max_depth': np.arange(3, 7)}
        
    model = RandomizedSearchCV(
            pipe, 
            param_distributions = param_grid, 
            scoring = 'neg_log_loss',
            n_iter = 2, 
            cv = 3, 
            verbose = 3, 
            random_state = 123)
    
    model.fit(X_train, y_train)
    scoring = model.score(X_test, y_test)
    
    print('The best score: ', model.best_score_)
    print()
    print('The best parameters: ', model.best_params_)
    
    return model, scoring



def multi_class_LSTM(X, y):
    
    seqs = [text_to_word_sequence(t) for t in X]
    new_seqs = sum(seqs, [])

    vocab_size = len(set(new_seqs))
    embed_dim = 100
    max_length = 3000
    batch_size = 50
    
    labels = pd.get_dummies(y).values
    
    tokenized = Tokenizer(num_words = vocab_size)
    tokenized.fit_on_texts(X)
    text = tokenized.texts_to_sequences(X)
    text = pad_sequences(text, maxlen = max_length, padding='post')
    
    X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size = 0.3, random_state = 123, shuffle = True)
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
       
    model = Sequential()
    model.add(lay.Embedding(vocab_size, embed_dim, input_length = max_length))
    model.add(lay.LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))
    model.add(lay.Dense(5, activation = 'softmax'))    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


    model.fit(X_train, y_train, epochs = 10, batch_size = batch_size, verbose=3)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=3)
    

    return model, accuracy

