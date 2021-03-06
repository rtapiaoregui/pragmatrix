#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 12:04:34 2018

@author: rita

My classifiers
"""

import os
import pandas as pd
import numpy as np
import pickle
import itertools

import matplotlib.pyplot as plt
import pydotplus

from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression as lreg
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as viz_tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers as lay
from keras.callbacks import EarlyStopping, ModelCheckpoint




class Dropping(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols_to_remove):
        self.cols_to_remove = cols_to_remove
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.drop([c for c in self.cols_to_remove if c in X.columns], axis = 1)      
        return X
    
    

# Models
def log_reg(X, y, cols_to_remove):
       
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123, shuffle = True)
    
    steps = [('colls_dropper', Dropping(cols_to_remove)),
             ('clf', lreg())]
    
    pipe = Pipeline(steps)
    
    param_grid = {'clf__C': np.linspace(5000, 11000, 1000)}
 
    model = GridSearchCV(estimator = pipe, param_grid = param_grid,
                       cv = 5, scoring = "roc_auc", refit = True)    

    model.fit(X_train, y_train)
    scoring = model.score(X_test, y_test)
    
    print('The best score: ', model.best_score_)
    print()
    print('The best parameters: ', model.best_params_)
    
    return model, scoring



def rooted(paths, X, y, cols_to_remove, plot_name):
    
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123, shuffle = True)
    
    X_train = Dropping(cols_to_remove).fit_transform(X_train)
    X_test = Dropping(cols_to_remove).fit_transform(X_test)
    model = viz_tree(max_leaf_nodes = 6, random_state = 123)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    
    dot_data = tree.export_graphviz(model,
                                feature_names = list(X_train),
                                out_file = None,
                                filled = True,
                                proportion = True,
                                rounded = True)
    
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    graph.write_png(os.path.join(paths['tree_plots_path'], str(plot_name) + '.png'))
    
    return model, preds, y_test
    
    

def xgbooster(X, y, cols_to_remove, metric, clf_type = XGBClassifier()):
   
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123, shuffle = True)
      
    # Building the feature extraction pipeline
    steps = [('colls_dropper', Dropping(cols_to_remove)),
             ('clf', clf_type)]
    
    pipe = Pipeline(steps)
       
    param_grid = {'clf__n_estimators': np.arange(50, 200),
                  'clf__max_depth': np.arange(3, 7)}
        
    model = RandomizedSearchCV(
            pipe, 
            param_distributions = param_grid, 
            scoring = metric,
            n_iter = 3, 
            cv = 4,
            n_jobs = 3,
            verbose = 3, 
            random_state = 123)
    
    print(X_train.columns)
    model.fit(X_train, y_train)
    performance = model.score(X_test, y_test)
    
    print('The best score: ', model.best_score_)
    print()
    print('The best parameters: ', model.best_params_)
    
    if (clf_type != XGBClassifier()):
        preds = model.predict(X_test)
    else:
        preds = model.predict_proba(X_test)
    
    return model, performance, preds, y_test



def NN_prepper(paths, tokenizer_path, X):
        
    embeddings_index = {}
    with open(paths['glove_path']) as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            embeddings_index[word] = coefs
            
    tokenizer = Tokenizer(lower = False, filters = '', char_level = False)
    
    if not isinstance(X, list):    
        # Preparing my data set so that it has the same structure and appearance as the nli corpora
        t_in_sents = [[' '.join(word_tokenize(b)).lower() for b in sent_tokenize(a)] for a in X]
        
        doc_n = []
        sents_x_doc = []
        for i in range(len(t_in_sents)):
            doc_n.append(i)
            sents_x_doc.append(len(t_in_sents[i]))
        
        structured_df = pd.DataFrame({'multiplier': sents_x_doc}, index = doc_n)            
    
    sents = sum(t_in_sents, [])      
    tokenizer.fit_on_texts(sents)
    
    with open(paths.get(tokenizer_path), 'wb') as handler:
        pickle.dump(tokenizer, handler, protocol=pickle.HIGHEST_PROTOCOL)

    return embeddings_index, tokenizer, structured_df, sents
    
    

def LSTMer(paths, X, y, n_output, my_loss, cm_plot_labels):

    embeddings_index, tokenizer, structured_df, sents = NN_prepper(paths, 'lstm_tokenizer_path', X)

# This commented section is needed to give sentences as input instead of whole documents. 
#    structured_df = structured_df.assign(target = y.values)
#    y = []
#    for i in range(len(structured_df)):
#        a = [structured_df.target[i]] * structured_df.multiplier[i]
#        y.append(a)
#    
#    y = sum(y, [])
    
    vocab_size = len(tokenizer.word_index) + 1
    embed_dim = 300
    max_length = 150
    batch_size = 20

    # prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector
        else:
            print('Missing from GloVe: {}'.format(word))
    
    print('Total number of null word embeddings:')
    print(np.sum(np.sum(embedding_matrix, axis = 1) == 0))
    
#    text = tokenizer.texts_to_sequences(sents)
    text = tokenizer.texts_to_sequences(X)
    text = pad_sequences(text, maxlen = max_length, padding = 'post')
    labels = pd.get_dummies(y).values
    
    X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size = 0.3, random_state = 123, shuffle = True)
    
    # Defining callbacks
    early_stopping = EarlyStopping(monitor='acc', patience=4)
    checkpoint = ModelCheckpoint(paths.get('lstm_weights_path'), monitor='acc', verbose = 2, save_best_only = True, mode = 'max')

    class_weight = {i: max(labels.sum(0))/labels.sum(0)[i] for i in range(n_output)}
    class_weight = {key: val/sum(class_weight.values()) for key, val in class_weight.items()}
    
    model = Sequential()
    model.add(lay.Embedding(vocab_size, embed_dim, 
                            weights=[embedding_matrix], 
                            input_length = max_length, 
                            trainable = False))
    model.add(lay.LSTM(55, activation = 'relu'))
    model.add(lay.Dense(n_output, activation = 'softmax'))    
    model.compile(loss = my_loss, optimizer = 'adam', metrics = ['accuracy'])
    
    model.fit(X_train, 
              y_train, 
              epochs = 15, 
              batch_size = batch_size,
              validation_data = (X_test, y_test),
              verbose = 2, 
              callbacks=[early_stopping, checkpoint],
              class_weight = class_weight)
    
    
    model.load_weights(paths.get('lstm_weights_path'))
    model.compile(loss = my_loss, optimizer = 'adam', metrics = ['accuracy'])
    
    preds = model.predict(X_test)
    
    return model, preds, y_test



def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    