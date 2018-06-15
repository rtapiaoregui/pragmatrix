#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:34:28 2018

@author: rita

Classes and functions to generate features based on the chief data set
"""


import re
import pandas as pd
import numpy as np
from scipy import spatial
from tqdm import tqdm
from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from gensim import models
from gensim.models.doc2vec import Doc2Vec
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin



# Functions
def bassicker(X):
       
    sentences = [sent_tokenize(obs) for obs in X['text']]
    words = [word_tokenize(obs) for obs in X['text']]  
    
    # Here I want to compute the standard deviation from 
    # the average difference between the lengths of each document's sentences, 
    # as well as the average difference between the sentences' lengths.
    n_sentences = []
    len_sents = []
    dists = []
    
    std_len_sents = []
    mean_dists = []
    
    for doc in sentences:
        n_sentences.append(len(doc))
        for i in range(len(doc)):
            length_s = len(doc[i])
            len_sents.append(length_s)
            if not (i == 0):
                dist = abs(len(doc[i]) - len(doc[i-1])) / max(len(doc[i]), len(doc[i-1]))
                dists.append(dist)
        std_len_sents.append(np.std(len_sents))
        mean_dists.append(np.mean(dists))
        
    
    X = X.assign(length_text = [len(t) for t in X['text']]) 
    X = X.assign(n_words = [len(word) for word in words])  
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
    for doc in tqdm(parsed_text):
        is_digit = []
        is_punct = []
        log_probability = []
        is_out_of_vocabulary = []
        pos_tag = []
        syntactic_dependency = []
        for word in doc:
            is_digit.append(word.is_digit)
            is_punct.append(word.is_punct)
            log_probability.append(word.prob)
            is_out_of_vocabulary.append(word.is_oov)
            pos_tag.append(word.tag_)
            syntactic_dependency.append(word.dep_)
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

    feat_spacy = {
            'spacy' : list(spacy_df)
            }
    
    return X, feat_spacy



def colls_checker(words, eligible, dict_colls, verbose = False):

    matches_per_el = []
    colls = []
    phatic_particle = 0
    for w in eligible:
        phatic_particle += 1
        if (verbose == True) and (phatic_particle % 100 == 0):
            print("""
                  Percentage of words that have already been checked to see
                  whether they belong to collocations: {}/{}
                  """.format(
                    phatic_particle, len(eligible)+1))
        w_idx = np.where(words == w)[0]
        for idx in w_idx:
            match_per_w = 0
            low = max(idx - 5, 0)
            high = min(idx + 6, len(words))
            segment = ' '.join(words[low:high])
            for col in dict_colls[w]:
                col = re.compile(col)
                if re.search(col, segment):
                    match_per_w += 1
                    colls.append(str(col))
#                    print('{} is a collocation of {}'.format(col, w))
            if match_per_w > 1:
                match_per_w = 1
                
            matches_per_el.append(match_per_w)
        
    return colls, matches_per_el



def doc2vec_prep_for_fit(X):
    
    texts = [word_tokenize(str(i)) for i in X['text']]
    tags = [word_tokenize(str(i)) for i in X['POS_tag']]
    syn_deps = [word_tokenize(str(i)) for i in X['syntactic_dependency']]

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



def NLIyer(X, snli_model, tokenizer, MAX_LEN = 150):
    
    avg_pred_contra = []
    avg_pred_neutrino = []
    avg_pred_assent = []
    std_pred_contra = []
    std_pred_neutrino = []
    std_pred_assent = []
    contras = []
    neutrinos = []
    assent = []
    docs = [[' '.join(word_tokenize(b)) for b in sent_tokenize(a) if len(word_tokenize(b)) <= MAX_LEN] for a in X['text']]
    for doc in docs:
        if len(doc) < 2:
            avg_pred_contra.append(0)
            avg_pred_neutrino.append(0)
            avg_pred_assent.append(0)
            std_pred_contra.append(0)
            std_pred_neutrino.append(0)
            std_pred_assent.append(0)
            contras.append(0)
            neutrinos.append(0)
            assent.append(0)
        else:
            sents = tokenizer.texts_to_sequences(doc)
            sents = pad_sequences(sents, maxlen = MAX_LEN)
            snli_preds = snli_model.predict([np.array(sents[:(len(sents)-1)]), np.array(sents[1:len(sents)])])
            means = np.mean(snli_preds, axis = 0)
            stds = np.std(snli_preds, axis = 0)
            maxes = np.argmax(snli_preds, axis = 1)
            mean_contra = np.mean(maxes == 0)
            mean_neut = np.mean(maxes == 1)
            mean_ent = np.mean(maxes == 2)
            avg_pred_contra.append(means[0])
            avg_pred_neutrino.append(means[1])
            avg_pred_assent.append(means[2])
            std_pred_contra.append(stds[0])
            std_pred_neutrino.append(stds[1])
            std_pred_assent.append(stds[2])
            contras.append(mean_contra)
            neutrinos.append(mean_neut)
            assent.append(mean_ent)
        
    df_nli = pd.DataFrame({
            'avg_pred_contra' : avg_pred_contra, 
            'avg_pred_neutrino' : avg_pred_neutrino,
            'avg_pred_assent' : avg_pred_assent,
            'std_pred_contra' : std_pred_contra,
            'std_pred_neutrino' : std_pred_neutrino,
            'std_pred_assent' : std_pred_assent,
            'contras' : contras,
            'neutrinos' : neutrinos,
            'assents' : assent
            })
            
    X.reset_index(inplace = True, drop = True)    
    X = X.merge(df_nli, left_index=True, right_index=True, how='outer')
    feat_nli = {'nli' : list(df_nli)}
    
    return X, feat_nli



# Classes
class collocater(BaseEstimator, TransformerMixin):
    
    def __init__(self, dict_colls):
        self.dict_colls = dict_colls
        self.freqs = {}


    def fit(self, X, y = None):
        
        whole_corpus = ' '.join(X['text'])
        whole_in_words = np.array(word_tokenize(whole_corpus))
        whole_eligible = list(set(whole_in_words) & self.dict_colls.keys())
        whole_colls_by_freq, _ = colls_checker(whole_in_words, whole_eligible, self.dict_colls, verbose = True)
        count_colls_freq = Counter(whole_colls_by_freq)
        # Sorting the collocations that appear in the whole corpus 
        # according to the frequency with which they appear
        for a, c in count_colls_freq.most_common()[::-1]:
            if c == 1:
                self.freqs.setdefault('super_rare_colloc', []).append(a)
            elif (c > 1 and c <= 7):
                self.freqs.setdefault('rare_colloc', []).append(a)
            elif (c > 7 and c <= 25): 
                self.freqs.setdefault('rarish_colloc', []).append(a)
            elif (c > 25 and c <= 55):
                self.freqs.setdefault('commonish_colloc', []).append(a)
            elif (c > 55 and c <= 150):
                self.freqs.setdefault('common_colloc', []).append(a)
            else:
                self.freqs.setdefault('super_common_colloc', []).append(a)
        
        return self


    def transform(self, X):
        
        eligible_ratio_total = []
        matches_per_doc = []
        ding_dong0 = []
        ding_dong1 = []
        ding_dong2 = []
        ding_dong3 = []
        ding_dong4 = []
        ding_dong5 = []
        for i in tqdm(range(X.shape[0])):
            text = str(X.iloc[i]['text'])
            words = np.array(word_tokenize(str(text)))
            eligible = list(set(words) & self.dict_colls.keys())
            colls_by_freq, matches_per_el = colls_checker(words, eligible, self.dict_colls)
            colls_by_freq = list(set(colls_by_freq))
            by_freq = {}
            ding_ding0 = 0
            ding_ding1 = 0
            ding_ding2 = 0
            ding_ding3 = 0
            ding_ding4 = 0
            ding_ding5 = 0
            for coll in colls_by_freq:
                if coll in self.freqs.get('super_rare_colloc'):
                    ding_ding0 += 1
                elif coll in self.freqs.get('rare_colloc'):
                    ding_ding1 += 1
                elif coll in self.freqs.get('rarish_colloc'):
                    ding_ding2 += 1
                elif coll in self.freqs.get('commonish_colloc'):
                    ding_ding3 += 1
                elif coll in self.freqs.get('common_colloc'):
                    ding_ding4 += 1
                else:
                    ding_ding5 += 1
            if len(colls_by_freq) > 0:
                    ding_dong0.append(ding_ding0/np.sum(matches_per_el))
                    ding_dong1.append(ding_ding1/np.sum(matches_per_el))
                    ding_dong2.append(ding_ding2/np.sum(matches_per_el))
                    ding_dong3.append(ding_ding3/np.sum(matches_per_el))
                    ding_dong4.append(ding_ding4/np.sum(matches_per_el))
                    ding_dong5.append(ding_ding5/np.sum(matches_per_el))
            else:
                ding_dong0.append(0)
                ding_dong1.append(0)
                ding_dong2.append(0)
                ding_dong3.append(0)
                ding_dong4.append(0)
                ding_dong5.append(0)
                
            if len(eligible) > 0:
                matches_per_doc.append(np.sum(matches_per_el))
            else:
                matches_per_doc.append(0)
                
            eligible_ratio_total.append(float(len(eligible) + 0.000001))  
            
        by_freq['super_rare_colloc'] = ding_dong0
        by_freq['rare_colloc'] = ding_dong1
        by_freq['rarish_colloc'] = ding_dong2
        by_freq['commonish_colloc'] = ding_dong3
        by_freq['common_colloc'] = ding_dong4
        by_freq['super_common_colloc'] = ding_dong5  

        X = X.assign(matches_per_doc = matches_per_doc)
        X = X.assign(eligible_ratio_total = eligible_ratio_total/np.asarray(X.n_words))
        X = X.assign(colls_per_eligible_wds = np.asarray(X.matches_per_doc)/eligible_ratio_total)
        
        freq_df = pd.DataFrame(by_freq)
        X.reset_index(inplace=True, drop=True)
        X = pd.concat([X, freq_df], axis = 1)
        
        print("\nThe sources ranked according to the texts' percentage of least-frequent collocations: \n{}\n".format(
                X.groupby('source')['super_rare_colloc'].mean().sort_values()
                ))
        
        feat_colls = ['matches_per_doc', 'eligible_ratio_total', 'colls_per_eligible_wds']
        feat_colls.extend(list(freq_df))
        feat_colls = {
                'collocations' : feat_colls
                }
    
        return X, feat_colls

    
    
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
        
        df_tx1 = pd.DataFrame(tf_tx1.toarray(), 
                              columns = ['tfidf_tx1_'+n for n in self.model_tx1.get_feature_names()])
        df_tx2 = pd.DataFrame(tf_tx2.toarray(), 
                              columns = ['tfidf_tx2_'+n for n in self.model_tx2.get_feature_names()])
        
        df_ps1 = pd.DataFrame(tf_pt1.toarray(), 
                              columns = ['tfidf_ps1_'+n for n in self.model_pt1.get_feature_names()])
        df_ps2 = pd.DataFrame(tf_pt2.toarray(), 
                              columns = ['tfidf_ps2_'+n for n in self.model_pt2.get_feature_names()])
    
        df_sd1 = pd.DataFrame(tf_sd1.toarray(), 
                              columns = ['tfidf_sn1_'+n for n in self.model_sd1.get_feature_names()])
        df_sd2 = pd.DataFrame(tf_sd2.toarray(), 
                              columns = ['tfidf_sn2_'+n for n in self.model_sd2.get_feature_names()])
    
        X.reset_index(inplace = True, drop = True)
        X = X.merge(df_tx1, left_index=True, right_index=True).\
        merge(df_tx2, left_index=True, right_index=True).\
        merge(df_ps1, left_index=True, right_index=True).\
        merge(df_ps2, left_index=True, right_index=True).\
        merge(df_sd1, left_index=True, right_index=True).\
        merge(df_sd2, left_index=True, right_index=True)
        
        feat_tfidf = {
                'text_1gram' : list(df_tx1),
                'text_2gram' : list(df_tx2),
                'pos_tags_1gram' : list(df_ps1),
                'pos_tags_2gram' : list(df_ps2),
                'syn_dep_1gram' : list(df_sd1),
                'syn_dep_2gram' : list(df_sd2)
                }
        
        return X, feat_tfidf



# The Doc2wecker is used for feature engineering, as well as as,  
# as a classifier, for target-variable prediction.
class Doc2wecker(BaseEstimator, TransformerMixin):
    
    def __init__(self, window = 7, min_count = 5, epochs = 100, vector_size = 200):
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

        raw_text = [word_tokenize(str(i)) for i in X['text']]  
        raw_tags = [word_tokenize(str(i)) for i in X['POS_tag']] 
        raw_syn_deps = [word_tokenize(str(i)) for i in X['syntactic_dependency']] 
        
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
            feat_wecker['doc2vec_vecs_' + names[key]] = list(df_vecs)
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
            feat_wecker['soft_doc2vec_dists_' + names[key]] = list(df_dists)
            df_list.append(df_dists)      
        
        X.reset_index(inplace = True, drop = True)
        new_df = pd.concat(df_list, axis = 1)
        X = X.merge(new_df, left_index=True, right_index=True)

        preds_dict = {}
        for cat in category_tags:
            preds_dict[cat] = new_df[[i for i in new_df.columns if re.search(str(cat), i)]].mean(axis = 1)
        
 
        return X, feat_wecker, preds_dict

 
          