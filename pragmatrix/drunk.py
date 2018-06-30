#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:41:07 2018

@author: rita
"""

from flask import Flask, render_template, jsonify, request
from flask_bootstrap import Bootstrap
import pickle
import numpy as np
import pandas as pd
import feature_trove as feat
from context import contextualizer
import classifiers as cla

app = Flask(__name__)
Bootstrap(app)



@app.route("/")
@app.route("/home", methods=['POST', 'GET'])
def hello():
    if request.method == 'GET':
        return render_template('index.html', title = 'pragmatrix')
           

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.form['user_submission']
        new_text = str(data)
        if not (len(new_text) >= 3000) and (len(new_text) <= 6000):
            raise ValueError
        df = pd.DataFrame({'text' : [new_text], 'source' : [''], 'source_cat' : ['']})
        
        (paths, spacy_model,
         columns_to_idf, cols_to_remove,
         dict_colls, nli_tokenizer, nli_model) = contextualizer()
        featurizer = feat.add_features(columns_to_idf, dict_colls, nli_model, nli_tokenizer, paths)
        featurizer.load()
        new_obs, feature_dict = featurizer.transform(df)
        new_obs = cla.Dropping(cols_to_remove).fit_transform(new_obs)
        
        xgb_obs = new_obs.loc[:, ~new_obs.columns.isin(['d2v_dist_text_literariness_1',
                                                       'd2v_dist_text_literariness_0',
                                                       'd2v_dist_pos_tags_literariness_1',
                                                       'd2v_dist_pos_tags_literariness_0', 
                                                       'd2v_dist_syn_deps_literariness_1',
                                                       'd2v_dist_syn_deps_literariness_0'])]
       
        with open(paths.get('terminator_path'), "rb") as filehandler:
            terminator = pickle.load(filehandler)
         
        preds = terminator.predict_proba(xgb_obs)[:, int(np.where(terminator.classes_ == 1)[0])]

        return render_template('prediction.html', title = 'prediction', pred = preds[0])

    except ValueError:
        return jsonify("Error: Please enter a text of length ranging from 3000 to 6000 characters.")


if __name__ == '__main__':
    app.run(debug = True, threaded = False)
    
