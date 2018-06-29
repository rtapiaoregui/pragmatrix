#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:41:07 2018

@author: rita
"""

from flask import Flask, render_template, jsonify, request
from flask_bootstrap import Bootstrap
import pickle
import os

import pandas as pd
import feature_trove as feat
from context import contextualizer
import classifiers as cla

app = Flask(__name__)
Bootstrap(app)

form = """
<form action = "/predict" method = "POST">
    <label for = "user_submission"> Submit a 3000 to 6000 characters-long text to obtain predictions: </label>
    <input id = "user_submission" type = "text" name = "user_submission" />
    <input type = "submit" />
</form>

"""

@app.route("/")
@app.route("/home", methods=['POST', 'GET'])
def hello():
    if request.method == 'GET':
        return render_template('index.html', title = 'About')
           

@app.route("/predict", methods=['POST'])
def prdict():
    try:
        data = request.form['user_submission']
        new_text = str(data)
        df = pd.DataFrame({'text' : [new_text], 'source' : [''], 'source_cat' : ['']})
        
        (paths, spacy_model,
         columns_to_idf, cols_to_remove,
         dict_colls, nli_tokenizer, nli_model) = contextualizer()
        featurizer = feat.add_features(columns_to_idf, dict_colls, nli_model, nli_tokenizer, paths)
        featurizer.load()
        new_obs, feature_dict = featurizer.transform(df)
        new_obs = cla.Dropping(cols_to_remove).fit_transform(new_obs)
        
        with open(paths.get('terminator_path'), "rb") as filehandler:
            terminator = pickle.load(filehandler)
        
        xgb_obs = new_obs.loc[:, ~new_obs.columns.isin(['d2v_dist_text_literariness_1',
                                                       'd2v_dist_text_literariness_0',
                                                       'd2v_dist_pos_tags_literariness_1',
                                                       'd2v_dist_pos_tags_literariness_0', 
                                                       'd2v_dist_syn_deps_literariness_1',
                                                       'd2v_dist_syn_deps_literariness_0'])]
         
        preds = terminator.predict_proba(xgb_obs)[:, 1].tolist()

        return jsonify("These are the predictions.",format(preds))

    except ValueError:
        return jsonify("Please enter a text of length ranging from 3000 to 6000 characters.")


if __name__ == '__main__':
    app.run(debug = True)
    
