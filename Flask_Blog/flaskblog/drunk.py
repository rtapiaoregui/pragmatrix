#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:41:07 2018

@author: rita
"""

from flask import Flask, render_template, jsonify, request
from sklearn.externals import joblib
import os

os.chdir('/Users/rita/Google Drive/DSR/DSR Project')
from pragmatrix import feature_trove as feat
from pragmatrix.context import contextualizer
from pragmatrix import classifiers as cla

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def hello():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            new_text = str(data["text"])
            (paths, spacy_model,
             columns_to_idf, cols_to_remove,
             dict_colls, nli_tokenizer, nli_model) = contextualizer()
            featurizer = feat.add_features(columns_to_idf, dict_colls, nli_model, nli_tokenizer, paths)
            featurizer.load()
            new_obs, feature_dict = featurizer.transform(new_text)
            new_obs = cla.Dropping(cols_to_remove).fit_transform(new_obs)
            
            terminator = joblib.load(paths.get('terminator_path'))
            
        except ValueError:
            return jsonify("Please enter a text of length ranging from 3000 to 6000 characters.")

        return jsonify(terminator.predict(new_obs).tolist())
    

if __name__ == '__main__':
    app.run(debug = True)
    
