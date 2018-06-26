#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:41:07 2018

@author: rita
"""

from flask import Flask, render_template, jsonify, request
from sklearn.externals import joblib
import os

os.chdir('/Users/rita/Google Drive/DSR/DSR Project/Flask_Blog/')
from flaskblog import app
    

if __name__ == '__main__':
    app.run(debug = True)
    
