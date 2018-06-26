#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:16:05 2018

@author: rita
"""


from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from flaskblog import routes
import os

os.chdir('/Users/rita/Google Drive/DSR/DSR Project/Code')
import feature_trove as feat
from context import contextualizer
import classifiers as cla


app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)