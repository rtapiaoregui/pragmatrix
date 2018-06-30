#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 08:51:49 2018

@author: rita

EDA
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text

# Loading the data sets
os.chdir('/Users/rita/Google Drive/DSR/DSR Project/Data/datasets/')
train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')

plot_path = "/Users/rita/Google Drive/DSR/DSR Project/Data/plots"

# Training set plots
data_s = train.source.value_counts().reset_index().rename(
        columns = {"index":"Sources", "source":"Amount of observations"}
        )

data_sc = train.source_cat.value_counts().reset_index().rename(
        columns = {"index":"Source categories", "source_cat":"Amount of observations"}
        )

sns.set()
sns.set_context("talk")
sns.set_style("white")
#sns.despine(offset = 10, trim=True)
plt.figure(figsize=(15, 8))
plt.subplot(1, 3, 1)
source_plot = sns.barplot(
        x = "Amount of observations", 
        y = "Sources", 
        data = data_s)
source_plot.set_xticklabels(source_plot.get_xticklabels(), rotation=90)

plt.subplot(1, 3, 2)
source_cat_plot = sns.barplot(
        x = "Amount of observations",
        y = "Source categories", 
        data = data_sc)
source_cat_plot.set_xticklabels(source_cat_plot.get_xticklabels(), rotation=90)

plt.subplot(1, 3, 3)
sns.countplot(x = 'literariness', data = train)
plt.xlabel("Degree of literariness")
plt.ylabel("Amount of observations")
labels = ["Non-literary texts", "Literary texts"]
plt.legend(labels, loc = 'best')

plt.tight_layout()
plt.savefig(os.path.join(plot_path, "labels.png"), transparent=True)

# Test set plots
amazon = test.loc[test['source'] == 'amazon_reviews'][:]
yelp = test.loc[test['source'] == 'yelp_reviews'][:]

plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)
sns.distplot(amazon.helpfulness, color = 'green')
plt.title('Ratings on helpfulness of Amazon-movie and TV reviews')
plt.subplot(2, 2, 2)
sns.distplot(yelp.usefulness, color = 'orange')
plt.title('Ratings on usefulness of Yelp-business reviews')
plt.subplot(2, 2, 3)
sns.distplot(yelp.comicality, color = 'purple')
plt.title('Ratings on comicality of Yelp-business reviews')
plt.subplot(2, 2, 4)
sns.distplot(yelp.ice_ice_baby)
plt.title('Ratings on coolness of Yelp-business reviews')
plt.xlabel("coolness")
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, "reviews.png"), transparent=True)




