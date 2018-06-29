#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:55:15 2018

@author: rita
"""

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

import plotly
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go


# Loading the data sets
os.chdir('/Users/rita/Google Drive/DSR/DSR Project/Dataset/datasets/')
train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')

plot_path = "/Users/rita/Google Drive/DSR/DSR Project/Dataset/plots"

# Training set plots
data_s = train.source.value_counts()

data_sc = train.source_cat.value_counts()


trace1 = go.Pie(
        labels=data_s.index, 
        values=data_s
        )

trace2 = go.Pie(
        labels=data_sc.index, 
        values=data_sc
        )

#fig = tools.make_subplots(rows=1, cols=2)
#
#fig.append_trace([trace1], 1, 1)
#fig.append_trace([trace2], 1, 2)

layout1 = go.Layout(title="My observations' sources")
layout2 = go.Layout(title="The categories my observations' sources belong to")

fig1 = go.Figure(data=[trace1], layout=layout1)
fig2 = go.Figure(data=[trace2], layout=layout2)

#plotly.offline.plot(fig, filename='pie_charts.html')
plotly.offline.plot(fig1, filename='pie_chart1.html')
plotly.offline.plot(fig2, filename='pie_chart2.html')


