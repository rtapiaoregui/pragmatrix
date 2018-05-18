#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:14:38 2018

@author: rita
"""

from nltk.tokenize import sent_tokenize
import numpy as np

def docs_normalizer(piece):
    
    if len(piece) < 3000:
        print("Sorry, the text couldn't be used. It was too short.")
        return None
    
    else:
        sent = sent_tokenize(piece)
       
        length = 0
        batches = []
        pieces = []
        for s in sent:
            #accumulate length for this group of sentences (batches)
            length += len(s)
            batches.append(s)
            
            #once we are over 3k characters, add the group of sentences to output
            #and reset batches to start over
            if (length > 3000):
                batch = ' '.join(batches)
                pieces.append(batch)
                length = 0
                batches = []
                    
        #include only up to 3 excerpts per text
        if len(pieces)>3:
            idx = np.random.choice(len(pieces), 3, replace = False)
            pieces = [pieces[i] for i in idx]
        return pieces





          