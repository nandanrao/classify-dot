#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:49:08 2018

@author: elliott
"""


import os
import pandas as pd
import numpy as np
from glob import glob
from nltk.stem import WordNetLemmatizer
import re

PATH = '/home/elliott/Dropbox/DOT_regulation/'
os.chdir(PATH)

regex = r'\w+'

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()        

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in re.findall(regex,doc)]


vectorizer = pd.read_pickle('analysis/tfidf.pkl')
policy_prob = pd.read_pickle('analysis/X-desc_soc2/logistic-model.pkl')        

fnames = glob('data/samples/*txt')
fnames.sort()
rows = []

for fname in fnames:
    txt = open(fname).read()
    Xtfidf = vectorizer.transform([txt])    
    top = policy_prob.predict(Xtfidf)[0]    
    row = [fname.split('/')[-1][:-4]]
    row.append(top)
    row = row + list(policy_prob.predict_proba(Xtfidf)[0])
    print(row[:2])
    
    rows.append(row)
    
cols = ['fname','top_policy'] + list(policy_prob.classes_)
df = pd.DataFrame(rows,columns=cols)

df.to_excel('analysis/samples-predicted.xlsx')