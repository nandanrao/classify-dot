#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:03:33 2018

@author: elliott
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from wordcloud import WordCloud
from multiprocessing import cpu_count
from nltk.stem import WordNetLemmatizer
import spacy
import re
regex = r'\w+'
nlp = spacy.load('en')

cpus = round(cpu_count() / 2) - 1

PATH = '/home/elliott/Dropbox/DOT_regulation/'

os.chdir(PATH)

dot = pd.read_csv('data/dot_1991/dot_test_2.csv')

xwalk = pd.read_stata('data/dot1991_census_soc_crosswalk.dta')
xwalk['dot_code'] = xwalk['dot_code9']
xwalk = xwalk[['dot_code','occ2000','soc6', 'occ1990dd',
               'soc2', 'desc_soc2',
               'soc3', 'desc_soc3',
               'soc6', #'desc_soc6',
               'ad_task_abstract', 'ad_task_routine', 'ad_task_manual',
               'ad_task_offshorability']]
xwalk = xwalk.drop_duplicates()

df = pd.merge(dot,xwalk,how='left',on='dot_code')

cbow = lambda x: nlp(x).vector

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in re.findall(regex,doc)]

#def cbow(tokens,weights):
#    V = np.mean([weights[w] * nlp(w) for w in tokens])

make_new = False
if make_new:
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                       use_idf=True, 
                                       ngram_range=(1,3),
                                       min_df = 25,
                                       max_df = .4,
                                       tokenizer = LemmaTokenizer()
                                       )
    
    
    W = tfidf_vectorizer.fit_transform(df['job_description'])
    pd.to_pickle(tfidf_vectorizer,'analysis/tfidf.pkl')
    
    V = df['job_description'].apply(cbow).values.tolist()
    pd.to_pickle(V,'analysis/V.pkl')
else:
    tfidf = pd.read_pickle('analysis/tfidf.pkl')
    W = tfidf.transform(df['job_description'])
    V = np.array(pd.read_pickle('analysis/V.pkl'))

for yvar in ['desc_soc2','desc_soc3', 'occ1990dd']:  #'ad_task_abstract', 'ad_task_routine', 'ad_task_manual',
    for xvar in ['X', 'V']:
        print()
        print(xvar,yvar)
        print()
        Yraw = df[yvar]

       
        keeplist = [x[0] for x in Yraw.value_counts().items() if x[1] >= 10]
        keep = np.array(pd.notnull(Yraw) & Yraw.isin(keeplist))    
        
        #keep = (df['desc_soc2']!='Production Occupations')
    
        Y = Yraw[keep]
        if xvar == 'X':
            X = W[keep]          
        else:
            X = V[keep]       
        
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                             Y,
                                                            train_size = .75)
        
        # defaults
        model = LogisticRegression(solver="newton-cg", 
                                   multi_class="multinomial", 
                                   n_jobs=cpus)
        
        model.fit(X_train, Y_train)
        print('Default Model:',model.score(X_test,Y_test))
        
        # pick out parameters using 3-fold CV
        param_grid = {'C':[.1, 1, 2, 10,100, 500], 'class_weight':[None,'balanced'] }
        CV_model = GridSearchCV(estimator=model, param_grid=param_grid) #default scorer is used, classification accuracy
        CV_model.fit(X_train, Y_train)
        # C = 100, class_weight=None
        print('Best Params:',CV_model.best_params_)
        print(CV_model.best_score_)
        
        bestmodel = LogisticRegression(C=CV_model.best_params_['C'], 
                                       class_weight=CV_model.best_params_['class_weight'],
                                       solver="newton-cg", multi_class="multinomial",
                                       n_jobs=cpus)
        
        bestmodel.fit(X_train, Y_train)
        Y_pred = bestmodel.predict(X_test)
        print('In-Sample Accuracy:',bestmodel.score(X_train,Y_train)) # in sample accuracy = 0.686
        print('Out-of-Sample Accuracy:',bestmodel.score(X_test,Y_test)) # out of sample accuracy = 0.504
        print('F1 Score (Micro):',f1_score(Y_pred,Y_test,average='micro')) # same as accuracy
        print('F1 Score (Macro):',f1_score(Y_pred,Y_test,average='macro')) # 0.387 
        print('F1 Score (Weighted):',f1_score(Y_pred,Y_test,average='weighted')) # 0.521
        
        pd.to_pickle(bestmodel,'analysis/%s-%s/logistic-model.pkl'%(xvar,yvar))
        M = pd.DataFrame(confusion_matrix(Y_test,Y_pred),columns=list(bestmodel.classes_))
        M.to_excel('analysis/%s-%s/confusion-matrix.xlsx'%(xvar,yvar))
        
        policy_probs = bestmodel.predict_proba(X)
        pd.to_pickle(policy_probs,'analysis/%s-%s/policy-probs.pkl'%(xvar,yvar))
        C = pd.DataFrame(policy_probs).corr()
        C.to_excel('analysis/%s-%s/pwcorr.xlsx'%(xvar,yvar))
        
        continue
        print('skipping word clouds')
    
        if xvar == 'X' :
            topics = set(Y)            
            words = tfidf_vectorizer.get_feature_names()
            
            words = [w.replace(' ', '_') for w in words]
            
            tstats = {k:{} for k in topics}
            betas = {k:{} for k in topics}
                        
            for topic in topics:                        
                topicY = Y == topic
                #print(topic)   
                tstats[topic] = {}
                betas[topic] = {}
                for i,word in enumerate(words):            
                    #print(xvar)
                    #if i % 100 == 0:
                    #    print(i)
                    
                    model = OLS(topicY,sm.add_constant(X[:,i].toarray())) 
                    result = model.fit() 
                    t = result.tvalues[1]
                    if pd.isnull(t):
                        t = 0
                    tstats[topic][word] = t
                    betas[topic] = result.params[1]
                        
            pd.to_pickle(tstats,'analysis/tstats-policy-%s.pkl'%yvar)    
            pd.to_pickle(betas,'analysis/betas-policy-%s.pkl'%yvar)
            
            tdict_topics = pd.read_pickle('analysis/tstats-policy-%s.pkl'%yvar)
            
            for topic, tdict in tdict_topics.items():
                #topic = topic.replace('topicprob_','')
                #print(topic)
                if len(tdict) == 0:
                    continue
                words = [w for w in tdict if len(w) > 1]# if termfreqs[w] > 200]    
                
                tstats = np.array([tdict[w] for w in words])
                
                # split up positive-effect and negative-effect words
                pos = tstats > .5
                neg = tstats < -.5
                
                tpos = tstats[pos]
                wordpos = [words[a] for a in [i for i, x in enumerate(pos) if x]]
                tneg = np.abs(tstats[neg]) # reverse sign for negative effect words
                wordneg = [words[a] for a in [i for i, x in enumerate(neg) if x]]
                
                maincol = 240#np.random.randint(0,360) # this is the "main" color
                def colorfunc(word=None, font_size=None, position=None,
                                  orientation=None, font_path=None, random_state=None):       
                    color = np.random.randint(maincol-10, maincol+10) 
                    if color < 0:
                        color = 360 + color
                    return "hsl(%d, %d%%, %d%%)" % (color,np.random.randint(65, 75)+font_size / 7, np.random.randint(35, 45)-font_size / 10)   
                
                # build scores tuples
                # positive effect words  
                scores = list(zip(tpos,wordpos))
                scores = [s for s in scores if np.isfinite(s[0])] 
                #if ONLY_PHRASES:
                #    scores = [s for s in scores if '_' in s[1]]
                scores.sort()
                scores.reverse()
                scores = [(b,np.log(a)) for (a,b) in scores]       

                wordcloud = WordCloud(background_color="white", ranks_only=False,max_font_size=100,
                                        color_func=colorfunc,
                                        height=600,width=1000).generate_from_frequencies(dict(scores[:100]))
                wordcloud.to_file('analysis/%s/'%yvar+topic+'-pos-words.png')
                
                # negative effect words
                scores = list(zip(tneg,wordneg))
                scores = [s for s in scores if np.isfinite(s[0])] 
                maincol = 0#np.random.randint(0,360) # this is the "main" color
                
                #if ONLY_PHRASES:
                #    scores = [s for s in scores if '_' in s[1]]
                scores.sort()
                scores.reverse()
                scores = [(b,np.log(a)) for (a,b) in scores]
                #print(scores[:10]) 
                wordcloud = WordCloud(background_color="white", ranks_only=False,max_font_size=100,
                                        color_func=colorfunc,
                                        height=600,width=1000).generate_from_frequencies(dict(scores[:100]))
                wordcloud.to_file('analysis/%s/'%yvar+topic+'-neg-words.png')
                
        