from sklearn.linear_model import LassoLarsCV, LassoCV, Lasso
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hashlib import md5
from hdmedians import medoid
from functools import partial
import pandas as pd
import numpy as np

from joblib import Memory
mem = Memory('../tasks-joblib', verbose=0)

def hash(s):
    m = md5()
    m.update(s.encode('utf-8'))
    return m.hexdigest()

@mem.cache()
def label_clusters_bow(socd, clusters):
    # Top 5 positive words for each cluster via tfidf logistic regression
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(socd.task)

    logistic = LogisticRegression(penalty='l1', C=2)
    logistic.fit(tfidf, clusters)

    rev_lookup = dict([(idx, w) for w,idx in tfidf_vectorizer.vocabulary_.items()])
    a = np.array([[rev_lookup[j] for j in i]
                  for i in np.flip(np.argsort(logistic.coef_, 1), 1)[:, :5]])

    return np.apply_along_axis(lambda x: '; '.join(x), 1, a)

@mem.cache()
def label_clusters_medoid(socd, clusters, vecs):
    flattened = np.array([np.hstack(i) for i in vecs])
    lookup = dict(zip([hash(str(i)) for i in flattened], socd.task))

    da = (pd.DataFrame(np.hstack([clusters.reshape(-1, 1), flattened]))
          .rename(columns = {0: 'cluster'})
          .sort_values('cluster')
          .groupby('cluster')
          .apply(lambda df: medoid(df.iloc[:, 1:].values.T))
          .map(lambda a: lookup[hash(str(a))]))

    d = sorted(list(da.to_dict().items()), key=lambda t: t[0])
    return np.array([b for a,b in d])


def vectorize_tasks(socd, clusters, key):
    cluster_one_hot = (OneHotEncoder()
                       .fit_transform(clusters.reshape(-1, 1))
                       .todense())

    return (pd.DataFrame(np.hstack([socd[key].values.reshape(-1,1),
                                    cluster_one_hot]))
            .rename(columns = {0: key})
            .groupby(key)
            .sum()
            .applymap(lambda v: np.minimum(v,1))
            .reset_index())

def get_task_overlap(socd, clusters):
    v = vectorize_tasks(socd, clusters)
    tasks_per_soc = v.iloc[:,1:].sum(1)
    socs_per_task = v.iloc[:,1:].sum(0)
    return tasks_per_soc, socs_per_task

@mem.cache()
def make_regression_data(occ_xwalk, socd, wages_per_occ, clusters):
    xx = occ_xwalk.merge(socd.assign(cluster = clusters.astype(int)),
                         how='inner',
                         on='soc')
    xx = xx[xx.cluster != 1]
    v = vectorize_tasks(xx, xx.cluster.values, 'occ1990dd')
    d = wages_per_occ.merge(v, how = 'inner', on='occ1990dd')
    X,y = d.iloc[:,2:], d.ln_hrwage_sic_purge
    return X, y, d.occ1990dd

def hh(subval, df):
    shares = df.groupby(subval).soc.count() / df.shape[0]
    return np.sum(shares**2)

def describe_clusters(socd, clusters, vecs):
    HH = socd.assign(cluster = clusters).groupby('cluster').apply(partial(hh, 'soc'))
    return pd.DataFrame({
        'cluster': np.unique(clusters),
        'label_bow': label_clusters_bow(socd, clusters),
        'label_medoid': label_clusters_medoid(socd, clusters, vecs),
        'HH': HH.values
    })

def get_occupational_error(preds, y, occ_codes):
    title_lookup = (pd.read_stata('census_us/dot1991_census_soc_crosswalk.dta')
                    .groupby('occ1990dd')
                    .head(1)[['occ1990dd', 'dot_title']])

    return (pd.DataFrame({'occ1990dd': occ_codes,
                   'err': preds - y,
                   'abs_err': np.abs(preds - y) })
     .merge(title_lookup, on = 'occ1990dd')
     .sort_values('abs_err', ascending=False))

def interpret_weights(w, lookup, thresh):
    descs = [lookup[i] for i in np.argwhere(np.abs(w) > thresh)[:,0]]
    dat = [[v, *d]for v,d in zip(w[np.abs(w) > thresh], descs)]
    return pd.DataFrame(dat, columns = ['coef', 'idx', 'bow_desc', 'medoid_desc', 'HH']).sort_values('coef')

def make_lookup(socd, clusters, vecs):
    lookup = describe_clusters(socd, clusters, vecs)
    return {i: v.values for i,v in lookup.iterrows()}


def score_model(alpha, X, y, occ_codes, lookup):
    rg = Lasso(alpha = alpha, max_iter=10000)
    preds = cross_val_predict(rg, X, y, cv=LeaveOneOut())
    occ_errors = get_occupational_error(preds, y, occ_codes)
    os_error = 1 - np.mean((preds - y)**2) / np.var(y)
    preds = rg.fit(X, y).predict(X)
    is_error = 1 - np.mean((preds - y)**2) / np.var(y)
    interpreted = interpret_weights(rg.coef_, lookup, 0.00001)
    return os_error, is_error, occ_errors, interpreted

@mem.cache()
def train_and_score(X, y, occ_codes, lookup):
    lasso = LassoCV(max_iter=10000, cv = LeaveOneOut(), n_jobs = -1)
    lasso.fit(X, y)
    return score_model(lasso.alpha_, X, y, occ_codes, lookup)

def score_by_cluster_size(socd, vecs, occ_xwalk, wages_per_occ, clusterer, **kwargs):
    clusters = clusterer(vecs, **kwargs)
    X, y, occ_codes = make_regression_data(occ_xwalk, socd, wages_per_occ, clusters)
    occd = occ_xwalk.merge(socd.assign(cluster = clusters.astype(int)),
                           how='inner',
                           on='soc')
    lookup = make_lookup(socd, clusters, vecs)

    return train_and_score(X, y, occ_codes, lookup)
