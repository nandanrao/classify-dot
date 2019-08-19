from sklearn.base import BaseEstimator, TransformerMixin
from embed_software.utils import get_embeddings, embed_docs
from diskcache import FanoutCache
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from math import ceil
from collections import deque

class PreEmbeddedVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model, cache_dir, chunk_size=1000):
        self.model = model
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        self.cache = FanoutCache(cache_dir, shards=24, size_limit=2**32) # 4GB cache

    def fit(self, X, y=None):
        return self

    def _embed_docs(self, docs):
        return embed_docs(self.model, '\n'.join(docs))

    def embed_docs(self, docs):
        cache = self.cache
        cached = [cache.get(doc) for doc in docs]
        to_embed = [doc for doc,c in zip(docs, cached) if c is None]
        if to_embed:
            embedded = deque(self._embed_docs(to_embed))
        else:
            embedded = []

        for doc,e in zip(to_embed, embedded):
            cache.set(doc, e)

        return np.array([c if c is not None else embedded.popleft() for c in cached])

    def transform(self, X):
        # rase Attribute error....
        if type(X) == str:
            return self.embed_docs([X])

        if type(X) == list:
            X = np.array(X)

        if len(X) < 200:
            return self.embed_docs(X)

        with ProcessPoolExecutor() as pool:
            chunks = ceil(X.shape[0] / self.chunk_size)
            chunked = np.array_split(X, chunks)
            embeds = pool.map(self.embed_docs, chunked)

        return np.vstack(list(embeds))

class Embedding():
    def __init__(self, path, sep='\t'):
        embedding = pd.read_csv(path, sep=sep, header=None)
        keys = embedding.iloc[:,0]
        vals = embedding.iloc[:,1:].values
        self.lookup = {k:v for k,v in zip(keys, vals)}

    def embed_paragraph(self, doc):
        sents = doc.split('\t')
        vecs = [self.embed_sent(sent) for sent in sents]
        vecs = [v for v in vecs if v is not None] # check if sentence is empty
        return np.array(vecs)            

    def embed_sent(self, sent):
        vec = self.embed_doc(sent)
        if len(vec):
            return vec.sum(0) / np.linalg.norm(vec)
        else:
            return None

    def embed_doc(self, doc, return_words = False):
        words = []
        vecs = []
        for word in doc.split():
            try:
                vecs.append(self.lookup[word])
                words.append(word)
            except KeyError:
                pass
        if not return_words: 
            return np.array(vecs)
        return np.array(vecs), words


class WordEmbeddingVectorizer(PreEmbeddedVectorizer):
    def __init__(self, vec_path, cache_dir, sep=' ', chunk_size=1000):
        self.embedding = Embedding(vec_path, sep=sep)
        self.chunk_size = chunk_size
        self.vec_path = vec_path
        self.cache_dir = cache_dir
        self.sep = sep
        self.cache = FanoutCache(cache_dir, shards=24, size_limit=2**32)

    def embed_doc(self, doc):
        e = self.embedding.embed_doc(doc)
        e = e.sum(0) / np.linalg.norm(e)
        return e
    
    def _embed_docs(self, docs):
        return np.array([self.embed_doc(doc) for doc in docs])
