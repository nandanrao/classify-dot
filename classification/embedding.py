from sklearn.base import BaseEstimator, TransformerMixin
from embed_software.utils import get_embeddings, embed_docs
from diskcache import FanoutCache
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from math import ceil
from collections import deque
import xxhash

class PreEmbeddedVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model, cache_dir, chunk_size=1000, max_workers=None):
        self.model = model
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        self.cache = FanoutCache(cache_dir, shards=24, size_limit=2**32) # 4GB cache
        self.max_workers = max_workers
 
    def fit(self, X, y=None):
        return self

    def _embed_docs(self, docs):
        return embed_docs(self.model, '\n'.join(docs))

    def cached_embed_docs(self, docs):
        cache = self.cache
        cached = [cache.get(xxhash.xxh64(doc).hexdigest()) for doc in docs]
        to_embed = [doc for doc,c in zip(docs, cached) if c is None]

        if to_embed:
            embedded = deque(self._embed_docs(to_embed))
        else:
            embedded = []

        for doc,e in zip(to_embed, embedded):
            xid = xxhash.xxh64(doc).hexdigest()
            cache.set(xid, e)

        return np.array([c if c is not None else embedded.popleft() for c in cached])

    def embed_docs(self, docs):
        if self.cache_dir:
            return self.cached_embed_docs(docs)

        return self._embed_docs(docs)


    def transform(self, X):
        # rase Attribute error....
        if type(X) == str:
            return self.embed_docs([X])

        if type(X) == list:
            X = np.array(X)

        if len(X) < 200 or self.max_workers == 1:
            return self.embed_docs(X)

        with ProcessPoolExecutor(self.max_workers) as pool:
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
        self.size = vals.shape[1]
        self.default = np.array([np.zeros(self.size)])

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
        vecs = np.array(vecs) if len(vecs) > 0 else self.default
        if not return_words: 
            return vecs 
        return vecs, words

from numba import njit

@njit
def normalize(e):
        e = e.sum(0)
        norm = np.linalg.norm(e)
        if norm > 0.0000001:
            e = e / norm
        return e

class WordEmbeddingVectorizer(PreEmbeddedVectorizer):
    def __init__(self, vec_path, cache_dir, sep=' ', chunk_size=1000, max_workers = None):
        self.embedding = Embedding(vec_path, sep=sep)
        self.chunk_size = chunk_size
        self.vec_path = vec_path
        self.sep = sep
        self.cache = FanoutCache(cache_dir, shards=24, size_limit=2**32)
        self.cache_dir = cache_dir
        self.max_workers = max_workers

    def embed_doc(self, doc):
        e = self.embedding.embed_doc(doc)
        return normalize(e)
    
    def _embed_docs(self, docs):
        return np.array([self.embed_doc(doc) for doc in docs])
