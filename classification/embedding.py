from sklearn.base import BaseEstimator, TransformerMixin
from embed_software.utils import get_embeddings, embed_docs
from diskcache import FanoutCache
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from math import ceil
from collections import deque

class PreEmbeddedVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model, dims, cache_dir, chunk_size=1000):
        self.model = model
        self.dims = dims
        self.chunk_size = chunk_size
        self.cache = FanoutCache(cache_dir, shards=24, size_limit=2**32) # 4GB cache

    def fit(self, X, y=None):
        return self

    def embed_docs(self, docs):
        cache = self.cache
        cached = [cache.get(doc) for doc in docs]
        to_embed = [doc for doc,c in zip(docs, cached) if c is None]
        if to_embed:
            embedded = deque(embed_docs(self.model, '\n'.join(to_embed)))
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
