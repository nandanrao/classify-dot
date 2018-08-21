import pandas as pd
import numpy as np
import re
from toolz import curry
from sklearn.feature_extraction.text import VectorizerMixin
from multiprocessing import Pool

punctuation = re.compile(r"[^0-9a-zA-Z\s]")

@curry
def exact_matcher(lookup, df):
    d = df.merge(lookup, how='left', on=['title'])
    d['assigned_title'] = d.title
    return d[~d.code.isna()]

@curry
def punct_lookup(lookup, t):
    z = zip(lookup.code, lookup.title)
    grams = [s.strip() for s in re.split(punctuation, t)]
    options = [(code, title) for code,title in z if title in grams]
    if len(options) > 0:
        options = sorted(options, key = lambda t: -len(t[1]))
    return options[0] if options else (None,None)

@curry
def title_matcher(lookup, match_fn, df):
    pool = Pool()
    df['code'], df['assigned_title'] = zip(*pool.map(match_fn, df.title))
    pool.close()
    pool.join()
    return df[~df.code.isna()]

@curry
def layered_matcher(fns, to_match, matched = []):
    if not fns:
        return pd.concat(matched, sort=False)
    fn = fns[0]
    new_matches = fn(to_match)
    new_to_match = to_match[~to_match.index.isin(new_matches.index)]
    new_matched = matched + [new_matches.reset_index(drop=True)]
    to_match = new_to_match.reset_index(drop=True)
    return layered_matcher(fns[1:], to_match, new_matched)


class NGrammer(VectorizerMixin):
    def __init__(self, ngram_range):
        self.ngram_range = ngram_range
        self.tokenizer = None
        self.token_pattern=r"(?u)\b\w\w+\b"
        self.tokenizer = self.build_tokenizer()

    def ngram_it(self, s):
        return self._word_ngrams(self.tokenizer(s))

from fuzzywuzzy import fuzz, utils, process

@curry
def lookup_code(lookup, t):
    try:
        prop, score = process.extractOne(t, lookup.title, score_cutoff = 0)
        return prop
    except ValueError:
        return None

# t = utils.full_process(j[j.code.isnull()].title.values[5])
# lookup_two(lookup)(t)
