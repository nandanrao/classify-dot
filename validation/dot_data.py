from os.path import join
import pandas as pd

def get_desc_lookup(path, soc_n):
    lookup = {
        2: 'soc2',
        3: 'soc3',
        6: 'soc6'
    }
    desc_lookup = {
        2: 'desc_soc2',
        3: 'desc_soc3',
        6: 'soc_title6'
    }
    key = lookup[soc_n]
    desc_key = desc_lookup[soc_n]
    return (pd
            .read_stata(path)
            .drop_duplicates([key, 'dot_code9'])[list(lookup.values()) + [desc_key, 'dot_code9', 'occ1990dd']]
            .rename(columns = {key: 'soc'})
            .pipe(lambda df: df.assign(soc = df.soc.astype(int))))

def get_dictionary(path, soc_n):
    xwalk = get_desc_lookup(join(path, 'crosswalks', 'dot1991_census_soc_crosswalk.dta'), soc_n)
    dot_dict = pd.read_csv(join(path, 'dot_test_2.csv'))
    return dot_dict.merge(xwalk, how='left', left_on='dot_code', right_on='dot_code9')

class LemmaTokenizer(object):
    regex = r'\w+'
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in re.findall(self.regex,doc)]
