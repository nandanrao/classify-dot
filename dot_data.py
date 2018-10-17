from os.path import join
import pandas as pd

def get_desc_lookup(path):
    xwalk = pd.read_stata(path)
    return xwalk.drop_duplicates(['soc2', 'dot_code9'])[['desc_soc2', 'soc2', 'soc6', 'dot_code9']]

def get_dictionary(path):
    xwalk = get_desc_lookup(join(path, 'crosswalks', 'dot1991_census_soc_crosswalk.dta'))
    dot_dict = pd.read_csv(join(path, 'dot_test_2.csv'))
    return dot_dict.merge(xwalk, how='left', left_on='dot_code', right_on='dot_code9')


class LemmaTokenizer(object):
    regex = r'\w+'
    def __init__(self):
        self.wnl = WordNetLemmatizer()        

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in re.findall(self.regex,doc)]      