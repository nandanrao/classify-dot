import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from toolz import curry
from validation.dot_data import get_dictionary, LemmaTokenizer
from statsmodels.iolib.summary2 import _df_to_simpletable, _formatter
from statsmodels.iolib.table import SimpleTable
import re

def bubbleup_score(y_train, X_test, y_test, model, N=3):
    labels = np.unique(y_train)
    preds = model.predict_proba(X_test)
    df = pd.DataFrame(preds)
    df.columns = labels
    return accuracy_score(get_soc_n_preds(df, N).values, 
                          y_test.astype(str).map(lambda s: s[0:N]))

class BubbleUpMixin():
    def set_bubbles(self, soc_n = 3, top_x = None):
        self.soc_n = soc_n
        self.top_x = top_x
        return self

    def predict(self, X):
        if not hasattr(self, 'soc_n'):
            raise Exception('Call set_bubbles before predicting!')
        preds = self.predict_proba(X)        
        df = pd.DataFrame(preds)
        df.columns = self.classes_
        if self.top_x is not None:
            return get_top_soc_n_preds(df, self.soc_n, self.top_x)
        else:
            return get_soc_n_preds(df, self.soc_n).values


def get_soc_n_preds(df, n):
    return (df.T
            .reset_index()
            .pipe(lambda df: df.assign(soc = df['index'].map(lambda i: str(i)[0:n])))
            .set_index('soc')
            .drop('index', 1)
            .groupby('soc')
            .sum().T
            .idxmax(1))

def get_top_soc_n_preds(df, soc_n, N):
    dd = (df.T
          .reset_index()
          .pipe(lambda df: df.assign(soc = df['index'].map(lambda i: str(i)[0:soc_n])))
          .set_index('soc')
          .drop('index', 1)
          .groupby('soc')
          .sum().T)

    idx = np.argsort(-dd.values, axis=1)[:,:N]
    predictions = dd.columns.values[idx]
    
    # TODO: Add probs
    # dd.values[idx]
    return predictions

def counts(arr, i):
    tp = arr[i,i]
    fp = np.sum(arr[:,i]) - tp
    fn = np.sum(arr[i,:]) - tp
    return tp, fp, fn

def prec(tp, fp):
    return tp/(tp+fp) if (tp+fp) > 0 else 0.

def recall(tp, fn):
    return tp/(tp+fn) if (tp+fn) > 0 else 0.

def scores_and_weights(df):
    idx = df.values.sum(1) != 0
    m = df.iloc[idx,:].values
    weights = m.sum(1) / m.sum()
    c = [counts(m, i) for i in np.arange(m.shape[0])]
    return c, weights

def micro(df):
    c,_ = scores_and_weights(df)
    tp, fp, fn = np.array(c).sum(0)
    micro_precision, micro_recall = tp / (tp + fp), tp / (tp + fn)
    return micro_precision, micro_recall

def macro(df, mode='weighted'):
    """ mode is {'weighted', 'raw', 'macro'} """
    c,weights = scores_and_weights(df)
    precisions = np.array([prec(tp,fp) for tp,fp,fn in c])
    recalls = np.array([recall(tp,fn) for tp,fp,fn in c])
    if mode == 'raw':
        return precisions, recalls
    elif mode == 'weighted':
        return precisions.dot(weights), recalls.dot(weights)
    else:
        return np.mean(precisions), np.mean(recalls)

def get_percentage(df, score, t, s):
    get_idx = lambda t: np.argwhere(t == df.columns)[0][0]
    get_trues = lambda i: df.iloc[i,:].sum()
    get_classified = lambda i: df.iloc[:,i].sum()

    i = get_idx(t)
    if score == 'recall':
        tot = get_trues(i)
    elif score == 'precision':
        tot = get_classified(i)
    return s/tot

def get_score(df, t, sdf, score):
    get_idx = lambda t: np.argwhere(t == df.columns)[0][0]

    i = get_idx(t)
    return sdf[score][i]

def map_series(ser, df):
    ser.index = df.columns[ser.index]
    return ser

def single_tabular(s, title, score):
    beg = 'begin{tabular}|end{tabular}'
    a = [re.search(beg, i) for i in s.split('\n')]
    tabulars = np.argwhere(np.array(a) != None).reshape(-1)
    insides = tabulars[1:-1]
    rows = [e for i,e in enumerate(s.split('\n')) if i not in insides]
    rows = rows[2:]
    rows = rows[:-2]
    pre = ['\\begin{subtable}[t]{\linewidth}',
           '\\begin{tabular*}{\\textwidth}{l @{\\extracolsep{\\fill}} c c c}']

    post = ['\\end{tabular*}', 
            '\caption{{ {} }}'.format(title),
            '\end{subtable}',
            '\\vspace{5mm}']
    rows = pre + rows + post
    return '\n'.join(rows)

def print_tables(x, score):
    for title,value,d in x:
        table = _df_to_simpletable(d, float_format="%.2f", index=False)
        s = table.as_latex_tabular()
        s = single_tabular(s, title, score)
        print(s)
        print('\n')

def make_code_lookup(SOC_LEVEL):
    dot_dict = get_dictionary('', SOC_LEVEL)
    di = (dot_dict[[f'desc_soc{SOC_LEVEL}', 'soc']]
          .groupby('soc')
          .head(1)
          .set_index('desc_soc3')
          .to_dict(orient='index'))
    return {k:v['soc'] for k,v in di.items()}

@curry
def truncate(lim, s):
    if len(s) > lim:
        return s[0:lim-2] + chr(8230)
    return s

def format_scores(s, code_lookup, count_lookup, test_count):
    dots = [str(count_lookup[code_lookup[c]]) for c in s.index ]
    tests = [str(test_count[c]) for c in s.index ]
    
    return pd.DataFrame({'Occupation': s.index.map(truncate(30)), 
                         'Percentage': s.values, 
                         'DOT/Test': [f'{d}/{t}' for d,t in zip(dots, tests)],
                         'SOC': [str(code_lookup[c]) for c in s.index ]}).reset_index(drop=True)

def format_dfs(score, sdf, df, idx, code_lookup, count_lookup, test_count):



    low = sdf[idx].sort_values(score).index
    df.index = df.columns
    x = [(df.columns[i], df.iloc[i,:].sort_values(ascending=False)) 
         for i in low[0:5]]

    if score == 'precision':
        x = [(df.columns[i], df.iloc[:,i].sort_values(ascending=False)) 
             for i in low[0:5]]
        # x = [(t, map_series(ser,df)) for t,ser in x]

    x = [(t,s[0:5]) for t,s in x]

    # x = [(t,s[s.index != t][0:5]) for t,s in x]

    # Get score and percentage
    x = [(t,get_score(df,t,sdf,score),get_percentage(df,score,t,s)) 
         for t,s in x]

    format_title = lambda t: f'{code_lookup[t]} - ({count_lookup[code_lookup[t]]}/{test_count[t]}) - {truncate(50, t)}'    

    x = [(format_title(t),score,format_scores(s, code_lookup, count_lookup, test_count)) for t,score,s in x]

    return x

def get_scores(country):
    df = pd.read_csv(f'confusion-matrices/soc-3/sentencespace_100_{country}.csv')
    vals = f1(*micro(df)), f1(*macro(df, 'macro')), f1(*macro(df, 'weighted'))
    cols = 'micro', 'macro', 'weighted-macro'
    return pd.Series((dict(zip(cols, vals))))

f1 = lambda p,r: 2*p*r / (p+r)
