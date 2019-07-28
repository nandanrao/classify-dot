import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def bubbleup_score(y_train, X_test, y_test, model, N=3):
    labels = np.unique(y_train)
    preds = model.predict_proba(X_test)
    df = pd.DataFrame(preds)
    df.columns = labels
    return accuracy_score(get_soc_n_preds(df, N).values, 
                          y_test.astype(str).map(lambda s: s[0:N]))

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

    predictions = dd.columns.values[np.argsort(-dd.values, axis=1)[:,:N]]

    return predictions
