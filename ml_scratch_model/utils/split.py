import numpy as np
import pandas as pd
def train_test_split(X, y, train_size=0.8):
    """
    学習用データを分割する。

    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, )
      正解値
    train_size : float (0<train_size<1)
      何割をtrainとするか指定

    Returns
    ----------
    X_train : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    X_test : 次の形のndarray, shape (n_samples, n_features)
      検証データ
    y_train : 次の形のndarray, shape (n_samples, )
      学習データの正解値
    y_test : 次の形のndarray, shape (n_samples, )
      検証データの正解値
    """
    
    sel_num = int(len(X) * train_size)
    if shuffle:
        data = np.concatenate([X,y[:,np.newaxis]], axis=1)
        np.random.shuffle(data) #shuffle
        X_train, X_test, y_train, y_test = data[:sel_num, :-1],data[sel_num:, :-1], data[:sel_num, -1], data[sel_num:, -1]
    else:
        X_train, X_test, y_train, y_test = X[:sel_num], X[sel_num:], y[:sel_num], y[sel_num:]
    return X_train, X_test, y_train, y_test
    