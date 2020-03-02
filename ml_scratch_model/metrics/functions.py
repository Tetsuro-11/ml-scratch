import numpy as np

def MSE(y_pred, y_true):
        mse = np.average(np.square(y_pred - y_true), axis=0)/2
        return mse
    
    
def cross_entropy(y_pred, y_true):
    delta = 1e-7
    return -np.sum(y_true*np.log(y_pred+delta))