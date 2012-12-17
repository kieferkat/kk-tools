
import numpy as np



def simple_normalize(X, axis=0):
    
    std_devs = np.std(X, axis=axis)
    means = np.mean(X, axis=axis)
    
    X = X-means
    X = X/std_devs
    
    # remove any nans or infinities:
    X = np.nan_to_num(X)
    
    return X

