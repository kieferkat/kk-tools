
import numpy as np



def simple_normalize(data, axis=0):
    
    X = np.array(data)
    std_devs = np.std(X, axis=axis)
    means = np.mean(X, axis=axis)
    
    Xnorm = X-means
    Xnorm = Xnorm/std_devs
    
    # remove any nans or infinities:
    Xnorm = np.nan_to_num(Xnorm)
    
    return Xnorm

