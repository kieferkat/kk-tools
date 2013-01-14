
import numpy as np



def simple_normalize(X, axis=0):
    
    print 'normalizing X'
    print 'previous X sum', np.sum(X)
    
    std_devs = np.std(X, axis=axis)
    means = np.mean(X, axis=axis)
    
    Xnorm = np.zeros(X.shape)
    Xnorm = X-means
    Xnorm = Xnorm/std_devs

    # remove any nans or infinities:
    Xnorm = np.nan_to_num(Xnorm)
    
    print 'post-normalization X sum', np.sum(Xnorm)
    
    return Xnorm

