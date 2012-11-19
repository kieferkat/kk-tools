
import pylab
import numpy as np
from pprint import pprint
from ..base.crossvalidation import CVObject
from ..utilities.csv import CsvTools
from regression import Regression
from sklearn.pls import PLSCanonical, PLSRegression, CCA



class PartialLeastSquares(Regression):
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(PartialLeastSquares, self).__init__(data_obj=data_obj,
                                                  variable_dict=variable_dict,
                                                  folds=folds)
        
        
    
    def pls_univariate_train(self, X, Y, n_components=3, verbose=True):
        
        pls = PLSRegression(n_components=n_components)
        
        if verbose:
            print 'fitting canonical pls...'
            
        pls.fit(X, Y)
        
        return pls
    
    
    def pls_univariate_test(self, X, Y, pls):
        
        predicted_Y = pls.predict(X)
        
        pred_Y_sign = np.sign(predicted_Y)
        Y_sign = np.sign(Y)
        
        accuracy = (Y_sign == pred_Y_sign).sum()*1. /Y.shape[0]
        