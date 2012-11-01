
import pylab, pandas
import statsmodels.api
import numpy as np
from pprint import pprint
from ..data.crossvalidation import CVObject
from ..utilities.csv import CsvTools




class Regression(CVObject):
    
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(Regression, self).__init__(variable_dict=variable_dict, data_obj=data_obj)
        self.set_folds(folds)
        
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        
            
        
    
    



class LogisticRegression(Regression):
    
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(LogisticRegression, self).__init__(variable_dict=variable_dict,
                                                 data_obj=data_obj, folds=folds)
        
        if self.Y is not None:
            self.replace_Y_negative_ones()
            
        
        
    def logistic_train(self, X, Y, intercept=True, verbose=True):
        print X
        print type(X), type(X[1]), np.dtype(X.dtype), np.dtype(X[1].dtype)
        
        n, p = X.shape
    
        if intercept:
            X = np.hstack((np.ones((n,1)), X))
            
        if verbose:
            print 'Y shape: ', Y.shape
            print 'X shape: ', X.shape
            
        model = statsmodels.api.Logit(Y, X)
        if verbose:
            print 'Fitting logistic model...'
            
        results = model.fit()
        return results
    
    
    def logistic_test(self, X, Y, train_results):
        pprint(train_results.params)
        training_betas = train_results.params
        test_eta = np.dot(X, training_betas)
        test_p = np.exp(test_eta) / (1. + np.exp(test_eta))
        test_predict = (test_p > 0.5)
        return (Y == test_predict).sum()*1. / Y.shape[0]



    def crossvalidate(self, indices_dict=None, folds=None, leave_mod_in=False):
        
        indices_dict = indices_dict or getattr(self,'subject_indices', None)
        leave_mod_in = leave_mod_in or getattr(self,'leave_mod_in', False)
        folds = folds or getattr(self,'folds',None)
        
        if indices_dict is None:
            print 'No indices dict for crossvalidation (usually subject_indices).'
            return False
        else:
            if not self.crossvalidation_ready:
                self.prepare_folds(indices_dict=indices_dict, folds=folds, leave_mod_in=leave_mod_in)
        
        trainresults, testresults = self.traintest_crossvalidator(self.logistic_train,
                                                                  self.logistic_test,
                                                                  self.trainX, self.trainY,
                                                                  self.testX, self.testY)
        
        cv_average = sum(testresults)/len(testresults)
        print 'Crossvalidation accuracy average: '
        pprint(cv_average)
    
    
    
    
    