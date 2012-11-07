
import pylab, pandas
import statsmodels.api
import numpy as np
from pprint import pprint
from ..base.crossvalidation import CVObject
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
            
        self.fixed_effects_set = False
        self.intercept_set = False
        
        
    def add_fixed_effects(self):
        
        fixed_effects = np.zeros((self.X.shape[0], len(self.subject_indices.keys())))
        for i, (subject, trial_inds) in enumerate(self.subject_indices.items()):
            for ind in trial_inds:
                fixed_effects[ind, i] = 1.
                
        print np.sum(fixed_effects, axis=1)
        print np.shape(fixed_effects)
        print np.shape(self.X)
        
        self.X = np.hstack((fixed_effects, self.X))
        
        print np.shape(self.X)
        
        self.fixed_effects_set = True

        
        
    def add_intercept(self):
        
        print np.shape(self.X)
        self.X = np.hstack((np.ones(self.X.shape[0]), self.X))
        self.intercept_set = True
        print np.shape(self.X)
        
        
        
    def logistic_train(self, X, Y, verbose=True):
        #print X
        #print type(X), type(X[1]), np.dtype(X.dtype), np.dtype(X[1].dtype)
        
        n, p = X.shape
            
        if verbose:
            print 'Y shape: ', Y.shape
            print 'X shape: ', X.shape
            
        model = statsmodels.api.Logit(Y, X)
        if verbose:
            print 'Fitting logistic model...'
            
        results = model.fit()
        return results
    
    
    
    def logistic_test(self, X, Y, train_results, predict_with_intercept=True):
        
        training_betas = train_results.params
        
        # please add fixed effects BEFORE intercept, for now! 
        if self.fixed_effects_set:
            training_betas = training_betas[len(self.subject_indices.keys()):]
            X = np.hsplit((X, len(self.subject_indices.keys())))
        
        
        if self.intercept_set:
            if not predict_with_intercept:
                X = np.hsplit((X, 1))
                training_betas = training_betas[1:]
                
        
        test_eta = np.dot(X, training_betas)
        test_p = np.exp(test_eta) / (1. + np.exp(test_eta))
        test_predict = (test_p > 0.5)
        return (Y == test_predict).sum()*1. / Y.shape[0]



    def crossvalidate(self, indices_dict=None, folds=None, leave_mod_in=False,
                      intercept=True, predict_with_intercept=True):
        
        indices_dict = indices_dict or getattr(self,'subject_indices', None)
        leave_mod_in = leave_mod_in or getattr(self,'leave_mod_in', False)
        folds = folds or getattr(self,'folds',None)
        
        
        if indices_dict is None:
            print 'No indices dict for crossvalidation (usually subject_indices).'
            return False
        else:
            if not self.crossvalidation_ready:
                self.prepare_folds(indices_dict=indices_dict, folds=folds, leave_mod_in=leave_mod_in)
        

        train_kwargs = {'intercept':intercept}
        test_kwargs = {'intercept':intercept, 'predict_with_intercept':predict_with_intercept}

        
        self.trainresults, self.testresults = self.traintest_crossvalidator(self.logistic_train,
                                                                  self.logistic_test,
                                                                  self.trainX, self.trainY,
                                                                  self.testX, self.testY,
                                                                  train_kwargs, test_kwargs)
        
        self.cv_average = sum(self.testresults)/len(self.testresults)
        print 'Crossvalidation accuracy average: '
        pprint(self.cv_average)
    
    
    
    
    