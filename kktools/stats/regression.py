
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
        self.fixed_effects_set = False
        self.intercept_set = False
        
        
    def add_intercept(self, verbose=True):
        
        if verbose:
            print 'shape of X:', np.shape(self.X)
        
        intercept = np.ones((self.X.shape[0],1))
        
        if verbose:
            print 'shape of intercept:', np.shape(intercept)
        
        self.X = np.hstack((intercept, self.X))
        
        if verbose:
            print 'new X shape with intercept:', np.shape(self.X)
            
        self.intercept_set = True
        
        
    
    def add_fixed_effects(self, subject_indices=None, verbose=True):
        
        subject_indices = subject_indices or getattr(self, 'subject_indices', None)
        if subject_indices is None:
            print 'cannot add fixed effects without subject indices set...'
            return False
        
        # fixed effects shares row size with X, columns are subjects
        fixed_effects = np.zeros((self.X.shape[0], len(subject_indices.keys())))
        
        # add 1s for each subject in their row:
        for i, (subject, trial_inds) in enumerate(subject_indices.items()):
            for ind in trial_inds:
                fixed_effects[ind, i] = 1.
                
        if verbose:
            print 'sum of fixed effects column:'
            print np.sum(fixed_effects, axis=0)
            print 'shape of fixed effects column:', np.shape(fixed_effects)
        
        self.X = np.hstack((fixed_effects, self.X))
        
        if verbose:
            print 'new X shape with fixed effects appended:', np.shape(self.X)
        
        self.fixed_effects_set = True
        
        
        
    def remove_zero_columns(self, X, subject_indices=None, verbose=True):
        
        if verbose:
            print 'removing all-0 columns (typically for fixed effects during crossval)...'
        
        sums = np.sum(X, axis=0)
        del_cols = [i for i,x in enumerate(sums) if x==0.]
        X = np.delete(X, del_cols, 1)
        
        if verbose:
            print 'new x shape:'
            print np.shape(X)
            
        return X, del_cols
    
    
    
            

class LogisticRegression(Regression):
    
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(LogisticRegression, self).__init__(variable_dict=variable_dict,
                                                 data_obj=data_obj, folds=folds)
        
        if self.Y is not None:
            self.replace_Y_negative_ones()
        
        
    def logistic_train(self, X, Y, verbose=True):
        
        n, p = X.shape
            
        if verbose:
            print 'Y shape: ', Y.shape
            print 'X shape: ', X.shape
            print 'Y sum: ', np.sum(Y)
            
        if self.fixed_effects_set:
            X, del_cols = self.remove_zero_columns(X)
            self.prior_beta_split = len(self.subject_indices.keys())-len(del_cols)
            
            
        model = statsmodels.api.Logit(Y, X)
        if verbose:
            print 'Fitting logistic model...'
            
        results = model.fit()
        
        return results
    
    
    
    def logistic_test(self, X, Y, train_results, predict_with_intercept=True,
                      predict_with_fixed_effects=True, use_prior_beta_split=True):
        
        training_betas = train_results.params
        print training_betas
        
        # please add fixed effects BEFORE intercept, for now! 
        if self.fixed_effects_set:
            if not predict_with_fixed_effects:
                if use_prior_beta_split:
                    print np.shape(X), self.prior_beta_split, len(training_betas)
                    X = np.hsplit(X, [len(self.subject_indices.keys())])[1]
                    training_betas = training_betas[self.prior_beta_split:]
                    print np.shape(X), len(training_betas)
                else:
                    X = np.hsplit(X, [len(self.subject_indices.keys())])[1]
                    training_betas = training_betas[len(self.subject_indices.keys()):]
        
        
        if self.intercept_set:
            if not predict_with_intercept:
                X = np.hsplit(X, 1)[1]
                training_betas = training_betas[1:]
                
        
        test_eta = np.dot(X, training_betas)
        test_p = np.exp(test_eta) / (1. + np.exp(test_eta))
        test_predict = (test_p > 0.5)
        return (Y == test_predict).sum()*1. / Y.shape[0]



    def crossvalidate(self, indices_dict=None, folds=None, leave_mod_in=False,
                      predict_with_intercept=True):
        
        indices_dict = indices_dict or getattr(self,'subject_indices', None)
        leave_mod_in = leave_mod_in or getattr(self,'leave_mod_in', False)
        folds = folds or getattr(self,'folds',None)
        
        
        if indices_dict is None:
            print 'No indices dict for crossvalidation (usually subject_indices).'
            return False
        else:
            if not self.crossvalidation_ready:
                self.prepare_folds(indices_dict=indices_dict, folds=folds, leave_mod_in=leave_mod_in)
        

        train_kwargs = {}
        test_kwargs = {'predict_with_intercept':predict_with_intercept,
                       'predict_with_fixed_effects':False,
                       'use_prior_beta_split':True}

        
        self.trainresults, self.testresults = self.traintest_crossvalidator(self.logistic_train,
                                                                            self.logistic_test,
                                                                            self.trainX, self.trainY,
                                                                            self.testX, self.testY,
                                                                            train_kwargs, test_kwargs)
        
        self.cv_average = sum(self.testresults)/len(self.testresults)
        print 'Crossvalidation accuracy average: '
        pprint(self.cv_average)
    
    
    
    
    