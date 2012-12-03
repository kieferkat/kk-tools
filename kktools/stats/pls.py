
import pylab
import numpy as np
from pprint import pprint
from ..base.crossvalidation import CVObject
from ..utilities.csv import CsvTools
from regression import Regression
from sklearn.pls import PLSCanonical, PLSRegression, CCA



class PLS(Regression):
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(PLS, self).__init__(data_obj=data_obj,
                                  variable_dict=variable_dict,
                                  folds=folds)
        
        
    def output_maps(self, X, Y, nifti_filepath, verbose=True):
        
        if not nifti_filepath.endswith('.nii'):
            nifti_filepath = nifti_filepath+'.nii'
            
        if verbose:
            print 'fitting prior to output...'
            
        pls = self.pls_train(X, Y, verbose=verbose)
        
        if verbose:
            print 'reshaping the coefs to the original brain shape...'
            
        unmasked = self.data.unmask_Xcoefs(pls.coefs, verbose=verbose)
        
        if verbose:
            print 'saving nifti to filename: ', nifti_filepath
            
        self.data.save_unmasked_coefs(unmasked, nifti_filepath)
        
    
    def pls_train(self, X, Y, verbose=True):
        
        pls = PLSRegression()
        
        if verbose:
            print 'fitting canonical pls...'
            
        pls.fit(X, Y)
        
        return pls
    
    
    def pls_test(self, X, Y, pls):
        
        predicted_Y = pls.predict(X)
        
        pred_Y_sign = np.sign(predicted_Y)
        Y_sign = np.sign(Y)
        
        accuracy = (Y_sign == pred_Y_sign).sum()*1. /Y.shape[0]
        
        
    def crossvalidate(self, indices_dict, folds=None, leave_mod_in=False,
                      verbose=True):
        
        
        if not self.crossvalidation_ready:
            if verbose:
                print 'preparing the crossvalidation folds...'
            self.prepare_folds(indices_dict=indices_dict, folds=folds,
                               leave_mod_in=leave_mod_in)
            
            
        train_kwargs = {}
        test_kwargs = {}
        
        self.trainresults, self.testresults = self.traintest_crossvalidator(self.pls_train,
                                                                            self.pls_test,
                                                                            self.trainX, self.trainY,
                                                                            self.testX, self.testY,
                                                                            train_kwargs, test_kwargs)
        
        self.cv_average = sum(self.testresults)/len(self.testresults)
        
        if verbose:
            print 'crossvalidation accuracy average'
            pprint(self.cvaverage)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        