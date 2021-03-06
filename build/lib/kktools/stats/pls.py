
import pylab
import numpy as np
from pprint import pprint
from ..base.crossvalidation import CVObject
from ..utilities.csv import CsvTools
from regression import Regression
from sklearn.pls import PLSCanonical, PLSRegression, CCA
from threshold import threshold_by_pvalue
from normalize import simple_normalize



class PLS(Regression):
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(PLS, self).__init__(data_obj=data_obj,
                                  variable_dict=variable_dict,
                                  folds=folds)
        
        
    def output_maps(self, X, Y, time_points, nifti_filepath,
                    threshold=0.01, two_tail=True, verbose=True):
        
        if not nifti_filepath.endswith('.nii'):
            nifti_filepath = nifti_filepath+'.nii'
            
        if verbose:
            print 'fitting prior to output...'
            
        pls = self.pls_train(X, Y, verbose=verbose)
        
        coefs = pls.coefs
        
        thresh_coefs = threshold_by_pvalue(coefs, threshold, two_tail=two_tail)
        
        if verbose:
            print 'reshaping the coefs to the original brain shape...'
            
        unmasked = self.data.unmask_Xcoefs(thresh_coefs, time_points, verbose=verbose)
        
        if verbose:
            print 'saving nifti to filename: ', nifti_filepath
            
        self.data.save_unmasked_coefs(unmasked, nifti_filepath)
        
    
    def pls_train(self, X, Y, verbose=True):
        
        Xn = simple_normalize(X)
        
        pls = PLSRegression()
        
        if verbose:
            print 'fitting canonical pls...'
            
        pls.fit(Xn, Y)
        
        return pls
    
    
    def pls_test(self, X, Y, pls):
        
        Xn = simple_normalize(X)
        
        predicted_Y = [x[0] for x in pls.predict(Xn)]
        
        pred_Y_sign = np.sign(predicted_Y)
        Y_sign = np.sign(Y)
        
        accuracy = (Y_sign == pred_Y_sign).sum()*1. /Y.shape[0]
        
        print 'accuracy in fold: ', accuracy
        return accuracy
        
        
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
        
        print self.testresults
        self.cv_average = sum(self.testresults)/len(self.testresults)
        
        if verbose:
            print 'crossvalidation accuracy average'
            pprint(self.cv_average)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        