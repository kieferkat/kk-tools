
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
import random
import itertools
import nibabel as nib
import scipy.stats as stats
from pprint import pprint, pformat
from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from normalize import simple_normalize
from threshold import threshold_by_pvalue, threshold_by_rawrange





class LassoClassifier(CVObject):
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(LassoClassifier, self).__init__(variable_dict=variable_dict, data_obj=data_obj)
        self.set_folds(folds)
        self.nifti = NiftiTools()
        self.alpha = 1.0        
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        #if self.Y:
        #    self.replace_Y_negative_ones()
        self.subject_indices = getattr(self.data, 'subject_indices', None)
        
    
    def determine_alpha(self, X, Y):
        print 'determining alpha with crossvalidation...'
        Xnorm = simple_normalize(X)
        clf = LassoCV(fit_intercept=False, verbose=True)
        clf.fit(Xnorm, Y)
        alpha = clf.alpha_ 
        self.alpha = alpha
        return alpha


    def fit_lasso(self, X, Y):
        print 'fitting Lasso...'
        print 'alpha:', self.alpha
        Xnorm = simple_normalize(X)
        clf = Lasso(alpha=self.alpha, fit_intercept=False)
        clf.fit(Xnorm, Y)
        return clf
    
    
    def test_lasso(self, X, Y, clf):

        X = simple_normalize(X)
        correct = []
        print 'Checking accuracy of next test group...'
        
        for trial, outcome in zip(X, Y):
            prediction = clf.predict(trial)
            if prediction <= 0:
                pred = -1
            elif prediction > 0:
                pred = 1
            #print prediction, outcome
            #print 'prediction, real:', prediction, outcome, np.sum(clf.coef_), np.sum(trial)
            correct.append((pred == outcome))
            
        accuracy = float(sum(correct))/float(len(correct))
        print 'Test group accuracy: ', accuracy
        return accuracy
    
    
    def train_lasso(self, X, Y):
        print 'Training next group...'
        clf = self.fit_lasso(X, Y)
        return clf
    
    
    def setup_crossvalidation(self, folds=None):
        folds = folds or self.folds
        if self.subject_indices:
            self.prepare_folds(folds=folds, indices_dict=self.subject_indices)
        else:
            print 'no subject indices set, cant setup cv folds'
        
        
    def crossvalidate(self, folds=None, logfile=None, ttest_mean=0.5):

        self.setup_crossvalidation(folds=folds)
        trainresults, testresults = self.traintest_crossvalidator(self.train_lasso, self.test_lasso,
                                                                  self.trainX, self.trainY,
                                                                  self.testX, self.testY)
        
        self.fold_accuracies = testresults
        self.average_accuracy = sum(self.fold_accuracies)/len(self.fold_accuracies)
        self.median_accuracy = np.median(self.fold_accuracies)
        self.accuracy_variance = np.var(self.fold_accuracies)
        self.accuracy_sem = stats.sem(self.fold_accuracies)
        self.tstat, self.pval = stats.ttest_1samp(self.fold_accuracies,ttest_mean)
        print 'Average accuracy: ', self.average_accuracy

        if logfile is not None:
            fid = open(logfile, 'w')
            fid.write('FOLD ACCURACIES:\n')
            fid.write(pformat(self.fold_accuracies))
            fid.write('\nAVERAGE ACCCURACY:\n')
            fid.write(pformat(self.average_accuracy))
            fid.write('\nMEDIAN ACCCURACY:\n')
            fid.write(pformat(self.median_accuracy))
            fid.write('\nACCCURACY VARIANCE:\n')
            fid.write(pformat(self.accuracy_variance))
            fid.write('\nACCCURACY ERROR:\n')
            fid.write(pformat(self.accuracy_sem))
            fid.write('\nACCURACY TSTAT:\n')
            fid.write(pformat(self.tstat))
            fid.write('\nACCURACY P-VAL:\n')
            fid.write(pformat(self.pval))
            fid.close()

        return self.average_accuracy
        
        
        
    def output_maps(self, X, Y, time_points, nifti_filepath, verbose=True):
        
        if not nifti_filepath.endswith('.nii'):
            nifti_filepath = nifti_filepath+'.nii'
            
        if verbose:
            print 'fitting to output...'
            
        clf = self.fit_lasso(X, Y)
        self.coefs = clf.coef_[0]
        
        #thresholded_coefs = threshold_by_pvalue(self.coefs, threshold, two_tail=two_tail)
        
        if verbose:
            print 'reshaping the coefs to original brain shape...'
            
        unmasked = self.data.unmask_Xcoefs(self.coefs, time_points, verbose=verbose)
        
        if verbose:
            print 'saving nifti to filename:', nifti_filepath
            
        self.data.save_unmasked_coefs(unmasked, nifti_filepath)

        
        
        
        
        
        
        
        
        
    
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
    
    
    
    
                
            
