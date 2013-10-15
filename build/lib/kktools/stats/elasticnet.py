
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
import random
import itertools
import nibabel as nib
import scipy.stats as stats
from pprint import pprint, pformat
from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from normalize import simple_normalize
from threshold import threshold_by_pvalue, threshold_by_rawrange





class ElasticNetClassifier(CVObject):
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(ElasticNetClassifier, self).__init__(variable_dict=variable_dict, data_obj=data_obj)
        self.set_folds(folds)
        self.nifti = NiftiTools()
        self.alpha = 1.0
        self.l1ratio = 0.5
        
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        #if self.Y:
        #    self.replace_Y_negative_ones()
        self.subject_indices = getattr(self.data, 'subject_indices', None)
        
    
    def determine_alpha_ratio(self, X, Y):
        print 'determining alpha and l1_ratio with crossvalidation...'
        Xnorm = simple_normalize(X)
        clf = ElasticNetCV(l1_ratio=[.01,.1], fit_intercept=False, verbose=True)
        #clf = ElasticNetCV(l1_ratio=[.1,.5,.9], fit_intercept=False, verbose=True)
        clf.fit(Xnorm, Y)
        alpha = clf.alpha_ 
        l1ratio = clf.l1_ratio_
        self.alpha = alpha
        self.l1ratio = l1ratio
        return alpha, l1ratio


    def fit_elasticnet(self, X, Y):
        print 'fitting Elastic Net...'
        print 'alpha:', self.alpha, 'l1ratio:', self.l1ratio
        Xnorm = simple_normalize(X)
        clf = ElasticNet(alpha=self.alpha, l1_ratio=self.l1ratio, fit_intercept=False)
        clf.fit(Xnorm, Y)
        return clf
    
    
    def test_enet(self, X, Y, clf):

        X = simple_normalize(X)
        correct = []
        print 'Checking accuracy of next test group...'
        
        for trial, outcome in zip(X, Y):
            prediction = clf.predict(trial)
            print prediction, outcome
            #print 'prediction, real:', prediction, outcome, np.sum(clf.coef_), np.sum(trial)
            correct.append((prediction == outcome))
            
        accuracy = float(sum(correct))/float(len(correct))
        print 'Test group accuracy: ', accuracy
        return accuracy
    
    
    def train_enet(self, X, Y):
        print 'Training next group...'
        clf = self.fit_elasticnet(X, Y)
        return clf
    
    
    def setup_crossvalidation(self, folds=None):
        folds = folds or self.folds
        if self.subject_indices:
            self.prepare_folds(folds=folds, indices_dict=self.subject_indices)
        else:
            print 'no subject indices set, cant setup cv folds'
        
        
    def crossvalidate(self, folds=None, logfile=None, ttest_mean=0.5):

        self.setup_crossvalidation(folds=folds)
        trainresults, testresults = self.traintest_crossvalidator(self.train_enet, self.test_enet,
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
        
        
        
    def output_maps(self, X, Y, time_points, nifti_filepath, threshold=0.01,
                    two_tail=True, verbose=True):
        
        if not nifti_filepath.endswith('.nii'):
            nifti_filepath = nifti_filepath+'.nii'
            
        if verbose:
            print 'fitting to output...'
            
        clf = self.fit_linearsvc(X, Y)
        self.coefs = clf.coef_[0]
        
        thresholded_coefs = threshold_by_pvalue(self.coefs, threshold, two_tail=two_tail)
        
        if verbose:
            print 'reshaping the coefs to original brain shape...'
            
        unmasked = self.data.unmask_Xcoefs(thresholded_coefs, time_points, verbose=verbose)
        
        if verbose:
            print 'saving nifti to filename:', nifti_filepath
            
        self.data.save_unmasked_coefs(unmasked, nifti_filepath)

        
        
        
        
        
        
        
        
        
    
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
    
    
    
    
                
            
