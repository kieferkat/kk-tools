
import numpy as np
from sklearn import svm
import random
import itertools
import nibabel as nib
import scipy.stats as stats
from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from normalize import simple_normalize
from threshold import threshold_by_pvalue, threshold_by_rawrange





class ScikitsSVM(CVObject):
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(ScikitsSVM, self).__init__(variable_dict=variable_dict, data_obj=data_obj)
        self.set_folds(folds)
        self.nifti = NiftiTools()
        
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        #if self.Y:
        #    self.replace_Y_negative_ones()
        self.subject_indices = getattr(self.data, 'subject_indices', None)
        
            
    def fit_svc(self, X, Y, cache_size=5000, class_weight='auto'):
        X = simple_normalize(X)
        clf = svm.SVC(cache_size=cache_size, class_weight=class_weight)
        clf.fit(X, Y)
        return clf
    
    
    def fit_linearsvc(self, X, Y, class_weight='auto'):
        print 'fitting linearsvm'
        X = simple_normalize(X)
        clf = svm.LinearSVC(class_weight=class_weight)
        clf.fit(X, Y)
        return clf
    
    
    def test_svm(self, X, Y, clf):

        X = simple_normalize(X)
        correct = []
        print 'Checking accuracy of next test group...'
        
        for trial, outcome in zip(X, Y):
            prediction = clf.predict(trial)
            correct.append((prediction[0] == outcome))
            
        accuracy = float(sum(correct))/float(len(correct))
        print 'Test group accuracy: ', accuracy
        return accuracy
    
    
    def train_svm(self, X, Y):
        
        print 'Training next group...'
        #X = simple_normalize(X)
        clf = self.fit_linearsvc(X, Y)
        return clf
    
    
    def setup_crossvalidation(self, folds=None):
        folds = folds or self.folds
        if self.subject_indices:
            self.prepare_folds(folds=folds, indices_dict=self.subject_indices)
        else:
            print 'no subject indices set, cant setup cv folds'
        
        
    def crossvalidate(self, folds=None):
        self.setup_crossvalidation(folds=folds)
        trainresults, testresults = self.traintest_crossvalidator(self.train_svm, self.test_svm,
                                                                  self.trainX, self.trainY,
                                                                  self.testX, self.testY)
        
        self.fold_accuracies = testresults
        self.average_accuracy = sum(self.fold_accuracies)/len(self.fold_accuracies)
        print 'Average accuracy: ', self.average_accuracy
        
        
        
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
        
        


class SVMRFE(CVObject):
    
    
    def __init__(self, data_obj=None):
        
        super(SVMRFE, self).__init__(data_obj=data_obj)
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        self.subject_indices = getattr(self.data, 'subject_indices', None)
        self.verbose = True
        
    
    def fit_linearsvc(self, Xnormed, Y, class_weight='auto'):
        print 'fitting linearsvm for coefficients'
        clf = svm.LinearSVC(class_weight=class_weight)
        clf.fit(Xnormed, Y)
        return clf.coef_[0]
        
        
    def normalize_xset(self, Xset):
        Xnormed = simple_normalize(Xset)
        return Xnormed
    
    
    def initialize_xmap(self, Xfull):
        xmap = np.ones(Xfull.shape[1])
        self.full_xmap = xmap
        self.current_xmap = xmap
        self.use_inds = np.where(self.current_xmap == 1)
    
    
    def justify_removal_inds(self, remove_inds):
        xmap_active = np.where(self.current_xmap == 1)
        return xmap_active[remove_inds]
    
    
    def excise_from_xmap(self, justified_remove_inds):
        self.current_xmap[justified_remove_inds] = 0.
        self.use_inds = np.where(self.current_xmap == 1.)
        
    
    def subselect(self, X, Y):
        return X[self.use_inds], Y[self.use_inds]
        
    
    def place_coefs(self, coefs):
        coefmap = np.zeros(self.current_xmap.shape)
        coefmap[self.current_xmap.astype(np.bool)] = np.squeeze(np.array(coefs))
        return coefmap
    
    
    def determine_weak_inds(self, coefs):
        inds = range(len(coefs))
        abs_coefs = np.abs(coefs)
        coefs_inds = zip(abs_coefs, inds)
        ranked = sorted(coefs_inds, key=lambda x: x[0])
        inds_to_remove = [ranked[i][1] for i in range(self.remove_per_iteration)]
        return inds_to_remove
    
    
    def initialize_removal(self):
        
        if self.removal_type == 'amount':
            self.remove_per_iteration = self.removal_coef
        elif self.removal_type == 'percent':
            coef_len = len(self.full_xmap)
            self.remove_per_iteration = round(float(coef_len)*self.removal_coef)


    def initialize_stop_condition(self):
        
        if self.stop_type == 'amount':
            self.stop_length = self.stop_coef
        elif self.stop_type == 'percent':
            coef_len = len(self.full_xmap)
            self.stop_length = round(float(coef_len)*self.stop_coef)
    
    
    def initialize(self, X, removal_criterion='amount', removal_coef=25,
                   stop_criterion='percent', stop_coef=0.05):
        
        self.initialize_xmap(X)
        
        self.removal_type = removal_criterion
        self.removal_coef = removal_coef
        self.stop_type = stop_criterion
        self.stop_coef = stop_coef
        
        self.initialize_removal()
        self.initialize_stop_condition()
        
    
    def run(self, X, Y):
        
        Xnorm = self.normalize_xset(X)
        self.Xuse = Xnorm.copy()
        self.Yuse = Y.copy()
        
        while len(self.use_inds) >= self.stop_length:
            
            self.current_coefs = self.fit_linearsvc(self.Xuse, self.Yuse)
            inds_to_remove = self.determine_weak_inds(self.current_coefs)
            justified_inds = self.justify_removal_inds(inds_to_remove)
            self.excise_from_xmap(justified_inds)
            self.Xuse, self.Yuse = self.subselect(Xnorm, Y)
            
            if self.verbose:
                print 'current coef length:', len(self.current_coefs)
                print 'stop length condition:', self.stop_length
                print 'next useable X:', len(self.Xuse)
                
        if self.verbose:
            print 'final run...'
            self.current_coefs = self.fit_linearsvc(self.Xuse, self.Yuse)
                
        if self.verbose:
            print 'Completed SVM RFE'
            print 'current_coefs length:', len(self.current_coefs)
            
                
    
    def output_maps(self, X, Y, time_points, nifti_filepath):
        
        if not nifti_filepath.endswith('.nii'):
            nifti_filepath = nifti_filepath+'.nii'
        
        self.reshaped_coefs = self.place_coefs(self.current_coefs)
        
        if self.verbose:
            print 'reshaping the coefs to original brain shape...'
            
        unmasked = self.data.unmask_Xcoefs(self.reshaped_coefs, time_points, verbose=verbose)
        
        if self.verbose:
            print 'saving nifti to filename:', nifti_filepath
            
        self.data.save_unmasked_coefs(unmasked, nifti_filepath)
        
        
        
        
        
        
        
        
        
        
    
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
    
    
    
    
                
            
