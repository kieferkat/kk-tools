
import numpy as np
from sklearn import svm
import random
import itertools
import nibabel as nib
import scipy.stats as stats
from ..data.crossvalidation import Crossvalidation
from ..data.nifti import NiftiTools
from normalize import simple_normalize
from threshold import threshold_by_pvalue, threshold_by_rawrange




class ScikitsSVM(object):
    
    def __init__(self, data, folds=None):
        super(ScikitsSVM, self).__init__()
        self.data= data
        self.nifti = NiftiTools()
        
        
    def setup_crossvalidation(self, folds=None):
        self.cv = Crossvalidation(indices_dict=self.data.subject_trial_indices,
                                  folds=folds)
        self.cv.create_crossvalidation_folds()
        self.folds = self.cv.folds
        
            
    def fit_svc(self, X, Y, cache_size=5000, class_weight='auto'):
        clf = svm.SVC(cache_size=cache_size, class_weight=class_weight)
        clf.fit(X, Y)
        return clf
    
    
    def fit_linearsvc(self, X, Y, class_weight='auto'):
        print 'fitting linearsvm'
        clf = svm.LinearSVC(class_weight=class_weight)
        clf.fit(X, Y)
        return clf
    
    
    def test_accuracy(self, clf, testX, testY):
        correct = []
        print 'Checking accuracy of next test group...'
        
        for trial, real_outcome in zip(testX, testY):
            prediction = clf.predict(trial)
            correct.append((prediction[0] == real_outcome))
            
        accuracy = float(sum(correct))/float(len(correct))
        print 'Test group accuracy: ', accuracy
        
        return accuracy
    
    
    def assign_groups(self, train_indices, test_indices):
        trainX, trainY = [], []
        testX, testY = [],[]
        
        negative_replace = lambda value: 1. if (value==1.) else 0.
        
        for trial in train_indices:
            trainX.append(self.data.X[trial])
            trainY.append(negative_replace(self.data.Y[trial]))
            
        for trial in test_indices:
            testX.append(self.data.X[trial])
            testY.append(negative_replace(self.data.Y[trial]))
            
        return trainX, trainY, testX, testY
    
    
    def crossvalidate_linearsvm(self, folds=None, accuracy_weight=None):
        self.setup_crossvalidation(folds=folds)
        
        self.fold_accuracies = []
        
        for i in range(self.folds):
            train_indices = self.cv.train_dict[i]
            test_indices = self.cv.test_dict[i]
            
            trainX, trainY, testX, testY = self.assign_groups(train_indices, test_indices)
            
            print 'Normalizing training data...'
            trainX = simple_normalize(trainX)
            print 'Normalizing testing data...'
            testX = simple_normalize(testX)
            
            curr_clf = self.fit_linearsvc(trainX, trainY)
            curr_acc = self.test_accuracy(curr_clf, testX, testY)
            
            if not accuracy_weight:
                self.fold_accuracies.append(curr_acc)
            elif accuracy_weight == 'group_trials':
                for trial in testY:
                    self.fold_accuracies.append(curr_acc)
                
                current_weighted_accuracy = sum(self.fold_accuracies)/len(self.fold_accuracies)
                print 'Current weighted accuracy: ', current_weighted_accuracy
                    
            
            
        self.average_accuracy = sum(self.fold_accuracies)/len(self.fold_accuracies)
        print 'Average accuracy: ', self.average_accuracy
        
        
        
    def output_maps(self, nifti_filepath, threshold=0.01, two_tail=True,
                    threshold_type='pvalue'):
        print 'Normalizing X matrix...'
        Xnorm = simple_normalize(self.data.X)
        print 'Classifying with linear svm...'
        clf = self.fit_linearsvc(Xnorm, self.data.Y)
        print 'Thresholding and dumping coefficients to file...'
        self.coefs = clf.coef_[0]
        
        if threshold_type == 'pvalue':
            thresholded_coefs = threshold_by_pvalue(self.coefs, threshold, two_tail=two_tail)
        elif threshold_type == 'raw_percentage':
            thresholded_coefs = threshold_by_rawrange(self.coefs, threshold, two_tail=two_tail)
        
        self.nifti.output_nifti_thrumask(thresholded_coefs, self.data.trial_mask,
                                         self.data.mask_shape, len(self.data.selected_trs),
                                         self.data.raw_affine, nifti_filepath)
        
        self.nifti.convert_to_afni(nifti_filepath, nifti_filepath[:-4])
        self.nifti.adwarp_to_template_talairach(nifti_filepath[:-4]+'+orig', None,
                                                self.data.talairach_template_path,
                                                self.data.dxyz, overwrite=True)
    
    
                
            