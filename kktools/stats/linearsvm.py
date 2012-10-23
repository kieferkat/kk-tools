
import numpy as np
from sklearn import svm
import random
import itertools
import nibabel as nib
import scipy.stats as stats
from ..data.crossvalidation import CVObject
from ..data.nifti import NiftiTools
from normalize import simple_normalize
from threshold import threshold_by_pvalue, threshold_by_rawrange





class ScikitsSVM(CVobject):
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(CVObject, self).__init__(variable_dict=variable_dict, data_obj=data_obj)
        self.set_folds(folds)
        self.nifti = NiftiTools()
        
        if getattr(self.data, 'X', None):
            self.X = self.data.X
        else:
            self.X = None
        if getattr(self.data, 'Y', None):
            self.Y = self.data.Y
            self.replace_negative_Y()
        else:
            self.Y = None
        if getattr(self.data, 'subject_trial_indices', None):
            self.indices_dict = self.data.subject_trial_indices
        else:
            self.indices_dict = None
        
        
    def replace_negative_Y(self):
        self.Y = self.replace_Y_vals(self.Y, -1., 0.)
        
            
    def fit_svc(self, X, Y, cache_size=5000, class_weight='auto'):
        clf = svm.SVC(cache_size=cache_size, class_weight=class_weight)
        clf.fit(X, Y)
        return clf
    
    
    def fit_linearsvc(self, X, Y, class_weight='auto'):
        print 'fitting linearsvm'
        clf = svm.LinearSVC(class_weight=class_weight)
        clf.fit(X, Y)
        return clf
    
    
    def test_accuracy(self, X, Y, clf):
        if clf is not None:
            X = simple_normalize(X)
            correct = []
            print 'Checking accuracy of next test group...'
            
            for trial, outcome in zip(X, Y):
                prediction = clf.predict(trial)
                correct.append((prediction[0] == outcome))
                
            accuracy = float(sum(correct))/float(len(correct))
            print 'Test group accuracy: ', accuracy
            return accuracy
        else:
            print 'ERROR: must keyword specify the classifier'
    
    
    def train(self, X, Y):
        
        print 'Training next group...'
        X = simple_normalize(X)
        clf = self.fit_linearsvc(X, Y)
        return clf
    
    
    def setup_crossvalidation(self, folds=None):
        folds = folds or self.folds
        if self.indices_dict:
            self.prepare_folds(folds=folds, indices_dict=self.indices_dict)
        self.cv_group_XY(self.X, self.Y)
        
        
    def crossvalidate(self, folds=None):
        self.setup_crossvalidation(folds=folds)
        trainresults, testresults = self.traintest_crossvalidator(self.train, self.test_accuracy,
                                                                  self.cv_train_X, self.cv_train_Y,
                                                                  self.cv_test_X, self.cv_test_Y)
        
        self.fold_accuracies = testresults
        self.average_accuracy = sum(self.fold_accuracies)/len(self.fold_accuracies)
        print 'Average accuracy: ', self.average_accuracy
        
        
        
    def output_maps(self, nifti_filepath, threshold=0.01, two_tail=True,
                    threshold_type='pvalue'):
        print 'Normalizing X matrix...'
        Xnorm = simple_normalize(self.X)
        print 'Classifying with linear svm...'
        clf = self.fit_linearsvc(Xnorm, self.Y)
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
    
    
                
            