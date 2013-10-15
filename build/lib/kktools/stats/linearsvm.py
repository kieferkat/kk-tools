
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pls import PLSCanonical, PLSRegression, CCA
import random
import itertools
import nibabel as nib
import scipy.stats as stats
from pprint import pprint, pformat
from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from normalize import simple_normalize
from threshold import threshold_by_pvalue, threshold_by_rawrange
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostClassifier, ExtraTreesClassifier





class ScikitsSVM(CVObject):
    
    def __init__(self, data_obj=None, variable_dict=None, folds=None, regC=1.0):
        super(ScikitsSVM, self).__init__(variable_dict=variable_dict, data_obj=data_obj)
        self.set_folds(folds)
        self.regC = regC
        self.nifti = NiftiTools()
        
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        #if self.Y:
        #    self.replace_Y_negative_ones()
        self.subject_indices = getattr(self.data, 'subject_indices', None)
        
            
    def fit_svc(self, X, Y, cache_size=5000, class_weight=None):
        X = simple_normalize(X)
        clf = svm.SVC(cache_size=cache_size, class_weight=class_weight)
        clf.fit(X, Y)
        return clf
    
    
    def fit_linearsvc(self, X, Y, class_weight=None, C=None):
        if C is None:
            C = self.regC
        print 'fitting linearsvm'
        X = simple_normalize(X)
        #print Y
        clf = svm.LinearSVC(class_weight=class_weight, C=C)
        clf.fit(X, Y)
        return clf
    
    
    def test_svm(self, X, Y, clf):

        X = simple_normalize(X)
        correct = []
        print 'Checking accuracy of next test group...'
        
        for trial, outcome in zip(X, Y):
            prediction = clf.predict(trial)
            #print 'prediction, real:', prediction, outcome, np.sum(clf.coef_), np.sum(trial)
            correct.append((prediction[0] == outcome))
            
        accuracy = float(sum(correct))/float(len(correct))
        print 'Test group accuracy: ', accuracy
        return accuracy
    
    
    def train_svm(self, X, Y):
        
        print 'Training next group...'
        clf = self.fit_linearsvc(X, Y)
        return clf
    
    
    def setup_crossvalidation(self, folds=None):
        folds = folds or self.folds
        if self.subject_indices:
            self.prepare_folds(folds=folds, indices_dict=self.subject_indices)
        else:
            print 'no subject indices set, cant setup cv folds'
        
        
    def crossvalidate(self, folds=None, logfile=None, ttest_mean=0.5):
        self.setup_crossvalidation(folds=folds)
        trainresults, testresults = self.traintest_crossvalidator(self.train_svm, self.test_svm,
                                                                  self.trainX, self.trainY,
                                                                  self.testX, self.testY)
        
        self.fold_accuracies = testresults
        self.average_accuracy = sum(self.fold_accuracies)/len(self.fold_accuracies)
        self.median_accuracy = np.median(self.fold_accuracies)
        self.accuracy_variance = np.var(self.fold_accuracies)
        self.accuracy_std = np.var(self.fold_accuracies)
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
            fid.write('\nACCCURACY STDEV:\n')
            fid.write(pformat(self.accuracy_std))
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
        
        


class SVMRFE(CVObject):
    
    
    def __init__(self, data_obj=None):
        
        super(SVMRFE, self).__init__(data_obj=data_obj)
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        self.subject_indices = getattr(self.data, 'subject_indices', None)
        self.verbose = True
        
    
    def fit_linearsvc(self, Xnormed, Y, class_weight='auto',penalty='l2'):
        print 'fitting linearsvm for coefficients'
        print 'regularization: ', self.C
        dual = True
        if penalty == 'l1':
            dual = False
        clf = svm.LinearSVC(class_weight=class_weight, C=self.C, penalty=penalty, dual=dual)
        clf.fit(Xnormed, Y)
        return clf.coef_[0]


    def fit_SGD(self, Xnormed, Y, alpha=0.001, l1_ratio=0.5, loss='modified_huber', penalty='elasticnet'):
        print 'fitting SGD Classifier...'
        clf = SGDClassifier(loss=loss, alpha=alpha, l1_ratio=l1_ratio, penalty=penalty, fit_intercept=False,
            verbose=0, class_weight='auto', n_iter=50)
        clf.fit(Xnormed, Y)
        return clf.coef_[0]


    def fit_pls(self, Xnormed, Y):
        print 'fitting pls...'
        clf = PLSRegression(n_components=self.pls_components, scale=False)
        clf.fit(Xnormed, Y)
        return clf.coefs


    def fit_logreg(self, Xnormed, Y, class_weight='auto', penalty='l2', fit_intercept=False):
        print 'fitting logistic regression...'
        print 'regularization: ', self.C
        clf = LogisticRegression(penalty=penalty, class_weight=class_weight, C=self.C,
            fit_intercept=fit_intercept)
        clf.fit(Xnormed, Y)
        return clf.coef_[0]



    def fit_linearsvc_bootstrap(self, Xnormed, Y, class_weight='auto', use_median=False):
        print '\nbootstrapping linearsvc...'
        
        clf = svm.LinearSVC(class_weight=class_weight)
        coefs = []

        Xpos = Xnormed[(Y==1),:]
        Xneg = Xnormed[(Y==-1),:]
        Ypos = Y[(Y==1)]
        Yneg = Y[(Y==-1)]

        pos_xlen = Xpos.shape[0]
        neg_xlen = Xneg.shape[0]
        max_xlen = np.maximum(pos_xlen, neg_xlen)
        print pos_xlen, neg_xlen, max_xlen
        #print len(Ypos), len(Yneg)

        for bind in range(self.bootstrap_folds):
            print 'bootstrap fold', bind
            
            pos_randinds = np.random.randint(pos_xlen, size=max_xlen*self.bootstrap_fraction)
            neg_randinds = np.random.randint(neg_xlen, size=max_xlen*self.bootstrap_fraction)

            X_, Y_ = np.zeros((max_xlen*2*self.bootstrap_fraction, Xpos.shape[1])), np.zeros((max_xlen*2*self.bootstrap_fraction))

            for i, pri in enumerate(pos_randinds):
                X_[i,:] = Xpos[pri,:]
                Y_[i] = Ypos[pri]

            for i, nri in enumerate(neg_randinds):
                X_[i+len(pos_randinds),:] = Xneg[nri,:]
                Y_[i+len(pos_randinds)] = Yneg[nri]

            #print len(Y_[(Y_==1)]), len(Y_[(Y_==-1)])

            print X_.shape, Y_.shape, 'fitting...'
            clf.fit(X_, Y_)
            coefs.append(clf.coef_[0])

        coefs = np.array(coefs)
        if use_median:
            coefavg = np.median(coefs, axis=0)
        else:
            coefavg = np.mean(coefs, axis=0)
        return coefavg


    def fit_gradientboosting(self, X, Y):

        GBR = GradientBoostingRegressor(loss='huber',learning_rate=0.1, n_estimators=500,
            max_depth=3, subsample=0.5, verbose=1, max_features='sqrt')
        GBR.fit(X, Y)
        return GBR.feature_importances_

    def fit_adaboost(self, X, Y):
        ADA = AdaBoostClassifier(n_estimators=100)
        ADA.fit(X, Y)
        return ADA.feature_importances_


    def fit_extratrees(self, X, Y):
        print 'fitting extratress classifier...'
        ET = ExtraTreesClassifier(verbose=True, n_estimators=100)
        ET.fit(X, Y)
        return ET.feature_importances_
        
        
    def normalize_xset(self, Xset):
        if self.verbose:
            print 'normalizing X...'
        Xnormed = simple_normalize(Xset)
        return Xnormed
    
    
    def initialize_xmap(self, Xfull):
        if self.verbose:
            print 'initializing xmap, use_inds...'
        xmap = np.ones(Xfull.shape[1])
        self.full_xmap = xmap
        self.current_xmap = xmap
        self.use_inds = np.where(self.current_xmap == 1)[0]
        
        if self.verbose:
            print 'xmap shape:', self.full_xmap.shape
            print 'current xmap shape', self.current_xmap.shape
            print 'use inds:', self.use_inds
    
    
    def justify_removal_inds(self, remove_inds):
        if self.verbose:
            print 'justifying removal inds with xmap...'
        xmap_active = np.where(self.current_xmap == 1)[0]
        return xmap_active[remove_inds]
    
    
    def excise_from_xmap(self, justified_remove_inds):
        if self.verbose:
            print 'setting xmap remove inds to zero...'
        self.current_xmap[justified_remove_inds] = 0.
        self.use_inds = np.where(self.current_xmap == 1.)[0]
        
        if self.verbose:
            print 'new use inds length:', len(self.use_inds)
        
    
    def subselect(self, X):
        if self.verbose:
            print 'subselecting new X...'
        return X[:,self.use_inds]
        
    
    def place_coefs(self, coefs):
        if self.verbose:
            print 'placing coefs with xmap...'
        coefmap = np.zeros(self.current_xmap.shape)
        coefmap[self.current_xmap.astype(np.bool)] = np.squeeze(np.array(coefs))
        return coefmap
    
    
    def determine_weak_inds(self, coefficients):
        if self.verbose:
            print 'Determining weak inds to cut...'

        if self.mean_deviance_weights:
            print 'using mean_deviance weights calculation...'
            coefs = coefficients*self.X_mean_deviance_use
        else:
            coefs = coefficients

        inds = range(len(coefs))
        abs_coefs = np.abs(coefs)
        coefs_inds = zip(abs_coefs, inds)
        ranked = sorted(coefs_inds, key=lambda x: x[0])

        if self.only_remove_zeros:
            self.remove_per_iteration = np.sum([1 for x in ranked if x[0] == 0.])
            print 'ZEROS:', self.remove_per_iteration
        elif self.removal_type == 'percent':
            coef_len = self.Xuse.shape[1]
            self.remove_per_iteration = int(round(float(coef_len)*self.removal_coef))
            #print 'readjusting removal amount:', self.remove_per_iteration
        
        if self.Xuse.shape[1]-self.remove_per_iteration > self.stop_length:
            to_remove = self.remove_per_iteration
        else:
            to_remove = self.Xuse.shape[1]-self.stop_length
        
        inds_to_remove = [ranked[i][1] for i in range(int(to_remove))]
        
        if self.verbose:
            print 'removal inds length:', len(inds_to_remove)
        
        return inds_to_remove
    
    
    def initialize_removal(self):
        
        if self.removal_type == 'amount':
            self.remove_per_iteration = int(self.removal_coef)
        elif self.removal_type == 'percent':
            coef_len = len(self.full_xmap)
            self.remove_per_iteration = int(round(float(coef_len)*self.removal_coef))
            
        if self.verbose:
            print 'removal type:', self.removal_type
            print 'removal amount:', self.remove_per_iteration


    def initialize_stop_condition(self):
        
        if self.stop_type == 'amount':
            self.stop_length = int(self.stop_coef)
        elif self.stop_type == 'percent':
            coef_len = len(self.full_xmap)
            self.stop_length = int(round(float(coef_len)*self.stop_coef))
            
        if self.verbose:
            print 'stop type:', self.stop_type
            print 'stop length:', self.stop_length
    

    def initialize_mean_deviance(self, X, Y):

        print '\ninitializing mean deviance...'
        
        print X.shape
        X_c1 = X[(Y==1),:]
        X_c2 = X[(Y==-1),:]
        print X_c1.shape, X_c2.shape

        #X_c1 = np.where(class1, X)
        #X_c2 = np.where(class2, X)

        X_c1m = np.median(X_c1, axis=0)
        X_c2m = np.median(X_c2, axis=0)

        print 'mean vec shapes:', X_c1m.shape, X_c2m.shape

        self.X_mean_deviance = X_c1m - X_c2m
        self.X_mean_deviance_use = self.X_mean_deviance.copy()

        print 'mean deviance shape:', self.X_mean_deviance.shape
        print ''


    
    def initialize(self, X, removal_criterion='amount', removal_coef=25,
                   stop_criterion='percent', stop_coef=0.05, start_C=1.0,
                   C_change_percent=0.0, C_max=1.0, C_min=0.01, only_remove_zeros=False,
                   pls_components=6, mean_deviance_weights=False, bootstrap=False, 
                   bootstrap_folds=50, bootstrap_fraction=0.1):
        
        if self.verbose:
            print 'PREFORMING INITIALIZATIONS...\n'
        self.initialize_xmap(X)
        
        self.removal_type = removal_criterion
        self.removal_coef = removal_coef
        self.stop_type = stop_criterion
        self.stop_coef = stop_coef
        self.C = start_C
        self.C_change_percent = C_change_percent
        self.C_max = C_max
        self.C_min = C_min
        self.only_remove_zeros = only_remove_zeros
        self.pls_components = pls_components
        self.mean_deviance_weights = mean_deviance_weights
        self.bootstrap = bootstrap
        self.bootstrap_folds = bootstrap_folds
        self.bootstrap_fraction = bootstrap_fraction

        self.initialize_removal()
        self.initialize_stop_condition()

        
    
    def run(self, X, Y, testX=None, testY=None, use_gbr=False, use_ada=False, use_sgd=False,
        use_pls=False, use_logreg=False, use_trees=False):
        
        Xnorm = self.normalize_xset(X)
        if testX is not None:
            Xtest_norm = self.normalize_xset(testX)
        self.Xuse = Xnorm.copy()
        self.Yuse = Y.copy()
        
        if self.verbose:
            print 'Xnorm shape:', Xnorm.shape

        if self.mean_deviance_weights:
            self.initialize_mean_deviance(Xnorm, Y)

        test_records = {}
        
        while len(self.use_inds) > self.stop_length:
            
            #self.current_coefs = self.fit_linearsvc_bootstrap(self.Xuse, self.Yuse)
            #if not use_gbr and not use_ada:
            if use_sgd:
                self.current_coefs = self.fit_SGD(self.Xuse, self.Yuse)
            elif use_pls:
                self.current_coefs = self.fit_pls(self.Xuse, self.Yuse)
            elif use_logreg:
                self.current_coefs = self.fit_logreg(self.Xuse, self.Yuse)
            elif use_trees:
                self.current_coefs = self.fit_extratrees(self.Xuse, self.Yuse)
            else:
                if not self.bootstrap:
                    self.current_coefs = self.fit_linearsvc(self.Xuse, self.Yuse)
                else:
                    self.current_coefs = self.fit_linearsvc_bootstrap(self.Xuse, self.Yuse)

            if testX is not None and testY is not None:
                if use_gbr or use_ada or use_trees:
                    self.current_coefs = self.fit_linearsvc(self.Xuse, self.Yuse)
                test_acc = self.test_coefs(Xtest_norm, testY)
                test_records[self.Xuse.shape[1]] = test_acc

            #if use_gbr:
            #    print '\nUsing Gradient Boosting Regression to determine feature importances...'
            #    self.current_coefs = self.fit_gradientboosting(self.Xuse, self.Yuse)
            #    #pprint(self.current_coefs)

            #if use_ada:
            #    print '\nUsing AdaBoost to determine feature importances...'
            #    self.current_coefs = self.fit_adaboost(self.Xuse, self.Yuse)


            inds_to_remove = self.determine_weak_inds(self.current_coefs)
            justified_inds = self.justify_removal_inds(inds_to_remove)
            self.excise_from_xmap(justified_inds)
            self.Xuse = self.subselect(Xnorm)

            if self.mean_deviance_weights:
                self.X_mean_deviance_use = self.subselect(self.X_mean_deviance)

            self.C = self.C*self.C_change_percent
            if self.C > self.C_max:
                self.C = self.C_max
            if self.C < self.C_min:
                self.C = self.C_min
            
            if self.verbose:
                print 'stop length:', self.stop_length
                print 'current coef length:', len(self.current_coefs)
                print 'stop length condition:', self.stop_length
                print 'next useable X:', self.Xuse.shape[1]

                
        if self.verbose:
            print 'final run with RFE...'
        
        #if use_sgd:
        #    self.current_coefs = self.fit_SGD(self.Xuse, self.Yuse)
        #else:
        if use_pls:
            self.current_coefs = self.fit_pls(self.Xuse, self.Yuse)
        elif use_logreg:
            self.current_coefs = self.fit_logreg(self.Xuse, self.Yuse)
        else:
            self.current_coefs = self.fit_linearsvc(self.Xuse, self.Yuse, penalty='l2')


        if testX is not None and testY is not None:
            test_acc = self.test_coefs(Xtest_norm, testY)
            test_records[self.Xuse.shape[1]] = test_acc
                
        if self.verbose:
            print 'Completed RFE'
            print 'current_coefs length:', len(self.current_coefs)

        return test_records
            
    
    def test_coefs(self, testX, testY):
        reshaped = self.place_coefs(self.current_coefs)
        accuracy = 0.0
        for X, Y in zip(testX, testY):
            output = np.sum(X*reshaped)
            if np.sign(output) == np.sign(Y):
                accuracy += 1.0
        accuracy = accuracy / len(testY)
        print '\ntest accuracy:', accuracy, '\n'
        return accuracy

    
    def output_maps(self, time_points, nifti_filepath):
        
        if not nifti_filepath.endswith('.nii'):
            nifti_filepath = nifti_filepath+'.nii'
        
        self.reshaped_coefs = self.place_coefs(self.current_coefs)
        
        if self.verbose:
            print 'reshaping the coefs to original brain shape...'
            
        unmasked = self.data.unmask_Xcoefs(self.reshaped_coefs, time_points, verbose=True)
        
        if self.verbose:
            print 'saving nifti to filename:', nifti_filepath
            
        self.data.save_unmasked_coefs(unmasked, nifti_filepath)
        
        
        
        
        
        
        
        
        
        
    
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
    
    
    
    
                
            
