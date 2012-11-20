

import numpy as np
import random
import itertools
import functools
from pprint import pprint
from process import Process




class CVObject(Process):
    
    
    def __init__(self, variable_dict=None, data_obj=None):
        super(CVObject, self).__init__(variable_dict=variable_dict)
        self.crossvalidator = Crossvalidation()
        if data_obj:
            self.data = data_obj
            self.subject_indices = getattr(self.data, 'subject_indices', None)
            self.indices_dict = self.subject_indices
        else:
            self.data = None
        self.crossvalidation_ready = False
    
    
    def set_folds(self, folds):
        self.crossvalidator.folds = folds
        self.folds = folds
        
        
    def replace_Y_vals(self, Y, original_val, new_val):
        replace = lambda val: new_val if (val==original_val) else val
        return np.array([replace(v) for v in Y])
        
        
    def replace_Y_negative_ones(self):
        self.Y = self.replace_Y_vals(self.Y, -1., 0.)
        
        
    def prepare_folds(self, indices_dict=None, folds=None, leave_mod_in=False):
        # indices dict must be a python dictionary with keys corresponding to
        # some kind of grouping (typically keys for subjects/brains).
        # the values for those keys in the dict are the indices of the X and Y
        # matrices in the data object "attached" to these subjects.
        # this allows for flexible and decently clear upsampling, downsampling,
        # and crossvalidation across folds of these "keys"
        
        # if no indices dict specified, try and get it from self.data's
        # subject_indices, assuming it has been made.
        if not indices_dict:
            if not self.data:
                print 'Unable to find indices_dict, quitting crossvalidation preparation'
                return False
            else:
                if not getattr(self.data, 'subject_indices', None):
                    print 'Unable to find indices_dict, quitting crossvalidation preparation'
                    return False
                else:
                    indices_dict = self.data.subject_indices
        
        # set folds:
        if folds:
            self.set_folds(folds)
        else:
            self.folds = None
            
        # have the crossvalidator object make training and testing dicts:
        self.crossvalidator.create_crossvalidation_folds(indices_dict=indices_dict,
                                                         folds=self.folds,
                                                         leave_mod_in=leave_mod_in)
            
        # reassign variables of CVObject from the crossvalidator:
        self.folds = self.crossvalidator.folds
        self.train_dict = self.crossvalidator.train_dict
        self.test_dict = self.crossvalidator.test_dict
        
        if (getattr(self, 'X', None) is not None) and (getattr(self, 'Y', None) is not None):
            print 'assigning to trainX, trainY, testX, testY...'
            self.cv_group_XY(self.X, self.Y)
            self.crossvalidation_ready = True
        else:
            print 'X and Y matrices, unset.. run cv_group_XY(X, Y) when ready to get cv groups'
        
        
        return True
            
            
    def subselect(self, data, indices):
        return np.array([data[i].tolist() for i in indices])

    
    def cv_group_XY(self, X, Y):
        if getattr(self, 'train_dict', None) and getattr(self, 'test_dict', None):
            fold_inds = self.train_dict.keys()
            assert fold_inds == self.test_dict.keys()
            
            self.trainX = [self.subselect(X, self.train_dict[tg]) for tg in fold_inds]
            self.trainY = [self.subselect(Y, self.train_dict[tg]) for tg in fold_inds]
            self.testX = [self.subselect(X, self.test_dict[tg]) for tg in fold_inds]
            self.testY = [self.subselect(Y, self.test_dict[tg]) for tg in fold_inds]
            
            self.subjects_in_folds = [self.crossvalidator.cv_sets[i] for i in fold_inds]
            
            self.crossvalidation_ready = True
            print 'completed groupings into trainX/Y, testX/Y'
        else:
            print 'Could not make train/test X Y matrices'
        
        
    
    def statsfunction_over_folds(self, statsfunction, Xgroups, Ygroups, **kwargs):
        # the statsfunction ported in must contain ONLY 2 NON-KEYWORD ARGUMENTS:
        # X data and Y data. The rest of the arguments MUST BE KEYWORDED.
        # you can pass the keyword arguments to this function that you would
        # have passed to the statsfunction originally. Note that the keyword
        # arguments (obviously) have to have the same name as they did in
        # statsfunction since they will be passed along to statsfunction soon
        # enough.
        results = []
        for X, Y in zip(Xgroups, Ygroups):
            statspartial = functools.partial(statsfunction, X, Y, **kwargs)
            results.append([statspartial()])
        return results
        
        
    def traintest_crossvalidator(self, trainfunction, testfunction, trainXgroups,
                                 trainYgroups, testXgroups, testYgroups, train_kwargs_dict={},
                                 test_kwargs_dict={}, verbose=False):
        
        trainresults = []
        testresults = []
        for trainX, trainY, testX, testY in zip(trainXgroups, trainYgroups,
                                                testXgroups, testYgroups):
            if verbose:
                print 'Crossvalidating next group.'
            trainpartial = functools.partial(trainfunction, trainX, trainY, **train_kwargs_dict)
            trainresult = trainpartial()
            testpartial = functools.partial(testfunction, testX, testY, trainresult, **test_kwargs_dict)
            testresult = testpartial()
            if verbose:
                print 'this groups\' test result:'
                pprint(testresult)
            trainresults.append(trainresult)
            testresults.append(testresult)
            
        return trainresults, testresults
    



class Crossvalidation(object):
    
    def __init__(self, indices_dict=None, folds=None):
        super(Crossvalidation, self).__init__()
        self.indices_dict = indices_dict
        self.folds = folds
            
            
    def chunker(self, indices, chunksize):
        # chunker splits indices into equal sized groups, returns dict with IDs:
        groups = [indices[i:i+chunksize] for i in range(0, len(indices), chunksize)]
        cv_sets = {}
        for i, group in enumerate(groups):
            cv_sets[i] = group
        return cv_sets
        
        
    def excise_remainder(self, indices, folds):
        modulus = len(indices) % folds
        random.shuffle(indices)
        return indices[modulus:], indices[0:modulus]
        
        
    def generate_sets(self, cv_sets, perms, mod_keys, include_mod):
        
        train_dict = {}
        test_dict = {}
        
        self.test_subject_byfold = []
        
        for p, groups in enumerate(perms):
            
            train_dict[p] = []
            test_dict[p] = []
            
            training_keys = []
            testing_keys = []
            
            for gkey in groups:
                training_keys.extend(cv_sets[gkey])
            
            for tr_key in training_keys:
                train_dict[p].extend(self.indices_dict[tr_key])
            
            for cv_key in cv_sets.keys():
                if cv_key not in groups:
                    testing_subjects = cv_sets[cv_key]
                    if include_mod:
                        testing_subjects.extend(mod_keys)
            
            for te_key in testing_subjects:
                test_dict[p].extend(self.indices_dict[te_key])
                
            self.test_subject_byfold.append(testing_subjects)
                
                
        return train_dict, test_dict
                
        
        
    def create_crossvalidation_folds(self, indices_dict=None, folds=None, leave_mod_in=False):
        
        self.folds = folds or getattr(self,'folds',None)
        self.indices_dict = indices_dict or getattr(self,'indices_dict',None)
        
        self.train_dict = {}
        self.test_dict = {}
        
        if self.indices_dict is None:
            print 'No indices dictionary provided, exiting...'
            return False
        
        index_keys = self.indices_dict.keys()
        
        if self.folds is None:
            print 'Folds unset, defaulting to leave one out crossvalidation...'
            self.folds = len(index_keys)
        
        divisible_keys, remainder_keys = self.excise_remainder(index_keys, self.folds)
        
        # cv_sets is a dict with group IDs and indices:
        self.cv_sets = self.chunker(divisible_keys, len(divisible_keys)/self.folds)
        
        # find the permutations of the group IDs, leaving one out:
        set_permutations = itertools.combinations(self.cv_sets.keys(), len(self.cv_sets.keys())-1)
        
        self.train_dict, self.test_dict = self.generate_sets(self.cv_sets, set_permutations,
                                                             remainder_keys, leave_mod_in)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    