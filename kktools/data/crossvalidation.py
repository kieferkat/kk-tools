

import numpy as np
import random
import itertools


# decorator supports leave one out only, at this point:
# function to be decorated must contain full X, full Y, training indices and
# testing indices!
#def crossvalidation_decorator(folds=1):
#    
#    def function_decorator(prediction_function):
#        def wrapper(X, Y, train_indices=None, test_indices=None):
    



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
        
        
    def generate_sets(self, cv_sets, perms, mod_inds, include_mod):
        
        train_dict = {}
        test_dict = {}
        
        for p, groups in enumerate(perms):
            
            train_dict[p] = []
            test_dict[p] = []
            
            training_subjects = []
            testing_subjects = []
            
            for ind in groups:
                training_subjects.extend(cv_sets[ind])
            
            for subject in training_subjects:
                train_dict[p].extend(self.indices_dict[subject])
            
            for key in cv_sets.keys():
                if key not in groups:
                    testing_subjects = cv_sets[key]
                    if include_mod:
                        testing_subjects.extend(mod_inds)
                        
            for subject in testing_subjects:
                test_dict[p].extend(self.indices_dict[subject])
                
        return train_dict, test_dict
                
        
        
    def create_crossvalidation_folds(self, indices_dict=None, folds=None, leave_mod_in=False):
        
        self.folds = folds or getattr(self,'folds',None)
        self.indices_dict = indices_dict or getattr(self,'indices_dict',None)
        
        self.train_dict = {}
        self.test_dict = {}
        
        if self.indices_dict is None:
            print 'No indices dictionary provided, exiting...'
            return False
        
        subject_inds = self.indices_dict.keys()
        
        if self.folds is None:
            print 'Folds unset, defaulting to leave one out crossvalidation...'
            self.folds = len(subject_inds)
        
        divisible_inds, remainder_inds = self.excise_remainder(subject_inds, self.folds)
        
        # cv_sets is a dict with group IDs and indices:
        cv_sets = self.chunker(divisible_inds, len(divisible_inds)/self.folds)
        
        # find the permutations of the group IDs, leaving one out:
        set_permutations = itertools.combinations(cv_sets.keys(), len(cv_sets.keys())-1)
        
        self.train_dict, self.test_dict = self.generate_sets(cv_sets, set_permutations,
                                                             remainder_inds, leave_mod_in)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    