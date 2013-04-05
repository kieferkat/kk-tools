
from pprint import pprint
import subprocess
import glob
import gc
import sys, os
import random
import time
import numpy as np
import scipy as sp
import nibabel as nib
import time

from process import Process
from nifti import NiftiTools
from sklearn import preprocessing
from nipy.io.api import load_image

from ..stats.normalize import simple_normalize
from ..utilities.cleaners import glob_remove
from ..utilities.csv import CsvTools
from ..afni.functions import AfniWrapper
from ..utilities.vector import VectorTools



class DataManager(Process):
    ''' 
    DataManager is the basic superclass for both the CsvData and BrainData classes
    (currently). It contains functions that both of those classes can use, particularly
    construction of the X and Y matrices.
    
    '''
    
    def __init__(self, variable_dict=None):

        super(DataManager, self).__init__(variable_dict=variable_dict)
        self.verbose = True
        
            
            
    def recode_variable(self, variable_list=[], variable_dict={}, allow_unspecified=True,
                        as_string=False):
        '''
        recode variable takes a list of variables to change and a dictionary that
        has old values as keys and new values as values. it iterates 
        through the variable_list and makes a new list of recoded varaibles.
        A bit clunky, yes? needs to be redone or taken out perhaps.

        allow_unspecified : this flag if set to true (a good move) will just keep a
        variable the same if it doesn't find it in the dict. a True value here will stop
        the function entirely.

        as_string: True here makes the variables strings, otherwise they are returned as
        floats.
        '''
        
        recoded = []
        for var in variable_list:
            if var in variable_dict.keys():
                nval = variable_dict[var]
                if as_string:
                    recoded.append(str(nval))
                else:
                    recoded.append(float(nval))
            else:
                if allow_unspecified:
                    if as_string:
                        recoded.append(string(var))
                    else:
                        recoded.append(float(var))
                else:
                    print 'variable not found in replacement dict'
                    return False
                
        if as_string:
            return recoded
        else:
            return np.array(recoded)
        
        
                
                
    def _xy_matrix_tracker(self, Ybinary):
        '''
        A function for verbosity in the create_XY_matrices.
        '''

        print 'X (trials) length: ', len(self.X)
        print 'Y (responses) length: ', len(self.Y)
        print 'positive responses: ', self.Y.count(Ybinary[0])
        print 'negative responses: ', self.Y.count(Ybinary[1])
        
        
        
        
        
    def create_XY_matrices(self, subject_design=None, downsample_type=None, with_replacement=False,
                           replacement_ceiling=None, random_seed=None, Ybinary=[1.,-1.], Yreplace=[1.,-1],
                           verbose=True):
        '''
        create_XY_matrices: the X and Y matrix creating super-function.

        this function will convert the subject_design dictionary into an X and Y matrix. It requires that
        subject_design be formatted properly. keys in subject_design must be some indicator of the subject,
        values are tuples or lists containing the trials list at [0] and the responses list at [1].

        ex:
        subject_design['ab040313'] = [trials, responses]

        downsample_type     :   downsample type can either be set to None/False, 'subject', or 'trial'.
                                'subject' downsampling will equalize the number of yes and no responses
                                within each subject (key) in the subject design dictionary.
                                'trial' downsampling will equalize the number of yes and no responses over 
                                the cumulative trials of all the subjects in the subject design dict.

        '''
        
        required_vars = {'subject_design':subject_design}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        
        self.random_seed = random_seed or getattr(self,'random_seed',None)
        
        
        if self.random_seed:
            print self.random_seed
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            
            
        self.X = []
        self.Y = []
        self.subject_indices = {}
        
        
        if not downsample_type:
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_indices[subject] = []
                
                if not with_replacement:
                    for trial, response in zip(trials, responses):
                        self.subject_indices[subject].append(len(self.X))
                        self.X.append(trial)
                        if float(response) == float(Ybinary[0]):
                            self.Y.append(Yreplace[0])
                        elif float(response) == float(Ybinary[1]):
                            self.Y.append(Yreplace[1])
                        else:
                            print 'y value unaccounted for, setting to 0...'
                            self.Y.append(0.)
                        #self.Y.append(response)
                        
                elif with_replacement:
                    
                    positive_trials = []
                    negative_trials = []
                    
                    for trial, response in zip(trials, responses):
                        if float(response) == float(Ybinary[0]):
                            positive_trials.append(trial)
                        elif float(response) == float(Ybinary[1]):
                            negative_trials.append(trial)
                            
                    if min(len(positive_trials), len(negative_trials)) == 0:
                        del self.subject_indices[subject]
                    
                    else:
                        if not replacement_ceiling:
                            upper_length = max(len(positive_trials), len(negative_trials))
                        else:
                            upper_length = replacement_ceiling
                        
                        for set in [positive_trials, negative_trials]:
                            random.shuffle(set)
                            
                            for i, trial in enumerate(set):
                                if i < upper_length:
                                    self.subject_indices[subject].append(len(self.X))
                                    self.X.append(trial)
                                
                            for rep_trial in [random.sample(set, 1)[0] for i in range(upper_length-len(set))]:
                                self.subject_indices[subject].append(len(self.X))
                                self.X.append(rep_trial)
                                
                        self.Y.extend([Yreplace[0] for x in range(upper_length)])
                        self.Y.extend([Yreplace[1] for x in range(upper_length)])
        
                if verbose:
                    self._xy_matrix_tracker(Yreplace)
                    
                    
                    
        elif downsample_type == 'group':
            
            positive_trials = []
            negative_trials = []
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_indices[subject] = []
                
                for trial, response, in zip(trials, responses):
                    if float(response) == float(Ybinary[0]):
                        positive_trials.append([subject,trial])
                    elif float(response) == float(Ybinary[1]):
                        negative_trials.append([subject,trial])
                        
            random.shuffle(positive_trials)
            random.shuffle(negative_trials)
            
            if not with_replacement:
                for i in range(min(len(positive_trials), len(negative_trials))):
                    [psub, ptrial] = positive_trials[i]
                    [nsub, ntrial] = negative_trials[i]
                    self.subject_indices[psub].append(len(self.X))
                    self.X.append(ptrial)
                    self.subject_indices[nsub].append(len(self.X))
                    self.X.append(ntrial)
                    self.Y.extend([Yreplace[0],Yreplace[1]])
                    
                if verbose:
                    self._xy_matrix_tracker(Ybinary)
                    
            elif with_replacement:
                
                if not replacement_ceiling:
                    upper_length = max(len(positive_trials), len(negative_trials))
                else:
                    upper_length = replacement_ceiling
                
                for set in [positive_trials, negative_trials]:
                    random.shuffle(set)
                    
                    for i, (sub, trial) in enumerate(set):
                        if i < upper_length:
                            self.subject_indices[sub].append(len(self.X))
                            self.X.append(trial)
                                                
                    for sub, trial in [random.sample(set, 1)[0] for i in range(upper_length-len(set))]:
                        self.subject_indices[sub].append(len(self.X))
                        self.X.append(trial)
                        
                self.Y.extend([Yreplace[0] for x in range(upper_length)])
                self.Y.extend([Yreplace[1] for x in range(upper_length)])
                
                if verbose:
                    self._xy_matrix_tracker(Yreplace)
                    
                    
                    
        elif downsample_type == 'subject':

            print 'inside'
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_indices[subject] = []
                
                subject_positives = []
                subject_negatives = []
                
                for trial, response in zip(trials, responses):
                    if float(response) == float(Ybinary[0]):
                        subject_positives.append(trial)
                    elif float(response) == float(Ybinary[1]):
                        subject_negatives.append(trial)
                        
                random.shuffle(subject_positives)
                random.shuffle(subject_negatives)

                print len(subject_positives), len(subject_negatives)
                
                if min(len(subject_positives), len(subject_negatives)) == 0:
                    del self.subject_indices[subject]
                    
                else:
                    if not with_replacement:
                        for i in range(min(len(subject_positives), len(subject_negatives))):
                            self.subject_indices[subject].append(len(self.X))
                            self.X.append(subject_positives[i])
                            self.subject_indices[subject].append(len(self.X))
                            self.X.append(subject_negatives[i])
                            self.Y.extend([Yreplace[0],Yreplace[1]])
                            
                    elif with_replacement:
                        if not replacement_ceiling:
                            upper_length = max(len(subject_positives), len(subject_negatives))
                        else:
                            upper_length = replacement_ceiling
                        
                        for set in [subject_positives, subject_negatives]:
                            random.shuffle(set)
                            
                            for i, trial in enumerate(set):
                                if i < upper_length:
                                    self.subject_indices[subject].append(len(self.X))
                                    self.X.append(trial)
                                
                            print upper_length
                            print len(set)
                                
                            for trial in [random.sample(set, 1)[0] for i in range(upper_length-len(set))]:
                                self.subject_indices[subject].append(len(self.X))
                                self.X.append(trial)
                                
                        self.Y.extend([Yreplace[0] for x in range(upper_length)])
                        self.Y.extend([Yreplace[1] for x in range(upper_length)])
                        
                if verbose:
                    self._xy_matrix_tracker(Yreplace)
                    
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        
        
    def merge_datamanagers(self, mergeable_dm):
        
        # this assumes that the mergeable dm has an X, Y, and subject_indices dict.
        # those are the only things that are assimmilated.
        
        oX = mergeable_dm.X
        oY = mergeable_dm.Y
        oSI = mergeable_dm.subject_indices
        
        merge_rows = len(oX)
        if merge_rows == len(oY):
            pass
        else:
            print 'merge X and merge Y have different lengths ?!'
            return False
        
        self.X.tolist()
        self.Y.tolist()
        
        self.X.extend(oX.tolist())
        self.Y.extend(oY.tolist())
        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
                
        for subj, inds in oSI.items():
            ninds = [x+merge_rows for x in inds]
            if subj in self.subject_indices:
                self.subject_indices[subj].extend(ninds)
            else:
                print 'merged subject not in original subject_indices dict'
                self.subject_indices[subj] = ninds
        
        
        
    def delete_subject_design(self, checkpoint=False):
        if checkpoint:
            inp = raw_input('Press any key to delete subject design (preserves self.X and self.Y)')
        del self.subject_design
        
        
    def X_to_memmap(self, memmap_filepath, empty_X=True, verbose=True,
                    testmode_bypass_overwrite=False):
        
        if verbose: print 'Creating X memmap'
        self.X_memmap_path = memmap_filepath
        self.X_memmap_shape = self.X.shape
        
        if testmode_bypass_overwrite:
            print '!!! TEST MODE: BYPASSING MEMMAP OVERWRITE !!!'
        else:
            try:
                if verbose: print 'attempting to delete old memmap...'
                os.remove(self.X_memmap_path)
                if verbose: print 'old memmap deleted.'
            except:
                if verbose: print 'no memmap to delete'
                
            
            if verbose: print 'writing new memmap...'
            X_memmap = np.memmap(self.X_memmap_path, dtype='float64', mode='w+', shape=self.X_memmap_shape)
            X_memmap[:,:] = self.X[:,:]
            
            del X_memmap
        
        
    def empty_X(self):
        print 'emptying X'
        self.X = []
            
        
    def normalizeX(self):
        print 'normalizing X'
        print 'previous X sum', np.sum(self.X)
        #self.X = preprocessing.normalize(self.X, axis=1)
        self.X = simple_normalize(self.X, axis=0)
        print 'post-normalization X sum', np.sum(self.X)
        
        
    def scaleX(self):
        print 'scaling X'
        self.X = preprocessing.scale(self.X)
        
        
    def normalize_within_subject(self, normalizeY=False):
        '''
        Iterates through the subject trials and normalizes the X matrices
        individually for each subject. This should obviously be done prior
        to calling create_XY_matrices.
        
        Optionally normalizes the Y vector as well.
        '''
        
        for subject, [trials, resp_vec] in self.subject_design.items():
            print 'normalizing within subject: ', subject
            self.subject_design[subject][0] = preprocessing.normalize(trials)
            if normalizeY:
                print 'also normalizing Y...'
                self.subject_design[subject][1] = preprocessing.normalize(resp_vec)
                
                
                
    def scale_within_subject(self, scaleY=False):
        '''
        Iterates through subject X and Y matrices and scales the X matrix. Can
        also optionally scale the Y matrix. Should be done prior to calling
        create_XY_matrices, if done at all.
        '''
        
        for subject, [trials, resp_vec] in self.subject_design.items():
            print 'scaling within subject:', subject
            self.subject_design[subject][0] = preprocessing.scale(trials)
            if scaleY:
                print 'also scaling Y...'
                self.subject_design[subject][1] = preprocessing.scale(resp_vec)
        

    def recodeY(self, oldvalues, newvalues):
        '''
        Will iterate through oldvalues and newvalues (paired), replacing the old
        values with the new values.
        
        Keep in mind this replacement is iterative, so if you first replace 1 with
        0, then later replace 0 with -1, you will have -1s for 1s in the end.
        Thus the values that come last have precedence in replacement.
        '''
        
        if len(oldvalues) == len(newvalues):
            for ov, nv in zip(oldvalues, newvalues):
                self.Y = [nv if y == ov else y for y in self.Y]
        


class CsvData(DataManager):
    
    def __init__(self, variable_dict=None):
        super(CsvData, self).__init__(variable_dict=variable_dict)
        self.csv = CsvTools()
        self.vector = VectorTools()
        self.afni = AfniWrapper()
        self.independent_dict = {}
        self.dependent_dict = {}
        self.bysubject_data_dict = {}
        self.available_variables = []
        
        
    def load_subject_csvs(self, subject_dirs=None, behavior_csv_name=None):
        
        required_vars = {'subject_dirs':subject_dirs, 'behavior_csv_name':behavior_csv_name}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.bysubject_data_dict = {}        
        subject_csv_dict = self.csv.load_csvs(self.subject_dirs, self.behavior_csv_name)
        
        for subject, csvdict in subject_csv_dict.items():
            self.bysubject_data_dict[subject] = self.csv.csv_to_coldict(csvdict)
            
            

    
    
    def load_subject_raw_tcs(self, subject_dirs=None, tmp_tc_dir='raw_tc'):
        
        required_vars = {'subject_dirs':subject_dirs}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.subject_vector_dict = self.vector.subject_vector_dict(self.subject_dirs, tmp_tc_dir)
                


    def merge_in_vector_dict(self, subject_vector_dict=None, bysubject_data_dict=None):
        
        required_vars = {'subject_vector_dict':subject_vector_dict,
                         'bysubject_data_dict':bysubject_data_dict}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.bysubject_data_dict = self.csv.merge_csv_dicts(self.bysubject_data_dict,
                                                            self.subject_vector_dict,
                                                            keylevel=1)


    def load_allsubject_csv(self, csv_path):
        # must have subject in header!

        csv_lines = self.csv.read(csv_path)
        self.bysubject_data_dict = self.csv.subjectcsv_to_subjectdict(csv_lines)
        
        self.sparse_data_dict = self.bysubject_data_dict.copy()
        
    
    def normalize_columns_within_subject(self, columns):
        
        for subject, cols in self.bysubject_data_dict.items():
            for col in cols:
                if col in columns:
                    data = np.array([float(x) for x in self.bysubject_data_dict[subject][col]])
                    #print subject, data.shape
                    data.shape = (data.shape[0], 1)
                    #print data.shape
                    self.bysubject_data_dict[subject][col] = [str(x[0]) for x in preprocessing.normalize(data, axis=0)]
    
    
    
    def scale_columns_within_subject(self, columns):
        
        for subject, cols in self.bysubject_data_dict.items():
            for col in cols:
                if col in columns:
                    data = np.array([float(x) for x in self.bysubject_data_dict[subject][col]])
                    #print subject, data.shape
                    data.shape = (data.shape[0], 1)
                    #print data.shape
                    self.bysubject_data_dict[subject][col] = [str(x[0]) for x in preprocessing.scale(data, axis=0)]
    
    
    
    def write_csv_data(self, filepath, data_dict):
        csvlines = self.csv.subject_csvdicts_tolines(data_dict)
        self.csv.write(csvlines, filepath)

    
    def add_independent_variable(self, variable, conditional_dict={}):
        self.independent_dict[variable] = conditional_dict
        
        
    def add_dependent_variable(self, variable, conditional_dict={}):
        self.dependent_dict[variable] = conditional_dict
        
        
    def assess_available_variables(self, verbose=True):
        
        for i, subject in enumerate(self.bysubject_data_dict.keys()):
            if i == 0:
                self.available_variables.extend(self.bysubject_data_dict[subject].keys())
            else:
                del_vars = []
                for var in self.available_variables:
                    if var not in self.bysubject_data_dict[subject].keys():
                        del_vars.append(var)
                for dvar in del_vars:
                    if verbose:
                        print 'not all subjects have variable %s, removing' % (dvar)
                    self.available_variables.remove(dvar)
                    
        
    def _find_inds_where(self, datalist, condval, stringensure=True):
        if stringensure:
            if type(condval) in (list, tuple):
                condval = [str(x) for x in condval]
            else:
                condval = [str(condval)]
            datalist = [str(x) for x in datalist]
        #print len(datalist), condval
        out = [i for i,x in enumerate(datalist) if x in condval]
        #print len(out)
        return out
        
    
    def _slice_conditional_inds(self, indslist):
        basis_inds = indslist[0]
        for inds in indslist:
            basis_inds = [x for x in basis_inds if x in inds]
        return basis_inds
    
                
    def cut_data_dict(self, keep_only_where_dict):
        
        self.sparse_data_dict = self.bysubject_data_dict.copy()
        
        for spvar, spvals in keep_only_where_dict.items():
            
            spvar = spvar.lower()
            
            if not type(spvals) in (list, tuple):
                spvals = [spvals]
                
            for subject, variables in self.bysubject_data_dict.items():
                cinds = []
                
                for variable in variables:
                    if variable == spvar:
                        #print spvals
                        cinds.append(self._find_inds_where(self.bysubject_data_dict[subject][variable], spvals))
                        
                basis_inds = self._slice_conditional_inds(cinds)
                
                #print cinds
                #print basis_inds
                
                for v in variables:
                    self.sparse_data_dict[subject][v] = [x for i,x in enumerate(self.sparse_data_dict[subject][v]) if i in basis_inds]
                    #print len(self.sparse_data_dict[subject][v]), v
                    
        
                    
    
    def _get_vars_by_conds(self, subject, variable, conditionals):
        
        nvars = []
        
        for cvar, conds in conditionals.items():
            cvar = cvar.lower()
            for cond in conds:
                cinds = self._find_inds_where(self.sparse_data_dict[subject][cvar], cond)
                suffix = '_'.join([cvar, str(cond)])
                nvar = [x for i,x in enumerate(self.sparse_data_dict[subject][variable]) if i in cinds]
                nvars.append([nvar, '_'.join([variable, suffix])])
        
        return nvars
    
    
    def _delete_unused_vars(self, subject, keep_vars):
        
        del_vars = []
        for var in self.sparse_data_dict[subject]:
            if var not in keep_vars:
                if var not in del_vars:
                    del_vars.append(var)
                        
        for var in del_vars:
            del(self.sparse_data_dict[subject][var])
            
    
    
    def create_design(self, independent_dict=None, dependent_dict=None,
                      csv_filepath=None):
        
        required_vars = {'independent_dict':independent_dict,
                         'dependent_dict':dependent_dict}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.subject_design = {}
        self.assess_available_variables()
        self.independent_names = []
        self.dependent_name = []
        
        
        for subject in self.sparse_data_dict:
            use_vars = []
            
            for iv, conds in self.independent_dict.items():
                if not conds:
                    use_vars.append([self.sparse_data_dict[subject][iv], iv])
                else:
                    use_vars.extend(self._get_vars_by_conds(subject, iv, conds))
                    
            if not self.independent_names:
                self.independent_names = [x[1] for x in use_vars]
                
            for dv, conds in self.dependent_dict.items():
                if not conds:
                    use_vars.append([self.sparse_data_dict[subject][dv], dv])
                else:
                    use_vars.extend(self._get_vars_by_conds(subject, dv, conds))
                    
            if not self.dependent_name:
                self.dependent_name = [x[1] for x in use_vars if x[1] not in self.independent_names]
                    
            print self.independent_names
            print self.dependent_name

            self._delete_unused_vars(subject, [x[1] for x in use_vars])
            for [varlist, varname] in use_vars:
                self.sparse_data_dict[subject][varname] = varlist
                
        
        for sub, subdict in self.sparse_data_dict.items():
            print sub
            for k,v in subdict.items():
                print k, len(v)
         
        if csv_filepath:
            self.write_csv_data(csv_filepath, self.sparse_data_dict)
            
            
        
        
        self.independent_names = sorted(self.independent_names)
        self.dependent_name = sorted(self.dependent_name)
        
        for subject, variables in self.sparse_data_dict.items():
            varlen = np.unique([len(x) for x in self.sparse_data_dict[subject].values()])
            
            if len(varlen) == 1:
                trials = []
                responses = [] 
                
                for row in range(varlen[0]):
                    trial = []
                    for colv in self.independent_names:
                        trial.append(float(self.sparse_data_dict[subject][colv][row]))
                    trials.append(trial)
                    for colv in self.dependent_name:
                        responses.append(float(self.sparse_data_dict[subject][colv][row]))
                      
                self.subject_design[subject] = [trials, responses]  
                        

    
    
class BrainData(DataManager):
    '''
    BrainData
    ------------
    Recoded BrainData class in style of Jonathan Taylor's masking scheme. It
    preforms faster than my old one so I decided to go with this. However,
    his originally forced a matrix transposition (reversal) which may not
    be ideal. This has the option of either using his matrix format or
    preserving (as best as possible) the dimensionality when loading niftis
    using nipy or nibabel.
    
    '''
    
    def __init__(self, variable_dict=None):
        
        super(BrainData, self).__init__(variable_dict=variable_dict)
        self.subject_data_dict = {}
        self.nifti = NiftiTools()
        self.vector = VectorTools()
    
    
    
    def create_niftis(self, subject_dirs=None, functional_name=None, anatomical_name=None,
                      dxyz=None, talairach_template_path=None, nifti_name=None,
                      within_subject_warp=True, to_template_warp=False):
        
            
        if not nifti_name.endswith('.nii'):
            nifti_name = nifti_name+'.nii'
        
        self.nifti.create_talairach_niftis(subject_dirs, functional_name,
                                           anatomical_name, dxyz,
                                           talairach_template_path, nifti_name,
                                           within_subject_warp, to_template_warp)
        
        
        
        
    def load_niftis_vectors(self, directory, verbose=True):
        '''
        Loads niftis and response vectors from a directory. This function is
        fairly specific. The nifti files should be named in this manner:
        
        prefix_***.nii
        
        Such that prefix denotes a subject and an underscore splits this subject
        from the rest of the nifti filename.
        
        Likewise, the response vector file should be coded:
        
        prefix_***.1D
        
        Such that the prefix matches a prefix for a nifti file!
        NO DUPLICATE PREFIXES - WILL CHOOSE INDISCRIMINATELY
        '''
        
        nifti_paths = sorted(glob.glob(os.path.join(directory, '*.nii')))
        vector_paths = sorted(glob.glob(os.path.join(directory, '*.nii')))
        
        npre = [os.path.split(n)[1].split('_')[0] for n in nifti_paths]
        vpre = [os.path.split(v)[1].split('_')[0] for v in vector_paths]
        
        pairs = []
        for nifti, np in zip(nifti_paths, npre):
            for vector, vp in zip(vector_paths, vpre):
                if np == vp:
                    pairs.append([nifti, self.vector.read(vector)])
                    break
        
        return pairs
        

        
        
        
    def load_niftis_fromdirs(self, subject_dirs, nifti_name, response_vector,
                             verbose=True):
        '''
        Iterates through subject directories, parses the response vector,
        and appends the path to the nifti file for loading later.
        
        Basic support for multiple niftis per subject (just added as different
        key in the subject_data_dict).
        
        '''
        for subject in subject_dirs:
                
            nifti = os.path.join(subject, nifti_name)
            vec = os.path.join(subject, response_vector)

            if not os.path.exists(nifti):
                if verbose:
                    print 'nifti not found: ', nifti
            elif not os.path.exists(vec):
                if verbose:
                    print 'respnse vector not found: ', vec
            else:
                
                respvec = self.vector.read(vec, usefloat=True)
                subject_key = os.path.split(subject)[1]
                
                if verbose:
                    pprint(nifti)
                    print 'appending raw data for subject: ', subject_key
                
                if not subject_key in self.subject_data_dict:
                    self.subject_data_dict[subject_key] = [nifti, respvec]
                    
                else:
                    tag = 2
                    while subject_key+'_'+str(tag) in self.subject_data_dict:
                        tag += 2
                    self.subject_data_dict[subject_key+'_'+str(tag)] = [nifti, respvec]
                
        
        
        
    def parse_trialsvec(self, trialsvec):
        '''
        Simple function to find the indices where Y is not 0. Returns the indices
        vector and the stripped Y vector. Used by masked_data().
        '''
        
        inds = [i for i,x in enumerate(trialsvec) if x != 0.]
        y = [x for x in trialsvec if x != 0]
        return inds, y
    
    
        
    def unmask_Xcoefs(self, Xcoefs, time_points, mask=None, reverse_transpose=True,
                      verbose=True, slice_off_back=0, slice_off_front=0):
        '''
        Reshape the coefficients from a statistical method back to the shape of
        the original brain matrix, so it can be output to nifti format.
        '''
        if mask is None:
            mask = self.original_mask
            
        unmasked = [np.zeros(mask.shape) for i in range(time_points)]
        
        print 'xcoefs sum', np.sum(Xcoefs)
        
        if slice_off_back:
            print np.sum(Xcoefs[:-slice_off_back]), np.sum(Xcoefs[-slice_off_back:])
            Xcoefs = Xcoefs[:-slice_off_back]
        if slice_off_front:
            Xcoefs = Xcoefs[slice_off_front:]
        
        print 'xcoefs sum', np.sum(Xcoefs)
        Xcoefs.shape = (time_points, -1)
        print 'Xcoefs shape', Xcoefs.shape
        
        for i in range(time_points):
            print 'raw coef time sum', np.sum(Xcoefs[i])
            print 'mask, xind shapes', mask.shape, Xcoefs[i].shape
            
            unmasked[i][np.asarray(mask).astype(np.bool)] = np.squeeze(np.array(Xcoefs[i]))
            #unmasked[i][np.asarray(mask).astype(np.bool)] = np.squeeze(np.ones(np.sum(np.asarray(mask).astype(np.bool))))
            
            print 'time ind coef sum', np.sum(unmasked[i])
            if reverse_transpose:
                unmasked[i] = np.transpose(unmasked[i], [2, 1, 0])
        
        unmasked = np.transpose(unmasked, [1, 2, 3, 0])
        
        if verbose:
            print 'Shape of unmasked coefs: ', np.shape(unmasked)
        
        return np.array(unmasked)
        
        
        
    def save_unmasked_coefs(self, unmasked, nifti_filename, affine=None,
                            talairach_template_path='./TT_N27+tlrc.'):
        '''
        Simple function to save the unmasked coefficients to a specified nifti.
        Affine is usually self.mask_affine, but can be specified.
        '''
        
        if affine is None:
            affine = self.mask_affine
            
        if self.verbose:
            print 'erasing old files with prefix:', nifti_filename#[:-4]
            
        glob_remove(nifti_filename)#[:-4])
            
        self.nifti.save_nifti(unmasked, affine, nifti_filename)
        
        time.sleep(0.25)
        
        self.nifti.convert_to_afni(nifti_filename, nifti_filename)#[:-4])
        
        time.sleep(0.25)
        
        subprocess.call(['3drefit','-view','tlrc',nifti_filename+'+orig.'])
        
        
        
    def make_masks(self, mask_path, ntrs, reverse_transpose=True, verbose=True):
        
        '''
        A function that makes the various mask objects.
        '''
        if verbose:
            if reverse_transpose:
                print 'using time-first reverse transposition of nifti matrix'
            else:
                print 'preserving dimensionality of nifti matrix (nt last)'
        
        mask = load_image(mask_path)
        tmp_mask, self.mask_affine, tmp_shape = self.nifti.load_nifti(mask_path)
        mask = np.asarray(mask)
        self.raw_affine = self.mask_affine
            
        if verbose:
            print 'mask shape:', mask.shape
        self.mask_shape = mask.shape
            
        if reverse_transpose:
            mask = np.transpose(mask.astype(np.bool), [2, 1, 0])
        else:
            mask = mask.copy().astype(np.bool)
            
        self.original_mask = mask.copy()
        self.flat_mask = mask.copy()
        self.flat_mask.shape = np.product(mask.shape)
        
        if verbose:
            print 'flat mask shape:', self.flat_mask.shape
            
        nmask = np.not_equal(mask, 0).sum()
        
        if verbose:
            print 'mask shape', mask.shape
        
        self.trial_mask = np.zeros((ntrs, mask.shape[0], mask.shape[1], mask.shape[2]))
        
        if verbose:
            print 'trial mask shape', self.trial_mask.shape
        
        for t in range(ntrs):
            self.trial_mask[t,:,:,:] = mask
            
        self.trial_mask = self.trial_mask.astype(np.bool)   
        
        
    def prepare_greymatter_mask(self, mask_path, greymatter_prefix='greymatter_resamp',
                                afni_greymatter_dset='/Users/span/abin/TT_caez_gw_18+tlrc.',
                                afni_index=0, reverse_transpose=True):
        
        # resample the gray matter mask to the user's mask:
        
        gm_resample_mask = os.path.join(os.path.split(mask_path)[0],greymatter_prefix)
        old_resamps = glob.glob(gm_resample_mask+'*')
        for oresamp in old_resamps:
            try:
                os.remove(oresamp)
            except:
                pass
        
        cmd = ['3dresample','-master', mask_path, '-prefix', gm_resample_mask,
               '-inset', afni_greymatter_dset+'['+str(afni_index)+']']
        
        subprocess.call(cmd)
        
        niicmd = ['3dAFNItoNIFTI', '-prefix', greymatter_prefix, greymatter_prefix+'+tlrc.']
        
        subprocess.call(niicmd)
        
        # for now need to have made the trial mask, etc...
        
        self.grey_matter = np.zeros(self.trial_mask.shape)
        
        gm_mask = load_image(gm_resample_mask+'.nii')
        gm_mask = np.asarray(gm_mask)
        
        self.grey_matter_flat = gm_mask.copy()
        self.grey_matter_flat.shape = np.product(self.grey_matter_flat.shape)
            
        if reverse_transpose:
            gm_mask = np.transpose(gm_mask, [2, 1, 0])
            for tr in range(len(self.trial_mask)):
                self.grey_matter[tr,:,:,:] = gm_mask[:,:,:]
        
        
        
        
    def masked_data(self, nifti, trialsvec, selected_trs=[], mask_path=None, lag=2,
                    reverse_transpose=True, verbose=True):
        
        '''
        This function masks, transposes, and subselects the trials from the nifti
        data.
        --------
        nifti           :   a filepath to the nifti.
        trialsvec       :   numpy array denoting the response variable at the TR of the
                            trial onset.
        selected_trs    :   a list of the trs in the trial to be subselected
        mask_path       :   path to the mask (optional but recommended)
        lag             :   how many TRs to push out the trial (2 recommended)
        '''
        
        if verbose:
            if reverse_transpose:
                print 'using time-first reverse transposition of nifti matrix'
            else:
                print 'preserving dimensionality of nifti matrix (nt last)'
        
        image = load_image(nifti)
            
        if verbose:
            print 'nifti shape:', image.shape
            
        nmask = np.not_equal(self.original_mask, 0).sum()
        
        ntrs = len(selected_trs)
        
        p = np.prod(image.shape[:-1])
        
        trial_inds, response = self.parse_trialsvec(trialsvec)
        
        ntrials = len(trial_inds)
        
        if reverse_transpose:
            X = np.zeros((ntrials, ntrs, nmask))
        else:
            X = np.zeros((ntrials, nmask, ntrs))
        Y = np.zeros(ntrials)
        
        reselect_trs = [x-1 for x in selected_trs]
        
        if reverse_transpose:
            im = np.transpose(np.asarray(image), [3, 2, 1, 0])
        
            for i in range(ntrials):
                if len(im) > trial_inds[i]+reselect_trs[-1]+lag:
                    # OLD VERSION: could only do a continuous range
                    #row = im[trial_inds[i]+reselect_trs[0]+lag:trial_inds[i]+reselect_trs[-1]+1+lag].reshape((ntrs,p))
                    
                    # NEW VERSION: uses list comprehension for any index range
                    row_inds = [trial_inds[i]+lag+x for x in reselect_trs]
                    row = im[row_inds].reshape((ntrs,p))
                    
                    X[i] = row[:,self.flat_mask]
                    Y[i] = response[i]
            
        else:
            im = np.asarray(image)
            
            for i in range(ntrials):
                if im.shape[3] > trial_inds[i]+reselect_trs[-1]+lag:
                    # OLD VERSION: could only do a continuous range
                    #row = im[:,:,:,trial_inds[i]+reselect_trs[0]+lag:trial_inds[i]+reselect_trs[-1]+1+lag].reshape((p,ntrs))

                    # NEW VERSION: uses list comprehension for any index range
                    row_inds = [trial_inds[i]+lag+x for x in reselect_trs]
                    row = im[:,:,:,row_inds].reshape((p,ntrs))
                    
                    X[i] = row[self.flat_mask,:]
                    Y[i] = response[i]
            
        return X, Y
    
    
    
    def create_trial_mask(self, mask_path, ntrs, reverse_transpose=True):
        
        mask = load_image(mask_path)
        tmp_mask, self.mask_affine, tmp_shape = self.nifti.load_nifti(mask_path)
        mask = np.asarray(mask)
        
        if reverse_transpose:
            mask = np.transpose(mask.astype(np.bool), [2, 1, 0])
        else:
            mask = mask.copy().astype(np.bool)
        
        self.original_mask = mask.copy()
        
        print mask.shape
        
        self.trial_mask = np.zeros((ntrs, mask.shape[0], mask.shape[1], mask.shape[2]))
        
        for t in range(ntrs):
            self.trial_mask[t,:,:,:] = mask
            
        self.trial_mask = self.trial_mask.astype(np.bool)
        print self.trial_mask.shape
    
            

    def create_design(self, subject_dirs, nifti_name, respvec_name, selected_trs,
                      mask_path=None, lag=2, reverse_transpose=True):
        
        self.selected_trs = selected_trs
        
        self.load_niftis_fromdirs(subject_dirs, nifti_name, respvec_name)
        
        self.subject_design = {}
        for subject, [image, respvec] in self.subject_data_dict.items():
            sX, sY = self.masked_data(image, respvec, selected_trs=selected_trs,
                                      mask_path=mask_path, lag=lag, reverse_transpose=reverse_transpose)
            sX.shape = (sX.shape[0], np.prod(sX.shape[1:]))
            print 'subject X shape:', sX.shape
            self.subject_design[subject] = [np.array(sX), np.array(sY)]
            
        del(self.subject_data_dict)
            
            
    def create_design_logan_npy(self, subject_npys):
        
        self.subject_design = {}
        for npy in subject_npys:
            subject = npy.split('.')[0]
            cur_data = np.load(npy)
            sX, sY = [], []
            for ind in range(len(cur_data)):
                sY.append(cur_data[ind]['Y'])
                i_X = cur_data[ind]['X'].copy()
                i_X.shape = (i_X.shape[0]*i_X.shape[1])
                #print i_X.shape
                sX.append(i_X)
            self.subject_design[subject] = [np.array(sX), np.array(sY)]
        
        del(self.subject_data_dict)
        


            
        
        
        
        
        
        
                

                                    
                    
                
        
    
            
    
