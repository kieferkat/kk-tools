
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
from process import Process
from nifti import NiftiTools
from ..utilities.cleaners import glob_remove
from ..utilities.csv import CsvTools
from ..afni.functions import FractionizeMask, MaskAve, MaskDump
from ..utilities.vector import vecread
from ..utilities.vector import subject_vector_dict as make_vector_dict



            

        

class DataManager(Process):
    
    def __init__(self, variable_dict=None):
        super(DataManager, self).__init__(variable_dict=variable_dict)
        self.nifti = NiftiTools()
        
            
            
    def recode_variable(self, var_list, oldnew_valdict, allow_unspecified=True,
                        as_string=False):
        
        recoded = []
        for var in var_list:
            if var in oldnew_valdict.keys():
                nval = oldnew_valdict[var]
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
                
                      
        
    def _xy_matrix_tracker(self):
        print 'X (trials) length: ', len(self.X)
        print 'Y (responses) length: ', len(self.Y)
        print 'positive responses: ', self.Y.count(1)
        print 'negative responses: ', self.Y.count(-1)
        
        
        
    def create_XY_matrices(self, subject_design=None, downsample_type=None, with_replacement=False,
                           replacement_ceiling=None, random_seed=None, Ybinary=[1.,-1.], verbose=True):
        
        
        required_vars = {'subject_design':subject_design}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        
        self.random_seed = random_seed or getattr(self,'random_seed',None)
        
        
        if self.random_seed:
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
                        self.Y.append(response)
                        
                elif with_replacement:
                    positive_trials = []
                    negative_trials = []
                    
                    for trial, response in zip(trials, responses):
                        if response > 0:
                            positive_trials.append(trial)
                        elif response < 0:
                            negative_trials.append(trial)
                    
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
                            
                    self.Y.extend([1. for x in range(upper_length)])
                    self.Y.extend([-1. for x in range(upper_length)])
        
                if verbose:
                    self._xy_matrix_tracker()
                    
                    
                    
        elif downsample_type == 'group':
            
            positive_trials = []
            negative_trials = []
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_indices[subject] = []
                
                for trial, response, in zip(trials, responses):
                    if response > 0:
                        positive_trials.append([subject,trial])
                    elif response < 0:
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
                    self.Y.extend([1.,-1.])
                    
                if verbose:
                    self._xy_matrix_tracker()
                    
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
                        
                self.Y.extend([1. for x in range(upper_length)])
                self.Y.extend([-1. for x in range(upper_length)])
                
                if verbose:
                    self._xy_matrix_tracker()
                    
                    
                    
        elif downsample_type == 'subject':
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_indices[subject] = []
                
                subject_positives = []
                subject_negatives = []
                
                for trial, response in zip(trials, responses):
                    if response > 0:
                        subject_positives.append(trial)
                    elif response < 0:
                        subject_negatives.append(trial)
                        
                random.shuffle(subject_positives)
                random.shuffle(subject_negatives)
                
                if min(len(subject_positives), len(subject_negatives)) == 0:
                    del self.subject_indices[subject]
                    
                else:
                    if not with_replacement:
                        for i in range(min(len(subject_positives), len(subject_negatives))):
                            self.subject_indices[subject].append(len(self.X))
                            self.X.append(subject_positives[i])
                            self.subject_indices[subject].append(len(self.X))
                            self.X.append(subject_negatives[i])
                            self.Y.extend([1.,-1.])
                            
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
                                
                        self.Y.extend([1. for x in range(upper_length)])
                        self.Y.extend([-1. for x in range(upper_length)])
                        
                if verbose:
                    self._xy_matrix_tracker()
                    
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        
        



class CsvData(DataManager):
    
    def __init__(self, variable_dict=None):
        super(CsvData, self).__init__(variable_dict=variable_dict)
        self.csv = CsvTools()
        self.maskdump = MaskDump(variable_dict=variable_dict)
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
        
        self.subject_vector_dict = make_vector_dict(self.subject_dirs, tmp_tc_dir)
        


    def merge_in_vector_dict(self, subject_vector_dict=None, bysubject_data_dict=None):
        
        required_vars = {'subject_vector_dict':subject_vector_dict,
                         'bysubject_data_dict':bysubject_data_dict}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.bysubject_data_dict = self.csv.merge_csv_dicts(self.bysubject_data_dict,
                                                            self.subject_vector_dict,
                                                            keylevel=1)

    
    
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
        
        out = [i for i,x in enumerate(datalist) if x in condval]
        return out
        
    
    def _slice_conditional_inds(self, indslist):
        basis_inds = indslist[0]
        for inds in indslist:
            basis_inds = [x for x in basis_inds if x in inds]
        return basis_inds
    
                
    def cut_data_dict(self, keep_only_where_dict):
        
        self.sparse_data_dict = self.bysubject_data_dict.copy()
        
        for spvar, spvals in keep_only_where_dict.items():
            
            if not type(spvals) in (list, tuple):
                spvals = [spvals]
                
            for subject, variables in self.bysubject_data_dict.items():
                cinds = []
                
                for variable in variables:
                    if variable == spvar:
                        cinds.append(self._find_inds_where(self.bysubject_data_dict[subject][variable], spvals))
                        
                basis_inds = self._slice_conditional_inds(cinds)
                
                
                for v in variables:
                    self.sparse_data_dict[subject][v] = [x for i,x in enumerate(self.sparse_data_dict[subject][v]) if i in basis_inds]
                    
    
    def _get_vars_by_conds(self, subject, variable, conditionals):
        
        nvars = []
        
        for cvar, conds in conditionals.items():
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
    
    def __init__(self, variable_dict=None):
        super(BrainData, self).__init__(variable_dict=variable_dict)
        self.nifti = NiftiTools()
        
            
            
    def create_niftis(self, subject_dirs=None, functional_name=None, anatomical_name=None,
                      dxyz=None, talairach_template_path=None, nifti_name=None,
                      within_subject_warp=True, to_template_warp=False):
        
        required_vars = {'subject_dirs':subject_dirs, 'functional_name':functional_name,
                         'anatomical_name':anatomical_name, 'dxyz':dxyz,
                         'nifti_name':nifti_name}
        
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
            
        self.talairach_template_path = talairach_template_path or getattr(self,'talairach_template_path',None)
        if not self.nifti_name.endswith('.nii'):
            self.nifti_name = self.nifti_name+'.nii'
        
        self.nifti.create_talairach_niftis(self.subject_dirs, self.functional_name,
                                           self.anatomical_name, self.dxyz,
                                           self.talairach_template_path, self.nifti_name,
                                           within_subject_warp, to_template_warp)
    
        
    def create_trial_mask(self, mask_path=None, selected_trs=None):
        
        required_vars = {'mask_path':mask_path, 'selected_trs':selected_trs}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.mask_data, self.mask_affine, self.mask_shape = self.nifti.load_nifti(self.mask_path)
        
        self.trial_mask = np.zeros((self.mask_shape[0], self.mask_shape[1], self.mask_shape[2],
                                    len(self.selected_trs)), np.bool)
                
        for i in range(len(self.selected_trs)):
            self.trial_mask[:,:,:,i] = self.mask_data[:,:,:]
            
            
    
    def create_experiment_mask(self, mask_path=None, experiment_trs=None):
        
        required_vars = {'mask_path':mask_path, 'experiment_trs':experiment_trs}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
                
        self.mask_data, self.mask_affine, self.mask_shape = self.nifti.load_nifti(self.mask_path)
        
        self.full_mask = np.zeros((self.mask_shape[0], self.mask_shape[1],
                                   self.mask_shape[2], self.experiment_trs), np.bool)
        
        for i in range(self.experiment_trs):
            self.full_mask[:,:,:,i] = self.mask_data[:,:,:]
        
        
    
    def parse_vector(self, vector_path, verbose=False):
        vfid = open(vector_path, 'rb')
        lines = vfid.readlines()
        vfid.close()
        vec = np.zeros(len(lines))
        
        for i in range(len(lines)):
            vl = int(lines[i].strip('\n'))
            if vl == 1 or vl == -1:
                vec[i] = vl
        
        if verbose:
            print vec
                
        return vec
    
    
        
    def load_nifti_data(self, subject_dirs=None, nifti_name=None, response_vector=None):
        
        required_vars = {'subject_dirs':subject_dirs, 'nifti_name':nifti_name,
                         'response_vector':response_vector}
        
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return None
            
        self.subject_data_dict = {}
            
        # iterate through niftis
        for subject in self.subject_dirs:
            
            if type(self.nifti_name) in (list, tuple) and type(self.response_vector) in (list, tuple):
                for nifti, respvec in zip(self.nifti_name, self.response_vector):
                    
                    if not nifti.endswith('.nii'):
                        nifti = nifti+'.nii'
                    
                    self.nifti_adder(subject, nifti, respvec)
                    
            else:
                
                if not self.nifti_name.endswith('.nii'):
                    nifti_name = self.nifti_name+'.nii'
                    
                self.nifti_adder(subject, nifti_name, response_vector)
            
            
                
    
    def nifti_adder(self, dir, nifti, vector, suffix=''):
        
        nifti = os.path.join(dir, nifti)
        vec = os.path.join(dir, vector)
        
        if not os.path.exists(nifti):
            print 'not found: ', nifti
        elif not os.path.exists(vec):
            print 'not found: ', vec
        else:
            
            respvec = self.parse_vector(vec)
            pprint(nifti)
            idata, affine, ishape = self.nifti.load_nifti(nifti)
            
            if getattr(self, 'raw_affine', None) is None:
                pprint(affine)
                self.raw_affine = affine
                
            if getattr(self, 'raw_data_shape', None) is None:
                pprint(ishape)
                self.raw_data_shape = ishape
                
            if getattr(self, 'experiment_trs', None) is None:
                print 'experiment trs: ', ishape[3]
                self.experiment_trs = ishape[3]
                
            subject_key = os.path.split(dir)[1]+suffix
            print 'appending raw data for subject: ', subject_key
            
            if not subject_key in self.subject_data_dict:
                self.subject_data_dict[subject_key] = [[np.array(idata)], [respvec]]
            else:
                self.subject_data_dict[subject_key][0].append(np.array(idata))
                self.subject_data_dict[subject_key][1].append(respvec)
    
 
    
    def free_nifti_data(self):
        self.subject_data_dict = None
        del(self.subject_data_dict)
        gc.collect()
        
    
    
    def subselect_data(self, selected_trs=None, trial_mask=None, lag=None, delete_data_dict=True,
                       verbose=True):
        
        required_vars = {'selected_trs':selected_trs, 'trial_mask':trial_mask,
                         'lag':lag}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        justified_trs = [x-1 for x in self.selected_trs]
        
        self.subject_design = {}
        
        print 'lag: ', self.lag
        print 'experiment trs', self.experiment_trs
        
        #for subject, [nifti_data, resp_vec] in self.subject_data_dict.items():
        
        for subject, [nifti_datas, resp_vecs] in self.subject_data_dict.items():
                        
            for nifti_data, resp_vec in zip(nifti_datas, resp_vecs):
                
                print 'Subselecting and masking trials for: ', subject
                
                onsetindices = np.nonzero(resp_vec)[0]
                responses = resp_vec[onsetindices]
                trials = []
                
                for i, ind in enumerate(onsetindices):
                    trs = [ind+tr+self.lag for tr in justified_trs]
                    if trs[-1] < self.experiment_trs-1:
                        raw_trial = nifti_data[:,:,:,trs]
                        trials.append(raw_trial[self.trial_mask])
                    else:
                        if verbose:
                            print 'left trial out', ind
                        responses = responses[0:i]
                
                if not subject in self.subject_design:
                    self.subject_design[subject] = [trials, responses]
                    
                else:
                    self.subject_design[subject][0].extend(trials)
                    self.subject_design[subject][1] = np.append(self.subject_design[subject][1], responses)
                
            self.subject_design[subject][0] = np.array(self.subject_design[subject][0])
            
            if verbose:
                print len(self.subject_design[subject][0])
                print len(self.subject_design[subject][1])
            
                
            if delete_data_dict:
                print 'Deleting data_dict entry for: ', subject
                # free some memory:
                self.subject_data_dict[subject] = None
                gc.collect()
        
        if delete_data_dict:
            self.free_nifti_data()
            
        
        
    def save_numpy_data(self, save_directory=None, suffix=None):
        
        required_vars = {'save_directory':save_directory}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        if suffix is None:
            try:
                suffix = '_lag'+str(self.lag)+'_trs'+str(len(self.selected_trs))
            except:
                suffix = ''
                
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            
        for subject, [trials, responses] in self.subject_design.items():
            
            print 'Saving numpy trials and responses for subject: ', subject
            
            trial_file = os.path.join(self.save_directory, subject+'_trials'+suffix+'.npy')
            resp_file = os.path.join(self.save_directory, subject+'_respvec'+suffix+'.npy')
            
            try:
                os.remove(trial_file)
            except:
                pass
            try:
                os.remove(resp_file)
            except:
                pass
                
            np.save(trial_file, trials)
            np.save(resp_file, responses)
            
        affine_file = os.path.join(self.save_directory, 'raw_affine'+suffix+'.npy')
        try:
            os.remove(affine_file)
        except:
            pass
        np.save(affine_file, self.raw_affine)
            
            
            
    def load_numpy_data(self, subjects=None, save_directory=None, suffix=None):
        
        required_vars = {'subjects':subjects, 'save_directory':save_directory}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.prediction_tr_length = getattr(self, 'prediction_tr_length', None) or len(self.selected_trs)
        
        if suffix is None:
            try:
                suffix = '_lag'+str(self.lag)+'_trs'+str(self.prediction_tr_length)
            except:
                suffix = ''
        
        self.subject_design = {}
        
        for subject in self.subjects:
            
            print 'Loading numpy trials and responses for subject: ', subject
            
            try:
                trial_file = os.path.join(self.save_directory, subject+'_trials'+suffix+'.npy')
                resp_file = os.path.join(self.save_directory, subject+'_respvec'+suffix+'.npy')
            
                trials = np.load(trial_file)
                responses = np.load(resp_file)
                
                self.subject_design[subject] = [trials, responses]
            except:
                print 'there was an error trying to load data & responses for subject: ', subject
            
        affine_file = os.path.join(self.save_directory, 'raw_affine'+suffix+'.npy')
        
        if not os.path.exists(affine_file):
            print 'No raw affine file found (suffix issue?), problematic for exporting maps.'
        else:
            self.raw_affine = np.load(affine_file)
            
                    

                    
            
        
        
        
        
        
        
                

                                    
                    
                
        
    
            
    