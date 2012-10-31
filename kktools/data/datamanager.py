
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
from ..base.process import Process
from ..data.nifti import NiftiTools
from ..utilities.cleaners import glob_remove
from ..utilities.csv import CsvTools
from ..afni.functions import FractionizeMask, MaskAve, MaskDump
from ..utilities.vector import read as vecread



class LogisticData(Process):
    
    def __init__(self, variable_dict=None):
        super(LogisticData, self).__init__(variable_dict=variable_dict)
        self.csv = CsvTools()
        self.maskdump = MaskDump()
        self.logistic_data_dict = {}
        
        
                
    def maskdump(self, subject_dirs=None, functional_name=None, anatomical_name=None,
                 mask_names=None, mask_dir=None, mask_area_strs=['l','r','b'],
                 mask_area_codes=[[1,1],[2,2],[1,2]]):
        
        required_vars = {'subject_dirs':subject_dirs, 'functional_name':functional_name,
                         'anatomical_name':anatmocial_name, 'mask_names':mask_names,
                         'mask_dir':mask_dir}
        self._assign_variables(required_vars)
        if not self._check_variables(requried_vars): return False
            
        mask_paths = [os.path.join(self.mask_dir, name) for name in self.mask_names]
        
        self.maskdump.run_over_subjects(self.subject_dirs, self.functional_name,
                                        self.anatomical_name, mask_paths,
                                        mask_area_strs=mask_area_strs,
                                        mask_area_codes=mask_area_codes)
        
        
        
    def load_behavioral_csv(self, csv_path, header=True, delimiter=',', newline='\n'):
        return self.csv.read(csv_path)
        
        
    def load_subject_csvs(self, subject_dirs=None, subject_csv_name=None):
        
        required_vars = {'subject_dirs':subject_dirs, 'behavior_csv_name':behavior_csv_name}
        self._assign_variables(required_vars)
        if not self._check_variables(requried_vars): return False
                
        for dir in self.subject_dirs:
            
            subject_name = os.path.split(dir)[1]
            subject_csv_path = os.path.join(dir, self.behavior_csv_name)
            
            if not subject_name in self.logistic_data_dict:
                self.logistic_data_dict[subject_name] = {}
            
            csv_aslist = self.load_behavioral_csv(subject_csv_path)
            header = csv_aslist[0]
            
            for col, title in enumerate(header):
                if title in self.logistic_data_dict:
                    print 'duplicate header, only last used'
                self.logistic_data_dict[subject][title] = []
                for row in csv_aslist[1:]:
                    self.logistic_data_dict[subject][title].append(row[col])
            

    
    
    def load_subject_raw_tcs(self, subject_dirs=None, tmp_tc_dir='raw_tc'):
        
        required_vars = {'subject_dirs':subject_dirs}
        self._assign_variables(required_vars)
        if not self._check_variables(requried_vars): return False
        
        
        for dir in self.subject_dirs:
            rawdir = os.path.join(dir, tmp_tc_dir)
            raw_tcs = glob.glob(os.path.join(rawdir,'*.tc'))
            
            for tcfile in raw_tcs:
                subject_name, area, mask_name = os.path.split(tcfile)[1].split('_')[0:3]
                rl = vecread(tcfile, float=True)
                
                if subject_name not in self.logistic_data_dict:
                    self.logistic_data_dict[subject_name] = {area+mask_name:rl}
                else:
                    self.logistic_data_dict[subject_name][area+mask_name] = rl

    
    
    def write_logistic_data(self, filepath, data_dict):
        
        columns = {'subject':[]}
        data = []
        
        for subject in data_dict:
            for key in data_dict[subject]:
                if key not in columns:
                    columns[key] = []
        
        for subject in data_dict:
            col_lens = []
            
            for key in columns:
                # ensure cols are same length:
                if key is not 'subject':
                    if key in data_dict[subject]:
                        col_lens.append(len(data_dict[subject][key]))
            
            if len(np.unique(col_lens)) == 1:
                
                for key in columns:
                    if key is not 'subject':
                        if key in data_dict[subject]:
                            columns[key].extend(data_dict[subject][key])
                        else:
                            columns[key].extend(['' for i in range(col_lens[0])])
                    else:
                        columns['subject'].extend([subject for i in range(col_lens[0])])
        
        
        header = [key for key in columns]
        data.append(header)
        # fill csv lines:
        for i in range(len(columns['subject'])):
            data.append([columns[key][i] for key in header])
        
        self.csv.write(data, filepath)
        
        
        
        
    def binarize_Y(self, Ylist):
        uniqueY = np.unique(Ylist)
        if uniqueY > 2:
            print 'more than 2 unique values in Y matrix/column'
            return False
        elif uniqueY < 2:
            print 'not enough unique Y values (all 1 value??)'
            return False
        else:
            if (1. in uniqueY) and (0. in uniqueY):
                return np.array(Ylist)
            elif (1. in uniqueY):
                return np.array([y if y == 1. else 0. for y in Ylist])
            elif (0. in uniqueY):
                return np.array([y if y == 0. else 1. for y in Ylist])
            else:
                print '1 will code for: ', uniqueY[0]
                print '0 will code for: ', uniqueY[1]
                return np.array([1. if y == uniqueY[0] else 0. for y in Ylist])
        
        
        
    def create_XY_matrices(self, independent_vars, dependent_var, conditional_dict={},
                           filepath=None):
        Ymatrix = []
        Xmatrix = []
        self.subject_indices = {}
        
        allvars = []
        if type(independent_vars) in (list, tuple):
            allvars.extend(independent_vars)
        else:
            allvars.append(independent_vars)
            independent_vars = [independent_vars]
        allvars.append(dependent_var)

        
        self.sparse_data_dict = self.logistic_data_dict.copy()
        
        if 'subjects' in conditional_dict:
            for subject in sparse_data_dict:
                if subject not in conditional_dict['subjects']:
                    del(sparse_data_dict[subject])
        
        
        for subject in self.sparse_data_dict:
            
            if subject not in self.subject_indices:
                self.subject_indices[subject] = []
            
            var_range = []
            cond_inds = []
            
            for variable in self.sparse_data_dict[subject]:
                varlist = self.sparse_data_dict[subject][variable]
                
                if not var_range:
                    var_range = range(len(varlist))
                    
                if variable in conditional_dict:
                    cond_inds.append([i for i,x in enumerate(varlist) if x in conditional_dict[variable]])
                    
                if variable not in allvars:
                    del(self.sparse_data_dict[subject][variable])
              
                
            var_range = [ind for ind in var_range if all([ind in x for x in cond_inds])]

            for variable in self.sparse_data_dict[subject]:
                self.sparse_data_dict[subject][variable] = [v for i,v in enumerate(self.sparse_data_dict[subject][variable]) if i in var_range]
                
            
        if filepath:
            self.write_logistic_data(filepath, self.sparse_data_dict)
                    
        
        for subject in self.sparse_data_dict:
            binY = self.binarize_Y(self.sparse_data_dict[subject][dependent_var])
            for i,Yval in enumerate(binY):
                self.subject_indices[subject].append(len(Ymatrix))
                Ymatrix.append(Yval)
                xrow = []
                for var in independent_vars:
                    xrow.append(self.sparse_data_dict[subject][var][i])
                Xmatrix.append(xrow)
                
        self.X = np.array(Xmatrix)
        self.Y = np.array(Ymatrix)
                        
    
    
    
    



class DataManager(Process):
    
    def __init__(self, variable_dict=None):
        super(DataManager, self).__init__(variable_dict=variable_dict)
        self.nifti = NiftiTools()
        
            
            
    def create_niftis(self, subject_dirs=None, functional_name=None, anatomical_name=None,
                      dxyz=None, talairach_template_path=None, nifti_name=None,
                      within_subject_warp=True, to_template_warp=False):
        
        required_vars = {'subject_dirs':subject_dirs, 'functional_name':functional_name,
                         'anatomical_name':anatomical_name, 'dxyz':dxyz,
                         'nifti_name':nifti_name}
        
        self._assign_variables(required_vars)
        if not self._check_variables(requried_vars): return False
            
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
            
                
        
    def _xy_matrix_tracker(self):
        print 'X (trials) length: ', len(self.X)
        print 'Y (responses) length: ', len(self.Y)
        print 'positive responses: ', self.Y.count(1)
        print 'negative responses: ', self.Y.count(-1)
        
        
        
    def create_XY_matrices(self, downsample_type=None, with_replacement=False,
                           replacement_ceiling=None, random_seed=None, verbose=True):
        
        self.random_seed = random_seed or getattr(self,'random_seed',None)
        # downsample type, with_replacement, verbose must be set EACH TIME!
        
        if self.random_seed:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            
        self.X = []
        self.Y = []
        self.subject_trial_indices = {}
        
        if not downsample_type:
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_trial_indices[subject] = []
                
                if not with_replacement:
                    for trial, response in zip(trials, responses):
                        self.subject_trial_indices[subject].append(len(self.X))
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
                                self.subject_trial_indices[subject].append(len(self.X))
                                self.X.append(trial)
                            
                        for rep_trial in [random.sample(set, 1)[0] for i in range(upper_length-len(set))]:
                            self.subject_trial_indices[subject].append(len(self.X))
                            self.X.append(rep_trial)
                            
                    self.Y.extend([1. for x in range(upper_length)])
                    self.Y.extend([-1. for x in range(upper_length)])
        
                if verbose:
                    self._xy_matrix_tracker()
                    
        elif downsample_type == 'group':
            
            positive_trials = []
            negative_trials = []
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_trial_indices[subject] = []
                
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
                    self.subject_trial_indices[psub].append(len(self.X))
                    self.X.append(ptrial)
                    self.subject_trial_indices[nsub].append(len(self.X))
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
                            self.subject_trial_indices[sub].append(len(self.X))
                            self.X.append(trial)
                                                
                    for sub, trial in [random.sample(set, 1)[0] for i in range(upper_length-len(set))]:
                        self.subject_trial_indices[sub].append(len(self.X))
                        self.X.append(trial)
                        
                self.Y.extend([1. for x in range(upper_length)])
                self.Y.extend([-1. for x in range(upper_length)])
                
                if verbose:
                    self._xy_matrix_tracker()
                    
                    
                    
        elif downsample_type == 'subject':
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_trial_indices[subject] = []
                
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
                    del self.subject_trial_indices[subject]
                    
                else:
                    if not with_replacement:
                        for i in range(min(len(subject_positives), len(subject_negatives))):
                            self.subject_trial_indices[subject].append(len(self.X))
                            self.X.append(subject_positives[i])
                            self.subject_trial_indices[subject].append(len(self.X))
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
                                    self.subject_trial_indices[subject].append(len(self.X))
                                    self.X.append(trial)
                                
                            print upper_length
                            print len(set)
                                
                            for trial in [random.sample(set, 1)[0] for i in range(upper_length-len(set))]:
                                self.subject_trial_indices[subject].append(len(self.X))
                                self.X.append(trial)
                                
                        self.Y.extend([1. for x in range(upper_length)])
                        self.Y.extend([-1. for x in range(upper_length)])
                        
                if verbose:
                    self._xy_matrix_tracker()
                    
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
                    
            

        
    
        
        
        
        
        
        
                

                                    
                    
                
        
    
            
    