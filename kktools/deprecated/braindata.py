





class BrainData(DataManager):
    
    def __init__(self, variable_dict=None):
        super(BrainData_old, self).__init__(variable_dict=variable_dict)
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
            
        self.trial_mask = np.array(self.trial_mask)
        tms = np.shape(self.trial_mask)
        #print tms
        np.transpose(self.trial_mask, (3, 2, 1, 0))
        
        self.trial_mask.shape = (tms[3], tms[0]*tms[1]*tms[2])
            
            
    
    def create_experiment_mask(self, mask_path=None, experiment_trs=None):
        
        required_vars = {'mask_path':mask_path, 'experiment_trs':experiment_trs}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
                
        self.mask_data, self.mask_affine, self.mask_shape = self.nifti.load_nifti(self.mask_path)
        
        self.full_mask = np.zeros((self.mask_shape[0], self.mask_shape[1],
                                   self.mask_shape[2], self.experiment_trs), np.bool)
        
        for i in range(self.experiment_trs):
            self.full_mask[:,:,:,i] = self.mask_data[:,:,:]

        fms = np.shape(self.full_mask)
        np.transpose(self.full_mask, (3,2,1,0))
            
        self.full_mask.shape = (fms[3], fms[0]*fms[1]*fms[2])
        
    
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
    
    
    def normalize_nifti_data(self):
        self.create_experiment_mask()
        for subject in self.subject_data_dict:
            niftis = self.subject_data_dict[subject][0]
            norm_niftis = []
            for nifti in niftis:
                norm_niftis.append(preprocessing.normalize(nifti))
            self.subject_data_dict[subject][0] = norm_niftis
                
                
        
    
        
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
        
        self.create_experiment_mask()
        
        #for subject, [nifti_data, resp_vec] in self.subject_data_dict.items():
        
        for subject, [nifti_datas, resp_vecs] in self.subject_data_dict.items():
                        
            for nifti_data, resp_vec in zip(nifti_datas, resp_vecs):
                
                #nifti_data = np.ma.array(nifti_data, mask=self.full_mask)
                
                print 'Subselecting and masking trials for: ', subject
                
                onsetindices = np.nonzero(resp_vec)[0]
                responses = resp_vec[onsetindices]
                trials = []
                
                for i, ind in enumerate(onsetindices):
                    trs = [ind+tr+self.lag for tr in justified_trs]
                    if trs[-1] < self.experiment_trs-1:
                        raw_trial = nifti_data[:,:,:,trs]
                        #print 'raw trial shape', np.shape(raw_trial)
                        rs = np.shape(raw_trial)
                        raw_trial = np.array(raw_trial)
                        np.transpose(raw_trial, (3, 2, 1, 0))
                        raw_trial.shape = (rs[3],rs[0]*rs[1]*rs[2])
                        #print 'new raw shape', np.shape(raw_trial)
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
            
                        
    
    def cut_subject_trials(self, include_to=40):
        for subject, [trials, responses] in self.subject_design:
            self.subject_design[subject] = [trials[0:include_to], responses[0:include_to]]
            
        
        
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
                
                print trial_file, resp_file
                
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
            
                    

                    