

import os
import sys
import time
import random
from pprint import pprint

import numpy as np
import numpy.lib.format as npf

from mvpa2.suite import *

from mvpa2.base import cfg
from mvpa2.datasets.base import Dataset
from mvpa2.datasets.mri import fmri_dataset

# See other tests and test_procrust.py for some example on what to do ;)
from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.mappers.zscore import zscore
from mvpa2.misc.support import idhash

from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.featsel.helpers import FixedNElementTailSelector
from mvpa2.datasets.mri import map2nifti
from mvpa2.base.hdf5 import h5save, h5load



class MemMapCommander(object):
    def __init__(self, directory=os.getcwd(), cmatrix_name='samplesmatrix.npy',
                 features_name='featurescores.npy'):
        
        self.topdir = directory
        
        self.cmatrix_allocated = False
        self.cmatrix = None
        self.cmatrix_filename = os.path.join(self.topdir, cmatrix_name)
        
        self.features_allocated = False
        self.features = None
        self.features_filename = os.path.join(self.topdir, features_name)
        
        
    def generic_load(self, filename, fullname):
        if os.path.exists(filename):
            pointer = npf.open_memmap(filename, mode='r+', dtype='float32')
            print 'Loaded '+fullname+' memmap.'
            return pointer
        else:
            print 'Unable to load '+fullname+'.'
            return False
        
        
    def generic_delete(self, filename):
        if os.path.exists(filename):
            os.remove(self.filename)
            
        
    def check_size(self, pointer, allocation_flag, x, y):
        if allocation_flag:
            if (x,y) == pointer.shape:
                return True
            else:
                return False
        else:
            print 'Object not allocated, cannot check size.'
            return False
        
        
    def allocate(self, filename, shape):
        self.generic_delete(filename)
        return npf.open_memmap(filename, mode='w+', dtype='float32', shape=shape)
        
  
    def load_cmatrix(self):
        self.cmatrix = self.generic_load(self.cmatrix_filename, 'correlation matrix')
        if self.cmatrix is not False:
            self.cmatrix_allocated = True
        
    def load_features(self):
        self.features =  self.generic_load(self.features_filename, 'feature scores matrix')
        if self.features is not False:
            self.features_allocated = True
        
    
    def allocate_cmatrix(self, x, y):
        self.cmatrix = self.allocate(self.cmatrix_filename, (x,y))
        self.cmatrix_allocated = True
        
        
    def allocate_features(self, x, y):
        self.features = self.allocate(self.features_filename, (x,y))
        self.features_allocated = True
    
        
    def memorysafe_dot(self, A, B):
        
        base_time = time.time()
        
        if self.cmatrix_allocated:
            
            Adim = A.shape[0]
            Bdim = B.shape[1]
            
            for x in range(Adim):
                elapsed_time = (time.time()-base_time)/3600.
                completed_fraction = 1./(Adim-(x+1))
                fraction_remaining = 1-completed_fraction
                time_left = str(elapsed_time*fraction_remaining)
                print 'Arow: %s\tEst. time left:\t %s hrs' % (str(x),time_left)
                
                for y in range(Bdim):
                    x_vec = A[x,:]
                    y_vec = B[:,y]
                    dot = np.dot(x_vec, y_vec)
                    self.cmatrix[x,y] = dot
            
            return True
            
        else:
            print 'No correlation matrix allocated.'
            return False
        
        
    def assign_featurescores(self, ind1, ind2):
        
        if self.features_allocated and self.cmatrix_allocated:
            self.features[ind1] = self.features[ind1] + np.max(self.cmatrix, axis = 1)
            self.features[ind2] = self.features[ind2] + np.max(self.cmatrix, axis = 0)
        
            print 'feature scores assigned to %s, %s' % (str(ind1), str(ind2))
            return True
        
        else:
            print 'features matrix or correlation matrix not allocated. Abort!!'
            return False
        
        
    def compute_correlations(self, datasets, features_shape=None):
    
        if not self.features_allocated:
            if features_shape is not None:
                self.allocate_features(features_shape[0], features_shape[1])
        
        for i, ds1 in enumerate(datasets):
            ds1_copy = ds1.copy()
            zscore(ds1_copy, chunks_attr=None)
            ds1_samples = ds1_copy.samples
            
            for j, ds2 in enumerate(datasets[i+1:]):
                ds2_copy = ds2.copy()
                zscore(ds2_copy, chunks_attr=None)
                ds2_samples = ds2_copy.samples
                
                matrixA = ds1_samples.T
                matrixB = ds2_samples
                
                if self.cmatrix_allocated:
                    if self.check_size(self.cmatrix, self.cmatrix_allocated,
                                       matrixA.shape[0], matrixB.shape[1]):
                        print 'size ok'
                    else:
                        print 'size incorrect'
                        generic_delete(self.cmatrix_allocated, self.cmatrix_filename)
                        self.allocate_cmatrix(matrixA.shape[0], matrixB.shape[1])
                else:
                    self.allocate_cmatrix(matrixA.shape[0], matrixB.shape[1])
                    
                self.memorysafe_dot(matrixA, matrixB)
                
                self.assign_feature_scores(i, j+i+1)




class DatasetManager(object):
    
    def __init__(self, directory=os.getcwd(), passive_dsets=None,
                 active_dsets=None, mask=None,
                 active_target_files=None, passive_targets=None):
        
        self.topdir = directory
        self.passive_dsets = passive_dsets
        self.active_dsets = active_dsets
        self.mask_name = mask
        self.active_target_files = active_target_files
        self.passive_target_file = passive_targets
                
        self.tailselector = FractionTailSelector(0.05, tail='upper',
                                                 mode='select', sort=False)
        self.hyper = Hyperalignment(alignment=ProcrusteanMapper())
        self.classifier = LinearCSVMC()
        self.crossvalidator = CrossValidation(self.classifier, NFoldPartitioner(attr='subject'),
                                              errorfx=mean_match_accuracy)
        
    
    @property
    def passive_dset_paths(self):
        if self.passive_dsets:
            return [os.path.join(self.topdir, x) for x in self.passive_dsets]
        else:
            return None
        
    @property
    def active_dset_paths(self):
        if self.active_dsets:
            return [os.path.join(self.topdir, x) for x in self.active_dsets]
        else:
            return None
        
    @property
    def mask_path(self):
        if self.mask_name:
            return os.path.join(self.topdir, self.mask_name)
        else:
            return None
        
    @property
    def active_target_paths(self):
        if self.active_target_files:
            return [os.path.join(self.topdir, x) for x in self.active_target_files]
        else:
            return None
        
    @property
    def passive_target_path(self):
        if self.passive_target_file:
            return os.path.join(self.topdir, self.passive_target_file)
        else:
            return None
    
    @property
    def passive_labels(self):
        if self.passive_target_path:
            return self.parse_targets(self.passive_target_path)
        else:
            return None
    
    @property
    def active_labels(self):
        if self.active_target_paths:
            return self.parse_targets(self.active_target_paths)
        else:
            return None
        
    
    def find_active_target_files(self):
        #! This function is contingent upon underscore naming conventions
        
        if any(self.active_dset_paths):
            prefixes = [(os.path.split(path)[1]).split('_')[0] for path in self.active_dset_paths]
            suffix = '_targets.txt'
            self.active_target_files = [x+suffix for x in prefixes]
    
    
    
    def parse_targets(self, targets_paths):
        # ensure paths a list
        if type(targets_paths) != type([]):
            targets_paths = [targets_paths]
        # get column data
        cols = [ColumnData(x, header=['targets']) for x in targets_paths]
        
        if len(targets_paths) > 1:
            labels = [[int(x) for x in c['targets']] for c in cols]
        else:
            cols = cols[0]
            labels = [int(x) for x in cols['targets']]
            
        return labels
    
    
    def load_passive_datasets(self):
        print 'loading passive datasets'
        self.ds_pas = []
        for i,nifti in enumerate(self.passive_dset_paths):
            ds = (fmri_dataset(samples=nifti, targets=self.passive_labels,
                               mask=self.mask_path))
            ds.sa['subject'] = np.repeat(i, len(ds))
            self.ds_pas.append(ds)
            
            
    def load_active_datasets(self):
        print 'loading active datasets'
        self.ds_act = []
        for i,nifti in enumerate(self.active_dset_paths):
            ds = (fmri_dataset(samples=nifti, targets=self.active_labels[i],
                               mask=self.mask_path))
            ds.sa['subject'] = np.repeat(i, len(ds))
            ds.sa['split'] = np.repeat([1,2], len(ds)/2)
            remove_invariant_features(ds)
            self.ds_act.append(ds)
            
            
    def load_datasets(self, dset_paths=None, targets=None,
                      mask=None):
        dsets = []
        for i, nifti in enumerate(dset_paths):
            ds = fmri_dataset(samples=nifti, targets=targets[i], mask=mask)
            ds.sa['subject'] = np.repeat(i, len(ds))
            ds.sa['split'] = np.repeat([1,2],len(ds)/2)
            dsets.append(ds)
        return dsets
            
    
    def split_datasets(self, datasets):
        train = [ds[ds.sa.split == 1,:] for ds in datasets]
        test = [ds[ds.sa.split == 2,:] for ds in datasets]
        return [train, test]

            
    def sample_mean_datasets(self, datasets):
        return [ds.get_mapped(mean_group_sample(['targets'])) for ds in datasets]
            
            
    def zscore_standard(self, datasets):
        [zscore(ds) for ds in datasets]
        
            
    def zscore_to_rest(self, datasets):
        print 'zscoring trs relative to rest trs'
        [zscore(ds, param_est=('targets',[0]), chunks_attr='subject') for ds in datasets]
                    
            
    def slice_out_code(self, datasets, code):
        return [ds[ds.sa.targets != code] for ds in datasets]

    
    def passive_anova_featureselection(self, passive_datasets):
        print 'Initiating passive anova feature selection...'
        anova = OneWayAnova()
        anova_features = [anova(ds) for ds in passive_datasets]
        feature_selector = [StaticFeatureSelection(self.tailselector(fs)) for fs in anova_features]
        fs_datasets = [feature_selector[x].forward(ds.copy()) for x,ds in enumerate(passive_datasets)]
        print 'Passive anova feature selction complete.'
        return [feature_selector, fs_datasets]

            
    def hyperalign(self, fs_datasets):
        print 'Beginning hyperalignment'
        return self.hyper(datasets=fs_datasets)

            
    def apply_feature_selector(self, feature_selector, datasets):
        print 'applying feature selector'
        return [feature_selector[i].forward(ds.copy()) for i,ds in enumerate(datasets)]


    def apply_hyperalignment(self, hyper_mapper, datasets):
        print 'applying hyperalignment mapper'
        return [hyper_mapper[i].forward(ds.copy()) for i,ds in enumerate(datasets)]
            
    
    def crossvalidate(self, datasets):
        ds_hyper = vstack(datasets)
        zscore(ds_hyper, chunks_attr='subject')
        return self.crossvalidator(ds_hyper)
        

    def hypertest(self):
        self.all_active = self.active_dsets[:]
        self.all_targets = self.active_target_files[:]
        
        self.successes = []
        self.failures = []
        max = len(self.all_active)
        
        critical_failure = False
        current_ind = []
        
        print 'BEGINNING ITERATIVE HYPERTEST'
        index_series = range(max)
        random.shuffle(index_series)
        print index_series
        
        
        while not critical_failure and len(index_series) > 0:
            
            if len(self.successes) > 0:
                test_inds = self.successes + [index_series.pop()]
            else:
                test_inds = [index_series.pop(), index_series.pop()]
                
            print test_inds
            
            self.active_dsets = [dset for i,dset in enumerate(self.all_active) if i in test_inds]
            self.active_target_files = [file for i,file in enumerate(self.all_targets) if i in test_inds]
            
            print self.active_dsets
            print self.active_target_files
            
            try:
                self.load_active_datasets()
                
                #self.zscore_to_rest(self.ds_act)
                #self.ds_act = self.slice_out_code(self.ds_act, -1)
                #self.ds_act = self.slice_out_code(self.ds_act, 0)
                
                
                [zscore(ds, chunks_attr='subject') for ds in self.ds_act]
                
                [self.feature_selector, self.ds_act_fs] = self.passive_anova_featureselection(self.ds_act)
                self.hyper_mapper = self.hyperalign(self.ds_act_fs)
                
                self.successes.extend([ind for ind in test_inds if ind not in self.successes])
                for ind in test_inds:
                    if ind in self.failures:
                        self.failures.remove(ind)
                        
                print '\nSUCCESS\n'
            
                
            except Exception, error:
                if len(self.successes) == 0:
                    self.failures.extend([ind for ind in test_inds if ind not in self.failures])
                    index_series = test_inds + index_series
                else:
                    self.failures.append(test_inds[-1])
                    index_series = [test_inds[-1]] + index_series
                print '\nFAILURE\n'
                pprint(error)
            
            
            print 'successful subjects:'
            print [dset for i,dset in enumerate(self.all_active) if i in self.successes]
            print 'failed subjects:'
            print [dset for i,dset in enumerate(self.all_active) if i in self.failures]
            
            self.ds_act = []
            self.feature_selector = []
            self.ds_act_fs = []
            self.hyper_mapper = []
                
    
    
    
        
        
    def hyperalignment_graphs(self):
        self.load_active_datasets()
        self.ds_act = self.slice_out_code(self.ds_act, -1)
        self.zscore_to_rest(self.ds_act)
        self.ds_act = self.slice_out_code(self.ds_act, 0)
        print self.ds_act[0].shape
        
        # SPLIT??
        #[self.ds_act, self.ds_test] = self.split_datasets(self.ds_act)
        
        [feature_selector, ds_fs] = self.passive_anova_featureselection(self.ds_act)
        
        print 'doing hyperalignment'
        mappers = self.hyperalign(ds_fs)
        
        print 'applying hyperalignment'
        ds_hyper = [mappers[i].forward(ds.copy()) for i,ds in enumerate(ds_fs)]
        
        print 'doing graph calcs'
        sm_orig = [np.corrcoef(ds.get_mapped(mean_group_sample(['targets'])).samples)
                   for ds in ds_fs]
        
        print np.shape(sm_orig[0])
        print np.shape(sm_orig[1])
        
        sm_orig_mean = np.mean(sm_orig, axis=0)
        
        sm_hyper_mean = np.mean([np.corrcoef(ds.get_mapped(mean_group_sample(['targets'])).samples)
                                 for ds in ds_hyper], axis=0)
        
        ds_hyper = vstack(ds_hyper)
        sm_hyper = np.corrcoef(ds_hyper.get_mapped(mean_group_sample(['targets'])))
        
        ds_fs = vstack(ds_fs)
        sm_anat = np.corrcoef(ds_fs.get_mapped(mean_group_sample(['targets'])))
        
        print 'time to draw the graph'
        labels = self.ds_act[0].UT
        pl.figure(figsize=(6,6))
        
        for i, sm_t in enumerate((
            (sm_orig_mean, "Average within-subject\nsimilarity"),
            (sm_anat, "Similarity of group average\ndata (anatomically aligned)"),
            (sm_hyper_mean, "Average within-subject\nsimilarity (hyperaligned data)"),
            (sm_hyper, "Similarity of group average\ndata (hyperaligned)"),
                      )):
            
            sm, title = sm_t
            pl.subplot(2,2,i+1)
            pl.imshow(sm, vmin=-1.0, vmax=1.0, interpolation='nearest')
            pl.colorbar(shrink=.4, ticks=[-1,0,1])
            pl.title(title, size=12)
            ylim = pl.ylim()
            pl.xticks(range(len(labels)), labels, size='small', stretch='ultra-condensed',
                      rotation=45)
            pl.yticks(range(len(labels)), labels, size='small', stretch='ultra-condensed',
                      rotation=45)
            pl.ylim(ylim)

        
        pl.show()

    def run(self):
        
        # load datasets and initialize target codes per TR:

        self.load_active_datasets()
        choice_labels = self.parse_targets([os.path.join(self.topdir, x) for x in self.choice_target_files])
        self.ds_test = self.load_datasets(dset_paths=self.active_dset_paths,
                                          targets=choice_labels, mask=self.mask_path)
        
        
        #self.ds_act = self.slice_out_code(self.ds_act, -1)
        #self.zscore_to_rest(self.ds_act)
        [zscore(ds, chunks_attr='subject') for ds in self.ds_act]
        
        self.ds_test = self.slice_out_code(self.ds_test, -1)
        self.zscore_to_rest(self.ds_test)
        
        
        #self.ds_act = self.slice_out_code(self.ds_act, 0)
        #print self.ds_act[0].shape
        
        self.ds_test = self.slice_out_code(self.ds_test, 0)
        print self.ds_test[0].shape
        
        self.ds_train = self.ds_act
        
        [self.feature_selector, self.train_fs] = self.passive_anova_featureselection(self.ds_train)
        self.hyper_mapper = self.hyperalign(self.train_fs)
        self.test_fs = self.apply_feature_selector(self.feature_selector, self.ds_test)
        self.test_hyper = self.apply_hyperalignment(self.hyper_mapper, self.test_fs)
        
        
        
        # crossvalidate the active hyperaligned data:
        cv_results = self.crossvalidate(self.test_hyper)
        
        print 'avg. between-subject classification accuracy hyperaligned: %.2f +/-%.3f' \
            % ( np.mean(cv_results), np.std(np.mean(cv_results, axis=1))/np.sqrt(len(self.ds_act)-1))
        

        # preform standard btwn subject analysis:
        sensitivity = SensitivityBasedFeatureSelection(OneWayAnova(), self.tailselector,
                                                       enable_ca=['sensitivities'])
        auto_featselect = FeatureSelectionClassifier(self.classifier, sensitivity)
        
        mni = vstack(self.ds_test)
        cv = CrossValidation(auto_featselect, NFoldPartitioner(attr='subject'),
                             errorfx=mean_match_accuracy)
        cv_results_mni = cv(mni)
        
        print 'avg. between-subject classification accuracy regular: %.2f +/-%.3f' \
            % ( np.mean(cv_results_mni), np.std(np.mean(cv_results_mni, axis=1))/np.sqrt(len(self.ds_act)-1))
        
        
        
    
    
if __name__ == '__main__':
    
    passive = ['ar_pas.nii','dd_pas.nii','as_pas.nii','dh_pas.nii','jb_pas.nii']
    #active = ['ar_act.nii','dd_act.nii','as_act.nii','dh_act.nii','jb_act.nii']
    
    
    #ALL:
    
    active = ['ar110_act.nii','as090_act.nii','as092_act.nii','dd083_act.nii',
              'dh110_act.nii','dk091_act.nii','gb090_act.nii','gb092_act.nii',
              'jb092_act.nii','jc092_act.nii','jr102_act.nii','ju083_act.nii',
              'kk102_act.nii','kr091_act.nii','ms101_act.nii','oo103_act.nii',
              'pl103_act.nii','ta092_act.nii','ti092_act.nii','tt092_act.nii',
              'zw092_act.nii']
    
    
    
    #TARGETS
    '''
    active = ['as090_act.nii', 'dd083_act.nii', 'jc092_act.nii', 'jr102_act.nii',
              'ju083_act.nii', 'kk102_act.nii', 'oo103_act.nii', 'zw092_act.nii']
    '''
    
    #FULL:
    #active = ['as090_act.nii', 'dh110_act.nii', 'gb090_act.nii', 'jc092_act.nii',
    #          'jr102_act.nii', 'kr091_act.nii', 'ta092_act.nii', 'ti092_act.nii']
    
    #ANT:
    #active = ['as090_act.nii', 'dk091_act.nii', 'gb092_act.nii', 'oo103_act.nii', 'pl103_act.nii']

    
    mask = 'mask.nii'
    
    atargets = [a.split('_')[0]+'_targets.txt' for a in active]
    chtargets = [a.split('_')[0]+'_choice.txt' for a in active]
    typetargets = [a.split('_')[0]+'_ttype.txt' for a in active]
    fulltargets = [a.split('_')[0]+'_full.txt' for a in active]
    ant = [a.split('_')[0]+'_ant.txt' for a in active]
    
    alag = [a.split('_')[0]+'_targets_lag.txt' for a in active]
    chlag = [a.split('_')[0]+'_choice_lag.txt' for a in active]
    typelag = [a.split('_')[0]+'_ttype_lag.txt' for a in active]
    fulllag = [a.split('_')[0]+'_full_lag.txt' for a in active]
    antlag = [a.split('_')[0]+'_ant_lag.txt' for a in active]
    
    #atargets = ['ar_targets.txt','dd_targets.txt','as_targets.txt','dh_targets.txt','jb_targets.txt']
    #chtargets = ['ar_choice.txt','dd_choice.txt','as_choice.txt','dh_choice.txt','jb_choice.txt']
    #typetargets = ['ar_ttype.txt','dd_ttype.txt','as_ttype.txt','dh_ttype.txt','jb_ttype.txt']
    ptarget = 'ptargets.txt'
    
    dsman = DatasetManager(passive_dsets=passive, active_dsets=active, mask=mask,
                           active_target_files=antlag, passive_targets=ptarget)
    
    dsman.choice_target_files = chlag
    
    
    dsman.hypertest()
    
    #dsman.run()
    
    #dsman.hyperalignment_graphs()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    