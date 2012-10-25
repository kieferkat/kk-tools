
import os, sys
import subprocess
import shutil
import glob
from ..base.process import Process
from ..base.afni.functions import MaskAve, FractionizeMask, MaskDump
from ..utilities.cleaners import glob_remove
from ..utilities.csv import CsvTools
from ..utilities.vector import read as vecread
from ..utilities.vector import write as vecwrite
from ..utilities.vector import makevecs



class RawTimecourse(Process):
    
    def __init__(self, variable_dict=None):
        super(RawTimecourse, self).__init__(variable_dict=variable_dict)
        self.csv = CsvTools()
        self.maskdump = MaskDump()
        
    
    def make_vectors(self, vector_model_path=None, subject_dirs=None):
        required_vars = {'vector_model_path':vector_model_path,
                         'subject_dirs':subject_dirs}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        makevecs(self.subject_dirs, self.vector_model_path)
            
            
            
    def mask_dump(self, subject_dirs=None, timecourse_functional=None,
                  anatomical_name=None, mask_paths=None, mask_area_strs=['l','r','b'],
                  mask_area_codes=[[1,1],[2,2],[1,2]], verbose=True):
        
        required_vars = {'subject_dirs':subject_dirs,
                         'timecourse_functional':timecourse_functional,
                         'anatomical_name':anatomical_name, 'mask_paths':mask_paths,
                         'mask_names':mask_names, 'mask_area_strs':mask_areas,
                         'mask_area_codes':mask_area_codes}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.maskdump.run_over_subjects(self.subject_dirs, self.timecourse_functional,
                                        self.anatomical_name, self.mask_paths,
                                        mask_area_strs=mask_area_strs,
                                        mask_area_codes=mask_area_codes)
                
                
                
    def parse_tc_file(self, filepath):
        # parses the tc file for relevant info
        filename = os.path.split(filepath)[1]
        split_name = filename.split('_')
        subject, area, mask = split_name[0:3]
        
        # get out activation:
        act = vecread(filepath, float=True)
        
        return mask, area, subject, act
        
        
            
    def average_activation(self, timecourse_ouput_dir=None, subject_dirs=None,
                           onset_vectors=None, timecourse_tr_range=None,
                           tmp_tc_dirname='raw_tcs/', verbose=True):
        
        required_vars = {'timecourse_ouput_dir':timecourse_output_dir,
                         'subject_dirs':subject_dirs, 'onset_vectors':onset_vectors,
                         'timecourse_tr_range':timecourse_tr_range}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        # prefer timecourse TR range to timecourse lag:
        
        if verbose:
            print 'preparing to average activation...'
            
        # create organization dicts:
        tc_dict, onset_dict = {}, {}
        
        # iterate averaging over subjects:
        for subjdir in self.subject_dirs:
            
            if verbose:
                print 'parsing subject ', os.path.split(subjdir)[1]
                
            # find tc files:
            tc_files = glob.glob(os.path.join(subjdir, tmp_tc_dirname,'*.tc'))
            
            # define the onset files in the subject directory:
            onset_files = [os.path.join(subjdir, ov+'.1D') for ov in onset_vectors]
            
            # parse each tc file:
            for tc in tc_files:
                mask, area, subject, act = self.parse_tc_file(tc)
                
                # parse the onset files with subject name:
                onset_dict[subject] = {}
                for onset_name, vecfile in zip(self.onset_vectors, onset_files):
                    onset_dict[subject][onset_name] = vecread(vecfile, float=False)
                    
                # add to dict in appropriate section:
                if mask in tc_dict:
                    if area in tc_dict[mask]:
                        tc_dict[mask][area][subject] = act
                    else:
                        tc_dict[mask][area] = {subject:act}
                else:
                    tc_dict[mask] = {area:{subject:act}}
                    
        
        self.create_timecourse_csvs(tc_dict, onset_dict)
        
        
        
    def create_timecourse_csvs(self, tc_dict, onset_dict, timecourse_output_dir=None,
                               timecourse_tr_range=None, onset_vectors=None, verbose=True):
        
        required_vars = {'timecourse_ouput_dir':timecourse_output_dir,
                         'onset_vectors':onset_vectors,
                         'timecourse_tr_range':timecourse_tr_range}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        if verbose:
            print 'creating timecourse csvs...'
            
        try:
            os.makedirs(self.timecourse_output_dir)
        except:
            pass
        
        for mask in tc_dict:
            for area in tc_dict[mask]:
                
                subjects_dict = tc_dict[mask][area]
                subject_keys = sorted(subjects_dict.keys())
                csv_dict = {}
                
                for ov_name in self.onset_vectors:
                    if verbose:
                        print 'writing all onset vectors for vector: ', ov_name
                        
                    csv_rows = []
                    
                    for subject in subject_keys:
                        act = subjects_dict[subject]
                        
                        # accumulator tracks activation and averages:
                        accumulator = []
                        for i in range(self.timecourse_lag):
                            accumulator.append([])
                            
                        # grab number vector from onset dict:
                        nvec = onset_dict[subject][ov_name]
                        
                        # iterate over nvec indices and activation, adding to
                        # accumulator where appropriate:
                        for i, ind in enumerate(nvec):
                            if ind == 1:
                                for trind, aind, in zip(self.timecourse_tr_range, range(len(self.timecourse_tr_range))):
                                    if len(act) > i+trind:
                                        accumulator[aind].append(act[i+trind])
                                        
                        # average the accumulator:
                        for i, actlist in enumerate(accumulator):
                            accumulator[i] = sum(actlist)/len(actlist)
                            
                        # create subject csv row:
                        row = [subject].extend([str(x) for x in accumulator])
                        csv_rows.append(row)
                        
                    # write out the csv:
                    csv_name = area+mask+'_'+ov_name+'.csv'
                    csv_path = os.path.join(self.timecourse_output_dir, csv_name)
                    self.csv.write(csv_rows, csv_path)
                    self.csv.append_row_stderr(csv_path)
                    
                    
                    
    def run(self, vector_model_path=None, subject_dirs=None,
            timecourse_functional=None, anatomical_name=None, mask_area_strs=None,
            mask_area_codes=None, mask_names=None, mask_paths=None, onset_vectors=None,
            timecourse_tr_range=None, timecourse_output_dir=None,
            scripts_dir=None):
        
        required_vars = {'subject_dirs':subject_dirs, 'scripts_dir':scripts_dir,
                         'timecourse_functional':timecourse_functional,
                         'anatomical_name':anatomical_name, 'mask_areas':mask_areas,
                         'mask_area_codes':mask_area_codes,'mask_names':mask_names,
                         'onset_vectors':onset_vectors,'timecourse_tr_range':timecourse_tr_range}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.subjects = subjects or [os.path.split(s)[1] for s in self.subject_dirs]
        self.mask_paths = mask_paths or [os.path.join(self.scripts_dir,m+'+tlrc') for m in self.mask_names]
        self.timecourse_output_dir = timecourse_output_dir or os.path.join(os.path.split(self.scripts_dir)[0], 'raw_timecourses')
        
        # assume timecourse range starts with 1, not 0...
        
        if type(self.timecourse_tr_range) is int:
            self.timecourse_tr_range = range(self.timecourse_tr_range)
        elif type(self.timecourse_tr_range) in (list, tuple):
            self.timecourse_tr_range = [x-1 for x in self.timecourse_tr_range]
        else:
            print 'timecourse tr range of incorrect type (should be int or list/tuple)'
            return False

        
        if getattr(self, 'vector_model_path', None):
            self.make_vectors()
            
        self.mask_dump()
        self.average_activation()
        
        
            
                
                

        
        
        
        
        
        
        