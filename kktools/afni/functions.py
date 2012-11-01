
import os, sys
import glob
import subprocess
import shutil
from ..base.process import Process





class AfniFunction(Process):
    
    
    def __init__(self, variable_dict=None):
        super(AfniFunction, self).__init__(variable_dict=variable_dict)
        
    
    def _clean_remove(self, files):
        for file in files:
            try:
                os.remove(file)
            except:
                pass
    
    def _clean(self, glob_prefix, type='standard', path=None):
        if path:
            prefix = os.path.join(path, glob_prefix)
        else:
            prefix = glob_prefix
            
        if type is 'standard':
            self._clean_remove(glob.glob(prefix))
            
        elif type is 'orig':
            self._clean_remove(glob.glob(prefix+'+orig*'))

        elif type is 'tlrc':
            self._clean_remove(glob.glob(prefix+'+tlrc*'))
            
        elif type is 'afni':
            self._clean_remove(glob.glob(prefix+'+orig*'))
            self._clean_remove(glob.glob(prefix+'+tlrc*'))
    
    
    
    
    
class MaskAve(AfniFunction):
    
    def __init__(self, variable_dict=None):
        super(MaskAve, self).__init__(variable_dict=variable_dict)
        
        
    def run(self, mask_path, dataset_path, mask_area_strs=['l','r','b'],
            mask_area_codes=[[1,1],[2,2],[1,2]], tmp_tc_dir='raw_tc'):
        
        subject_path = os.path.split(dataset_path)[0]
        subject_name = os.path.split(subject_path)[1]
        
        mask_name = os.path.split(mask_path)[1].split('+')[0]
        
        print 'maskave', subject_name, mask_name
        
        tmpdir = os.path.join(subject_path, tmp_tc_dir)
        try:
            os.makedirs(tmpdir)
        except:
            pass
        
        for area, codes in zip(mask_area_strs, mask_area_codes):
            
            raw_tc = '_'.join([subject_name, area, mask_name, 'raw.tc'])
            raw_tc = os.path.join(tmpdir, raw_tc)
            
            self._clean(raw_tc)
            
            cmd = ['3dmaskave', '-mask', mask_path, '-quiet', '-mrange',
                   str(codes[0]), str(codes[1]), dataset_path]
            
            fcontent = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            fcontent.wait()
            fcontent = fcontent.communicate()[0]
            
            fid = open(raw_tc, 'w')
            fid.write(fcontent)
            fid.close()
            
    
    
    
class FractionizeMask(AfniFunction):
    
    def __init__(self, variable_dict=None):
        super(FractionizeMask, self).__init__(variable_dict=variable_dict)
        
        
    def run(self, mask_path, dataset_path, anat_path, subject_mask_suffix='r'):
        
        subject_path = os.path.split(dataset_path)[0]
        mask_name = (os.path.split(mask_path)[1]).split('+')[0]
        
        subject_mask = os.path.join(subject_path, mask_name+subject_mask_suffix+'+orig')
        self._clean(subject_mask+'*')
        
        cmd = ['3dfractionize', '-template', dataset_path, '-input', mask_path,
           '-warp', anat_path, '-clip', '0.1', '-preserve', '-prefix',
           subject_mask]
        
        subprocess.call(cmd)
        return subject_mask
        
        
        
class MaskDump(AfniFunction):
    
    def __init__(self, variable_dict=None):
        super(MaskDump, self).__init__(variable_dict=variable_dict)
        self.fractionize = FractionizeMask(variable_dict=variable_dict)
        self.maskave = MaskAve(variable_dict=variable_dict)
        
        
    def run(self, dataset_path, anat_path, mask_paths, mask_area_strs=['l','r','b'],
            mask_area_codes=[[1,1],[2,2],[1,2]]):
        
        subject_path = os.path.split(dataset_path)[0]
        
        if type(mask_paths) in (list, tuple):
            for mask in mask_paths:
                subject_mask = self.fractionize.run(mask, dataset_path, anat_path)
                self.maskave.run(subject_mask, dataset_path, mask_area_strs=mask_area_strs,
                                 mask_area_codes=mask_area_codes)
                
        else:
            subject_mask = self.fractionize.run(mask_paths, dataset_path, anat_path)
            self.maskave.run(subject_mask, dataset_path, mask_area_strs=mask_area_strs,
                             mask_area_codes=mask_area_codes)
        
        
    def run_over_subjects(self, subjdirs, dataset_name, anat_name, mask_paths,
                          mask_area_strs=['l','r','b'], mask_area_codes=[[1,1],[2,2],[1,2]]):
        
        for dir in subjdirs:
            dataset_path = os.path.join(dir, dataset_name)
            anat_path = os.path.join(dir, anat_name)
            
            self.run(dataset_path, anat_path, mask_paths, mask_area_strs=mask_area_strs,
                     mask_area_codes=mask_area_codes)
        
        
    
        
        
        
    