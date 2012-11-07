
import os, sys
import subprocess
import shutil
import glob

from ..base.process import Process
from ..afni.functions import AfniWrapper
from ..base.scriptwriter import Scriptwriter
from ..utilities.cleaners import glob_remove
from ..defaults.lab_standard import preprocessing as preprocessing_defaults


class Preprocessor(Process):
    
    def __init__(self, variable_dict=None):
        super(Preprocessor, self).__init__(variable_dict=variable_dict)
        self._apply_defaults(preprocessing_defaults)
        self.script_name = 'preprocess'
        self.scriptwriter = Scriptwriter()
        self.afni = AfniWrapper()
        self.run_script = True
        self.write_script = True
        
        
        
    def _update_dset(self, suffix):
        if not getattr(self, 'current_functional', None):
            self.prior_functional = self.functional_name
            if suffix:
                self.current_functional = self.functional_name+'_'+suffix
            else:
                self.current_functional = self.functional_name
        else:
            self.prior_functional = self.current_functional
            self.current_functional = self.current_functional+suffix
            
    
    
    def convert_anatomical(self, subject_dir, nifti_name, anatomical_name=None):

        required_vars = {'anatomical_name':anatomical_name}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        print nifti_name, subject_dir
        nifti_path = os.path.join(subject_dir, nifti_name)
        anat_path = os.path.join(subject_dir, self.anatomical_name)
        
        if self.run_script:
            self.afni.copy3d(nifti_path, anat_path)
        
        if self.write_script:
            self.afni.copy3d.write(self.scriptwriter, nifti_name, self.anatomical_name)
            
        
        
        
    def cutoff_buffer(self, subject_dir, nifti_name, nifti_trs, leadin=None,
                      leadout=None, prefix='epi', write_cmd_only=False):
        
        required_vars = {'leadin':leadin, 'leadout':leadout}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        nifti_path = os.path.join(subject_dir, nifti_name)
        epi_path = os.path.join(subject_dir, prefix)
        
        if self.run_script:
            self.afni.tcatbuffer(nifti_path, epi_path, self.leadin, nifti_trs-self.leadout-1)
        
        if self.write_script:
            self.afni.tcatbuffer.write(self.scriptwriter, nifti_name, prefix, self.leadin,
                                       nifti_trs-self.leadout-1, cmd_only=write_cmd_only)
                
                
        
        
    def refit(self, subject_dir, epi_name, tr_length=None, write_cmd_only=False):
        
        required_vars = {'tr_length':tr_length}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        epi_path = os.path.join(subject_dir, epi_name)
        
        if self.run_script:
            self.afni.refit(epi_path, self.tr_length)
        
        if self.write_script:
            self.afni.refit.write(self.scriptwriter, epi_name, tr_length,
                                  cmd_only=write_cmd_only)
            
        
        
    def tshift(self, subject_dir, epi_name, tshift_slice=None, tpattern=None,
               prefix='epits', write_cmd_only=False):
        
        required_vars = {'tshift_slice':tshift_slice, 'tpattern':tpattern}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        epi_path = os.path.join(subject_dir, epi_name)
        tshift_path = os.path.join(subject_dir, prefix)
        
        if self.run_script:
            self.afni.tshift(epi_path+'+orig', tshift_path, self.tshift_slice,
                             self.tpattern)
        
        if self.write_script:
            self.afni.tshift.write(self.scriptwriter, epi_name, prefix,
                                   self.tshift_slice, self.tpattern,
                                   cmd_only=write_cmd_only)
        
                 
        
        
    def concatenate(self, subject_dir, epi_names, functional_name=None):
        
        required_vars = {'functional_name':functional_name}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        epi_paths = [os.path.join(subject_dir, epi) for epi in epi_names]
        functional_path = os.path.join(subject_dir, self.functional_name)
        
        if self.run_script:
            self.afni.tcatdsets(epi_paths, functional_path)
        
        if self.write_script:
            self.afni.tcatdsets.write(self.scriptwriter, epi_names,
                                      self.functional_name)
            
    
            
            
    def volreg(self, subject_dir, motionfile_name=None, functional_name=None,
               volreg_base=None, suffix='m'):
        
        required_vars = {'motionfile_name':motionfile_name,
                         'functional_name':functional_name,
                         'volreg_base':volreg_base}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self._update_dset(suffix)
        
        prior_path = os.path.join(subject_dir, self.prior_functional+'+orig')
        motion_path = os.path.join(subject_dir, self.motionfile_name)
        mfunc_path = os.path.join(subject_dir, self.current_functional)
        
        if self.run_script:
            self.afni.volreg(prior_path, mfunc_path, motion_path, self.volreg_base)
        
        if self.write_script:
            self.afni.volreg.write(self.scriptwriter, self.prior_functional, suffix,
                                   self.motionfile_name, self.volreg_base)
            

        
        
    def smooth(self, subject_dir, blur_kernel=None, functional_name=None,
               suffix='b'):
        
        required_vars = {'blur_kernel':blur_kernel,
                         'functional_name':functional_name}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self._update_dset(suffix)
        
        prior_path = os.path.join(subject_dir, self.prior_functional+'+orig')
        blur_path = os.path.join(subject_dir, self.current_functional)
        
        if self.run_script:
            self.afni.smooth(prior_path, blur_path, self.blur_kernel)
            
        if self.write_script:
            self.afni.smooth.write(self.scriptwriter, self.prior_functional, suffix,
                                   self.blur_kernel)
        
        
        
        
    def normalize(self, subject_dir, functional_trs, suffix='n', ave_suffix='_ave',
                normalize_expression='((a-b)/b)*100'):
        
        required_vars = {'normalize_expression':normalize_expression}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self._update_dset(suffix)
        
        prior_path = os.path.join(subject_dir, self.prior_functional+'+orig')
        ave_path = os.path.join(subject_dir, self.prior_functional+ave_suffix)
        norm_path = os.path.join(subject_dir, self.current_functional)
        
        if self.run_script:
            self.afni.normalize(prior_path, ave_path, norm_path, functional_trs-1)
        
        if self.write_script:
            self.afni.normalize.write(self.scriptwriter, self.prior_functional,
                                      suffix, ave_suffix, functional_trs,
                                      self.normalize_expression)
        
        
        
    def highpass_filter(self, subject_dir, highpass_value=None, functional_name=None,
                        suffix='f'):
        
        required_vars = {'highpass_value':highpass_value,
                         'functional_name':functional_name}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self._update_dset(suffix)
        
        prior_path = os.path.join(subject_dir, self.prior_functional+'+orig')
        filter_path = os.path.join(subject_dir, self.current_functional)
        
        self.afni.highpassfilter(prior_path, filter_path, self.highpass_value)
        
        
        
        
    def talairach_warp(self, subject_dir, anatomical_name=None, functional_name=None,
                       tt_n27_path=None):
        
        required_vars = {'anatomical_name':anatomical_name,
                         'functional_name':functional_name,
                         'tt_n27_path':tt_n27_path}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self._update_dset('')

        anat_path = os.path.join(subject_dir, self.anatomical_name+'+orig')
        func_path = os.path.join(subject_dir, self.current_functional+'+orig')        
        
        if self.run_script:
            self.afni.talairachwarp(func_path, anat_path, self.tt_n27_path)
            
        if self.write_script:
            self.afni.talairachwarp.write(self.scriptwriter, self.current_functional,
                                          self.anatomical_name, self.tt_n27_path)
        
        
        
    def write(selfanat_nifti, func_niftis, func_nifti_trs,
            subject_dirs=None, anatomical_name=None, functional_name=None,
            leadin=None, leadout=None, tr_length=None, tshift_slice=None,
            tpattern=None, motionfile_name=None, volreg_base=None, blur_kernel=None,
            normalize_expression=None, highpass_value=None, tt_n27_path=None):
        
        required_vars = {'anatomical_name':anatomical_name,
                         'functional_name':functional_name,
                         'subject_dirs':subject_dirs, 'leadin':leadin,
                         'leadout':leadout, 'tr_length':tr_length,
                         'tshift_slice':tshift_slice, 'tpattern':tpattern,
                         'motionfile_name':motionfile_name, 'volreg_base':volreg_base,
                         'blur_kernel':blur_kernel,
                         'normalize_expression':normalize_expression,
                         'highpass_value':highpass_value, 'tt_n27_path':tt_n27_path}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self.run_script = False
        self.write_script = True
        
        self.convert_anatomical()
        self.cutoff_buffer()
        self.refit()
        self.tshift()
        self.concatenate()
        self.volreg()
        self.smooth()
        self.normalize()
        self.highpass_filter()
        self.talairach_warp()
        
        
    
    def run(self, anat_nifti, func_niftis, func_nifti_trs,
            subject_dirs=None, anatomical_name=None, functional_name=None,
            leadin=None, leadout=None, tr_length=None, tshift_slice=None,
            tpattern=None, motionfile_name=None, volreg_base=None, blur_kernel=None,
            normalize_expression=None, highpass_value=None, tt_n27_path=None):
        
        required_vars = {'anatomical_name':anatomical_name,
                         'functional_name':functional_name,
                         'subject_dirs':subject_dirs, 'leadin':leadin,
                         'leadout':leadout, 'tr_length':tr_length,
                         'tshift_slice':tshift_slice, 'tpattern':tpattern,
                         'motionfile_name':motionfile_name, 'volreg_base':volreg_base,
                         'blur_kernel':blur_kernel,
                         'normalize_expression':normalize_expression,
                         'highpass_value':highpass_value, 'tt_n27_path':tt_n27_path}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        # iterate over subject dirs:
        for dir in self.subject_dirs:
            
            self.convert_anatomical(dir, anat_nifti)
            for i, (nifti, trlen) in enumerate(zip(func_niftis, func_nifti_trs)):
                self.cutoff_buffer(dir, nifti, trlen, prefix='epi'+str(i))
                self.refit(dir, 'epi'+str(i))
                self.tshift(dir, 'epi'+str(i), 'epits'+str(i))
            epi_names = ['epits'+str(i) for i in range(len(func_niftis))]
            self.concatenate(dir, epi_names)
            self.volreg(dir)
            self.smooth(dir)
            total_trs = sum(func_nifti_trs) - (self.leadin + self.leadout)*len(func_nifti_trs) - 1
            self.normalize(dir, total_trs)
            self.highpass_filter(dir)
            self.talairach_warp(dir)
            
        
        
        
        
        
        
        