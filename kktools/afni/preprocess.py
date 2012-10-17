
import os, sys
import subprocess
import shutil
import glob

from ..base.process import Process
from ..utilities.cleaners import glob_remove
from ..defaults.lab_standard import preprocessing as preprocessing_defaults


class Preprocessor(Process):
    
    def __init__(self, variable_dict=None):
        super(Preprocessor, self).__init__(variable_dict=variable_dict)
        self._apply_defaults(preprocessing_defaults)
        
        
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
        if not self._check_variables(requried_vars): return False
        
        nifti_path = os.path.join(subject_dir, nifti_name)
        anat_path = os.path.join(subject_dir, self.anatomical_name)
        
        glob_remove(anat_path, suffix='+orig*')
        
        cmd = ['3dcopy', nifti_path, anat_path]
        subprocess.call(cmd)
        
        
    def cutoff_buffer(self, subject_dir, nifti_name, nifti_trs, leadin=None,
                      leadout=None, prefix='epi'):
        
        required_vars = {'leadin':leadin, 'leadout':leadout}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        nifti_path = os.path.join(subject_dir, nifti_name)
        epi_path = os.path.join(subject_dir, prefix)
        glob_remove(epi_path, suffix='+orig*')
        
        nifti_cut = nifti_path+'['+str(self.leadin)+'..'+str(nifti_trs-self.leadout-1)+']'
        
        cmd = ['3dTcat', '-prefix', epi_path, nifti_cut]
        subprocess.call(cmd)
        
        
    def refit(self, subject_dir, epi_name, tr_length=None):
        
        required_vars = {'tr_length':tr_length}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        epi_path = os.path.join(subject_dir, epi_name)
        cmd = ['3drefit', '-TR', str(self.tr_length), epi_path]
        subprocess.call(cmd)
        
        
    def tshift(self, subject_dir, epi_name, tshift_slice=None, tpattern=None,
               prefix='epits'):
        
        required_vars = {'tshift_slice':tshift_slice, 'tpattern':tpattern}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        epi_path = os.path.join(subject_dir, epi_name)
        tshift_path = os.path.join(subject_dir, prefix)
        glob_remove(tshift_path, suffix='+orig*')
        
        cmd = ['3dTshift', '-slice', str(self.tshift_slice), '-tpattern',
               self.tpattern, '-prefix', tshift_path, epi_path]
        subprocess.call(cmd)
        
        
    def concatenate(self, subject_dir, epi_names, functional_name=None):
        
        required_vars = {'functional_name':functional_name}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        epi_paths = [os.path.join(subject_dir, epi) for epi in epi_names]
        functional_path = os.path.join(subject_dir, self.functional_name)
        glob_remove(functional_path, suffix='+orig*')
        
        cmd = ['3dTcat','-prefix', functional_path].extend(epi_paths)
        subprocess.call(cmd)
        
        for epi in epi_paths:
            glob_remove(epi, suffix='*')
            
            
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
        
        glob_remove(motion_path)
        glob_remove(mfunc_path, suffix='+orig*')
        
        cmd = ['3dvolreg', '-Fourier', '-twopass', '-prefix', mfunc_path,
               '-base', str(self.volreg_base), '-dfile', motion_path, prior_path]
        subprocess.call(cmd)
        
        
    def smooth(self, subject_dir, blur_kernel=None, functional_name=None,
               suffix='b'):
        
        required_vars = {'blur_kernel':blur_kernel,
                         'functional_name':functional_name}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self._update_dset(suffix)
        
        prior_path = os.path.join(subject_dir, self.prior_functional+'+orig')
        blur_path = os.path.join(subject_dir, self.current_functional)
        
        glob_remove(blur_path, suffix='+orig*')
        
        cmd = ['3dmerge', '-prefix', blur_path, '-1blur_fwhm', str(self.blur_kernel),
               '-doall', prior_path]
        subprocess.call(cmd)
        
        
    def normalize(self, subject_dir, functional_trs, suffix='n', ave_suffix='_ave',
                  functional_name=None, normalize_expression=None):
        
        required_vars = {'normalize_expression':normalize_expression,
                         'functional_name':functional_name}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self._update_dset(suffix)
        
        prior_path = os.path.join(subject_dir, self.prior_functional+'+orig')
        ave_path = os.path.join(subject_dir, self.functional_name+ave_suffix)
        norm_path = os.path.join(subject_dir, self.current_functional)
        
        glob_remove(norm_path, suffix='+orig*')
        glob_remove(ave_path, suffix='+orig*')
        
        prior_trrange = prior_path+'[0..'+str(functional_trs)+']'
        tstat = ['3dTstat', '-prefix', ave_path, prior_trrange]
        subprocess.call(tstat)
        
        refit = ['3drefit', '-abuc', ave_path+'+orig']
        subprocess.call(refit)
        
        calc = ['3dcalc', '-datum', 'float', '-a', prior_trrange, '-b',
                ave_path, '-expr', self.normalize_expression, '-prefix',
                norm_path]
        subprocess.call(calc)
        
        
    def highpass_filter(self, subject_dir, highpass_value=None, functional_name=None,
                        suffix='f'):
        
        required_vars = {'highpass_value':highpass_value,
                         'functional_name':functional_name}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        self._update_dset(suffix)
        
        prior_path = os.path.join(subject_dir, self.prior_functional+'+orig')
        filter_path = os.path.join(subject_dir, self.current_functional)
        
        glob_remove(filter_path, suffix='+orig*')
        
        cmd = ['3dFourier', '-prefix', filter_path, '-highpass',
               str(self.highpass_value), prior_path]
        subprocess.call(cmd)
        
        
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
                
        tlrc = ['@auto_tlrc', '-warp_orig_vol', '-suffix', 'NONE', '-base',
                self.tt_n27_path, '-input', ]
        subprocess.call(tlrc)
        
        refit = ['3drefit', '-apar', anat_path, func_path]
        subprocess.call(refit)
        
        
    
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
            
        
        
        
        
        
        
        