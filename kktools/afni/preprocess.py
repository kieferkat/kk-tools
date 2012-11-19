
import os, sys
import subprocess
import shutil
import glob

from ..afni.pipeline import AfniPipeline
from ..afni.functions import AfniWrapper
from ..base.scriptwriter import Scriptwriter
from ..utilities.cleaners import glob_remove


class Preprocessor(AfniPipeline):
    
    def __init__(self):
        super(Preprocessor, self).__init__()
        self.script_name = 'preprocess'
        self.scriptwriter = Scriptwriter()
        self.afni = AfniWrapper()
        self.run_script = True
        self.write_script = True
        
        
        
    def convert_anatomical(self, dir=None, dset_in=None, dset_out=None,
                           verbose=True):
        
        if verbose:
            print dset_in, dir
            
        nifti_path = os.path.join(dir, dset_in)
        anat_path = os.path.join(dir, dset_out)
        
        if self.run_script:
            self.afni.copy3d(nifti_path, anat_path)
        
        if self.write_script:
            self.afni.copy3d.write(self.scriptwriter, dset_in, dset_out)
            
        
        
        
    def cutoff_buffer(self, dir=None, dset_in=None, dset_out=None, nifti_trs=None,
                      leadin=None, leadout=None):
        
        
        nifti_path = os.path.join(dir, dset_in)
        epi_path = os.path.join(dir, dset_out)
        
        if self.run_script:
            self.afni.tcatbuffer(nifti_path, epi_path, leadin, nifti_trs-leadout-1)
        
        if self.write_script:
            self.afni.tcatbuffer.write(self.scriptwriter, dset_in, dset_out, leadin,
                                       nifti_trs-leadout-1, cmd_only=False)
                
        return dset_out
                
        
        
    def refit(self, dir=None, dset_in=None, tr_length=None):
        
        epi_path = os.path.join(dir, dset_in)
        
        if self.run_script:
            self.afni.refit(epi_path, tr_length)
        
        if self.write_script:
            self.afni.refit.write(self.scriptwriter, dset_in, tr_length,
                                  cmd_only=False)
            
        return dset_in
            
        
        
    def tshift(self, dir=None, dset_in=None, dset_out=None, tshift_slice=0,
               tpattern='altplus'):
        
        epi_path = os.path.join(dir, dset_in)
        tshift_path = os.path.join(dir, dset_out)
        
        if self.run_script:
            self.afni.tshift(epi_path+'+orig', tshift_path, tshift_slice,
                             tpattern)
        
        if self.write_script:
            self.afni.tshift.write(self.scriptwriter, dset_in, dset_out,
                                   tshift_slice, tpattern, cmd_only=False)
            
        return dset_out
        
        
        
    def concatenate(self, dir=None, dsets_in=[], dset_out=None):
                
        epi_paths = [os.path.join(dir, epi) for epi in dsets_in]
        dset_out_path = os.path.join(dir, dset_out)
        
        if self.run_script:
            self.afni.tcatdsets(epi_paths, dset_out_path)
        
        if self.write_script:
            self.afni.tcatdsets.write(self.scriptwriter, dsets_in, dset_out)
            
    
            
            
    def volreg(self, dir=None, dset_in=None, dset_out=None, motionfile_name='3dmotion.1D',
               volreg_base=3):
        
        
        in_path = os.path.join(dir, dset_in+'+orig')
        motion_path = os.path.join(dir, motionfile_name)
        out_path = os.path.join(dir, dset_out)
        
        if self.run_script:
            self.afni.volreg(in_path, out_path, motion_path, volreg_base)
        
        if self.write_script:
            self.afni.volreg.write(self.scriptwriter, dset_in, dset_out,
                                   motionfile_name, volreg_base)
            

        
        
    def smooth(self, dir=None, dset_in=None, dset_out=None, blur_kernel=4):
        
        prior_path = os.path.join(dir, dset_in+'+orig')
        blur_path = os.path.join(dir, dset_out)
        
        if self.run_script:
            self.afni.smooth(prior_path, blur_path, blur_kernel)
            
        if self.write_script:
            self.afni.smooth.write(self.scriptwriter, dset_in, dset_out, blur_kernel)
        
        
        
        
    def normalize(self, dir=None, dset_in=None, dset_out=None, functional_trs=None,
                  dset_ave='dset_average', normalize_expression='((a-b)/b)*100'):
    
        prior_path = os.path.join(dir, dset_in+'+orig')
        ave_path = os.path.join(dir, dset_ave)
        norm_path = os.path.join(dir, dset_out)
        
        if self.run_script:
            self.afni.normalize(prior_path, ave_path, norm_path, functional_trs-1,
                                normalize_expression)
        
        if self.write_script:
            self.afni.normalize.write(self.scriptwriter, dset_in, dset_out, dset_ave,
                                      functional_trs-1, normalize_expression)
        
        
        
    def highpass_filter(self, dir=None, dset_in=None, dset_out=None, highpass_value=None):
        
        prior_path = os.path.join(dir, dset_in+'+orig')
        filter_path = os.path.join(dir, dset_out)
        
        if self.run_script:
            self.afni.highpassfilter(prior_path, filter_path, highpass_value)
            
        if self.write_script:
            self.afni.highpassfilter.write(self.scriptwriter, dset_in, dset_out,
                                           highpass_value)
        
        
        
        
    def talairach_warp(self, dir=None, anatomical=None, functional=None,
                       template_path=None):

        anat_path = os.path.join(dir, anatomical+'+orig')
        func_path = os.path.join(dir, functional+'+orig')        
        
        if self.run_script:
            self.afni.talairachwarp(func_path, anat_path, template_path)
            
        if self.write_script:
            self.afni.talairachwarp.write(self.scriptwriter, functional,
                                          anatomical, template_path)
        
        
        
    
            
        
        
        
        
        
        
        