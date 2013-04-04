
import nibabel as nib
import numpy as np
import subprocess
import os
import shutil
from ..utilities.cleaners import glob_remove


class NiftiTools(object):
    
    
    def __init__(self):
        super(NiftiTools, self).__init__()
        
        
    def adwarp_to_template_talairach(self, dataset_in, dataset_out, talairach_path, dxyz,
                                     overwrite=False):
        if not overwrite:
            glob_remove(dataset_out)
            subprocess.call(['adwarp', '-apar', talairach_path, '-dpar', dataset_in,
                             '-dxyz', str(dxyz), '-prefix', dataset_out])
        else:
            subprocess.call(['adwarp','-apar',talairach_path,'-dpar',dataset_in,
                             '-dxyz', str(dxyz)])
                
        
    def convert_to_nifti(self, dataset_in, dataset_out):
        glob_remove(dataset_out)
        subprocess.call(['3dAFNItoNIFTI', '-prefix', dataset_out, dataset_in])
        
        
    def convert_to_afni(self, nifti_in, dataset_out):
        try:
            glob_remove(dataset_out+'+orig')
        except:
            pass
        try:
            glob_remove(dataset_out+'+tlrc')
        except:
            pass
        subprocess.call(['3dcopy', nifti_in, dataset_out])
        
        
    def refit(self, functional, anatomical):
        subprocess.call(['3drefit', '-apar', anatomical, functional])
        
        
    def adwarp_to_subject_talairach(self, dataset_in, dataset_out, anatomical, dxyz):
        glob_remove(dataset_out)
        subprocess.call(['adwarp', '-apar', anatomical, '-dpar', dataset_in,
                         '-dxyz', str(dxyz), '-prefix', dataset_out])
        
        
    def create_talairach_niftis(self, subject_dirs, functional_name, anatomical_name,
                                dxyz, talairach_path, output_name, within_subject_warp=True,
                                to_template_warp=True):
        
        if not output_name.endswith('.nii'):
            output_name = output_name+'.nii'
        
        # iterate subject directories:
        for s in subject_dirs:
            
            functional = os.path.join(s, functional_name)
            anatomical = os.path.join(s, anatomical_name)
            
            # refit the functional to the anatomical:
            self.refit(functional, anatomical)
                        
            # adwarp the functional to the anatomical
            if within_subject_warp:
                adwarp_temp = os.path.join(s, 'temp_adwarp_func')
                self.adwarp_to_subject_talairach(functional, adwarp_temp, anatomical,
                                                 dxyz)
            else:
                adwarp_temp = functional
            
            # warp the warped functional to the talairach template:
            if to_template_warp:
                template_temp = os.path.join(s, 'temp_template_func')
                self.adwarp_to_template_talairach(adwarp_temp+'+tlrc', template_temp, talairach_path,
                                                  dxyz)
            else:
                template_temp = adwarp_temp
                
            # make the nifti files
            nifti_path = os.path.join(s, output_name)
            try:
                glob_remove(nifti_path)
            except:
                pass
            self.convert_to_nifti(template_temp+'+tlrc', nifti_path)
            
            if within_subject_warp:
                glob_remove(adwarp_temp)
            if to_template_warp:
                glob_remove(template_temp)
                
        return [os.path.join(s, output_name) for s in subject_dirs]
        
    
    def save_nifti(self, data, affine, filepath):
        glob_remove(filepath)
        nii = nib.Nifti1Image(data, affine)
        nii.to_filename(filepath)
    
        
    def output_nifti_thrumask(self, data, mask, mask_3dshape, trlen, affine, output_filepath):
        
        shaped_matrix = np.zeros((mask_3dshape[0], mask_3dshape[1], mask_3dshape[2],
                                  trlen))
        shaped_matrix[np.where(mask)] = data
        self.save_nifti(shaped_matrix, affine, output_filepath)
        
        
    def load_nifti(self, nifti_path):
        image = nib.load(nifti_path)
        shape = image.get_shape()
        idata = image.get_data()
        affine = image.get_affine()
        return idata, affine, shape
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        