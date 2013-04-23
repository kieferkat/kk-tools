

import sys, os
import glob
from covariance_matrix import *
import kktools.api as kkapi









if __name__ == "__main__":
	topdir = os.path.split(os.getcwd())[0]
	#subs = ['ar110111','as090711']
	#subdirs = [os.path.join(topdir,x) for x in subs]
	#subniis = [os.path.join(x, 'actepin_reg.nii') for x in subdirs]


	subdirs = kkapi.parsers.subject_dirs(topdir=topdir, exclude=['mm'])
	subniis = [os.path.join(x, 'pasepin_reg.nii') for x in subdirs]

	cc = CovarianceCalculator()

	#cc.normalize_niftis(subdirs, '*_pas_falign.nii', 'ttmask_warp.nii')

	#cc.compare_covariance_dicts('pas_faligned_cov.pkl','pas_unaligned_cov.pkl', pickled=True)

	
	cc.load_mask('ttmask_warp.nii')
	cc.load_subject_niftis(subniis)
	cc.prepare_3d_adjacency(numx=1, numy=1, numz=1)
	cc.mask_flatten_subject_niftis(normalize=True)
	cc.assign_model_subject('ar110111')
	
	print cc.nifti_dict.keys()
	#cc.generate_mappings()
	cc.generate_super_mappings(cc.nifti_dict, exclude_current_subject=False)
	cc.save_mappings('pas_super_mappings.pkl')

	#cc.load_mappings('passive_mappings.pkl')

	cc.load_subject_niftis(subniis)
	cc.functional_alignment_bymapping()

	#for subj in cc.mappings.keys():
	#	cc.mapping_to_nifti(subj)
	

	




