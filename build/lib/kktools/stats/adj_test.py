

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

	cc.load_mask('ttmask_warp.nii')
	cc.load_subject_niftis(subniis)
	cc.prepare_3d_adjacency()
	cc.mask_flatten_subject_niftis(normalize=True)
	cc.assign_model_subject(cc.nifti_dict.keys()[0])
	cc.generate_mappings()
	cc.save_mappings('mappings_test.pkl')

	#cc.load_mappings('mappings_test.pkl')

	for subj in cc.mappings.keys():
		cc.mapping_to_nifti(subj)






