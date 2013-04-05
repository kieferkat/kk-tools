

from covariance_matrix import *
import os


if __name__ == "__main__":
	cc = CovarianceCalculator()
	nifti = os.path.join(os.getcwd(), 'as012012', 'parkuse_gnet.nii')
	niftis = [nifti]

	mask = os.path.join(os.getcwd(), 'graphnet', 'mask_resamp.nii')
	cc.load_mask(mask)
	cc.load_subject_niftis(niftis)
	cc.prepare_nifti_dict()
	cc.construct_X_matrix()
	cc.compute_empirical_covariance(cc.X)








