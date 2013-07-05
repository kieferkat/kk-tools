
import scipy as sp 
import numpy as np 
import os, glob
import nibabel as nib








def temporal_blur(nifti_path, mask_path, output_filepath, weight=0.5):

	nifti = nib.load(nifti_path)
	ndata = nifti.get_data()
	naffine = nifti.get_affine()

	mask = nib.load(mask_path)
	mdata = mask.get_data()
	maffine = mask.get_affine()

	data = np.transpose(ndata, [3,2,1,0])
	data = np.reshape(data.shape[0], np.product(data.shape[1:]))

	mask = np.transpose(mdata, [2,1,0])
	mask.shape = np.product(mask.shape)

	data = data[:, flat_mask]

	# detrend
	dt_data = sp.signal.detrend(data, axis=0)

	trs, voxels = dt_data.shape

	tb_data = dt_data.copy()

	for voxel in range(voxels):
		timecourse = dt_data[:,voxel]

		for i in range(trs-1):
			tb_data[i,voxel] += dt_data[i+1,voxel]*weight
			tb_data[i+1,voxel] += dt_data[i,voxel]*weight
		tb_data[0,voxel] += dt_data[0,voxel]*weight
		tb_data[-1,voxel] += dt_data[-1,voxel]*weight

		tb_data[:,voxel] = tb_data[:,voxel]/(1.+2*weight)

	unmasked = [np.zeros(mdata.shape) for i in range(trs)]
	for i in range(trs):
		unmasked[i][np.asarray(mdata).astype(np.bool)] = np.squeeze(np.array(tb_data[i]))
		unmasked[i] = np.transpose(unmasked[i],[2,1,0])

	unmasked = np.transpose(unmasked, [1,2,3,0])
	unmasked = np.array(unmasked, dtype=np.float32)

	nii = nib.Nifti1Image(unmasked, naffine)
	if os.path.exists(output_filepath):
		os.remove(output_filepath)
	nii.to_filename(output_filepath)
