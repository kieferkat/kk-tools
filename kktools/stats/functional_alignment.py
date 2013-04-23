import os, glob
from pprint import pprint
import nibabel as nib
import numpy as np
import sklearn.covariance as skcov
import sklearn.linear_model as linmod
import time
import cPickle
import pyximport
pyximport.install()
import cloops
from nngarotte import NonNegativeGarrote




class CloopsInterface(object):

	def __init__(self):
		super(CloopsInterface, self).__init__()


	def construct_ridge_regressions(self, Alist, Blist, adj, trs, voxels):
		return cloops.construct_ridge_regressions(Alist, Blist, adj, trs, voxels)


	def expected_voxels_bytr(self, data, mapping, tr, voxels):
		return cloops.expected_voxels_bytr(data, mapping, tr, voxels)

	def calculate_correlation(self, dataA, dataB):
		return cloops.calculate_correlation(dataA, dataB)



class FAAssistant(object):

	def __init__(self):
		super(FAAssistant, self).__init__()


	def _nifti_parse(self, nifti_path):

		nifti = nib.load(nifti_path)
		return nifti.get_data(), nifti.shape, nifti.get_affine()


	def flatten_mask(self, mask):

		flattened = mask.copy()
		flattened.shape = np.product(flattened.shape)
		return flattened


	def load_mask(self, mask_path):

		mdata, mshape, maffine = self._nifti_parse(mask_path)
		self.original_mask_shape = mshape
		self.original_mask_affine = maffine

		mdata = np.transpose(mdata, [2,1,0])
		mdata_bool = mdata.astype(np.bool)
		return mdata_bool


	def load_data(self, data_path):

		ddata, dshape, daffine = self._nifti_parse(data_path)
		self.original_data_shape = dshape
		self.original_data_affine = daffine

		ddata = np.transpose(ddata, [3,2,1,0])

		return ddata


	def load_subject_niftis(self, nifti_paths, individual_folders=True):

		subject_data = {}
		subject_keys = []

		if individual_folders:
			for np in nifti_paths:
				subject_keys.append(os.path.split(os.path.split(np)[0])[1])
		else:
			subject_keys = range(len(nifti_paths))

		for sk, npath in zip(subject_keys, nifti_paths):
			subject_data[sk] = self.load_data(npath)

		return subject_data


	def flatten_data(self, nifti_dict):

		print 'flattening data...'

		flat_niftis = {}

		for key, data in nifti_dict.items():
			print 'flattening', key
			flat_niftis[key] = data.reshape(data.shape[0], np.product(data.shape[1:]))

		return flat_niftis


	def mask_data(self, nifti_dict, flat_mask):

		print 'masking data...'

		masked_niftis = {}

		for key, data in nifti_dict.items():
			print 'masking', key
			masked_niftis[key] = data[:, flat_mask]

		return masked_niftis


	def split_flattened_data(self, data, fraction=0.5):

		trs, voxels = data.shape

		A_trs = int(round(float(trs)*fraction))
		B_trs = trs-A_trs

		print 'closest trs split:', A_trs, B_trs

		dataA = data[:A_trs,:]
		dataB = data[A_trs:,:]

		return dataA, dataB


	def split_nifti_dict(self, nifti_dict, fraction=0.5):

		print 'splitting nifti dict by fraction', fraction

		ndictA = {}
		ndictB = {}

		for skey, data in nifti_dict.items():
			print 'splitting', skey
			dataA, dataB = self.split_flattened_data(data, fraction=fraction)
			ndictA[skey] = dataA
			ndictB[skey] = dataB

		return ndictA, ndictB


	def data_to_nifti(self, data, mask_3d, mask_affine, filepath, save_dtype=np.float32):

		print 'saving data to file', filepath
		trs, voxels = data.shape

		unmasked = [np.zeros(mask_3d.shape) for i in range(trs)]

		for i in range(trs):
			unmasked[i][np.asarray(mask_3d).astype(np.bool)] = np.squeeze(np.array(data[i]))
			unmasked[i] = np.transpose(unmasked[i],[2,1,0])

		unmasked = np.transpose(unmasked, [1,2,3,0])
		unmasked = np.array(unmasked, dtype=save_dtype)

		nii = nib.Nifti1Image(unmasked, mask_affine)

		if os.path.exists(filepath):
			print 'overwriting...'
			os.remove(filepath)

		nii.to_filename(filepath)


	def normalize_data(self, data):

		print 'normalizing column-wise...'
		print 'pre-normalization sum:', np.sum(data)

		std_dev = np.std(data, axis=0)
		means = np.mean(data, axis=0)

		normed = data-means
		normed = normed/std_dev
		normed = np.nan_to_num(normed)

		print 'post-normalization sum:', np.sum(normed)
		return normed


	def normalize_nifti_dict(self, nifti_dict):

		normed_niftis = {}

		for skey, data in nifti_dict.items():
			normed_niftis[skey] = self.normalize_data(data)

		return normed_niftis


	def pickle_object(self, py_object, filepath):

		if os.path.exists(filepath):
			print 'overwriting...'
			os.remove(filepath)

		print 'pickling', filepath, '...'
		fid = open(filepath,'w')
		cPickle.dump(py_object, fid)
		fid.close()


	def unpickle_object(self, filepath):

		print 'unpickling', filepath, '...'
		fid = open(filepath,'r')
		py_object = cPickle.load(fid)
		fid.close()

		return py_object


	




class FunctionalAlignment(object):

	def __init__(self):
		super(FunctionalAlignment, self).__init__()
		self.cloops = CloopsInterface()
		self.assistant = FAAssistant()


	def prepare_mask_fromnifti(self, mask_path, save=True):

		# prepare mask:
		self.mask = self.assistant.load_mask(mask_path)
		self.flat_mask = self.assistant.flatten_mask(self.mask)

		if save:
			self.assistant.pickle_object(self.mask, 'mask.pkl')
			self.assistant.pickle_object(self.flat_mask, 'flat_mask.pkl')
			self.assistant.pickle_object(self.assistant.original_mask_affine, 'mask_affine.pkl')

		return self.flat_mask


	def prepare_data_fromnifti(self, nifti_paths, flat_mask, individual_folders=True, save=True,
							   normalize=True, filename='nifti_dict.pkl'):

		# prepare data
		nifti_dict = self.assistant.load_subject_niftis(nifti_paths, individual_folders=individual_folders)
		nifti_dict = self.assistant.flatten_data(nifti_dict)
		nifti_dict = self.assistant.mask_data(nifti_dict, flat_mask)

		if save:
			self.assistant.pickle_object(nifti_dict, 'unnormed_'+filename)

		if not normalize:
			return nifti_dict

		else:
			normed_nifti_dict = self.assistant.normalize_nifti_dict(nifti_dict)

			if save:
				self.assistant.pickle_object(normed_nifti_dict, 'normed_'+filename)

			return normed_nifti_dict


	def split_nifti_dict(self, nifti_dict, fraction=0.5, save=True):

		nifti_dictA, nifti_dictB = self.assistant.split_nifti_dict(nifti_dict, fraction=fraction)

		if save:
			self.assistant.pickle_object(nifti_dictA, 'nifti_dict_A.pkl')
			self.assistant.pickle_object(nifti_dictB, 'nifti_dict_B.pkl')


	def load_presaved_data(self, pickled_nifti_dict):

		nifti_dict = self.assistant.unpickle_object(pickled_nifti_dict)
		return nifti_dict


	def load_presaved_masks(self, pickled_mask, pickled_flat_mask, pickled_affine):
		self.mask = self.assistant.unpickle_object(pickled_mask)
		self.flat_mask = self.assistant.unpickle_object(pickled_flat_mask)
		self.mask_affine = self.assistant.unpickle_object(pickled_affine)
		return self.mask, self.flat_mask, self.mask_affine


	def piecewise_covariance(self, dataA, dataB, correlation=False):

		#print 'computing by-voxel covariances piecewise...'

		dataA = np.array(dataA)
		dataB = np.array(dataB)

		assert dataA.shape == dataB.shape
		voxel_covariances = []

		for voxel in range(dataA.shape[1]):
			A = []
			B = []

			for tr in range(dataA.shape[0]):
				A.append(dataA[tr, voxel])
				B.append(dataB[tr, voxel])

			if correlation:
				astd = np.std(A)
				bstd = np.std(B)
				A = A/astd
				B = B/bstd

			voxel_covariances.append(np.cov(A,B)[0,1])

		return voxel_covariances


	def piecewise_correlation(self, dataA, dataB):

		dataA = np.array(dataA).tolist()
		dataB = np.array(dataB).tolist()

		#print np.sum(dataA)
		#print np.sum(dataB)
		#print dataA[0:3]
		#print dataB[0:3]
		#stop

		voxelwise_correlation = self.cloops.calculate_correlation(dataA, dataB)

		return voxelwise_correlation




	def compute_all_covariances(self, nifti_dict, save=True, filename='sub_to_sub_covariances.pkl',
							    correlation=False):

		subjects = nifti_dict.keys()
		covariance_dict = {}

		for subject in subjects:
			covariance_dict[subject] = {}
			for sub in subjects:
				covariance_dict[subject][sub] = None

		for subject in subjects:
			if not correlation:
				print 'covariance of', subject
			else:
				print 'correlation of', subject

			nsubs = [x for x in subjects if x != subject]

			for nsub in nsubs:

				if covariance_dict[subject][nsub] is None:

					if not correlation:
						print 'computing by-voxel covariances piecewise to subject..', nsub
						sub_to_nsub_covars = self.piecewise_covariance(nifti_dict[subject], nifti_dict[nsub],
																	   correlation=False)
						covariance_dict[subject][nsub] = sub_to_nsub_covars
						covariance_dict[nsub][subject] = sub_to_nsub_covars
					else:
						print 'computing by-voxel correlations piecewise to subject..', nsub
						sub_to_nsub_corr = self.piecewise_correlation(nifti_dict[subject], nifti_dict[nsub])
						covariance_dict[subject][nsub] = sub_to_nsub_corr
						covariance_dict[nsub][subject] = sub_to_nsub_corr

		if save:
			self.assistant.pickle_object(covariance_dict, filename)

		return covariance_dict



	def create_covariance_csv(self, covariance_dict, filename='AtoB_covariance.csv',
							  method='mean'):

		print 'creating csv of covariance dict...'

		fid = open(filename,'w')

		header = sorted(covariance_dict.keys())
		fid.write(','+','.join(header)+'\n')

		for sub in header:
			values = []
			for nsub in header:
				if sub == nsub:
					values.append(1.)
				else:
					if method == 'mean':
						values.append(np.mean(covariance_dict[sub][nsub]))
					elif method == 'median':
						values.append(np.median(covariance_dict[sub][nsub]))
			fid.write(sub+','+','.join([str(x) for x in values])+'\n')

		fid.close()






	def construct_adjacency(self, mask, numx=1, numy=1, numz=1):

		print 'numx, numy, numz:', numx, numy, numz
		print 'constructing adjacency matrix...'

		vmap = np.cumsum(mask).reshape(mask.shape)

		bmask = np.bool_(mask.copy())

		vmap[~bmask] = -1
		vmap -= 1

		nx, ny, nz = bmask.shape

		adj = []

		for x in range(nx):
			for y in range(ny):
				for z in range(nz):
					if mask[x,y,z]:

						# local map specifies all adjacent voxels in range:
						local_map = vmap[max((x-numx),0):(x+numx+1),
						                 max((y-numy),0):(y+numy+1),
						                 max((z-numz),0):(z+numz+1)]

						# inds are all the local map indices that are not
						# -1 (valid brain voxels)
						inds = (local_map>-1)
						inds = np.bool_(inds)

						# append the adjacent voxels for this current voxel:
						adjacency_row = np.array(local_map[inds], dtype=int)
						adj.append(adjacency_row)

		print len(adj)

		for i, a in enumerate(adj):
			adj[i] = a.tolist()

		return adj




	def construct_average_subject(self, subject_keys, nifti_dict, save=True, filename='average_subject.pkl'):

		mapsubs = []
		for msk in subject_keys:
			mapsubs.append(np.array(nifti_dict[msk]))

		average_subject = []

		for i, msub in enumerate(mapsubs):
			if i == 0:
				average_subject = msub
			else:
				average_subject = average_subject+msub

		average_subject = average_subject / float(len(mapsubs))

		if save:
			self.assistant.pickle_object(average_subject, filename)

		return average_subject


	def construct_median_subject(self, subject_keys, nifti_dict, save=True, filename='median_subject.pkl'):

		mapsubs = []
		for msk in subject_keys:
			mapsubs.append(np.array(nifti_dict[msk]))

		trs, voxels = mapsubs[0].shape

		median_subject = np.zeros((trs, voxels))

		print 'constructing median subject...'

		for tr in range(trs):
			if (tr % 20) == 0:
				print 'tr', tr, '...'
			for voxel in range(voxels):
				vox_array = []
				for msub in mapsubs:
					vox_array.append(msub[tr,voxel])
				median_subject[tr, voxel] = np.median(vox_array)

		if save:
			self.assistant.pickle_object(median_subject, filename)

		return median_subject




	def create_mapping(self, dataA, dataB, adjacency, method='ridge',
					   ridge_alphas=[.01,.1,1.,10.], elastic_net_cv_ratios=[.1,.5,.9,.95],
					   elastic_net_alpha=1.0, elastic_net_ratio=0.5, garotte_alpha=0.0005):
		
		print 'mapping A to B...'
		mapstart = time.time()

		print dataA.shape
		assert dataA.shape == dataB.shape
		assert dataA.shape[1] == dataB.shape[1] == len(adjacency)

		mapping = []
		mapping_score = []

		trs, voxels = dataB.shape

		dataA_list = dataA.tolist()
		dataB_list = dataB.tolist()

		# assumes data does NOT need intercept / is normalized!
		if method == 'ridge':
			regression = linmod.RidgeCV(alphas=ridge_alphas, fit_intercept=False)
		elif method == 'elastic_net_cv':
			regression = linmod.ElasticNetCV(l1_ratio=elastic_net_cv_ratios,
											 fit_intercept=False,
											 cv=3, n_jobs=8)
		elif method == 'garotte':
			regression = NonNegativeGarrote(garotte_alpha, fit_intercept=False,
											estimate_method='ols')
		elif method == 'elastic_net':
			regression = linmod.ElasticNet(alpha=elastic_net_alpha, l1_ratio=elastic_net_ratio,
										   fit_intercept=False, normalize=False)


		all_zero_errors = 0
		solution_zero_errors = 0

		A_iv_matrices, B_dv_vectors = self.cloops.construct_ridge_regressions(dataA_list,
																			  dataB_list,
																			  adjacency,
																			  trs,
																			  voxels)

		for iv, dv, vox in zip(A_iv_matrices, B_dv_vectors, range(voxels)):
			if (vox % 5000) == 0:
				if method == 'ridge':
					print 'preforming ridge regression:', vox
				elif method == 'elastic_net':
					print 'preforming elastic net regression:', vox
					print 'solution zeros:', solution_zero_errors
				elif method == 'garotte':
					print 'preforming non-negative garotte regression:', vox
				elif method == 'elastic_net_cv':
					print 'preforming elastic net with cross-validation:', vox
					print 'solution zeros:', solution_zero_errors


			iv = np.array(iv)
			dv = np.array(dv)

			if np.sum(iv) == 0 or np.sum(dv) == 0:
				all_zero_errors += 1
				mapping.append([[vox, 1.]])
				mapping_score.append(0.0)

			else:
				regression.fit(iv, dv)
				mscore = regression.score(iv, dv)
				coefs = regression.coef_

				# testing only!!
				#print 'alpha, ratio:', regression.alpha_, regression.l1_ratio_

				if np.sum(coefs) == 0.:
					solution_zero_errors += 1
					mapping.append([[vox,1.]])
					mapping_score.append(0.0)

				else:
					mapping.append(zip(adjacency[vox], coefs))
					mapping_score.append(mscore)

		mapend = time.time()

		print 'all zero iv or dv errors:', all_zero_errors
		print 'solution zero errors:', solution_zero_errors
		print 'total time elapsed:', mapend-mapstart, '\n'
		print 'average mapping score:', np.mean(mapping_score)

		return mapping, mapping_score



	def generate_mappings(self, nifti_dict, model_data, adjacency, ridge_alphas=[.01,.1,1.,10.],
						  elastic_net_cv_ratios=[.1,.5,.9,.95], garotte_alpha=0.0005, method='ridge', 
						  conserve_memory=False, elastic_net_alpha=1.0, elastic_net_ratio=0.5):

		print 'generating mappings...'
		mappings_dict = {}
		mappings_scores = {}

		skeys = nifti_dict.keys()

		for skey in skeys:
			print 'creating mapping for subject:', skey
			mapping, mapping_score = self.create_mapping(nifti_dict[skey], model_data, adjacency, ridge_alphas=ridge_alphas,
										  				 elastic_net_cv_ratios=elastic_net_cv_ratios, method=method,
										  				 garotte_alpha=garotte_alpha, elastic_net_alpha=elastic_net_alpha,
										  				 elastic_net_ratio=elastic_net_ratio)
			mappings_dict[skey] = mapping
			mappings_scores[skey] = mapping_score

			if conserve_memory:
				print 'deleting', skey, 'from nifti dict...'
				del nifti_dict[skey]

		print 'average map scores:'
		for skey in skeys:
			print skey, np.mean(mappings_scores[skey])
		print 'total average score:', np.mean([np.mean(x) for k,x in mappings_scores.items()])

		return mappings_dict, mappings_scores



	def test_coefs_forzero(self, mapping):

		allzero_coefs = 0
		for items in mapping:
			coefs = [x[1] for x in items]
			if np.sum(coefs) == 0:
				allzero_coefs += 1
		print 'allzero coefs:', allzero_coefs



	def test_mappings_forzero(self, mappings):

		for skey, maps in mappings.items():
			print skey
			self.test_coefs_forzero(maps)



	def data_from_mapping(self, data, mapping, test_row_sums=False, r2_score_cutoff=0.0,
						  map_scores=[]):

		print 'creating new data from mapping...'

		trs, voxels = data.shape
		data_list = data.tolist()
		mapped = []

		if not r2_score_cutoff:
			r2_score_cutoff = 0.0
			map_scores = []

		convertstart = time.time()

		for tr in range(trs):
			if (tr % 20) == 0:
				print 'mapping tr:', tr

			tr_row = self.cloops.expected_voxels_bytr(data_list, mapping, tr, voxels, r2_score_cutoff, map_scores)
			if test_row_sums:
				print np.sum(tr_row)
			else:
				mapped.append(tr_row)

		mapped = np.array(mapped)

		convertend = time.time()
		print 'conversion time:', convertend-convertstart

		return mapped


	def map_data(self, nifti_dict, mappings, conserve_memory=False, test_row_sums=False,
				 r2_score_cutoff=0.0, map_scores=[]]):

		print 'mapping nifti_dict...'

		mapped_niftis = {}

		skeys = nifti_dict.keys()
		for skey in skeys:
			print 'map of subject', skey
			data = nifti_dict[skey]
			mapped_data = self.data_from_mapping(data, mappings[skey], test_row_sums=test_row_sums,
												 r2_score_cutoff=r2_score_cutoff,
												 map_scores=map_scores)
			mapped_niftis[skey] = mapped_data

			if conserve_memory:
				del nifti_dict[skey]
				del mappings[skey]

		return mapped_niftis



	# BROKEN?
	def map_data_memorysafe(self, nifti_dict, mappings, directory='Bdata_through_Amaps',
							conserve_memory=True):

		print 'mapping sequentially to directory', directory

		try:
			os.makedirs(directory)
		except:
			print 'directory already created'
			pass

		skeys = nifti_dict.keys()

		for skey in skeys:
			print 'mapping', skey, '...'
			filename = os.path.join(directory, skey+'.npy')


			if os.path.exists(filename):
				print 'overwriting...'
				os.remove(filename)

			mapped_data = self.data_from_mapping(nifti_dict[skey], mappings[skey])

			np.save(filename, mapped_data)

			if conserve_memory:
				del nifti_dict[skey]
				del mappings[skey]



	def convert_subject_npys_tonifti(self, subject_npys, mask, mask_affine):

		print 'converting .npys to nifti...'

		for snpy in subject_npys:

			print 'converting', snpy, '...'
			subject = os.path.split(snpy)[1].rstrip('.npy')
			data = np.load(snpy)
			print data.shape

			self.assistant.data_to_nifti(data, mask, mask_affine, subject+'.nii')


	def niftidict_to_nifti(self, nifti_dict, mask, mask_affine, suffix='_aligned.nii',
						   directory=os.getcwd()):

		print 'converting dict to niftis'

		for subject_key, data in nifti_dict.items():

			print 'converting', subject_key, 'to nifti...'

			try:
				os.makedirs(directory)
			except:
				pass

			filepath = os.path.join(directory, subject_key+suffix)

			self.assistant.data_to_nifti(data, mask, mask_affine, filepath)




	# INCOMPLETE:
	def load_subject_npys(self, subject_npys):

		print 'loading subject data...'

		nifti_dict = {}

		for npf in subject_npys:
			skey = os.path.split(npf)[1].rstrip('.npy')
			print 'loading', skey, '...'

			nifti_dict[skey] = np.load(skey)

		return nifti_dict














