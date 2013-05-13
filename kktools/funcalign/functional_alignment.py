import os, glob
from pprint import pprint, pformat
import nibabel as nib
import numpy as np
import sklearn.covariance as skcov
import sklearn.linear_model as linmod
from sklearn import cross_validation
import time
import cPickle
import pyximport
pyximport.install()
import cloops
import scipy.stats
from nngarotte import NonNegativeGarrote
import math
import multiprocessing
import Queue
#from theil_sen import theil_sen




class CloopsInterface(object):

	def __init__(self):
		super(CloopsInterface, self).__init__()


	def construct_ridge_regressions(self, Alist, Blist, adj, trs, voxels):
		return cloops.construct_ridge_regressions(Alist, Blist, adj, trs, voxels)

	def construct_voxel_job(self, Alist, Blist, adj, trs, voxels):
		return cloops.construct_voxel_job(Alist, Blist, adj, trs, voxels)

	def expected_voxels_bytr(self, data, mapping, tr, voxels, scoring_mask):
		return cloops.expected_voxels_bytr(data, mapping, tr, voxels, scoring_mask)

	def calculate_correlation(self, dataA, dataB):
		return cloops.calculate_correlation(dataA, dataB)

	def calculate_theil_sen_slope(self, x, y):
		return cloops.calculate_theil_sen_slope(x, y)






class MappingSlave(multiprocessing.Process):

	def __init__(self, in_queue, out_queue, trs, adjacency, dataA, dataB, regression, mask):
		multiprocessing.Process.__init__(self)
		self.in_queue = in_queue
		self.out_queue = out_queue
		self.kill_recieved = False
		self.adjacency = adjacency
		self.regression = regression
		self.trs = trs
		self.dataA = dataA
		self.dataB = dataB
		self.cloops = CloopsInterface()
		self.mask = mask
		print 'REGRESSION SLAVE ACTIVATED'


	def run(self):

		while not self.kill_recieved:

			try:
				voxel = self.in_queue.get_nowait()

			except Queue.Empty:
				break

			errors_dict = {}
			alpha = 0.
			coefs = [0.]

			if self.mask[voxel] == 0:
				errors_dict['masked_out'] = 1
				mapping = [[vox,1.]]
				mapping_score = 0.

			else:

				[iv, dv, vox] = self.cloops.construct_voxel_job(self.dataA, self.dataB, self.adjacency, 
					self.trs, voxel)

				iv = np.array(iv)
				dv = np.array(dv)

				if np.sum(iv) == 0 or np.sum(dv) == 0:
					errors_dict['iv_or_dv_allzero'] = 1
					mapping = [[vox, 1.]]
					mapping_score = 0.

				else:
					try:
						self.regression.fit(iv, dv)
					
						try:
							mscore = self.regression.score(iv, dv)
						except:
							errors_dict['scoring_error'] = 1
							mscore = 0.

						coefs = self.regression.coef_
						alpha = self.regression.alpha_

						# testing only!!
						#print 'alpha, score, coefsum:', alpha, mscore, np.sum(coefs)

						if mscore == 0.:
							errors_dict['solution_zero_score'] = 1
							mapping = [[vox,1.]]
							mapping_score = 0.

						else:
							mapping = zip(self.adjacency[vox], coefs)
							mapping_score = mscore

					except:
						errors_dict['regression_error'] = 1
						mapping = [[vox,1.]]
						mapping_score = 0.

			self.out_queue.put([vox, mapping, mapping_score, alpha, np.sum(coefs), errors_dict])




class QueueSlave(multiprocessing.Process):

	def __init__(self, in_queue, voxels):
		multiprocessing.Process.__init__(self)
		self.in_queue = in_queue
		self.voxels = voxels
		self.kill_recieved = False

	def run(self):
		voxind = 0
		while not self.kill_recieved:
			if voxind < self.voxels:
				try:
					self.in_queue.put(voxind)
					voxind += 1
				except:
					pass

			else:
				break




class MappingThreadManager(object):

	def __init__(self, threadlimit=6):
		super(MappingThreadManager, self).__init__()
		self.thread_limit = threadlimit
		self.in_queue = multiprocessing.Queue()
		self.out_queue = multiprocessing.Queue()
		self.cloops = CloopsInterface()
		self.completed_voxels = []


	def run(self, dataA, dataB, adj, trs, voxels, regression_method, mapping_mask):

		self.trs = trs
		self.voxels = voxels
		self.dataA = dataA
		self.dataB = dataB
		self.adjacency = adj
		self.method = regression_method
		self.mask = mapping_mask

		start_time = time.time()

		queueslave = QueueSlave(self.in_queue, self.voxels)
		queueslave.start()
		time.sleep(0.5)

		self.run_slaves()
		self.monitor_slaves()

		end_time = time.time()

		print 'regression completion in', end_time-start_time, 'seconds.'

		print 'sorting voxels mappings and scores...'
		self.completed_voxels = sorted(self.completed_voxels, key=lambda x: x[0])
		organized_mappings = [x[1] for x in self.completed_voxels]
		organized_scores = [x[2] for x in self.completed_voxels]

		return organized_mappings, organized_scores



	def run_slaves(self):

		for i in range(self.thread_limit):

			if self.method == 'ridge':
				regression = linmod.RidgeCV(alphas=ridge_alphas, fit_intercept=False)
			elif self.method == 'bayes_ridge':
				regression = linmod.BayesianRidge(n_iter=500, normalize=False, fit_intercept=False)
			elif self.method == 'ard':
				regression = linmod.ARDRegression(n_iter=500,fit_intercept=False, normalize=False)
			elif self.method == 'elastic_net_cv':
				regression = linmod.ElasticNetCV(l1_ratio=elastic_net_cv_ratios,
												 fit_intercept=False,
												 cv=3, n_jobs=8)
			elif self.method == 'garotte':
				regression = NonNegativeGarrote(garotte_alpha, fit_intercept=False,
												estimate_method='ols')
			elif self.method == 'elastic_net':
				regression = linmod.ElasticNet(alpha=elastic_net_alpha, l1_ratio=elastic_net_ratio,
											   fit_intercept=False, normalize=False)

			worker = MappingSlave(self.in_queue, self.out_queue, self.trs, 
				self.adjacency, self.dataA, self.dataB, regression, self.mask)

			worker.start()



	def monitor_slaves(self):

		master_errors_dict = {'iv_or_dv_allzero':0, 'scoring_error':0, 
							  'solution_zero_score':0, 'regression_error':0}
		coefs = []
		alphas = []

		while len(self.completed_voxels) < self.voxels:

			[vox, mapping, mapping_score, alpha, coefsum, errors_dict] = self.out_queue.get()

			for error, errflag in errors_dict.items():
				master_errors_dict[error] += errflag

			coefs.append(coefsum)
			alphas.append(alpha)
			self.completed_voxels.append([vox, mapping, mapping_score])

			if (len(self.completed_voxels) % 5000) == 0:

				if self.method == 'ard':
					print 'performing ARD:', vox, '( median alpha:', np.median(alphas), '- median coefsum:', np.median(coefs), ')'
				elif self.method == 'bayes_ridge':
					print 'performing bayesian ridge:', vox, '( median alpha:', np.median(alphas), '- median coefsum:', np.median(coefs), ')'

				pprint(master_errors_dict)








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


	def data_to_nifti(self, data, mask_3d, mask_affine, filepath, save_dtype=np.float32,
					  normalize=True):

		print 'saving data to file', filepath
		trs, voxels = data.shape

		if normalize:
			data = self.normalize_data(data)

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


	def mapscore_to_nifti(self, data, mask_3d, mask_affine, filepath, save_dtype=np.float32,
						  normalize=False):

		print 'saving data to file', filepath

		voxels = data.shape[0]

		if normalize:
			data = self.normalize_data(data)

		unmasked = np.zeros(mask_3d.shape)
		unmasked[np.asarray(mask_3d).astype(np.bool)] = np.squeeze(np.array(data))

		unmasked = np.transpose(unmasked, [2,1,0])
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
		print 'trs, voxels:', trs, voxels

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


	def construct_correlation_brain(self, nifti_dict):

		print 'constructing the correlation brain...'

		trs, voxels = nifti_dict[nifti_dict.keys()[0]].shape
		corr_brain = np.zeros(voxels)

		for voxel in range(voxels):
			if (voxel % 5000) == 0:
				print 'voxel', voxel, '...'
			vox_arrays = []
			for sub, data in nifti_dict.items():
				vox_arrays.append(data[:,voxel])
			vox_arrays = np.array(vox_arrays)
			corr_table = np.corrcoef(vox_arrays)
			corr_sums = (np.sum(corr_table, axis=0)-1.)/(len(corr_table)-1)
			tstat, pval = scipy.stats.ttest_1samp(corr_sums, 0.)
			corr_brain[voxel] = tstat

		return corr_brain


	def mask_from_correlation_brain(self, correlation_brain, threshold=1.96, absolute=True):

		print 'converting correlation statistics to mask...'

		corr_brain = correlation_brain.copy()

		if absolute:
			corr_brain = np.absolute(corr_brain)

		mask_brain = np.array([0 if x < threshold else 1 for x in corr_brain])

		return mask_brain



	def construct_theil_sen_mask(self, comparator_dict, median_subject, pvalue_thresh=0.05, verbose=True):

		# must have more than one subject in comparator dict!

		trs, voxels = median_subject.shape

		theil_sen_mask = np.zeros(voxels)

		pvalues = []
		slope_means = []

		for voxel in range(voxels):
			x_vectors = []
			y_vector = np.array(median_subject[:,voxel]).tolist()

			if not np.sum(y_vector) == 0:
				for skey in comparator_dict.keys():
					subdata = comparator_dict[skey]
					x_vectors.append(np.array(subdata[:,voxel]).tolist())

				#assert x_vectors[0].shape == y_vector.shape

				theil_sen_slopes = []
				#print y_vector

				for xv in x_vectors:
					if np.sum(xv) == 0:
						theil_sen_slopes.append(0.)
					else:
						#output = theil_sen(y_vector, xv, sample=False)
						#theil_sen_slopes.append(output[0])
						slopes = self.cloops.calculate_theil_sen_slope(xv, y_vector)
						medslope = np.median(slopes)
						#print medslope
						theil_sen_slopes.append(medslope)

				#print theil_sen_slopes
				#print scipy.stats.ttest_1samp(np.array(theil_sen_slopes),0.)
				t_, p_ = scipy.stats.ttest_1samp(np.array(theil_sen_slopes),0.)
				p_ = np.nan_to_num(p_)

				if p_ < pvalue_thresh:
					theil_sen_mask[voxel] = 1

				pvalues.append(p_)
				slope_means.append(np.mean(theil_sen_slopes))

				if verbose:
					#print 'mean slope, p', np.mean(theil_sen_slopes), p_
					if (voxel % 500) == 0:
						print 'voxel', voxel, 'slope, p:', np.mean(slope_means), np.mean(pvalues)


		return theil_sen_mask




	def create_mapping(self, dataA, dataB, adjacency, method='ridge',
					   ridge_alphas=[.01,.1,1.,10.], elastic_net_cv_ratios=[.1,.5,.9,.95],
					   elastic_net_alpha=1.0, elastic_net_ratio=0.5, garotte_alpha=0.0005,
					   threaded=False, mapping_mask=None):
		
		print 'mapping A to B...'
		mapstart = time.time()

		print dataA.shape, dataB.shape
		assert dataA.shape == dataB.shape
		assert dataA.shape[1] == dataB.shape[1] == len(adjacency)

		mapping = []
		mapping_score = []

		trs, voxels = dataB.shape

		dataA_list = dataA.tolist()
		dataB_list = dataB.tolist()

		if mapping_mask is None:
			print 'setting mapping mask to all 1s (allow all voxels)...'
			mapping_mask = np.ones(voxels)


		if threaded:

			mapping_threader = MappingThreadManager()
			mapping, mapping_score = mapping_threader.run(dataA_list, dataB_list, adjacency, trs, voxels, method,
				mapping_mask)
			
			mapend = time.time()

			print 'total time elapsed:', mapend-mapstart, '\n'
			print 'average mapping score:', np.mean(mapping_score)


		else:

			# assumes data does NOT need intercept / is normalized!
			if method == 'ridge':
				regression = linmod.RidgeCV(alphas=ridge_alphas, fit_intercept=False)
			elif method == 'bayes_ridge':
				regression = linmod.BayesianRidge(n_iter=500, normalize=False, fit_intercept=False)
			elif method == 'ard':
				regression = linmod.ARDRegression(n_iter=500,fit_intercept=False, normalize=False)
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
			zero_within_errors = 0
			alphas_count = [ 0 for x in ridge_alphas ]
			alpha_history = []
			lambda_history = []
			sigma_history = []
			scoring_errors = 0
			regression_errors = 0
			mapping_skips = 0
		

			A_iv_matrices, B_dv_vectors = self.cloops.construct_ridge_regressions(dataA_list,
																				  dataB_list,
																				  adjacency,
																				  trs,
																				  voxels)


			for iv, dv, vox in zip(A_iv_matrices, B_dv_vectors, range(voxels)):

				if (vox % 5000) == 0:
					if method == 'ridge':
						print 'preforming ridge regression:', vox, '(zeroed: ', zero_within_errors, ') alphas:', alphas_count
					elif method == 'bayes_ridge':
						print 'performing bayesian ridge:', vox, '(zeroed:', zero_within_errors, ' - mapping skips:', mapping_skips, ' - median alpha:', np.median(alpha_history), ')'
					elif method == 'ard':
						print 'performing automatic relevance determination:', vox, '(solution zeros:', solution_zero_errors, ' - scoring errors:', scoring_errors, ' - median alpha:', np.median(alpha_history), ')'
						#print '\t\t median alpha (precision of noise): ', np.median(alpha_history)
						#print '\t\t median lambda (precision of weights): ', np.median(lambda_history)
						#print '\t\t median sigma (var/covar of weights): ', np.median(sigma_history)
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

				elif mapping_mask[vox] == 0.:
					mapping_skips += 1
					mapping.append([[vox,1.]])
					mapping_score.append(0.)

				else:
					try:
						regression.fit(iv, dv)

						if method == 'ard':
							try:
								mscore = regression.score(iv, dv)
							except:
								mscore = 0.
								scoring_errors += 1

						else:
							try:
								mscores = cross_validation.cross_val_score(regression, iv, dv, cv=6)
								mscore = np.mean(mscores)
							except:
								scoring_errors += 1
								mscore = 0.
								mscores = np.array([0, 0, 0, 0, 0, 0])
								zero_within_errors += 1

							# simple check to see if 0 is within the standard deviation of the mean
							# score. if so, set to zero:
							if mscore - (mscores.std() / math.sqrt(6)) <= 0.:
								zero_within_errors += 1
								mscore = 0.


						coefs = regression.coef_
						alpha = regression.alpha_

						if method == 'ridge':
							for i,a in enumerate(ridge_alphas):
								if a == alpha:
									alphas_count[i] += 1

						elif method == 'bayes_ridge' or method == 'ard':
							alpha_history.append(alpha)


						# testing only!!
						#print 'alpha, score, coefsum:', alpha, mscore, np.sum(coefs)

						if mscore == 0.:
							solution_zero_errors += 1
							mapping.append([[vox,1.]])
							mapping_score.append(0.0)

						else:
							mapping.append(zip(adjacency[vox], coefs))
							mapping_score.append(mscore)
							
					except:
						regression_errors += 1
						mapping.append([[vox,1.]])
						mapping_score.append(0.0)
					

			mapend = time.time()

			print 'all zero iv or dv errors:', all_zero_errors
			print 'solution zero errors:', solution_zero_errors
			print 'percentage zeroed due to error:', float(zero_within_errors)/float(voxels)
			print 'total time elapsed:', mapend-mapstart, '\n'
			print 'average mapping score:', np.mean(mapping_score)

		return mapping, mapping_score





	def create_average_score(self, mapping_scores, include_nonzero_only=False, normalize=False,
							 apply_logistic_function=True):

		print 'creating average score from mapping scores...'

		zero_inds = None
		average = None

		for subk, mapscore in mapping_scores.items():

			print 'on', subk
			
			mapscore = np.array(mapscore)

			if include_nonzero_only:

				if zero_inds is None:
					zero_inds = np.zeros(mapscore.shape[0])

				for i, elem in enumerate(mapscore):
					if elem == 0.:
						zero_inds[i] = 1

			if average is None:
				average = mapscore
			else:
				average = average+mapscore


		if include_nonzero_only:
			for i, b in enumerate(zero_inds):
				if b == 1:
					average[i] = 0.

		average = average/len(mapping_scores)

		if normalize:
			avgmean = np.mean(average)
			avgstd = np.std(average)
			average = average-avgmean
			average = average/avgstd

			# apply the logistic function
			if apply_logistic_function:
				average = np.array(1. / (1. + np.exp(-1. * average)))


		return average



	def create_median_score(self, mapping_scores, normalize=False, apply_logistic_function=True):

		print 'creating median score from mapping scores...'

		subkeys = mapping_scores.keys()

		voxels = len(mapping_scores[subkeys[0]])
		median = np.zeros(voxels)

		for voxel in range(voxels):
			vox_vals = []
			for sk in subkeys:
				vox_vals.append(mapping_scores[sk][voxel])
			median[voxel] = np.median(np.array(vox_vals))

		if normalize:
			avg_med = np.mean(median)
			std_med = np.std(median)
			median = median-avg_med
			median = median/std_med

			# apply the logistic function
			if apply_logistic_function:
				median = np.array(1. / (1. + np.exp(-1. * median)))


		return median




	def generate_mappings(self, nifti_dict, model_data, adjacency, ridge_alphas=[.01,.1,1.,10.],
						  elastic_net_cv_ratios=[.1,.5,.9,.95], garotte_alpha=0.0005, method='ridge', 
						  conserve_memory=False, elastic_net_alpha=1.0, elastic_net_ratio=0.5,
						  specific_keys=None, threaded=False, mapping_mask=None):

		print 'generating mappings...'
		mappings_dict = {}
		mappings_scores = {}

		if specific_keys is None:
			skeys = nifti_dict.keys()
		else:
			skeys = specific_keys

		for skey in skeys:
			print 'creating mapping for subject:', skey
			mapping, mapping_score = self.create_mapping(nifti_dict[skey], model_data, adjacency, ridge_alphas=ridge_alphas,
										  				 elastic_net_cv_ratios=elastic_net_cv_ratios, method=method,
										  				 garotte_alpha=garotte_alpha, elastic_net_alpha=elastic_net_alpha,
										  				 elastic_net_ratio=elastic_net_ratio, threaded=threaded,
										  				 mapping_mask=mapping_mask)
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




	def check_median_brain_significance(self, median_brain, absolute=True, keep='dev', inversep=True):

		print 'checking median brain voxels for significance...'

		median_brain = np.array(median_brain)
		voxels = median_brain.shape[1]
		print median_brain.shape
		check_brain = np.zeros(median_brain.shape[1])

		for voxel in range(voxels):
			timeseries = median_brain[:,voxel]
			if absolute:
				timeseries = np.abs(timeseries)
			t, p = scipy.stats.ttest_1samp(timeseries,0.)
			if keep == 'p':
				check_brain[voxel] = (1-p)
			elif keep == 't':
				check_brain[voxel] = t
			elif keep == 'dev':
				check_brain[voxel] = np.mean(timeseries)

		return check_brain



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



	def data_from_mapping(self, data, mapping, test_row_sums=False, scoring_mask=None):

		print 'creating new data from mapping...'

		trs, voxels = data.shape
		data_list = data.tolist()
		mapped = []

		if scoring_mask == None:
			scoring_mask = [1 for x in range(voxels)]

		convertstart = time.time()

		for tr in range(trs):
			if (tr % 20) == 0:
				print 'mapping tr:', tr

			tr_row = self.cloops.expected_voxels_bytr(data_list, mapping, tr, voxels, scoring_mask)
			if test_row_sums:
				print np.sum(tr_row)
			else:
				mapped.append(tr_row)

		mapped = np.array(mapped)

		convertend = time.time()
		print 'conversion time:', convertend-convertstart

		return mapped


	def map_data(self, nifti_dict, mappings, conserve_memory=False, test_row_sums=False,
				 scoring_mask=None):

		print 'mapping nifti_dict...'

		mapped_niftis = {}

		skeys = nifti_dict.keys()
		for skey in skeys:
			print 'map of subject', skey
			data = nifti_dict[skey]

			mapped_data = self.data_from_mapping(data, mappings[skey], test_row_sums=test_row_sums,
												 scoring_mask=None)
			mapped_niftis[skey] = mapped_data

			if conserve_memory:
				del nifti_dict[skey]
				del mappings[skey]

		return mapped_niftis


	def score_parser(self, scores, cutoff, cutoff_type):

		if cutoff_type == 'percent':

			minimum = min(scores)
			maximum = max(scores)
			ptp = maximum-minimum

			assert cutoff >= 0.0 and cutoff <= 1.0

			cutmax = maximum - ptp*cutoff

			return [1 if x >= cutmax else 0 for x in scores]

		elif cutoff_type == 'raw':

			return [1 if x >= cutoff else 0 for x in scores]

		elif cutoff_type == 'pvalue':

			zscores = scipy.stats.zscore(scores)
			zthreshold = scipy.stats.norm.isf(cutoff)

			return [1 if x >= zthreshold else 0 for x in zscores]

		elif cutoff_type == 'nonzero':

			return [1 if x > 0. else 0 for x in scores]



	def map_data_score_cutoff(self, nifti_dict, mappings, mappings_scores, cutoff, conserve_memory=False,
							  cutoff_type='percent', single_score=False):

		print 'mapping nifti dict with score cutoff:', cutoff
		print 'score cutoff type is:', cutoff_type

		mapped_niftis = {}
		skeys = nifti_dict.keys()

		if single_score:
			nmappings_scores = {}
			for skey in skeys:
				nmappings_scores[skey] = mappings_scores
			mappings_scores = nmappings_scores

		for skey in skeys:
			print 'mapping subject', skey

			data = nifti_dict[skey]
			scores = mappings_scores[skey]
			scores_bool = self.score_parser(scores, cutoff, cutoff_type)
			print 'Sum scores:', np.sum(scores_bool)
			print len(scores_bool)
			mapped_data = self.data_from_mapping(data, mappings[skey], scoring_mask=scores_bool)
			mapped_niftis[skey] = mapped_data

			if conserve_memory:
				del nifti_dict[skey]
				del mappings[skey]
				del mappings_scores[skey]

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
						   directory=os.getcwd(), normalize=True):

		print 'converting dict to niftis'

		for subject_key, data in nifti_dict.items():

			print 'converting', subject_key, 'to nifti...'

			try:
				os.makedirs(directory)
			except:
				pass

			filepath = os.path.join(directory, subject_key+suffix)

			self.assistant.data_to_nifti(data, mask, mask_affine, filepath, normalize=normalize)




	# INCOMPLETE:
	'''
	def load_subject_npys(self, subject_npys):

		print 'loading subject data...'

		nifti_dict = {}

		for npf in subject_npys:
			skey = os.path.split(npf)[1].rstrip('.npy')
			print 'loading', skey, '...'

			nifti_dict[skey] = np.load(skey)

		return nifti_dict
	'''













