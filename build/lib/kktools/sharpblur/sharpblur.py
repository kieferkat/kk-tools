import scipy as sp 
import numpy as np 
import os, glob
import nibabel as nib
import time
import scipy.stats
import math
import itertools
import pprint
import scipy.signal as signal

import regreg.api as rr 
from statsmodels.robust.norms import HuberT
from statsmodels.robust.scale import mad


class ComputeFunctions(object):

	def __init__(self, step_size=1):
		super(ComputeFunctions, self).__init__()
		self.step = step_size
		self.step_range = range(-step_size,step_size+1)
		self.steps = self.compute_steps(self.step_range)


	def compute_steps(self, step_range):
		steps = [(di ,dj, dk) for di, dj, dk in itertools.product(step_range, step_range, step_range) if (di, dj, dk) != (0, 0, 0)]
		return steps

	def compute_XB(self, X, B):
	    total = 0
	    I, J, K = X.shape[1:]
	    for b, s in enumerate(self.steps):
	        di, dj, dk = s
	        Xs = X[:,(self.step+di):(I-self.step+di),(self.step+dj):(J-self.step+dj),(self.step+dk):(K-self.step+dk)]
	        Bs = B[b]
	        total += Xs * Bs
	    return total

	def compute_XsqrtB(self, X, B):
	    total = 0
	    I, J, K = X.shape[1:]
	    for b, s in enumerate(self.steps):
	        di, dj, dk = s
	        Xs = X[:,(self.step+di):(I-self.step+di),(self.step+dj):(J-self.step+dj),(self.step+dk):(K-self.step+dk)]
	        Bs = np.sqrt(B[b])
	        total += Xs * Bs
	    return total

	def compute_XsqB(self, X, B):
	    total = 0
	    I, J, K = X.shape[1:]
	    for b, s in enumerate(self.steps):
	        di, dj, dk = s
	        Xs = X[:,(self.step+di):(I-self.step+di),(self.step+dj):(J-self.step+dj),(self.step+dk):(K-self.step+dk)]
	        Bs = B[b]**2
	        total += Xs * Bs
	    return total

	def compute_XtY(self, X, Y):
	    I, J, K = X.shape[1:]
	    total = np.zeros((len(self.steps),I-2*self.step,J-2*self.step,K-2*self.step))
	    for b, s in enumerate(self.steps):
	        di, dj, dk = s
	        Xs = X[:,(self.step+di):(I-self.step+di),(self.step+dj):(J-self.step+dj),(self.step+dk):(K-self.step+dk)]
	        total[b] = (Xs * Y).sum(0)
	    return total

	def fwhm_to_stdev(self, fwhm):
		return float(fwhm)/2.3548201

	def pdf_center_at_fwhm(self, fwhm):
		return scipy.stats.norm.pdf(0,loc=0,scale=self.fwhm_to_stdev(fwhm))



class ImageHuber(rr.smooth_atom):

	def __init__(self, X, t=1.345, offset=None, quadratic=None, initial=None, coef=1.,
		scale=None, step_size=1, verbose=True):

		self.verbose = verbose

		# set step size
		if self.verbose: print 'step size:', step_size
		self.step = step_size

		# determine dimensions:
		if self.verbose: print 'X shape:', X.shape
		T, I, J, K, = self.Xshape = X.shape

		# calculate neighborhood size based on the step size
		self.neighborhood_size = (step_size*2+1)**3 - 1
		if self.verbose: print 'neighborhood size:', self.neighborhood_size 

		# determine the usable area for coefficient calculation
		shape = (self.neighborhood_size, I-(step_size*2), J-(step_size*2), K-(step_size*2))
		if self.verbose: print 'regression shape:', shape

		# set up X, Y, huber loss w/ t
		self.X = X
		self.Y = X[:, step_size:(I-step_size), step_size:(J-step_size), step_size:(K-step_size)]
		self.huber = HuberT(t=t)

		if scale is None:
			self.scale = np.ones((I-(step_size*2), J-(step_size*2), K-(step_size*2)))
		else:
			self.scale = scale.reshape((I-(step_size*2), J-(step_size*2), K-(step_size*2)))

		rr.smooth_atom.__init__(self, shape, initial=initial, coef=coef, offset=offset,
			quadratic=quadratic)

		self.compute = ComputeFunctions(step_size=step_size)


	def smooth_objective(self, coef, mode='both', check_feasibility=False):
		Yhat = self.compute.compute_XB(self.X, coef)
		R = self.Y - self.Yhat(coef)
		s = self.scale[np.newaxis,...]

		if mode == 'both':
			f = self.huber.rho(R * s).sum()
			g = -self.compute.compute_XtY(self.X, s * self.huber.psi(R * s))
			return f, g
		elif mode == 'grad':
			g = -self.compute.compute_XtY(self.X, s * self.huber.psi(R * s))
			return g
		elif mode == 'func':
			f = self.huber.rho(R * s).sum()
			return f
		else:
			raise ValueError("mode incorrectly specified")


	def estimate_scale(self, coef):
		Yhat = self.compute.compute_XB(self.X, coef)
		R = self.Y - Yhat
		self.scale = mad(R)
		return self.scale


	def Yhat(self, coef):
		return self.compute.compute_XB(self.X, coef)



class SharpBlur(object):

	def __init__(self, step_size=1, Y_weight=1.):
		super(SharpBlur, self).__init__()
		self.step = step_size
		self.compute = ComputeFunctions(step_size=step_size)
		self.Y_weight = Y_weight
		self.verbose = True


	def load_data_mask(self, data_path, mask_path, detrend=True, zscore_in=False):
		'''
		Function to load in mask and nifti data. Will detrend & z-score if desired.
		'''

		# load nifti data:
		ndata = nib.load(data_path).get_data()

		# load mask, mask affine
		mask = nib.load(mask_path)
		mdata = mask.get_data()
		maffine = self.mask_affine = mask.get_affine()

		if self.verbose:
			print 'data shape:', ndata.shape
			print 'mask shape:', mdata.shape

		# reverse-transpose the data
		data = np.transpose(ndata, [3,2,1,0])
		
		if self.verbose:
			print 'transposed data shape, trs, datapoints:', data.shape, data.shape[0], np.product(data.shape[1:])

		# self.data_shape set to transposed shape:
		self.data_shape = data.shape

		# transpose and store the mask:
		mask = self.mask = np.transpose(mdata, [2,1,0])

		# Detrend the data along time-axis, if desired
		if detrend:
			if self.verbose:
				print 'detrending...'
			data = signal.detrend(data, axis=0)

		# z-score the data along time-axis, if desired
		if zscore_in:
			if self.verbose:
				print 'zscoring...'
			zdata = (data-np.mean(data, axis=0)) / np.std(data, axis=0)
			data = np.nan_to_num(zdata)

		return data


	def estimate_step_size(self, image_huber, X, num_its=10):
		'''
		JT's algorithim for determining the initial step size.
		'''
		B = np.random.standard_normal(image_huber.shape)
		for _ in range(num_its):
			XtXB = self.compute.compute_XtY(X, self.compute.compute_XB(X, B))
			L = np.linalg.norm(XtXB) / np.linalg.norm(B)
			B = XtXB / np.linalg.norm(XtXB)
		return 1. / L


	def blur(self, X, huber_t=1.345, nonnegative=True, force_coefs_positive=False,
		scale_huber=True, beta_boost=1.):

		# Get an approximate OLS fit to find out the rough scale of the data.
		# Basically squared error with t=1e6:
		if self.verbose: print 'Initializing Huber loss class...'
		loss = ImageHuber(X, step_size=self.step, t=huber_t)

		# Set up non-negativity constraint:
		if nonnegative:
			if self.verbose: print 'Initializing non-negative constraint...'
			constraint = rr.nonnegative(loss.shape)

		# Set up regreg problem with the Huber Loss image class and the constraint:
		if self.verbose: print 'Initializing regreg problem class...'
		if nonnegative:
			problem = rr.simple_problem(loss, constraint) #constrained
		else:
			problem = rr.simple_problem.smooth(loss) #unconstrained
		
		# Initial guess at the step size. The estimate used is the square of the
		# largest singular value of the linear map: B -> Yhat(B)
		if self.verbose: print 'Estimating initial step size...'
		step = self.estimate_step_size(loss, X)
		
		# Initialize the regreg solver:
		if self.verbose: print 'Initializing FISTA solver...'
		solver = rr.FISTA(problem)

		# Solve the problem:
		if self.verbose: print 'Solving initial problem...'
		solver.fit(tol=1.e-3, max_its=200, debug=True, start_step=step)

		# Estimate the scale for each voxel individually based on the coefficients
		if scale_huber:
			if self.verbose: print 'Estimating voxel-wise scale from initial coefficients...'
			scale = loss.estimate_scale(solver.composite.coefs)
		
		# Initialize the second loss function with the proper scale and the problem
		# class:
		if self.verbose: print 'Initializing final problem with scale & warm-start...'
		
		if scale_huber:
			final_loss = ImageHuber(X, step_size=self.step, t=huber_t, scale=scale)
		else:
			final_loss = ImageHuber(X, step_size=self.step, t=huber_t)

		if nonnegative:
			final_problem = rr.simple_problem(final_loss, constraint) #constrained
		else:
			final_problem = rr.simple_problem.smooth(final_loss) #unconstrained

		final_problem.coefs[:] = problem.coefs

		# Solve for the coefficients:
		if self.verbose: print 'Solving final problem...'
		nonzero_scale = final_loss.scale[np.absolute(final_loss.scale) > 1.e-6]
		print np.median(nonzero_scale)
		final_coefs = final_problem.solve(tol=5e-5, debug=True, start_step=step/np.median(nonzero_scale),
			max_its=100, min_its=20) #5e-5

		# Produce the output blurred data with the found coefficients.
		# The Y value is added in with a weight of 1. and the we normalize by 
		# 1. + sum of weights:
		if self.verbose: print 'Computing final data from coefficients...'
		if force_coefs_positive:
			poscoefs = final_coefs.copy()
			poscoefs[poscoefs < 0.] = 0.
			final_coefs = poscoefs

		if beta_boost != 1.:
			if self.verbose: print 'Boosting betas by factor:', beta_boost, '...'
			final_coefs**(1./beta_boost)

		final_wsum = final_coefs.sum(0)
		final_Yhat = self.compute.compute_XB(final_loss.X, final_coefs)

		#final_Yhat = self.compute.compute_XsqrtB(final_loss.X, final_coefs)
		#final_Yhat = self.compute.compute_XB_posonly(final_loss.X, final_coefs)

		#final_data = (final_loss.Y + final_Yhat) / (1 + final_wsum)

		final_data = ( (final_loss.Y * self.Y_weight) + final_Yhat ) / (self.Y_weight + final_wsum)

		return final_data



	def output_maps(self, data_out, output_filename, zscore_out=True):

		# z-score output if desired:
		if self.verbose: print 'zscoring output...'
		if zscore_out:
			zdata_out = (data_out-np.mean(data_out, axis=0)) / np.std(data_out, axis=0)
			zdata_out = np.nan_to_num(zdata_out)
			data_out = zdata_out

		# initialize data_output
		full_data_out = np.zeros(self.data_shape)
		if self.verbose:
			print 'data output shape:', full_data_out.shape
			print 'data current shape:', data_out.shape
		full_data_out[:,self.step:-self.step,self.step:-self.step,self.step:-self.step] = data_out
		data_out = full_data_out
		if self.verbose: print 'shape-justified data output shape:', data_out.shape

		# create reverse-mask to zero non-brain areas:
		if self.verbose: print 'Creating reverse-mask...'
		out_mask = [np.ones(self.mask.shape) for i in range(data_out.shape[0])]
		for i in range(data_out.shape[0]):
			out_mask[i][np.asarray(self.mask).astype(np.bool)] = 0
		out_mask = np.array(out_mask)

		# zero out non-brain areas:
		if self.verbose: print 'Zeroing out non-brain areas...'
		data_out[np.asarray(out_mask).astype(np.bool)] = 0.

		# transpose back to nifti format:
		if self.verbose: print 'Transposing back to nifti format...'
		data_out = np.transpose(data_out, [3,2,1,0])

		# save nifti to file using mask affine
		if self.verbose: print 'Saving nifti', output_filename, '...'
		nii = nib.Nifti1Image(data_out, self.mask_affine)
		if os.path.exists(output_filename):
			os.remove(output_filename)
		nii.to_filename(output_filename)



	def run(self, nifti_path, mask_path, output_path, detrend=True, zscore_in=False,
		zscore_out=True, nonnegative=True, force_coefs_positive=False, scale_huber=True):

		# load in the X data matrix and the mask, detrend and zscore if desired:
		X = self.load_data_mask(nifti_path, mask_path, detrend=detrend, zscore_in=zscore_in)

		# preform the sharpblur on X
		Xout = self.blur(X, nonnegative=nonnegative, force_coefs_positive=force_coefs_positive,
			scale_huber=scale_huber)

		# output the blurred data:
		self.output_maps(Xout, output_path, zscore_out=zscore_out)












