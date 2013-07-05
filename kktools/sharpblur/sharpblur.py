import scipy as sp 
import numpy as np 
import os, glob
import scipy.signal as signal
import nibabel as nib
import sklearn.covariance as skcov
import sklearn.linear_model as linear_model
from sklearn import cross_validation
import time
from matplotlib import pyplot as pl
import scipy.stats
import math
import itertools
import pprint

import regreg.api as rr 
from statsmodels.robust.norms import HuberT
from statsmodels.robust.scale import mad


class ComputeFunctions(object):

	def __init__(self):
		super(ComputeFunctions, self).__init__()
		self.steps = [(di ,dj, dk) for di, dj, dk in itertools.product([-1,0,1], [-1,0,1], [-1,0,1]) if (di, dj, dk) != (0, 0, 0)]

	def compute_XB(self, X, B):
	    total = 0
	    I, J, K = X.shape[1:]
	    for b, s in enumerate(self.steps):
	        di, dj, dk = s
	        Xs = X[:,(1+di):(I-1+di),(1+dj):(J-1+dj),(1+dk):(K-1+dk)]
	        Bs = B[b]
	        total += Xs * Bs
	    return total

	def compute_XtY(self, X, Y):
	    I, J, K = X.shape[1:]
	    total = np.zeros((len(self.steps),I-2,J-2,K-2))
	    for b, s in enumerate(self.steps):
	        di, dj, dk = s
	        Xs = X[:,(1+di):(I-1+di),(1+dj):(J-1+dj),(1+dk):(K-1+dk)]
	        total[b] = (Xs * Y).sum(0)
	    return total


class ImageHuber(rr.smooth_atom):

	def __init__(self, X, t=1.345, offset=None, quadratic=None, initial=None, coef=1.,
		scale=None):

		T, I, J, K, = self.Xshape = X.shape

		shape = (26, I-2, J-2, K-2)
		self.X = X
		self.Y = X[:, 1:(I-1), 1:(J-1), 1:(K-1)]
		self.huber = HuberT(t=t)

		if scale is None:
			self.scale = np.ones((I-2, J-2, K-2))
		else:
			self.scale = scale.reshape((I-2, J-2, K-2))

		rr.smooth_atom.__init__(self, shape, initial=initial, coef=coef, offset=offset,
			quadratic=quadratic)

		self.compute = ComputeFunctions()


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

	def __init__(self):
		super(SharpBlur, self).__init__
		self.compute = ComputeFunctions()
		self.verbose = True


	def load_data_mask(self, data_path, mask_path, detrend=True, zscore_in=False):

		nifti = nib.load(data_path)
		ndata = nifti.get_data()
		naffine = nifti.get_affine()

		mask = nib.load(mask_path)
		mdata = mask.get_data()
		maffine = self.mask_affine = mask.get_affine()

		if self.verbose:
			print ndata.shape

		data = np.transpose(ndata, [3,2,1,0])
		
		if self.verbose:
			print data.shape, data.shape[0], np.product(data.shape[1:])

		self.data_shape = data.shape
		d0,d1,d2,d3 = data.shape

		#data.shape = (d0, d1*d2*d3)

		mask = self.mask = np.transpose(mdata, [2,1,0])

		self.nmask = np.zeros((26, self.mask.shape[0]-2, self.mask.shape[1]-2, self.mask.shape[2]-2))
		for i in range(26):
			self.nmask[i,:,:,:] = self.mask[1:-1,1:-1,1:-1]

		#fmask = mask.astype(np.bool)
		#fmask.shape = np.product(mask.shape)
		#self.fmask = fmask

		#data = data[:, fmask]
		#trs, voxels = data.shape

		if self.verbose:
			print d0, d1, d2, d3

		if detrend:
			if self.verbose:
				print 'detrending...'
			data = signal.detrend(data, axis=0)

		if zscore_in:
			if self.verbose:
				print 'zscoring...'
			zdata = (data-np.mean(data, axis=0)) / np.std(data, axis=0)
			data = np.nan_to_num(zdata)

		return data


	def estimate_step_size(self, image_huber, X, num_its=10):
		B = np.random.standard_normal(image_huber.shape)
		for _ in range(num_its):
			XtXB = self.compute.compute_XtY(X, self.compute.compute_XB(X, B))
			L = np.linalg.norm(XtXB) / np.linalg.norm(B)
			B = XtXB / np.linalg.norm(XtXB)
		return 1. / L


	def blur(self, X):

		# Get an approximate OLS fit to find out the rough scale of the data.
		# Basically squared error with t=1e6:
		loss = ImageHuber(X)

		# Set up non-negativity constraint:
		constraint = rr.nonnegative(loss.shape)

		# Set up regreg problem with the Huber Loss image class and the constraint:
		problem = rr.simple_problem(loss, constraint)

		# Initial guess at the step size. The estimate used is the square of the
		# largest singular value of the linear map: B -> Yhat(B)
		step = self.estimate_step_size(loss, X)

		# Initialize the regreg solver:
		solver = rr.FISTA(problem)

		# Solve the problem:
		solver.fit(tol=1.e-3, max_its=200, debug=True, start_step=step)
		#solver.fit(tol=1., max_its=2, debug=True, start_step=step*5)

		# Assume that there is a roughly constant scale factor across the image:
		#print 'coefs shape:', solver.composite.coefs.shape
		#print 'medians:', np.median(solver.composite.coefs), np.median(solver.composite.coefs[np.asarray(self.nmask).astype(np.bool)])
		#scale = np.median(loss.estimate_scale(solver.composite.coefs[np.asarray(self.nmask).astype(np.bool)]))
		#print 'SCALE:', scale

		scale = loss.estimate_scale(solver.composite.coefs)
		print 'Scale shape:', scale.shape


		# Initialize the second loss function with the proper scale and the problem
		# class:
		final_loss = ImageHuber(X, scale=scale)
		final_problem = rr.simple_problem(final_loss, constraint)

		final_problem.coefs[:] = problem.coefs

		# Solve for the coefficients:
		nonzero_scale = final_loss.scale[np.absolute(final_loss.scale) > 1.e-8]
		print np.median(final_loss.scale), np.median(nonzero_scale)
		final_coefs = final_problem.solve(tol=5e-5, debug=True, start_step=step/np.median(nonzero_scale),
			max_its=100, min_its=20)

		# Produce the output blurred data with the found coefficients.
		# The Y value is added in with a weight of 1. and the we normalize by 
		# 1. + sum of weights:
		final_wsum = final_coefs.sum(0)
		final_Yhat = self.compute.compute_XB(final_loss.X, final_coefs)
		final_data = (final_loss.Y + final_Yhat) / (1 + final_wsum)

		return final_data



	def output_maps(self, data_out, output_filename, zscore_out=True):

		if zscore_out:
			zdata_out = (data_out-np.mean(data_out, axis=0)) / np.std(data_out, axis=0)
			zdata_out = np.nan_to_num(zdata_out)
			data_out = zdata_out

		full_data_out = np.zeros(self.data_shape)
		print 'full out', full_data_out.shape
		print 'data out', data_out.shape
		full_data_out[:,1:-1,1:-1,1:-1] = data_out
		data_out = full_data_out
		print 'new data out', data_out.shape


		out_mask = [np.ones(self.mask.shape) for i in range(data_out.shape[0])]
		for i in range(data_out.shape[0]):
			out_mask[i][np.asarray(self.mask).astype(np.bool)] = 0
		out_mask = np.array(out_mask)

		data_out[np.asarray(out_mask).astype(np.bool)] = 0.

		#print 'formatting for nifti...'
		#unmasked = [np.zeros(self.mask.shape) for i in range(trs)]
		#for i in range(trs):
		#	unmasked[i][np.asarray(self.mask).astype(np.bool)] = np.squeeze(np.array(data_out[i]))
		#	unmasked[i] = np.transpose(unmasked[i],[2,1,0])

		#print 'transposing...'
		#unmasked = np.transpose(unmasked, [1,2,3,0])
		#print unmasked.shape, ndata.shape
		#print 'changing data type...'
		#unmasked = np.array(unmasked, dtype=np.float32)

		data_out = np.transpose(data_out, [3,2,1,0])

		print 'saving nifti...'
		nii = nib.Nifti1Image(data_out, self.mask_affine)
		if os.path.exists(output_filename):
			os.remove(output_filename)
		nii.to_filename(output_filename)



	def run(self, nifti_path, mask_path, output_path, detrend=True, zscore_in=False,
		zscore_out=True):

		X = self.load_data_mask(nifti_path, mask_path, detrend=detrend, zscore_in=zscore_in)
		Xout = self.blur(X)
		self.output_maps(Xout, output_path, zscore_out=zscore_out)












