

import math
import numpy as np



def construct_voxel_job(list dataA, list dataB, list adjacency, int trs, int voxel):

	cdef int tr, voxel_adj_ind
	cdef list jobs, iv_matrix, dv_vector, voxels_adjacent, adjacency_row

	voxels_adjacent = adjacency[voxel]
	iv_matrix = []
	dv_vector = []

	for tr in range(trs):
		dv_vector.append(dataB[tr][voxel])

		adjacency_row = []
		for voxel_adj_ind in voxels_adjacent:
			adjacency_row.append(dataA[tr][voxel_adj_ind])

		iv_matrix.append(adjacency_row)

	job = [iv_matrix, dv_vector, voxel]

	return job


def construct_ridge_regressions(list dataA, list dataB, list adjacency, int trs, int voxels):

	cdef int voxel_ind, tr, voxel_adj_ind
	cdef list iv_matrix, dv_vector, iv_matrices, dv_vectors, voxels_adjacent, adjacency_row

	iv_matrices = []
	dv_vectors = []

	for voxel_ind in range(voxels):
		
		if (voxel_ind % 5000) == 0:
			print 'regressions prepared:', voxel_ind

		voxels_adjacent = adjacency[voxel_ind]
		iv_matrix = []
		dv_vector = []

		for tr in range(trs):
			dv_vector.append(dataB[tr][voxel_ind])

			adjacency_row = []
			for voxel_adj_ind in voxels_adjacent:
				adjacency_row.append(dataA[tr][voxel_adj_ind])

			iv_matrix.append(adjacency_row)

		iv_matrices.append(iv_matrix)
		dv_vectors.append(dv_vector)

	return iv_matrices, dv_vectors




def expected_voxels_bytr(list data, list mapping, int tr, int voxels, list scoring_mask):

	cdef double expected, adj_coef
	cdef int adj_ind, voxel_ind, score_bool
	cdef list outrow

	outrow = []

	for voxel_ind in range(voxels):
		expected = 0.
		score_bool = scoring_mask[voxel_ind]

		if score_bool == 1:
			for adj_ind, adj_coef in mapping[voxel_ind]:
				expected += (data[tr][adj_ind] * adj_coef)
		else:
			expected += data[tr][voxel_ind]

		outrow.append(expected)

	return outrow



def blur_voxels_bytr(list data, list mapping, int tr, int voxels, list scoring_mask):

	cdef double expected, adj_coef, covar_sum
	cdef int adj_ind, voxel_ind, score_bool
	cdef list outrow

	outrow = []

	for voxel_ind in range(voxels):
		expected = 0.
		covar_sum = 0.
		score_bool = scoring_mask[voxel_ind]

		if score_bool == 1:
			for adj_ind, adj_coef in mapping[voxel_ind]:
				#if adj_ind != voxel_ind:
				#	covar_sum += abs(adj_coef)
				
				covar_sum += abs(adj_coef)


			for adj_ind, adj_coef in mapping[voxel_ind]:
				#if adj_ind != voxel_ind:
				#	expected += data[tr][adj_ind] * (adj_coef/covar_sum)

				expected += data[tr][adj_ind] * (adj_coef/covar_sum)


		outrow.append(expected)

	return outrow




def calculate_correlation(list dataA, list dataB):


	cdef double sumxy, sumxsq, sumysq, sumx, sumy, voxel_r, x, y
	cdef list voxelwise_correlations
	cdef long trs, voxels

	voxelwise_correlations = []
	trs = len(dataA)
	voxels = len(dataA[0])

	for voxel in range(voxels):

		if (voxel % 5000) == 0:
			print 'calculating correlation for voxel', voxel

		sumxy = 0.
		sumxsq = 0.
		sumysq = 0.
		sumx = 0.
		sumy = 0.
		for tr in range(trs):
			x = dataA[tr][voxel]
			y = dataB[tr][voxel]
			#print x, y
			sumxy += x*y
			sumxsq += x*x
			sumysq += y*y
			sumx += x
			sumy += y
		#print 'sumx, sumy, sumxy, sumxsq, sumysq:', sumx, sumy, sumxy, sumxsq, sumysq
		try:
			voxel_r = ((trs*sumxy) - (sumx*sumy)) / math.sqrt(((trs*sumxsq)-(sumx*sumx)) * ((trs*sumysq)-(sumy*sumy)))
		except:
			voxel_r = 0.
		#print 'voxel r:', voxel_r
		voxelwise_correlations.append(voxel_r)

	return voxelwise_correlations



def calculate_theil_sen_slope(list x, list y):

	cdef list slopes
	cdef int nval, i, j
	cdef double y_1, y_2, x_1, x_2

	nval = len(x)

	slopes = []

	for i in range(nval-1):
		for j in range(i, nval):
			if x[i] != y[j]:
				y_1 = y[i]
				y_2 = y[j]
				x_1 = x[i]
				x_2 = x[j]
				if (x_2 - x_1) != 0.:
					slopes.append((y_2 - y_1) / (x_2 - x_1))
				else:
					slopes.append(0.)

	return slopes





















