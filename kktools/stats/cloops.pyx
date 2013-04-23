

import math



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




def expected_voxels_bytr(list data, list mapping, int tr, int voxels, double r2_cutoff, list map_scores):

	cdef double expected, adj_coef, score
	cdef int adj_ind, voxel_ind, cutoff
	cdef list outrow

	outrow = []

	for voxel_ind in range(voxels):
		expected = 0.
		cutoff = 0
		if r2_cutoff > 0:
			score = map_scores[voxel_ind]
			if score < r2_cutoff:
				cutoff = 1

		if cutoff == 0:
			for adj_ind, adj_coef in mapping[voxel_ind]:
				expected += (data[tr][adj_ind] * adj_coef)
		else:
			expected += data[tr][voxel_ind]
			
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






