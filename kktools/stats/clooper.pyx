




def adj_vox_calculator(list data, int voxelrange, list mapping, int tr):

	cdef double expected, adj_coef
	cdef int adjacent, voxel
	outrow = []

	for voxel in range(voxelrange):
		expected = 0.
		for adjacent, adj_coef in mapping[voxel]:
			expected += data[tr][adjacent] * adj_coef
		outrow.append(expected)
		
	return outrow






def construct_adjacency_forridge(list data, list adjacency, int mvoxels, int trs):

	cdef int mvox_ind, tr, vadj
	cdef list iv_matrix, model_vector, iv_matrices, model_vectors, voxel_adjacency, adj_row

	model_vectors = []
	iv_matrices = []

	for mvox_ind in range(mvoxels):

		if (mvox_ind % 1000) == 0:
			print 'Preparing for ridge iter:', mvox_ind

		voxel_adjacency = adjacency[mvox_ind]
		model_vector = []
		iv_matrix = []

		for tr in range(trs):

			model_vector.append(data[tr][mvox_ind])

			adj_row = []
			for vadj in voxel_adjacency:
				adj_row.append(data[tr][vadj])

			iv_matrix.append(adj_row)


		iv_matrices.append(iv_matrix)
		model_vectors.append(model_vector)

	return iv_matrices, model_vectors