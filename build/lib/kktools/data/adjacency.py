

import numpy as np


def construct_4d_adjacency_list(nx, ny, nz, nt, return_full=False):
    """
    Based on Logan's construct_adjacency_list, but for 4 dimension brain data
    instead of the 3d data going in to that function...
    """
    
    nvoxels = nx*ny*nz*nt
    
    from scipy.sparse import coo_matrix
    
    
    y_coords = np.reshape(
        np.tile(
            np.arange(ny, dtype=np.int32),
            (nx*nz, 1)),
        (nvoxels),
        order='f')
    
    
    x_coords = np.reshape(
        np.tile(
            np.reshape(
                np.tile(
                    np.arange(nx, dtype=np.int32),
                    (nz, 1)),
                (nz*nx),
                order='f'),
            (ny, 1)),
        (nx*ny*nz))

    
    z_coords = np.tile(
        np.arange(nz, dytpe=np.int32),
        (nx*ny))