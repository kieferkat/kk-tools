
import itertools
from scipy import sparse
import numpy as np


def construct_4d_adjacency_list(mask, numx=1, numy=1, numz=1):
    """
    Basically the prepare_adj function from regreg, but with less options.
    """
    
    regions = np.zeros(mask.shape)
    regions.shape = mask.shape
    reg_values = np.unique(regions)
    
    vmap = np.cumsum(mask).reshape(mask.shape)
    mask = np.bool_(mask.copy())
    vmap[~mask] = -1
    vmap -= 1 # sets vmap's values from 0 to mask.sum()-1
    
    adj = []
    nx, ny, nz, nt = mask.shape
    
    for i, j, k, t in itertools.product(range(nx), range(ny),
                                        range(nz), range(nt)):
        
        if mask[i, j, k, t]:
            
            local_map = vmap[max((i-numx), 0):(i+numx+1),
                             max((j-numy), 0):(j+numy+1),
                             max((k-numz), 0):(k+numz+1),
                             max((t-numt), 0):(t+numt+1)]
            
            local_reg = regions[max((i-numx), 0):(i+numx+1),
                                max((j-numy), 0):(j+numy+1),
                                max((k-numz), 0):(k+numz+1),
                                max((t-numt), 0):(t+numt+1)]
            
            region = regions[i, j, k, t]
            ind = (local_map > -1) * (local_reg == region)
            ind = np.bool_(ind)
            nbrs = np.array(local_map[ind], dtype=np.int)
            adj.append(nbrs)
            
    
    for i, a in enumerate(adj):
        a[np.equal(a, i)] = -1
        
    num_ind = np.max([len(a) for a in adj])
    adjarray = -np.ones((len(adj), num_ind), dtype=np.int)
    
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            adjarray[i,j] = adj[i][j]
            
    return adjarray
    
    
    
