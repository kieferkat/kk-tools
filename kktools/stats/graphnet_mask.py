import numpy as np
from nipy.io.api import load_image
from nipy.core.api import Image

def adj_from_nii(maskfile,num_time_points,numt=0,numx=1,numy=1,numz=1,regions=None):
    """
    Construct adjacency array from .nii mask file

    INPUT:

    maskfile: Path to mask file (.nii)

    Other parameters are passed directly to prepare_adj (see that function for docs)

    OUTPUT:

    adj: An array containing adjacency information
    """
    mask = load_image(maskfile)._data
    newmask = np.zeros(np.append(num_time_points,mask.shape))
    for i in range(num_time_points):
        newmask[i] = mask
    adj = prepare_adj(newmask,numt,numx,numy,numz,regions)
    adj = convert_to_array(adj)
    return adj


def prepare_adj(mask,numt=0,numx=1,numy=1,numz=1,regions=None, gm_mask=None):
    """
    Return adjacency list, where the voxels are considered
    neighbors if they fall in a ball of radius numt, numx, numy, and numz
    for time, x position, y position, and z position respectively.

    INPUT:

    X: a 5-dimensional ndarray. The first index is trial, the second index is time,
    the third index is x position, the fourth index is y position and the fifth
    position is z position.

    mask: a binary 4-dimensional ndarray, the same size as X[0,:,:,:,:] where
    1 indicates that the voxel-timepoint is included and 0 indicates that it is
    excluded. NOTE: Usually the mask is thought of as a 3-dimensional ndarray, since
    it is uniform across time. 

    regions: a multivalued array the same size as the mask that indicates different
    regions in the spatial structure. No adjacency edges will be made across region
    boundaries.

    numt: an integer, the radius of the "neighborhood ball" in the t direction
    numx: an integer, the radius of the "neighborhood ball" in the x direction                                                                
    numy: an integer, the radius of the "neighborhood ball" in the y direction                                                                
    numz: an integer, the radius of the "neighborhood ball" in the z direction                                                                
                                                                    
    OUTPUT:

    newX: The matrix X reshaped as a 2-dimensional array for analysis
    adj: The adjacency list associated with newX

    """
    
    #Create map going from X to predictor vector indices. The entries of
    # this array are -1 if the voxel is not included in the mask, and the 
    # index in the new predictor corresponding to the voxel if the voxel
    # is included in the mask.

    if regions == None:
        regions = np.zeros(mask.shape)
    regions.shape = mask.shape
    reg_values = np.unique(regions)
    
    vmap = np.cumsum(mask).reshape(mask.shape)
    mask = np.bool_(mask.copy())
    vmap[~mask] = -1
    vmap -= 1 # now vmap's values run from 0 to mask.sum()-1
    
    
    if gm_mask is not None:
        gm = True
        vgm_mask = gm_mask.copy()
        #vgm_mask[~mask] = -1

    else:
        gm = False

    # Create adjacency list
    
    adj = []
    #gm_adj = []

    nt,nx,ny,nz = mask.shape

    for t in range(nt):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if mask[t,i,j,k]:
                        
                        local_map = vmap[max((t-numt),0):(t+numt+1),
                                         max((i-numx),0):(i+numx+1),
                                         max((j-numy),0):(j+numy+1),
                                         max((k-numz),0):(k+numz+1)]
                        
                        local_gm = vgm_mask[max((t-numt),0):(t+numt+1),
                                            max((i-numx),0):(i+numx+1),
                                            max((j-numy),0):(j+numy+1),
                                            max((k-numz),0):(k+numz+1)]
                        
                        local_reg = regions[max((t-numt),0):(t+numt+1),
                                            max((i-numx),0):(i+numx+1),
                                            max((j-numy),0):(j+numy+1),
                                            max((k-numz),0):(k+numz+1)]
                        
                        region = regions[t,i,j,k]
                        ind = (local_map>-1)*(local_reg == region)
                        ind = np.bool_(ind)
                        adjrow = np.array(local_map[ind], dtype=int)
                        
                        if gm:
                            gmrow = np.array(local_gm[ind], dtype=float)
                        else:
                            gmrow = np.ones(len(adjrow), dtype=float)
                            
                        adj.append([[a,g] for a,g in zip(adjrow, gmrow)])

                        #adj.append(np.array(local_map[ind],dtype=int))
                        #if gm:
                        #    gm_adj.append(np.array(vgm_mask[ind], dtype=float))
                        #adj.append(local_map[ind])
                        
    
    accum = []
    for i, a in enumerate(adj):
        for [ax, g] in a:
            accum.append(g)
            
    print np.unique(g), np.unique(vgm_mask)
    print np.sum(g), np.sum(vgm_mask)
    stop
                        
    for i, a in enumerate(adj):
        for j, [ax, g] in enumerate(a):
            if ax == i:
                a[j] = [-1, g]
        adj[i] = a
        
        #a[np.equal(a,i)] = -1
        #adj[i] = a.tolist()
        
    #if gm:
    #    for i, g in enumerate(gm_adj):
    #        gm_adj[i] = g.tolist()
    #return convert_to_array(adj)
    
    #if gm:
    #    return adj, gm_adj
    #else:
    #    return adj


    return adj




def convert_to_array(adj):
    num_ind = np.max([len(a) for a in adj])
    adjarray = -np.ones((len(adj),num_ind))
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            adjarray[i,j] = adj[i][j]
    return adjarray

   
def test_prep(nt=0,nx=1,ny=1,nz=1):
    """
    Let's make this into a proper test...... what should newa, adj be in this case?
    """
    a = np.array(range(1,1+2*3*4*4*4)).reshape((2,3,4,4,4))
    mask = a[0]*0
    mask[:,0,0,0] = 1
    mask[:,1,1,:] = 1
#    print mask[0]
#    print a[0,0]
    adj = prepare_adj(mask,nt,nx,ny,nz)
#    print newa[0,0], adj[0], newa[0,adj[0]]
