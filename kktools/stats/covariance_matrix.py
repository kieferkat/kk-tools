
import os
import glob
from pprint import pprint
import nibabel as nib
import numpy as np
import sklearn.covariance as skcov
import numpy.lib.format as npf
import sklearn.linear_model as linmod
import time
import pickle, cPickle



class MemmapMatrix(object):

    def __init__(self, filepath, verbose=True):

        super(MemmapMatrix, self).__init__
        
        self.filepath = filepath
        self.matrix = None
        self.verbose = verbose


    def load_matrix(self, filepath):

        if os.path.exists(filepath):
            self.matrix = npf.open_memmap(filepath, mode='r+', dtype='float32')
            if self.verbose:
                print 'loaded memmap', filepath
        else:
            if self.verbose:
                print 'nonexistant memmap', filepath


    def delete_matrix(self, filepath):

        if os.path.exists(filepath):
            os.remove(filepath)
            if self.verbose:
                print 'deleted memmap', filepath

        else:
            if self.verbose:
                print 'no file to delete'


    def allocate_matrix(self, filepath, shape):

        if self.verbose:
            print 'shape of memmap will be', shape
            print 'allocating memmap...'

        self.delete_matrix(filepath)

        self.matrix = npf.open_memmap(filepath, mode='w+', dtype='float32', shape=shape)

        if self.verbose:
            print 'completed allocation of covtable to filename', filepath


    def memorysafe_cov(self, data):

        if self.verbose:
            print 'initiating memorysafe covariance calculation...'

        rows = data.shape[0]
        cols = data.shape[1]

        for cov_r in range(rows):

            if self.verbose:
                print 'ROW:\t', cov_r

            for cov_c in range(rows):

                mini_cov = np.cov(data[cov_r], data[cov_c])

                self.matrix[cov_r,cov_r] = mini_cov[0,0]
                self.matrix[cov_r,cov_c] = mini_cov[0,1]
                self.matrix[cov_c,cov_r] = mini_cov[1,0]
                self.matrix[cov_c,cov_c] = mini_cov[1,1]


        if self.verbose:
            print 'covariance table calculation is complete!'


    def memorysafe_dot(self, A, B):

        if self.verbose:
            print 'initiating memorysafe dot product...'

        Adim = A.shape[0]
        Bdim = B.shape[1]

        for x in range(Adim):
            if self.verbose:
                print 'Arow:', x

            for y in range(Bdim):
                x_vec = A[x,:]
                y_vec = B[:,y]
                dot = np.dot(x_vec, y_vec)
                self.matrix[x,y] = dot

        if self.verbose:
            print 'Memorysafe dot completed.' 







class CovarianceCalculator(object):

    def __init__(self):
        super(CovarianceCalculator, self).__init__()
        self.ntrs = None


    def load_nifti(self, niftipath):
        nifti = nib.load(niftipath)
        nifti_data = nifti.get_data()
        nifti_shape = nifti.shape
        nifti_affine = nifti.get_affine()
        return nifti_data, nifti_shape, nifti_affine



    def load_mask(self, mask_path, transpose=True, flatten=False):
        mdata, mshape, maffine = self.load_nifti(mask_path)
        self.original_mask_shape = mshape
        self.original_mask_affine = maffine

        mdata = mdata.astype(np.bool)

        if transpose:
            mdata = np.transpose(mdata, [2,1,0])

        self.mask = mdata

        if flatten:
            self.mask = self.mask.reshape(np.prod(self.mask.shape))



    def prepare_3d_adjacency(self, numx=1, numy=1, numz=1, verbose=True):

        if verbose:
            print 'initializing adjacency components...'

        vmap = np.cumsum(self.mask).reshape(self.mask.shape)
        mask = np.bool_(self.mask.copy())
        vmap[~mask] = -1.
        vmap -= 1 # vmap's values run from 0 to mask.sum()-1

        adj = []
        nx, ny, nz = mask.shape

        if verbose:
            print 'constructing adjacency list...'

        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if mask[x,y,z]:

                        local_map = vmap[max((x-numx),0):(x+numx+1),
                                         max((y-numy),0):(y+numy+1),
                                         max((z-numz),0):(z+numz+1)]

                        inds = (local_map>-1)
                        inds = np.bool_(inds)
                        adjacency_row = np.array(local_map[inds], dtype=int)
                        adj.append(adjacency_row)

        for i, a in enumerate(adj):
        #    a[np.equal(a,i)] = -1
            adj[i] = a.tolist()

        self.adjacency = adj


    def mask_and_flatten(self, data, maskdata, reverse_transposed=True, verbose=True):

        # currently only works with reverse transposition.
        # mask and data must have the same shape!

        if verbose:
            print 'masking and flattening data...'

        flat_mask = maskdata.copy()
        flat_mask.shape = np.product(flat_mask.shape)

        masked_data = data.copy()

        if reverse_transposed:
            masked_data = masked_data.reshape(masked_data.shape[0], np.product(masked_data.shape[1:]))
            masked_data = masked_data[:,flat_mask]
        else:
            masked_data = masked_data.reshape(np.product(masked_data.shape[:-1], masked_data.shape[-1]))
            masked_data = masked_data[flat_mask,:]

        return masked_data



    def load_data(self, data_path, reverse_transpose=True, verbose=True):
        ddata, dshape, daffine = self.load_nifti(data_path)
        self.original_data_shape = dshape
        self.original_data_affine = daffine

        if reverse_transpose:
            ddata = np.transpose(ddata, [3,2,1,0])
            if not getattr(self, 'ntrs', None):
                self.ntrs = ddata.shape[0]
        else:
            if not getattr(self, 'ntrs', None):
                self.ntrs = ddata.shape[3]

        data = ddata

        if verbose:
            print 'ntrs', self.ntrs
            print 'data shape', data.shape
            print 'masking...'


        return data


    def load_subject_niftis(self, niftipaths, verbose=True):

        self.nifti_dict = {}

        if verbose:
            print 'loading in subject niftis to dict...'

        for nifti in niftipaths:
            subject = os.path.split(os.path.split(nifti)[0])[1]
            subdata = self.load_data(nifti)
            self.nifti_dict[subject] = subdata


    def mask_flatten_subject_niftis(self, normalize=False, verbose=True):

        if verbose:
            print 'masking and flattening subject niftis...'

        for subject, dataobj in self.nifti_dict.items():
            self.nifti_dict[subject] = self.mask_and_flatten(dataobj, self.mask)

            if normalize:
                self.nifti_dict[subject] = self.normalize_data(self.nifti_dict[subject])


    def assign_model_subject(self, model_subject_key):

        self.model_subject_key = model_subject_key
        self.model_subject_data = self.nifti_dict[model_subject_key]
        #self.model_subject_data = self.normalize_data(self.model_subject_data)




    def create_mapping_to_model(self, subject_key, verbose=True, verbose_iter=1000, testing=False):

        # data should be flattened by this point to correspond to the adjacency
        # list!
        # this ONLY works for reverse transposed data, currently (the default behavior!)

        mapstart = time.time()

        subdata = self.nifti_dict[subject_key].copy()

        if not hasattr(self,'mappings'):
            self.mappings = {}
        self.mappings[subject_key] = []

        '''
        enet = linmod.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.],
                                   precompute='auto',
                                   max_iter=1000,
                                   tol=.0001,
                                   cv=3,
                                   n_jobs=2)
        '''

        ridge = linmod.RidgeCV(fit_intercept=False)
        allzero_errors = 0
        solzero_errors = 0

        for mvox_ind in range(self.model_subject_data.shape[1]):

            if verbose:
                if (mvox_ind % verbose_iter == 0):
                    print 'voxel index: ', mvox_ind
                    print 'all zero errors:', allzero_errors
                    print 'solution zero errors:', solzero_errors
                    print ''

            if testing:
                self.mappings[subject_key].append([[mvox_ind, 1.]])
            else:

                voxel_adj = self.adjacency[mvox_ind]
                model_vector = []
                iv_matrix = []

                for tr in range(self.model_subject_data.shape[0]):
                    model_vector.append(self.model_subject_data[tr, mvox_ind])

                    adj_row = []
                    for vadj in voxel_adj:
                        adj_row.append(subdata[tr,vadj])
                    iv_matrix.append(adj_row)

                model_vector = np.array(model_vector)
                iv_matrix = np.array(iv_matrix)

                if np.sum(model_vector) == 0 or np.sum(iv_matrix) == 0:
                    allzero_errors += 1
                    self.mappings[subject_key].append([[mvox_ind, 1.]])

                else:
                    ridge.fit(iv_matrix, model_vector)

                    if np.sum(ridge.coef_) == 0:
                        solzero_errors += 1
                        self.mappings[subject_key].append([[mvox_ind, 1.]])
                    else:
                        self.mappings[subject_key].append(zip(voxel_adj, ridge.coef_))


        mapend = time.time()

        if verbose:
            print 'TOTAL TIME ELAPSED', mapend-mapstart, subject_key


    def generate_mappings(self, testing=False):

        self.mappings = {}
        non_model_subs = [x for x in self.nifti_dict.keys() if x != self.model_subject_key]
        print non_model_subs

        for nmsub in non_model_subs:
            self.create_mapping_to_model(nmsub, testing=testing)


    def save_mappings(self, filepath, method='pickle', verbose=True):

        fid = open(filepath, 'w')

        if method == 'pickle':
            if verbose:
                print 'pickling self.mappings to file', filepath
            #pickler = pickle.Pickler(fid)
            #pickler.dump(self.mappings)
            cPickle.dump(self.mappings, fid)

        fid.close()

    def load_mappings(self, filepath, method='pickle', verbose=True):

        fid = open(filepath, 'r')

        if method == 'pickle':
            if verbose:
                print 'unpickling mappings from file', filepath
            #unpickler = pickle.Unpickler(fid)
            #self.mappings = unpickler.load()
            self.mappings = cPickle.load(fid)

        fid.close()



    def mapping_to_nifti(self, subject_key, suffix='pas_falign', reverse_transposed=True, verbose=True,
                         testing=False):

        if verbose:
            print 'converting map to new nifti', subject_key

        data = self.nifti_dict[subject_key]
        mapping = self.mappings[subject_key]
        ndata = []

        if testing:
            ndata = data
        else:
            for tr in range(data.shape[0]):
                if verbose:
                    print 'TR:', tr

                # slightly faster:
                tr_row = []
                for vox in range(data.shape[1]):
                    accumulation = 0.
                    for a, coef in mapping[vox]:
                        accumulation += data[tr,a]*coef
                    tr_row.append(accumulation)
                ndata.append(np.array(tr_row))

                # slightly slower:
                #ndata.append([sum([data[tr,a]*coef for a, coef in mapping[vox]]) for vox in range(data.shape[1])])

            ndata = np.array(ndata, dtype=np.float32)

        unmasked = [np.zeros(self.mask.shape) for i in range(ndata.shape[0])]

        if verbose:
            print 'unmasked shape:', unmasked

        for i in range(ndata.shape[0]):
            if verbose:
                print 'transposing tr:', i
            unmasked[i][np.asarray(self.mask).astype(np.bool)] = np.squeeze(np.array(ndata[i]))
            if reverse_transposed:
                unmasked[i] = np.transpose(unmasked[i],[2,1,0])

        unmasked = np.array(unmasked, dtype=np.float32)        
        unmasked = np.transpose(unmasked, [1,2,3,0])
        print unmasked.shape
        print self.original_mask_affine

        filepath = subject_key+'_'+suffix+'.nii'
        nii = nib.Nifti1Image(unmasked, self.original_mask_affine)

        nii.to_filename(filepath)





    def normalize_data(self, data, verbose=True):

        if verbose:
            print 'normalizing dataset column-wise:'
            print 'pre-normalization sum:', np.sum(data)

        stdev = np.std(data, axis=0)
        means = np.mean(data, axis=0)

        dnorm = np.zeros(data.shape)
        dnorm = data-means
        dnorm = dnorm/stdev
        
        dnorm = np.nan_to_num(dnorm)

        if verbose:
            print 'post-normalization sum:', np.sum(dnorm)
        
        return dnorm


    def standardize_data(self, data, verbose=True):
        
        if verbose:
            print 'standardizing data by column (makes correlation)'
        ndata = data.copy()
        ndata /= data.std(axis=0)

        return ndata


    def randomize_zero_rows(self, data, verbose=True):
        ''' applying a random gaussian to all-zero rows is useful 
        for some kinds of covariace matrices where singular value
        decomposition is needed. if there are multiple identical column-wise
        then the matrix will fail, so random gaussian is a hack-y way
        of getting around this'''
        
        if verbose:
            print 'random gaussian on all-zero rows'

        rows = data.shape[0]
        cols = data.shape[1]

        ndata = data.copy()
        
        for row in range(rows):
            if np.sum(data[row]) == 0.:
                gaussiand = np.random.normal(size=cols)
                ndata[row,:] = gaussiand

        if verbose:
            print 'checking for single-value columns...'
            for row in range(rows):
                if len(np.unique(ndata[row])) == 1:
                    print 'single value column found! (bad)'
                    print ndata[row]

        return ndata


    def prepare_nifti_dict(self, randomize_zero_rows=True, normalize_data=True, verbose=True):

        if verbose:
            print 'preforming preparatory steps on nifi data prior to concatenation...'

        if randomize_zero_rows:
            if verbose:
                print 'randomizing zero rows within nifti...'

            for subject in self.nifti_dict:
                if verbose:
                    print 'nifti for subject', subject

                subdata = self.nifti_dict[subject]
                rdata = self.randomize_zero_rows(subdata)
                self.nifti_dict[subject] = rdata


        if normalize_data:
            if verbose:
                print 'normalizing the columns (voxels/features) within nifti...'

            for subject in self.nifti_dict:
                if verbose:
                    print 'nifti for subject', subject

                subdata = self.nifti_dict[subject]
                normdata = self.normalize_data(subdata)
                self.nifti_dict[subject] = normdata



    def construct_X_matrix(self, to_numpy=True, verbose=True):

        if verbose:
            print 'constructing X matrix from nifti dict...'

        self.X = []

        for subject in self.nifti_dict:
            if verbose:
                print 'appending subject', subject

            subdata = self.nifti_dict[subject]

            for trdata in subdata:
                self.X.append(trdata)

        if to_numpy:
            if verbose:
                print 'forming X into numpy matrix...'

            self.X = np.array(self.X)
    
    

    def compute_empirical_covariance(self, data, verbose=True):

        if verbose:
            print 'data shape:', data.shape
            print 'computing the empirical covariance matrix...'

        empirical_covar = skcov.EmpiricalCovariance()
        empirical_covar.fit(data)

        if verbose:
            print 'covariance matrix shape', empirical_covar.covariance_.shape

        return empirical_covar


    def compute_minimum_covariance_determinant(self, data, assume_centered=False, verbose=True):

        if verbose:
            print 'data shape:', data.shape
            print 'computing the minimum covariance determinant...'

        mincovdet = skcov.MinCovDet(assume_centered=assume_centered)
        mincovdet.fit(data)

        if verbose:
            print 'covariance matrix shape', mincovdet.covariance_.shape

        return mincovdet


    def compute_oas_covariance(self, data, verbose=True):

        if verbose:
            print 'data shape:', data.shape
            print 'computing the oas covariance matrix...'

        oas = skcov.OAS()
        oas.fit(data)

        if verbose:
            print 'covariance matrix shape', oas.covariance_.shape

        return oas










