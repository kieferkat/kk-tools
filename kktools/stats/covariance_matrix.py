
import os
import glob
from pprint import pprint
import nibabel as nib
import numpy as np
import sklearn.covariance as skcov
import numpy.lib.format as npf



class CovarMemmap(object):

    def __init__(self, filepath, verbose=True):

        super(CovarMemmap, self).__init__
        
        self.filepath = filepath

        self.covtable_allocated = False
        self.covtable = None

        self.verbose = verbose


    def load_covtable(self, filepath):

        if os.path.exists(filepath):
            self.covtable = npf.open_memmap(filepath, mode='r+', dtype='float32')
            if self.verbose:
                print 'loaded memmap', filepath
            self.covtable_allocated = True
        else:
            if self.verbose:
                print 'nonexistant memmap', filepath
            return None


    def delete_covtable(self, filepath):

        if os.path.exists(filepath):
            os.remove(filepath)
            if self.verbose:
                print 'deleted memmap', filepath

        else:
            if self.verbose:
                print 'no file to delete'


    def allocate_covtable(self, filepath, shape):

        if self.verbose:
            print 'shape of covariance memmap will be', shape
            print 'allocating memmap for covariance table...'

        self.delete_covtable(filepath)

        self.covtable = npf.open_memmap(filepath, mode='w+', dtype='float32', shape=shape)

        if self.verbose:
            print 'completed allocation of covtable to filename', filepath

        self.covtable_allocated = True


    def memorysafe_cov(self, data):

        if self.covtable_allocated:
            if self.verbose:
                print 'initiating memorysafe covariance calculation...'

            rows = data.shape[0]
            cols = data.shape[1]

            for cov_r in range(rows):

                if self.verbose:
                    print 'ROW:\t', cov_r

                for cov_c in range(rows):

                    mini_cov = np.cov(data[cov_r], data[cov_c])

                    self.covtable[cov_r,cov_r] = mini_cov[0,0]
                    self.covtable[cov_r,cov_c] = mini_cov[0,1]
                    self.covtable[cov_c,cov_r] = mini_cov[1,0]
                    self.covtable[cov_c,cov_c] = mini_cov[1,1]


            if self.verbose:
                print 'covariance table calculation is complete!'

        else:
            print 'covariance table has not been allocated (or loaded?)'






class CustomNiftiObj(object):

    def __init__(self, data=[], affine=[], shape=[]):
        super(CustomNiftiObj, self).__init__()
        self.data = data
        self.affine = affine
        self.shape = shape
        self.shape_history = [shape]
        self.transposed = False








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



    def load_mask(self, mask_path, transpose=True, flatten=True):
        mdata, mshape, maffine = self.load_nifti(mask_path)
        
        mdata = mdata.astype(np.bool)

        if transpose:
            mdata = np.transpose(mdata, [2,1,0])

        self.mask = CustomNiftiObj(data=mdata, shape=mshape, affine=maffine)

        if flatten:
            self.mask.data = self.mask.data.reshape(np.prod(self.mask.data.shape))
            self.mask.shape_history.append(self.mask.shape)
            self.mask.shape = self.mask.data.shape



    def load_data(self, data_path, transpose=False, flatten=True, verbose=True):
        ddata, dshape, daffine = self.load_nifti(data_path)

        if transpose:
            ddata = np.transpose(ddata, [3,2,1,0])
            if not getattr(self, 'ntrs', None):
                self.ntrs = ddata.shape[0]
        else:
            if not getattr(self, 'ntrs', None):
                self.ntrs = ddata.shape[3]

        dataobj = CustomNiftiObj(data=ddata, shape=dshape, affine=daffine)

        if flatten:
            dataobj.data = dataobj.data.reshape(self.ntrs, np)
            dataobj.shape_history.append(dataobj.shape)
            dataobj.shape = dataobj.data.shape

        if verbose:
            print 'ntrs', self.ntrs
            print 'data shape', dataobj.shape
            print 'masking...'

        dataobj.data = dataobj.data[:,self.mask.data]

        if transpose:
            if verbose: print 'transposing...'
            dataobj.data = dataobj.data.T
            dataobj.shape = dataobj.data.shape
            dataobj.shape_history.append(dataobj.shape)
            dataobj.transposed = True

        return dataobj


    def load_subject_niftis(self, niftipaths, verbose=True):

        self.nifti_dict = {}

        if verbose:
            print 'loading in subject niftis to dict...'

        for nifti in niftipaths:
            subject = os.path.split(os.path.split(nifti)[0])[1]
            subdata = self.load_data(nifti)
            self.nifti_dict[subject] = subdata.data


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










