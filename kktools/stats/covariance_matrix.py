
import os
import glob
from pprint import pprint
import nibabel as nib
import numpy as np
import sklearn.covariance as skcov


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
        niftii_data = nifti.get_data()
        nifti_shape = nifti.shape
        nifti_affine = nifti.get_affine()
        return nifti_data, nifti_shape, nifti_affine



    def load_mask(self, mask_path, transpose=True, flatten=True):
        mdata, mshape, maffine = load_nifti(mask_path)
        
        mdata = mdata.astype(np.bool)

        if transpose:
            mdata = np.transpose(mdata, [2,1,0])

        self.mask = CustomNiftiObj(data=mdata, shape=mshape, affine=maffine)

        if flatten:
            self.mask.data = self.mask.reshape(np.prod(self.mask.data.shape))
            self.mask.shape_history.append(self.mask.shape)
            self.mask.shape = self.mask.data.shape



    def load_data(self, data_path, transpose=True, flatten=True, verbose=True):
        ddata, dshape, daffine = load_nifti(data_path)

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


    def load_subject_niftis(self, sdirs, niftiname, verbose=True):

        self.nifti_dict = {}

        if verbose:
            print 'loading in subject niftis to dict...'

        for sdir in sdirs:
            subject = os.path.split(sdir)[1]
            nifti = os.path.join(sdir, niftiname)
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

        return ndatat


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










