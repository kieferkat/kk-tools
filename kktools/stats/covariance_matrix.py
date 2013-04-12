
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
import pyximport
pyximport.install()
import clooper 



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
        '''
        constructs the 3D adjacency list. this functions basically the same as
        the graphnet l3 penalty adjacency matrix, except there are only 3 dimensions
        rather than 4.

        it iterates through the dimensions of the mask. if a voxel in the mask is true,
        it finds other adjacent voxels according to a user specified distance that are
        also true. The output is a list of lists, one adjacency list for each voxel.

        '''

        if verbose:
            print 'initializing adjacency components...'

        # vmap assigns a number for each voxel:
        vmap = np.cumsum(self.mask).reshape(self.mask.shape)

        # binarize the mask:
        mask = np.bool_(self.mask.copy())

        # set all 0 values to -1:
        vmap[~mask] = -1.
        vmap -= 1 # vmap's values run from 0 to mask.sum()-1

        # initialize the adjacency list and find the mask dimensions:
        adj = []
        nx, ny, nz = mask.shape

        if verbose:
            print 'constructing adjacency list...'

        # cascading iteration through the dimensions:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if mask[x,y,z]:

                        # local map specifies all adjacent voxels in range:
                        local_map = vmap[max((x-numx),0):(x+numx+1),
                                         max((y-numy),0):(y+numy+1),
                                         max((z-numz),0):(z+numz+1)]

                        # inds are all the local map indices that are not
                        # -1 (valid brain voxels)
                        inds = (local_map>-1)
                        inds = np.bool_(inds)

                        # append the adjacent voxels for this current voxel:
                        adjacency_row = np.array(local_map[inds], dtype=int)
                        adj.append(adjacency_row)

        # convert to list
        # NOTE: this is different than the graphnet l3 penalty because the 
        # current voxel is not removed from the list of adjacent voxels!
        for i, a in enumerate(adj):
            adj[i] = a.tolist()

        self.adjacency = adj


    def mask_and_flatten(self, data, maskdata, reverse_transposed=True, verbose=True):
        '''
        mask out the data according to the mask, then flatten it into a tr-by-voxel array.
        reverse_transposition is assumed to be true (and should be, really, since some of
        these functions currently only support that format)

        '''

        # currently only works with reverse transposition.
        # mask and data must have the same shape!

        if verbose:
            print 'masking and flattening data...'

        flat_mask = maskdata.copy()
        flat_mask.shape = np.product(flat_mask.shape)

        masked_data = data

        if reverse_transposed:
            masked_data = masked_data.reshape(masked_data.shape[0], np.product(masked_data.shape[1:]))
            masked_data = masked_data[:,flat_mask]
        else:
            masked_data = masked_data.reshape(np.product(masked_data.shape[:-1], masked_data.shape[-1]))
            masked_data = masked_data[flat_mask,:]

        return masked_data



    def load_data(self, data_path, reverse_transpose=True, verbose=True):
        '''
        loads in a nifti file from a data_path and returns the numpy array
        for the data. reverse_transposing it (to put time first) is recommended.
        '''

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
        '''
        iterates through the subject niftis, loads them in with the load_data function,
        and then adds them to the nifti_dict
        '''

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



    def construct_super_subject(self, mapsubs, nifti_dict):

        msubs = []
        for ms in mapsubs:
            msubs.append(np.array(nifti_dict[ms].copy())

        sublen = len(msubs)
        super_subject = []
        for i, msb in enumerate(msubs):
            if i == 0:
                super_subject = msb
            else:
                super_subject = super_subject+msb

        super_subject = super_subject/sublen

        return super_subject.tolist()



    def generate_super_mappings(self, nifti_dict, save=True, exclude_current_subject=False,
                                verbose=True):
        '''
        TODO
        '''
        
        self.mappings = {}
        subjects = nifti_dict.keys()

        if not exclude_current_subject:
            super_subject = self.construct_super_subject(subjects, nifti_dict)

        
        for sub in subjects:
            print 'on subject', sub

            mapstart = time.time()

            self.mappings[sub] = []

            subdatali = nifti_dict[sub].tolist()

            if exclude_current_subject:
                mapsubs = [s for s in subjects if s != sub]
            else:
                mapsubs = subjects
            
            voxels = nifti_dict[sub].shape[1]

            
            if exclude_current_subject:
                print 'constucting super subject...'
                super_subject = self.construct_super_subject(mapsubs, nifti_dict)

            ridge = linmod.RidgeCV(alphas=[.001,.01,.1,1.,10.,100.],fit_intercept=False)

            allzero_errors = 0
            solzero_errors = 0

            iv_matrices, sub_vectors = clooper.construct_adjacency_forridge(subdatali,
                                                                            super_subject,
                                                                            self.adjacency,
                                                                            self.model_subject_data.shape[1],
                                                                            self.model_subject_data.shape[0])


            for iv_matrix, sub_vector, vox in zip(iv_matrices, sub_vectors, range(voxels)):

                if (vox % 1000) == 0:
                    print 'Ridge voxel iter:', vox

                sub_vector = np.array(sub_vector)
                iv_matrix = np.array(iv_matrix)

                if np.sum(sub_vector) == 0 or np.sum(iv_matrix) == 0:
                    allzero_errors += 1
                    self.mappings[sub].append([[vox, 1.]])

                else:
                    ridge.fit(iv_matrix, sub_vector)

                    if np.sum(ridge.coef_) == 0:
                        solzero_errors += 1
                        self.mappings[sub].append([[vox, 1.]])

                    else:
                        self.mappings[sub].append(zip(self.adjacency[vox], ridge.coef_))

                mapend = time.time()

            if verbose:
                print 'TOTAL TIME ELAPSED', mapend-mapstart, sub

        del subdatalists






    def create_mapping_to_model(self, subject_key, verbose=True, verbose_iter=1000, testing=False):

        # data should be flattened by this point to correspond to the adjacency
        # list!
        # this ONLY works for reverse transposed data, currently (the default behavior!)

        # start the timer for this subject
        mapstart = time.time()

        # copy the subject's data from the nifti dict and convert it to a list for the C functions:
        subdata = self.nifti_dict[subject_key]
        subdatali = subdata.tolist()

        # copy the model subject's data and convert it to a list
        # (inefficient to do this every time, should be changed...)
        modeldatali = self.model_subject_data.tolist()

        # create the mappings class dict if not already made, then create the mappings list for 
        # the current subject:
        if not hasattr(self,'mappings'):
            self.mappings = {}
        self.mappings[subject_key] = []

        # originally was going to use ridge + lasso penalty for elastic net, but decided
        # to use just ridge for speed
        '''
        enet = linmod.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.],
                                   precompute='auto',
                                   max_iter=1000,
                                   tol=.0001,
                                   cv=3,
                                   n_jobs=2)
        '''

        # prepare the crossvalidated Ridge regression class to do the ridge regressions. 
        # set fit_intercept to false because the data has been normalized prior to this 
        # function being called (or at least SHOULD have been!):
        ridge = linmod.RidgeCV(alphas=[.01,.1,1.,10.],fit_intercept=False)

        # counters for different error types:
        allzero_errors = 0
        solzero_errors = 0

        # cython function call that collects all of the voxel independent variable matrices
        # and all of the model vectors for the corresponding voxel on the model data. These
        # are collected voxel by voxel in accordance with the mask:
        # faster cython version:
        iv_matrices, model_vectors = clooper.construct_adjacency_forridge(subdatali,
                                                                          modeldatali,
                                                                          self.adjacency,
                                                                          self.model_subject_data.shape[1],
                                                                          self.model_subject_data.shape[0])


        # zip together the iv_matrices, the model vectors, and the voxel numbers, then perform
        # the ridge regression on each:
        for iv_matrix, model_vector, vox in zip(iv_matrices, model_vectors, range(self.model_subject_data.shape[1])):

            if (vox % verbose_iter) == 0:
                print 'Ridge voxel iter:', vox

            # convert the model vector and iv matrix to numpy format (for scikits ridge):
            model_vector = np.array(model_vector)
            iv_matrix = np.array(iv_matrix)

            # in the case of either the model vector being all 0's or the iv_matrix being all
            # 0's, simply default the mappings to be 1-to-1.
            # 1-to-1 is equivalent to the mapping from this voxel (vox) to have a beta weight
            # of 1:
            if np.sum(model_vector) == 0 or np.sum(iv_matrix) == 0:
                allzero_errors += 1
                self.mappings[subject_key].append([[vox, 1.]])

            else:

                # fit the ridge, get beta weights for each adjacent voxel:
                ridge.fit(iv_matrix, model_vector)

                # in the case of all the ridge coefficients being 0, default to a 1-to-1
                # mapping:
                if np.sum(ridge.coef_) == 0:
                    solzero_errors += 1
                    self.mappings[subject_key].append([[vox, 1.]])

                # append the mappings to the subject list of mappings (a list that corresponds
                # to the number of voxels). Zip together the voxel numbers in the adjacency
                # matrix and the ridge regression coefficients. This SHOULD NEVER BREAK. 
                # (which it doesn't, so we know each adjacent voxel is getting a coefficient):
                else:
                    self.mappings[subject_key].append(zip(self.adjacency[vox], ridge.coef_))


        # slower python-only version
        '''
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
        '''

        mapend = time.time()

        if verbose:
            print 'TOTAL TIME ELAPSED', mapend-mapstart, subject_key





    def generate_mappings(self, testing=False, del_niftis=True):
        '''
        generate_mappings iterate through the subjects in the nifti_dict that are 
        NOT the model_subject and calls the create_mapping_to_model function to 
        generate a mapping for each subject.

        If del_niftis is set to true, this also deletes the nifti data from the nifti_dict 
        for each subject when the mapping is completed to save memory (recommended). 
        '''

        self.mappings = {}
        non_model_subs = [x for x in self.nifti_dict.keys() if x != self.model_subject_key]
        print non_model_subs
        
        for nmsub in non_model_subs:
            self.create_mapping_to_model(nmsub, testing=testing)
            if del_niftis:
                del self.nifti_dict[nmsub]




    def save_mappings(self, filepath, method='pickle', verbose=True):
        '''
        save_mappings pickles the mappings dict to a specified filepath using cPickle
        '''

        if os.path.exists(filepath):
            os.remove(filepath)

        fid = open(filepath, 'w')

        if method == 'pickle':
            if verbose:
                print 'pickling self.mappings to file', filepath
            #pickler = pickle.Pickler(fid)
            #pickler.dump(self.mappings)
            cPickle.dump(self.mappings, fid)

        fid.close()



    def load_mappings(self, filepath, method='pickle', verbose=True):
        '''
        load the mappings stored in a filepath into self.mappings
        '''

        fid = open(filepath, 'r')

        if method == 'pickle':
            if verbose:
                print 'unpickling mappings from file', filepath
            #unpickler = pickle.Unpickler(fid)
            #self.mappings = unpickler.load()
            self.mappings = cPickle.load(fid)

        fid.close()


    def pickle_object(self, filepath, item):

        if os.path.exists(filepath):
            os.remove(filepath)

        fid = open(filepath, 'w')
        print 'pickling', filepath
        cPickle.dump(item, fid)

        fid.close()


    def unpickle_object(self, filepath):
        print 'unpickling...'

        fid = open(filepath, 'r')
        item = cPickle.load(fid)
        fid.close()
        return item



    def mapping_to_nifti(self, subject_key, suffix='pas_falign', reverse_transposed=True, verbose=True,
                         testing=False, save_to_file=False, test_dict=None):
        '''
        mapping_to_nifti takes a subject_key and converts their nifti, stored in self.nifti_dict,
        to a new nifti according to the voxel ajacency mappings stored in self.mappings.
        '''

        if verbose:
            print 'converting map to new nifti', subject_key

        # get the nifti data for the subject and the mappings for the subject. Convert the 
        # subject's nifti from numpy format to list for the C function:
        if test_dict is None:
            data = self.nifti_dict[subject_key]
        else:
            data = test_dict[subject_key]

        datali = data.tolist()
        mapping = self.mappings[subject_key]

        # initialize the new pre-nifti list ndata:
        ndata = []

        # in testing mode the ndata is just set to be the nifti:
        if testing:
            ndata = data

        # iterate through the trs. For each tr calculate the new voxel activation for 
        # every voxel in that tr according to the mappings:
        else:
            for tr in range(data.shape[0]):
                if verbose:
                    print 'TR:', tr

                # fast C-function calculation of new voxel activation:
                c_s = time.time()
                tr_row = clooper.adj_vox_calculator(datali, data.shape[1], mapping, tr)
                ndata.append(np.array(tr_row))
                c_e = time.time()

                print 'cloop time:', c_e-c_s

                # deprecated slower python-only versions: you can use these if cython is
                # not installed on the computer or for whatever reason you cannot compile the
                # clooper.pyx files:
                '''
                # slightly faster:
                #tr_row = []
                #for vox in range(data.shape[1]):
                #    accumulation = 0.
                #    for a, coef in mapping[vox]:
                #        accumulation += data[tr,a]*coef
                #    tr_row.append(accumulation)
                #ndata.append(np.array(tr_row))

                # slightly slower:
                #ndata.append([sum([data[tr,a]*coef for a, coef in mapping[vox]]) for vox in range(data.shape[1])])
                '''

            # convert ndata to a numpy array with float32 (rather than float64)
            ndata = np.array(ndata, dtype=np.float32)

        if save_to_file:
            self.save_mapped_tonifti(subject_key, ndata, suffix=suffix, reverse_transposed=reverse_transposed)

        return ndata




    def save_mapped_tonifti(self, subject_key, ndata, suffix='faligned', verbose=True, reverse_transposed=True):

        # initialize the un-flattened version of the new nifti:
        unmasked = [np.zeros(self.mask.shape) for i in range(ndata.shape[0])]

        # iterate through the trs. for each tr squeeze that tr's voxels into the unmasked
        # array according to where the mask is "True". This functions the same way as 
        # datamanager's unmasking scheme but it should be fully encapsulated into this class:
        for i in range(ndata.shape[0]):
            #if verbose:
            #    print 'transposing tr:', i
            unmasked[i][np.asarray(self.mask).astype(np.bool)] = np.squeeze(np.array(ndata[i]))
            if reverse_transposed:
                unmasked[i] = np.transpose(unmasked[i],[2,1,0])

        # ensure that unmasked is datatype float32 instead of float64
        unmasked = np.array(unmasked, dtype=np.float32)        

        # transpose unmasked back to standard nifti format with time as the 4th dimension:
        unmasked = np.transpose(unmasked, [1,2,3,0])

        print unmasked.shape
        print self.original_mask_affine

        # write the nifti to filepath:
        filepath = subject_key+'_'+suffix+'.nii'
        nii = nib.Nifti1Image(unmasked, self.original_mask_affine)

        if os.path.exists(filepath):
            os.remove(filepath)

        nii.to_filename(filepath)



    ''' TODO:
    def check_covariance_testset(self, test_nifti_dict, normalize=False):

        self.faligned_cov_to_model = {}
        self.unaligned_cov_to_model = {}

        for sub in self.mappings.keys():

            origdata = test_nifti_dict[sub]
            ndata = self.mapping_to_nifti(sub, save_to_file=False, test_dict=test_nifti_dict)

            if normalize:
                ndata = self.normalize_data(ndata)
    '''




    def functional_alignment_bymapping(self, suffix='pas_super_falign',
                                       verbose=True, save=True, del_niftis=True):

        #non_model_subs = [x for x in self.nifti_dict.keys() if x != self.model_subject_key]
        #print non_model_subs

        self.faligned_cov_to_model = {}
        self.unaligned_cov_to_model = {}


        for nmsub in self.mappings.keys():
            if verbose:
                print 'MAPPING', nmsub
            

            origdata = self.nifti_dict[nmsub]
            ndata = self.mapping_to_nifti(nmsub, save_to_file=True, suffix=suffix)
            ndata = self.normalize_data(ndata)
            #self.save_mapped_tonifti(nmsub, ndata, suffix='')

            if verbose:
                print 'calculating covariance for', nmsub
            self.unaligned_cov_to_model[nmsub] = self.simple_covariance(origdata, self.model_subject_data)
            self.faligned_cov_to_model[nmsub] = self.simple_covariance(ndata, self.model_subject_data)

            print 'Unaligned covariance metrics:', nmsub
            self.covariance_metrics(self.unaligned_cov_to_model[nmsub])
            print ''
            print 'Aligned covariance metrics:', nmsub
            self.covariance_metrics(self.faligned_cov_to_model[nmsub])
            print ''

            if del_niftis:
                del self.nifti_dict[nmsub]


        if save:
            if verbose:
                print 'pickling covariance dicts...'
            self.pickle_object('pas_super_unaligned_cov.pkl', self.unaligned_cov_to_model)
            self.pickle_object('pas_super_faligned_cov.pkl', self.faligned_cov_to_model)




    def covariance_metrics(self, covar, verbose=True):

        covsum = np.sum(covar)
        covmean = np.mean(covar)
        covmedian = np.median(covar)

        if verbose:
            print 'covariance sum:', covsum
            print 'covariance mean:', covmean
            print 'covariance median:', covmedian

        return covsum, covmean, covmedian



    def normalize_niftis(self, subjdirs, niftiglob, maskpath, suffix='_norm'):
        print 'loading mask...'
        self.load_mask(maskpath)

        for subjd in subjdirs:
            'normalizing subject nifti ', subjd
            subjname = os.path.split(subjd)[1]
            niftifile = glob.glob(os.path.join(subjd, niftiglob))[0]
            print niftifile
            newnifti = niftifile.rstrip('.nii')

            ldata = self.load_data(niftifile)
            maskdata = self.mask_and_flatten(ldata, self.mask)
            normdata = self.normalize_data(maskdata)

            self.save_mapped_tonifti(newnifti, normdata, suffix=suffix)






    def compare_covariance_dicts(self, covdictA, covdictB, pickled=False, verbose=True):

        if pickled:
            covdictA = self.unpickle_object(covdictA)
            covdictB = self.unpickle_object(covdictB)

        mean_differences = []
        median_differences = []

        for key in covdictA.keys():
            A = covdictA[key]
            B = covdictB[key]
            print len(A), len(B)
            amean = np.mean(A)
            bmean = np.mean(B)
            amedian = np.median(A)
            bmedian = np.median(B)

            A_min_B_mean = amean-bmean
            A_min_B_median = amedian-bmedian

            mean_differences.append(A_min_B_mean)
            median_differences.append(A_min_B_median)

            if verbose:
                print 'A minus B mean:', A_min_B_mean
                print 'A minus B median:', A_min_B_median, '\n'

        if verbose:
            print 'A minus B mean average:', sum(mean_differences)/len(mean_differences)
            print 'A minus B median average', sum(median_differences)/len(median_differences)




    def simple_covariance(self, dataA, dataB, verbose=True):

        assert dataA.shape == dataB.shape

        byvoxel_covariances = []

        for vox in range(dataA.shape[1]):

            A = []
            B = []

            for tr in range(dataA.shape[0]):
                A.append(dataA[tr,vox])
                B.append(dataB[tr,vox])

            byvoxel_covariances.append(np.cov(A, B)[0,1])

        

        return byvoxel_covariances




    def normalize_data(self, data, verbose=True):

        '''
        a typical normalization function: subtracts the means and divides by
        the standard deviation, column-wise
        '''

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










