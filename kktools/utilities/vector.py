
import os
import numpy as np
import glob as glob
import subprocess
from ..utilities.cleaners import glob_remove


def read(vectorfile, usefloat=True):
    fid = open(vectorfile,'r')
    lines = fid.readlines()
    fid.close()
    if not usefloat:
        lines = [int(l.strip('\n')) for l in lines]
    else:
        lines = [float(l.strip('\n')) for l in lines]
    return lines


def stringread(vectorfile):
    fid = open(vectorfile, 'r')
    lines = fid.readlines()
    fid.close()
    return [x.strip('\n') for x in lines]


def write(vector, filename):
    fid = open(filename,'w')
    for item in vector:
        fid.write(str(item)+'\n')
    fid.close()


def vectordir_todict(vector_dir, glob_prefix='*.tc', filename_split_keyinds=None,
                     filename_split='_', verbose=False):
    
    vector_dict = {}
    vectors = glob.glob(os.path.join(vector_dir, glob_prefix))
    for vector in vectors:
        vector_name = os.path.split(vector)[1].rstrip('.tc')
        try:
            if verbose:
                print 'Attempting to load vector:', vector_name
            vsplit = vector_name.split(filename_split)
            if filename_split_keyinds:
                id = ''.join([k for i,k in vsplit if i in filename_split_keyinds])
            else:
                id = vector_name
            vec_data = read(vector)
            vector_dict[id] = vec_data
        except:
            if verbose:
                print 'Error loading vector:', vector_name
                
    return vector_dict


def subject_vector_dict(subject_dirs, vector_dir, glob_prefix='*.tc',
                             filename_split_keyinds=[2,3], filename_split='_',
                             verbose=False):
    subject_dict = {}
    for dir in subject_dirs:
        id = os.path.split(dir)[1]
        vecdir = os.path.join(dir, vector_dir)
        try:
            if verbose:
                print 'Attempting to load vectors for subject:', id
            vector_dict = vectordir_todict(vecdir, glob_prefix=glob_prefix,
                                           filename_split_keyinds=filename_split_keyinds,
                                           filename_split=filename_split, verbose=verbose)
            subject_dict[id] = vector_dict
        except:
            if verbose:
                print 'error loading vectors for subject:', id
    
    return subject_dict
        
    


def replace(vector, oldvalue, newvalue):
    return [x if not x == oldvalue else newvalue for x in vector]
    
    
    
def replace_at_interval(vector, newvalue, interval, start=0):
    rewrite_inds = [x-1 for x in range(start, len(vector)+1, interval)]
    return [newvalue if (i in rewrite_inds) else x for i,x in enumerate(vector)]
    
    
# simply call stephanie's old makeVec.py for now.
# eventually create a new makevec class because stephanie's is limited...
def makevecs(dirs, vector_model_path):
    for dir in dirs:
        os.chdir(dir)
        subprocess.call(['/usr/local/bin/makeVec.py'], vector_model_path)
        
        
    
    

def offset(vector, offset):
    if offset < 0:
        return [x for x in vector[abs(offset):]] + [0]*abs(offset)
    elif offset > 0:
        return [0]*offset + [x for x in vector[:-1*offset]]
    else:
        return vector



def combine(vectors, operation_on_overlap='zero', usefloat=False, warning_on_overlap=True):
    
    if operation_on_overlap not in ['multiply','add','sum','subtract','zero']:
        print 'invalid operation specified'
        return None
    
    single_vector = []
    
    for i, item_set in enumerate(zip(*vectors)):
        
        if len(np.nonzero(item_set)[0]) == 0:
            if usefloat:
                single_vector.append(0.)
            else:
                single_vector.append(0)
                
        elif len(np.nonzero(item_set)[0]) == 1:
            single_vector.append(item_set[np.nonzero(item_set)[0]])
            
        else:
            if warning_on_overlap:
                print 'overlap at index: ', i
                
            if operation_on_overlap == 'multiply':
                val = reduce(lambda x, y: x*y, item_set)
                single_vector.append(val)
                
            elif operation_on_overlap == 'sum' or operation_on_overlap == 'add':
                single_vector.append(sum(item_set))
                
            elif operation_on_overlap == 'subtract':
                val = reduce(lambda x, y: x-y, item_set)
                single_vector.append(val)
                
            elif operation_on_overlap == 'zero':
                if usefloat:
                    single_vector.append(0.)
                else:
                    single_vector.append(0)
                    
    return single_vector
            
            
            
            
            
            
            