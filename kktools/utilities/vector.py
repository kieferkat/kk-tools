
import os
import numpy as np
import subprocess
from ..utilities.cleaners import glob_remove


def read(vectorfile, usefloat=False):
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
            
            
            
            
            
            
            