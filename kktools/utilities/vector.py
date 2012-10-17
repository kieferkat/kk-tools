
import os
import numpy as np
import subprocess
from ..utilities.cleaners import glob_remove


def read(vectorfile, float=False):
    fid = open(vectorfile,'r')
    lines = fid.readlines()
    fid.close()
    if not float:
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



def combine(vectors, operation_on_overlap='zero', float=False, warning_on_overlap=True):
    
    if operation_on_overlap not in ['multiply','add','sum','subtract','zero']:
        print 'invalid operation specified'
        return None
    
    single_vector = []
    
    for i, item_set in enumerate(zip(*vectors)):
        
        if len(np.nonzero(item_set)[0]) == 0:
            if float:
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
                if float:
                    single_vector.append(0.)
                else:
                    single_vector.append(0)
                    
    return single_vector
            
            
            
            
            
            
            