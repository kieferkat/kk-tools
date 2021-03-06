
import glob
import re
import os






def dirs(topdir=os.getcwd(), prefixes=[], exclude=[], regexp=None, initial_glob='*'):
    
    files = [f for f in glob.glob(os.path.join(topdir,initial_glob)) if os.path.isdir(f)]
    if regexp:
        files = [f for f in files if re.search(regexp, os.path.split(f)[1])]
    files = [f for f in files if not any([os.path.split(f)[1].startswith(ex) for ex in exclude])]
    if prefixes:
        files = [f for f in files if any([os.path.split(f)[1].startswith(pr) for pr in prefixes])]
        
    return sorted(files)


def subject_dirs(topdir=os.getcwd(), prefixes=[], exclude=[], initial_glob='*'):
    return dirs(topdir=topdir, prefixes=prefixes, exclude=exclude,
                initial_glob=initial_glob,regexp=r'[a-zA-Z]\d\d\d\d\d\d')
    
    
def subjects(max_length=None, topdir=os.getcwd(), prefixes=[], exclude=[],
             initial_glob='*'):
    subjdirs=subject_dirs(topdir=topdir, prefixes=prefixes, exclude=exclude,
                          initial_glob=initial_glob)
    if not max_length:
        return [os.path.split(x)[1] for x in subjdirs]
    else:
        return [os.path.split(x)[1][0:min(max_length,len(os.path.split(x)[1]))] for x in subjdirs]
        
def consprint(input_list, python=True, bash=True):
    
    if python:
        print '[\''+'\',\''.join([str(x) for x in input_list])+'\']'
        
    if bash:
        print '( '+' '.join([str(x) for x in input_list])+' )'

        
    
    
    
        