
import os, sys
import subprocess
import shutil
import glob
from pprint import pprint

from ..afni.pipeline import AfniPipeline


class Deconvolve(AfniPipeline):
    
    def __init__(self):
        super(Deconvolve, self).__init__()
        self.script_name = 'regression_auto'
        self.makevec_path = '/usr/local/bin/makeVec.py'
        
    
    def run(self, subject_dirs, subject_prefixes=None):
        
        if not subject_prefixes:
            subject_prefixes = [os.path.split(sd)[1] for sd in subject_dirs]
        
        # make the vectors for each subject
        self.vector.makevecs(subject_dirs, vector_model_path, makevec_path=self.makevec_path)
        
        
        