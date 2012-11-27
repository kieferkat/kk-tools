
import os
import subprocess
import glob
import shutil

from ..afni.pipeline import AfniPipeline
from ..afni.functions import AfniWrapper
from ..base.scriptwriter import Scriptwriter
from ..utilities.cleaners import glob_remove
from ..base.inspector import Inspector
from ..utilities.csv import CsvTools
from ..utilities.vector import VectorTools



class AfniPipeline(object):
    
    def __init__(self):
        super(AfniPipeline, self).__init__()
        self.scriptwriter = Scriptwriter()
        self.afni = AfniWrapper()
        self.run_script = True
        self.write_script = True
        self.inspector = Inspector()
        self.csv = CsvTools()
        self.vector = VectorTools()
        
    
    def justify_suffix(self, dset_path, suffix):
        pass