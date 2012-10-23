
import os, sys
import subprocess
import shutil
import glob

from ..base.process import Process
from ..base.scriptwriter import Scriptwriter
from ..utilities.cleaners import glob_remove




class Regression(Process):
    
    
    def __init__(self, variable_dict=None):
        super(Regression, self).__init__(variable_dict=variable_dict)
        self.script_name = 'regression'
        # note to self: incorporate makevec later!
        self.makevec_path = 'usr/local/bin/makeVec.py'
        self.scriptwriter = Scriptwriter()
        
        
    
    def _discover_waver_names(self, modelfile_path):
        vf = open(modelfile_path,'r')
        vl = vf.readlines()
        vf.close()
        self.waver_names = []
        for l in vl:
            if l.upper.startswith('OUTPUT:'):
                self.waver_names.append(l[7:].strip(' \"\'\n'))
                
                
                
    def run_makevec(self, modelfile_path):
        subprocess.call([self.makevec_path, modelfile_path])
        
        
    def waver(self, subject_dir, waver_names=None, waver_dt=2.0, waver_type='GAM'):
        
        required_vars = {'waver_names':waver_names, 'waver_dt':waver_dt,
                         'waver_type':waver_type}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        waver_in_paths = [os.path.join(subject_dir, w) for w in self.waver_names]
        waver_out_paths = [w[:-3]+'c.1D' for w in waver_out_paths]
        
        if not self.waver_type.startswith('-'):
            wtype = '-'+self.waver_type
        else:
            wtype = self.waver_type
            
        cmds = [['waver', '-dt', str(self.waver_dt), wtype, '-input', fid] for fid in waver_in_paths]
        
        for cmd, out_path in zip(cmds, waver_out_paths):
            out = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            out.wait()
            out = out.communicate()[0]
            fid_out = open(out_path, 'w')
            fid_out.write(out)
            fid_out.close()
            
            
    
        
        
        
        
        
        
        
        
        
        
        
                