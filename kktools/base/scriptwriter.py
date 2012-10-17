
import os
import glob
import subprocess
from ..base.process import Process


class ScriptWriter(Process):
    
    
    def __init__(variable_dict=None):
        super(ScriptWriter, self).__init__(variable_dict=variable_dict)
        
    
    def read_template(self, template_path):
        fid = open(template_path, 'r')
        lines = fid.readlines()
        fid.close()
        return [l.strip('\n') for l in lines]
        
        
    def determine_inclusions(self, lines, section_flags):
        section_flags = [flag.lower() for flag in section_flags]
        inclusions = []
        skip_flag = False
        for line in lines:
            sline = line.lower().strip()
            if sline.startswith('!{begin:'):
                section = sline[sline.index(':')+1:sline.index('}')]
                if not section in section_flags:
                    skip_flag = True
            elif sline.startswith('!{end'):
                skip_flag = False
            elif not skip_flag:
                inclusions.append(line)
        return inclusions
        
        
    def insert_variables(self, lines, **kwargs):
        if kwargs:
            self._assign_variables(kwargs)
            
        filled = []
        for line in lines:
            while not line.find('?{'
        
        
            
    def parse_template(self, template_path):
        self.raw_lines = self.read_template(template_path)
        self.script_lines = self.determine_inclusions(self.raw_lines, section_flags)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    