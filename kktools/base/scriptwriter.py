
import os
import subprocess
import glob


class Scriptwriter(object):
    
    
    def __init__(self):
        super(Scriptwriter, self).__init__()
        self.master = []
        self.sections = []
        self.unset_variables = []
        
        
    def add_topheader(self, scriptname='Script'):
        top = ['#! /bin/csh',
               '######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##',
               '####',
               '##\t\t'+scriptname+' auto-written by Scriptwriter',
               '##',
               '##',
               '##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##\n\n']
        self.master.extend(top)
        
        
    def add_subject_loop(self, subjects, command_block):
        subject_loop = []
        subject_loop.append('foreach subject ( '+' '.join(subjects)+' )\n\n')
        subject_loop.append(['cd ../${subject}*\n\n'])
        subject_loop.append(command_block)
        subject_loop.append('\n\nend\n\n')
        
        
    def add_unset_vars(self):
        varhead = ['##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##',
                   '##\t\tUnset variables:',
                   '##\n\n']
        for varname in self.unset_variables:
            varhead.append('set '+varname+' = #INSERT VALUE(S)\n')
        varhead.append('\n\n')
        
        
    def add_header(self, header):
        head = ['##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##',
                '##\t\t'+header,
                '##\n\n']
        self.sections.extend(head)    
    
        
    def fill_cmd(self, cmd, vars):
        filled_cmd = []
        for item in cmd:
            if type(item) in (list, tuple):
                filled_cmd.append(self.fill_cmd(item, vars))
            else:
                if vars:
                    for vname, var in vars.items():
                        if var:
                            if item.find('${'+vname+'}') != -1:
                                item = item.replace('${'+vname+'}', var)
                        else:
                            if vname not in self.unset_variables:
                                self.unset_variables.append(vname)
                else:
                    citem = item[:]
                    while citem.find('${') != -1:
                        self.unset_variables.append(citem[citem.find('${')+2:citem.find('}')])
                        citem = citem[citem.find('}'):]
                filled_cmd.append(item)
        return filled_cmd
    
    
    def add_cmd(self, cmd, vars):
        filled_cmd = self.fill_cmd(cmd, vars)
        self.sections.extend(filled_cmd)
                
        
    def add_cleaner(self, pieces, check='', find=''):
        for name, item in pieces.items():
            cleaner = ['if ( -e '+item+check+' ) then',
                       ['rm -rf '+item+find],
                       'endif']
            self.sections.extend(cleaner)
            
        
    def add_cleaners(self, clean):
        for key, pieces in clean.items():
            if key == 'afni':
                self.add_cleaner(pieces, check='+orig.HEAD', find='+orig*')
            elif key == 'afni_tlrc':
                self.add_cleaner(pieces, check='+tlrc.HEAD', find='+tlrc*')
            elif key == 'standard':
                self.add_cleaner(pieces)
                
    
    def write_section(self, header=None, cmd=None, vars=None, clean=None):
        if header: self.add_header(header)
        if clean: self.add_cleaners(clean)
        if cmd: self.add_cmd(cmd, vars)
        
        
    def recursive_flatten(self, container, tabs):
        out = ''
        for item in container:
            if type(item) in (list, tuple):
                out = out+self.recursive_flatten(item, tabs+1)+'\n'
            else:
                out = out+('\t'*tabs)+item+'\n'
        return out
                
        
        
    def write_out(self, filename):
        fid = open(filename,'w')
        script = self.recursive_flatten(self.master, 0)
        fid.write(script)
        fid.close()
        os.system('chmod 775 '+filename)
        
        
        
        
        
        
        
    
    