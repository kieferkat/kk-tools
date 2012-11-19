
import os
import subprocess
import glob


class Scriptwriter(object):
    
    
    def __init__(self):
        super(Scriptwriter, self).__init__()
        self.master = []
        self.blocks = []
        self.current_block = []
        self.unset_variables = []
        
        
    def add_topheader(self, scriptname='Script'):
        top = ['#! /bin/csh\n',
               '######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##',
               '####',
               '##\t\t'+scriptname+' auto-written by Scriptwriter',
               '##',
               '##',
               '##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##\n\n']
        return top
        
        
    def add_unset_vars(self):
        varhead = ['##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##',
                   '##\t\tUnset variables:',
                   '##\n']
        for varname in self.unset_variables:
            varhead.append('set '+varname+' = #INSERT VALUE(S)')
        varhead.append('\n')
        return varhead
                
                
                
    def create_header(self, header):
        head = ['\n',
                '##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##',
                '##\t\t'+header,
                '##\n']
        return head
    
    
    def justify_command_variables(self, command, variables):
        filled_cmd = []
        for item in cmd:
            if type(item) in (list, tuple):
                filled_cmd.append(self.justify_command_variables(item, variables))
            else:
                if variables:
                    for vname, var in variables.items():
                        if var is not None:
                            print vname, var
                            if item.find('${'+vname+'}') != -1:
                                item = item.replace('${'+vname+'}', str(var))
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
    
    
    def justify_cleaner_variables(self, cleaner, variables):
        filled_cleaner = []
        for [var, cltype] in cleaner:
            if variables:
                if var in variables:
                    filled_cleaner.append([variables[var], cltype])
                else:
                    filled_cleaner.append(['${'+var+'}', cltype])
            else:
                filled_cleaner.append(['${'+var+'}', cltype])
        return filled_cleaner
 
                
    
    def create_cleaner(self, cleanpairs):
        cleaner = []
        for [var, cltype] in cleanpairs:
            if not cltype:
                cltype = ''
            else:
                cltype = '*'+cltype+'*'
                
            cleaner.extend(['rm -rf '+var+cltype+'\n'])
        return cleaner

    
                
    def add_section(self, section_dict):
        current_section = []
        header = None
        section_variables = None
        section_command = None
        cleaner = None
        
        for keyword, vals in section_dict.items():
            if vals is not None:
                if keyword == 'header':
                    header = vals
                elif keyword == 'clean':
                    cleaner = vals
                elif keyword == 'command':
                    section_command = vals
                elif keyword == 'variables':
                    section_variables = vals
                    
                    
        if section_command is not None:
            section_command = self.justify_command_variables(section_command, section_variables)
        if cleaner is not None:
            cleaner = self.justify_cleaner_variables(cleaner, section_variables)
                    
        if header is not None:
            current_section.extend(self.create_header(header))
            
        if cleaner is not None:
            current_section.extend(self.create_cleaner(cleaner))
            
        if section_command is not None:
            current_section.extend(section_command)
            
        self.current_block.append(current_section)

            
    def loop_block_over_subjects(self, subjects):
        subject_loop = []
        subject_loop.append('foreach subject ( '+' '.join(subjects)+' )\n')
        subject_loop.append(['cd ../${subject}*\n'])
        subject_loop.append(self.current_block)
        subject_loop.append('\n\nend\n\n')
        self.current_block = subject_loop
            
            
    def next_block(self):
        if any(self.current_block):
            self.blocks.append(self.current_block)
        self.current_block = []
        
        
    def recursive_flatten(self, container, tabs):
        out = ''
        for item in container:
            if type(item) in (list, tuple):
                out = out+self.recursive_flatten(item, tabs+1)
            else:
                out = out+('\t'*tabs)+item+'\n'
        return out
                
        
    def prep_master(self):
        self.master.extend(self.add_topheader())
        self.master.extend(self.add_unset_vars())
        self.next_block()
        for block in self.blocks:
            self.master.extend(block)
        
        
    def write_out(self, filename):
        self.prep_master()
        fid = open(filename,'w')
        script = self.recursive_flatten(self.master, 0)
        fid.write(script)
        fid.close()
        os.system('chmod 775 '+filename)
        
        
                
    '''
    def write_section(self, header=None, cmd=None, vars=None, clean=None):
        if header: self.add_header(header)
        if clean: self.add_cleaners(clean)
        if cmd: self.add_cmd(cmd, vars)
        
    
    def write_line(self, line=None, vars=None):
        if line: self.add_cmd(line, vars)
    
    

    def add_cleaner(self, pieces, check='', find=''):
        for name, item in pieces.items():
            if item:
                cleaner = ['if ( -e '+item+check+' ) then',
                           ['rm -rf '+item+find],
                           'endif\n']
            else:
                cleaner = ['if ( -e ${'+name+'}'+check+' ) then',
                           ['rm -rf ${'+name+'}'+find],
                           'endif\n']
            self.sections.extend(cleaner)
    '''     
        
    
        
        
        
        
        
        
        
    
    