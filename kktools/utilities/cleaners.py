

import glob
import shutil
import os
from parsers import subject_dirs
from parsers import dirs as parse_dirs



def glob_remove(file_prefix, suffix='*'):
    candidates = glob.glob(file_prefix+suffix)
    for c in candidates:
        try:
            os.remove(c)
        except:
            pass
        
        
        
class DirectoryCleaner(object):
    
    def __init__(self, prefixes=[], exclude=[], topdir=None):
        super(DirectoryCleaner, self).__init__()
        self.topdir = topdir or os.getcwd()
        if prefixes:
            self.dirs = parse_dirs(topdir=self.topdir, prefixes=prefixes,
                                   exclude=exclude)
        else:
            self.dirs = subject_dirs(topdir=topdir, exclude=exclude)
        self.types, self.files = [], []
        
    def walk_directories(self, function):
        for dir in self.dirs:
            os.chdir()
            self.files = glob.glob('./*')
            function()
            os.chdir('..')
        self.files = []
        
    def action_flag(self, action):
        if action == 'remove':
            for file in self.files:
                for suffix in self.types:
                    if file.endswith(suffix): os.remove(file)
        elif action == 'move':
            for flag in self.types:
                if not flag.endswith('HEAD') or not flag.endswith('BRIK'):
                    dir_name = 'old_'+flag.strip('.')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    for file in self.files:
                        if file.endswith(flag): shutil.move(file, dir_name)
                else:
                    dir_name = 'old_afni'
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    for file in self.files:
                        if file.endswith(flag): shutil.move(file, dir_name)
        
    def remove(self, *args):
        print os.getcwd()
        if args:
            self.types = args
            print self.types
        if not self.files:
            self.walk_directories(self.remove)
        else:
            self.action_flag('rm')
            
    def move(self, *args):
        print os.getcwd()
        if args:
            self.types = args
            print self.types
        if not self.files:
            self.walk_directories(self.move)
        else:
            self.action_flag('mv')
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    