
import os
import glob
import re
from pprint import pprint, pformat



class Variables(dict):
    
    
    def __init__(self):
        super(Variables, self).__init__()
        
    
    def __getattr__(self, name):
        return self[name]


    def __setattr__(self, name, value):
        super(Variables, self).__setattr__(name, value)
        self[name] = value
        
        
        
        
        
        
        
        