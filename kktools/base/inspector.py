

import os, sys
import shutil, glob
import subprocess

from ..afni.functions import AfniWrapper



class Inspector(object):
    
    def __init__(self):
        super(Inspector, self).__init__()
        self.afni = AfniWrapper()
        
        
    def _split_by_colon(self, line, ind=1):
        spl = line.split(':')
        return spl[ind].strip()
        
    
    def _orientation_parse(self, line):
        spl = line.split(':')[1]
        spl = spl.split(' ')
        nums = []
        for i,x in enumerate(spl):
            if x in ['[R]','[L]','[A]','[P]','[I]','[S]','mm']:
                nums.append(float(spl[i-1]))
        return nums
        
        
    def parse_dataset_info(self, dataset_path, info_output_filepath=None,
                           verbose=True):
        
        dataset_info = self.afni.info(dataset_path, output_filepath=info_output_filepath)
        info_dict = {}
        info_dict['dxyz_shape'] = []
        
        li = dataset_info.split('\n')
        
        for line in li:
            #print line
            if line.find('Template Space') >= 0:
                if verbose: print 'Found template space.'
                info_dict['template_space'] = self._split_by_colon(line)
                
            elif line.find('Geometry String') >= 0:
                if verbose: print 'Found geometry string'
                shape = self._split_by_colon(line, ind=2).strip('\"')
                info_dict['shape'] = [int(x) for x in shape.split(',')]
                
            elif line.find('R-to-L') >= 0:
                if verbose: print 'Found R-to-L'
                nums = self._orientation_parse(line)
                info_dict['R'] = nums[0]
                info_dict['L'] = nums[1]
                info_dict['dxyz_shape'].append(nums[2])
                
            elif line.find('A-to-P') >= 0:
                if verbose: print 'Found A-to-P'
                nums = self._orientation_parse(line)
                info_dict['A'] = nums[0]
                info_dict['P'] = nums[1]
                info_dict['dxyz_shape'].append(nums[2])
                
            elif line.find('I-to-S') >= 0:
                if verbose: print 'Found I-to-S'
                nums = self._orientation_parse(line)
                info_dict['I'] = nums[0]
                info_dict['S'] = nums[1]
                info_dict['dxyz_shape'].append(nums[2])
                
            elif line.find('Number of time steps') >= 0:
                if verbose: print 'Found total trs'
                spl = line.split(' ')
                nums = []
                for i,x in enumerate(spl):
                    if x == '=':
                        nums.append(spl[i+1])
                if len(nums) > 0:
                    info_dict['total_trs'] = int(nums[0])
                if len(nums) > 1:
                    info_dict['tr_length'] = float(nums[1][:-1])
                if len(nums) > 3:
                    info_dict['nslices'] = int(nums[3])
                if len(nums) > 4:
                    info_dict['slice_thickness'] = float(nums[4])
                
        return info_dict
                
                
            
            
            
            
            
            
            