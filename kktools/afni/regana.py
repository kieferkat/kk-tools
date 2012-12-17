
import os, sys
import subprocess
import shutil
import glob
from pprint import pprint

from ..afni.pipeline import AfniPipeline


class RegAnaMaker(AfniPipeline):
    
    def __init__(self):
        super(RegAnaMaker, self).__init__()
        self.script_name = 'regana_auto'
        
        
    def xrows_from_inddiff_csv(self, csv, xcol_names, subjects, subject_colname='subject'):
        csvlines = self.csv.read(csv)
        coldict = self.csv.csv_to_coldict(csvlines)
        
        xrow_dict = {}
        for row, subject in enumerate(coldict[subject_colname]):
            xrow = []
            for xcol in xcol_names:
                xrow.append(coldict[xcol][row])
            xrow_dict[subject] = xrow
            
        return xrow_dict
                
                
    def pair_subjects_xrows(self, subjects, xrow_dict):
        subs = [os.path.split(sub)[1] for sub in subjects]
        xrows = []
        for subject in subs:
            if subject in xrow_dict:
                xrows.append(xrow_dict[subject])
            else:
                for keysub in xrow_dict:
                    if subject.lower().startswith(keysub.lower()):
                        xrows.append(xrow_dict[keysub])
                        break
        return xrows
    
    
        
    def run(self, subject_dirs, dataset_name, output_path, subject_Xrows, modelX=[], null=[0],
            rmsmin=0):
        
        if type(subject_Xrows) == type([]):
            Xrows = subject_Xrows
        elif type(subject_Xrows) == type({}):
            Xrows = self.pair_subjects_xrows(subject_dirs, subject_Xrows)
            
        self.afni.regana(subject_dirs, dataset_name, output_path, Xrows, modelX=modelX,
                         null=null, rmsmin=rmsmin)
        
        
    
    def write(self, subjects, dataset_name, dataset_ind, output_name, csv_path, xvar_names,
              modelX=[], null=[0], rmsmin=0, script_name=None, subject_colname='subject',
              verbose=True):
        
        if script_name:
            self.script_name = script_name
        
        xrow_dict = self.xrows_from_inddiff_csv(csv_path, xvar_names, subjects,
                                                subject_colname=subject_colname)
        
        if verbose:
            pprint(xrow_dict)
        
        Xrows = self.pair_subjects_xrows(subjects, xrow_dict)
        
        self.afni.regana.write(self.scriptwriter, subjects, dataset_name, dataset_ind,
                               output_name, Xrows, modelX=modelX, null=null, rmsmin=rmsmin)
        
        self.scriptwriter.write_out(self.script_name, add_unset_vars=False, topheader=False)
        
        
    