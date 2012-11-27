
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
                
                
    def pair_subjects_xrows(subjects, xrow_dict):
        subs = [os.path.split(sub)[1] for sub in subjects]
        xrows = []
        for subject in subs:
            xrows.append(xrow_dict[subject])
        return xrows
    
    
        
    def run(self, subject_dirs, dataset_name, output_path, subject_Xrows, modelX=[], null=[0],
            rmsmin=0):
        
        if type(subject_Xrows) == type([]):
            Xrows = subject_Xrows
        elif type(subject_Xrows) == type({}):
            Xrows = self.pair_subjects_xrows(subject_dirs, subject_Xrows)
            
        self.afni.regana(subject_dirs, dataset_name, output_path, Xrows, modelX=modelX,
                         null=null, rmsmin=rmsmin)
        
        
    
    def write(self, subjects, dataset_name, output_name, subject_Xrows, modelX=[], null=[0],
              rmsmin=0):
        
        if type(subject_Xrows) == type([]):
            Xrows = subject_Xrows
        elif type(subject_Xrows) == type({}):
            Xrows = self.pair_subjects_xrows(subjects, subject_Xrows)
        
        self.afni.regana.write(self.scriptwriter, subjects, dataset_name, output_name,
                               Xrows, modelX=modelX, null=null, rmsmin=rmsmin)
        
        
    