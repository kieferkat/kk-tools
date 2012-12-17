
import os, glob
import kktools.api as api




if __name__ == '__main__':
    
    regana = api.RegAnaMaker()
    
    subjects = api.parsers.subjects(topdir=os.path.split(os.getcwd())[0])
    csv_path = os.path.abspath('face_fmri_indvldiff_measures.csv')
    dataset_name = 'zleader_reg_rt'
    dataset_ind = 31
    xvar_names = ['r$hap1','r$lap1','i$hap1','i$lap1']
    script_name = 'regana_auto_test'
    
    regana.write(subjects, dataset_name, dataset_ind, 'test_out', csv_path,
                 xvar_names, script_name=script_name, subject_colname='initials')
    