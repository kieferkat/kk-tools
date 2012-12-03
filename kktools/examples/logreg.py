
import kktools.api as api
import os, sys, glob
from pprint import pprint

vars = api.Variables()

vars.topdir = os.path.split(os.getcwd())[0]
vars.scriptsdir = os.getcwd()
vars.subject_dirs = api.parsers.subject_dirs(topdir=vars.topdir, exclude=['nn'])

vars.anatomical_name = 'anat+tlrc'
vars.functional_name = 'actepif+orig'
vars.mask_dir = vars.scriptsdir
vars.mask_names = ['nacc8mm','ins','caudate','acing','mpfc']
vars.behavior_csv_name = 'actmatrix_indiff.csv'

logdata = api.LogisticData(variable_dict=vars)

#logdata.maskdump()
logdata.load_subject_csvs()
logdata.load_subject_raw_tcs()
logdata.write_logistic_data('all_log_data.csv',logdata.logistic_data_dict)
logdata.create_XY_matrices(['bnacc8mmr','binsr','bmpfcr'],'yesno',conditional_dict={'TR':[4],
    'yesno':[1,-1]},
    filepath='sparse_log_data2.csv')

reg = api.LogisticRegression(data_obj=logdata, variable_dict=vars)
reg.crossvalidate(predict_with_intercept=False)

pprint(zip(reg.subjects_in_folds, reg.testresults))