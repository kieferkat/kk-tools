

from ..base.variables import Variables

preprocessing = Variables()
preprocessing.anatomical_name = 'anat'
preprocessing.leadin = 6
preprocessing.leadout = 4
preprocessing.tr_length = 2.0
preprocessing.tshift_slice = 0
preprocessing.tpattern = 'altplus'
preprocessing.motionfile_name = '3dmotion.1D'
preprocessing.volreg_base = 3
preprocessing.blur_kernel = 4
preprocessing.normalize_expression = '((a-b)/b)*100'
preprocessing.highpass_value = 0.011
preprocessing.tt_n27_path = '/Users/span/abin/TT_N27+tlrc'