
# necessary imports
import os
import kktools.api as api


if __name__ == '__main__':
    
    # begin by creating the variable dictionary:
    vars = api.Variables()
    
    # assuming you are running this from the scripts directory...
    vars.scriptsdir = os.getcwd()
    vars.topdir = os.path.split(vars.scriptsdir)[0]
    
    # generate all the subject dirs from the topdir:
    vars.subject_dirs = api.parsers.subject_dirs(topdir=vars.topdir)
    vars.subjects = api.parsers.subjects(topdir=topdir)
    
    # save directory:
    vars.save_directory = os.path.join(vars.topdir, 'kktools_output')
    
    # file, path, afni variables:
    vars.functional_name = None
    vars.anatomical_name = None
    vars.dxyz = None
    vars.talairach_template_path = None
    vars.nifti_name = ['pop12_warp375.nii','pop34_warp375.nii']
    vars.response_vector = ['choice12.1D','choice34.1D']
    vars.mask_path = os.path.join(os.path.join(vars.topdir, 'reg_output'), 'talairach_mask375.nii')
    
    
    # datamanager trial selection variables:
    vars.selected_trs = [1,2,3,4,5,6,7]
    vars.lag = 2
    vars.downsample_type = 'subject'
    vars.with_replacement = False
    vars.replacement_ceiling = None
    vars.random_seed = None
    
    
    # graphnet specific variables:
    G = None
    l1 = None
    l2 = None
    l3 = None
    delta = None
    svmdelta = None
    initial = None
    adaptive = False
    svm = False
    scipy_compare = True
    tol = 1e-5
    
    
    # create the datamanager object that parses niftis:
    data = api.DataManager(variable_dict=vars)
    
    # do necessary data management prior to graphnet:
    #data.create_niftis()
    data.create_trial_mask()
    data.load_nifti_data()
    data.subselect_data()
    data.save_numpy_data()
    #data.load_numpy_data()
    data.create_XY_matrices(downsamples_type=vars.downsample_type,
                            with_replacement=vars.with_replacement,
                            replacement_ceiling=vars.replacement_ceiling)
    
    # create a graphnet interface object, give it the variables:
    gnet = api.GraphnetInterface(data_obj=data, variable_dict=vars)
    
    # run the graphnet with parameters:
    
    gnet.test_graphnet(data.X, data.Y, G=G, l1=l1, l2=l2, l3=l3, delta=delta,
                       svmdelta=svmdelta, initial=initial, adaptive=adaptive,
                       svm=svm, scipy_compare=scipy_compare, tol=tol)
    
    
    
    
    
    
    
    
    
    
    
    
    