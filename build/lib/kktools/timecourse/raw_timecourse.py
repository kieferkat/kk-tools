
import glob
import os
import subprocess
import shutil




subjects = ['ar110111','as090711','as092011','dd083111','dh110111',
            'dk091611','gb090711','gb092711','jb092811','ju083111',
            'kr091611','ms101911','oo103111','pl103111','ti092211',
            'tt092811','rl043013','tf050113','jt050113']

onset_vectors = ['varall_onsets','varchoosecert_onsets','varchoosevar_all_onsets',
                 'varchoosevar_win_onsets','varchoosevar_lose_onsets','posall_onsets',
                 'poschoosecert_onsets','poschoosepos_all_onsets',
                 'poschoosepos_win_onsets','poschoosepos_lose_onsets','negall_onsets',
                 'negchoosecert_onsets','negchooseneg_all_onsets',
                 'negchooseneg_win_onsets','negchooseneg_lose_onsets','risk_onsets',
                 'norisk_onsets','risk_switch_onsets','risk_noswitch_onsets',
                 'norisk_switch_onsets','norisk_noswitch_onsets']

vector_file = 'actreg_ch_onsets.txt'
#masks = ['nacc8mm','mpfc','acing','caudate','ins','dorins']
#masks = ['antins_desai_mpm']

masks = ['nacc_desai_mpm']
# NOT relative to scripts:
#output_dir = 'raw_both'
tr_lag = 10
dataset_name = 'actepif_warp.nii'
anat_name = 'anat+tlrc'

tmp_tc_dir = 'raw_tcs/'





def make_vectors(subjdirs, scriptsdir, vector_file):
    # make the absolute path of the vector file:
    vector_path = os.path.join(scriptsdir, vector_file)
    
    # iterate over subjects, using makeVec.py to create the onset vectors per
    # subject:
    for dir in subjdirs:
        os.chdir(dir)
        subprocess.call(['/usr/local/bin/makeVec.py', vector_path])



def fractionize_mask(mask, dataset_name, anat_name, scriptsdir):
    # function assumes to be located within subject directory
    
    # PRINT OUT:
    print 'fractionizing', mask
    
    # define scriptsdir mask location
    scripts_mask = os.path.join(scriptsdir, mask+'+tlrc.')
    
    # attempt to remove old fractionized mask files:
    try:
        os.remove(mask+'r+tlrc.HEAD')
        os.remove(mask+'r+tlrc.BRIK')
    except:
        pass
    
    # fractionize the mask to the functional dataset
    #cmd = ['3dfractionize','-overwrite', '-template', dataset_name, '-input', scripts_mask,
    #       '-warp', anat_name, '-clip', '0.1', '-preserve', '-prefix',
    #       mask+'r+tlrc']
    #subprocess.call(cmd)
    
    #print scripts_mask
    subprocess.call(['3dresample','-overwrite','-master',dataset_name,'-prefix',mask+'r+tlrc',
                     '-inset',scripts_mask])


    

def mask_average(subject, mask, dataset_name, tmp_tc_dir):
    # function assumes within subject folder
    
    # PRINT OUT:
    print 'maskave', subject, mask
    
    # define left, right, both:
    areas = ['l','r','b']
    area_codes = [[3,3],[4,4],[3,4]]
    area_codes = [[1,1],[2,2],[1,2]]
    
    '''
    tcfiles = glob.glob(os.path.join(tmp_tc_dir, '*.tc'))
    removed_count = 0
    for tcf in tcfiles:
        try:
            os.remove(tcf)
            removed_count += 1
        except:
            print 'could not remove:', tcf
            
    print 'tcfiles removed:', removed_count
    '''
    
    # iterate areas, complete mask ave:
    for area, codes in zip(areas, area_codes):
        # define the name of the raw tc file:
        raw_tc = '_'.join([subject, area, mask, 'raw.tc'])
        
        # attempt to remove the file if it already exists here or in the
        # temporary directory:
        try:
            os.remove(raw_tc)
        except:
            pass
        try:
            os.remove(os.path.join(tmp_tc_dir, raw_tc))
        except:
            pass
        
        cmd = ['3dmaskave', '-mask', mask+'r+tlrc', '-quiet', '-mrange',
               str(codes[0]), str(codes[1]), dataset_name]#, '>', raw_tc]
        #subprocess.call(cmd)
        
        fcontent = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        fcontent.wait()
        fcontent = fcontent.communicate()[0]
        
        fid = open(raw_tc,'w')
        fid.write(fcontent)
        fid.close()
        
        
        # move the raw tc file to the tmp tc directory:
        shutil.move(raw_tc, tmp_tc_dir)
        
    
    
        
def maskdump(topdir, subjdirs, subjects, dataset_name, anat_name, masks,
             scriptsdir, tmp_tc_dir):
    
    # PRINT OUT:
    print 'mask dump', subjects
    
    # iterate over subjects:
    for subjdir, subject in zip(subjdirs, subjects):
        # enter the subject directory:
        os.chdir(subjdir)
    
        # create the directory for raw tc files, if necessary:
        if not os.path.exists(tmp_tc_dir):
            os.mkdir(tmp_tc_dir)
			
        tcs = glob.glob(os.path.join(tmp_tc_dir,'*.tc'))
        for tcf in tcs:
            try:
                os.remove(tcf)
            except:
                pass
		
        
        # iterate over masks:
        for mask in masks:
            # fractionize the masks
            fractionize_mask(mask, dataset_name, anat_name, scriptsdir)
            
            # create the raw timecourses:
            mask_average(subject, mask, dataset_name, tmp_tc_dir)
    
    # return to the topdir (just in case):
    os.chdir(topdir)
        
    
    
def parse_tc_file(tc):
    # returns relevant info about and in tc file
    
    # get information on subject, area, and mask form filename:
    simple_fid = os.path.split(tc)[1]
    spl_fid = simple_fid.split('_')
    subject, area, mask = spl_fid[0:3]
    
    # get activation out of tc file:
    fid = open(tc,'r')
    lines = fid.readlines()
    fid.close()
    
    act = [float(x.strip('\n')) for x in lines]
    
    return mask, area, subject, act


def parse_onset_file(vecfile):
    
    # justify, parse onset vectors for each subject:
    num_vectors = []
    fid = open(vecfile,'r')
    num_vector = [int(x.strip('\n')) for x in (fid.readlines())]
    fid.close()
    
    return num_vector

        

def average_activation(output_dir, scriptsdir, subjdirs, tmp_tc_dir,
                       onset_vectors, tr_lag):
    
    # PRINT OUT:
    print 'average activation master function'
    
    # create tc dict to organize files:
    tc_dict = {}
    onset_dict = {}
    
    # iterate over subjects:
    for subjdir in subjdirs:
        
        # PRINT OUT:
        print 'parsing subject', os.path.split(subjdir)[1]
        
        # find the tc files in the tmp directory:
        tc_files = glob.glob(os.path.join(subjdir, tmp_tc_dir, '*.tc'))
        
        # define the onset files in the subject directory:
        onset_files = [os.path.join(subjdir,ov+'.1D') for ov in onset_vectors]
        
        # parse each tc file:
        for tc in tc_files:
            mask, area, subject, act = parse_tc_file(tc)
            
            # parse the onset files with the subject name:
            onset_dict[subject] = {}
            for onset_name, vecfile in zip(onset_vectors, onset_files):
                onset_dict[subject][onset_name] = parse_onset_file(vecfile)
            
            #add to dict in appropriate section:
            if mask in tc_dict:
                if area in tc_dict[mask]:
                    tc_dict[mask][area][subject] = act
                else:
                    tc_dict[mask][area] = {subject:act}
            else:
                tc_dict[mask] = {area:{subject:act}}
        
            
        
    create_timecourse_csvs(tc_dict, onset_dict, output_dir, scriptsdir,
                           tr_lag, onset_vectors)
    
    
    
def create_timecourse_csvs(tc_dict, onset_dict, output_dir, scriptsdir,
                           tr_lag, onset_vectors):
    
    # PRINT OUT:
    print 'create timecourses'
    
    # go into the output dir
    try:
        os.mkdir(output_dir)
    except:
        pass
    os.chdir(output_dir)
    
    # iterate over the vectors & tc_dict creating csvs:
    for mask in tc_dict:
        for area in tc_dict[mask]:
            generate_raw_avg_csv(tc_dict, onset_dict, mask, area, tr_lag,
                                 onset_vectors)
                
                
           
           
                
def generate_raw_avg_csv(tc_dict, onset_dict, mask, area, tr_lag, onset_vectors):
    # pull sub dict:
    subjects_dict = tc_dict[mask][area]
        
    # csv dict, by vector name:
    csv_dict = {}
    
    # iterate over the onset vectors
    for ovec_name in onset_vectors:
        
        # PRINT OUT:
        print 'writing all onset vectors for vector: ', ovec_name
        
        csv_rows = []
        for subject in subjects_dict:
            # get out the activation for the subject:
            act = subjects_dict[subject]
            
            # set up a flexible accumulator, for averaging (there is almost
            # certainly a better way to do this, but not worth worrying about a.t.m.)
            accumulator = []
            for i in range(tr_lag):
                accumulator.append([])
                
            # grab the appropriate number vector file from the onset dict:
            nvec = onset_dict[subject][ovec_name]
                
            # iterate through the nvec indices and the activation, adding to the
            # accumulator where appropriate according to the lag:
            for i, ind in enumerate(nvec):
                if ind == 1:
                    for j,k in zip(range(i,i+tr_lag),range(tr_lag)):
                        if len(act) > j:
                            accumulator[k].append(act[j])
                            
            # average the trs of the accumulator:
            for i,actlist in enumerate(accumulator):
                try:
                    accumulator[i] = sum(actlist)/len(actlist)
                except:
                    print ovec_name, subject, 'accumulator error'

            # create the subject csv row:
            row = subject+','+','.join([str(x) for x in accumulator])+'\n'
            csv_rows.append(row)
            
        # with the rows, write out the csv:
        csv_name = area+mask+'_'+ovec_name+'.csv'
        cfid = open(csv_name,'w')
        for row in csv_rows:
            cfid.write(row)
        



def find_subject_dirs(topdir, subjects):
    subjdirs = []
    for subject in subjects:
        dir = glob.glob(os.path.join(topdir,subject)+'*')[0]
        if dir:
            subjdirs.append(dir)
    return subjdirs



if __name__ == '__main__':
    
    output_dir = 'timecourses_desai'
    
    scriptsdir = os.getcwd()
    topdir = os.path.split(os.getcwd())[0]
    subjectdirs = find_subject_dirs(topdir, subjects)
    
    if vector_file:
        make_vectors(subjectdirs, scriptsdir, vector_file)
    
    maskdump(topdir, subjectdirs, subjects, dataset_name, anat_name, masks,
             scriptsdir, tmp_tc_dir)
    
    average_activation(output_dir, scriptsdir, subjectdirs, tmp_tc_dir,
                       onset_vectors, tr_lag)
    
    os.chdir(scriptsdir)
        
        
    
    
    
    
    
    
    
