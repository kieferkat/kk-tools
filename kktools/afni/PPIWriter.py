
import os, sys
import glob
import subprocess
import optparse


class PPIWriter(object):
    
    def __init__(self):
        super(PPIWriter, self).__init__()
        self.master = []
        self.script_name = 'PPI_auto'
        self.subjects = []
        self.anatomical = 'anat'
        self.functional = 'dataset'
        self.dxyz = 1.
        self.mask_dir = 'scripts'
        self.mask_name = 'nacc8mm'
        self.suffix = 'b'
        self.mrange = [1,2]
        
        
    def header(self):
        top = ['#! /bin/csh\n',
               '######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##',
               '####',
               '##\t\t'+self.script_name+' auto-written.',
               '##',
               '##',
               '##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##\n\n']
        return top
    
    
    def subject_loop(self, subjects, internal):
        subj_loop = ['foreach subject ( '+' '.join(subjects)+' )\n']
        subj_internal = ['cd ../${subject}*',
                         'echo processing ${subject}\n']
        subj_internal.extend(internal)
        subj_loop.append(subj_internal)
        subj_loop.append('end\n\n')
        return subj_loop
    
    
    def warp_functional(self):
        cmd = ['# warp functional data to tlrc space: ',
               'adwarp -apar '+self.anatomical+'+tlrc -dpar '+self.functional+'+orig -dxyz '+str(self.dxyz)+' -prefix '+self.functional+'_warp']
        return cmd
    
    
    def clean(self, find_file, flag_prefix):
        cmd = ['# cleanup',
               'if ( -e '+find_file+' ) then',
               ['rm -rf '+flag_prefix+'*'],
               'endif']
        return cmd
    
    
    def resample(self):
        
        cmd = ['# resample mask to dataset:']
        pieces = ['3dresample', '-master',  self.functional+'_warp+tlrc',
                  '-prefix', self.mask_name+'r', '-inset',
                  '../'+self.mask_dir+'/'+self.mask_name+'+tlrc']
        move = ['mv','../'+self.mask_dir+'/'+self.mask_name+'r*','./']
        cmd.append(' '.join(pieces))
        cmd.append(' '.join(move))
        return cmd
    
    
    def maskave(self, comment=True):
        
        if comment:
            cmd = ['# generate timecourses for VOI:']
        else:
            cmd = []
        pieces = ['3dmaskave','-mask',self.mask_name+'r+tlrc','-mrange']+[str(x) for x in self.mrange]+['-quiet',self.functional+'_warp+tlrc','>',self.mask_name+'_'+self.suffix+'.1D']
        cmd.append(' '.join(pieces))
        return cmd
    
    
    def detrend(self, polort=2, comment=True):
        
        if comment:
            cmd = ['# detrend VOI timeseries']
        else:
            cmd = []
        pieces = ['3dDetrend','-polort',str(polort),'-prefix',
                  self.mask_name+'_'+self.suffix+'R',
                  self.mask_name+'_'+self.suffix+'.1D\\\'']
        cmd.append(' '.join(pieces))
        return cmd
    
    
    def transpose(self, comment=True):
        
        if comment:
            cmd = ['# transpose detrended tc:']
        else:
            cmd = []
            
        pieces = ['1dtranspose',self.mask_name+'_'+self.suffix+'R.1D',
                  self.mask_name+'_'+self.suffix+'_ts.1D']
        cmd.append(' '.join(pieces))
        return cmd
    
    
    def generate_hrf(self, tr_length=2):
        
        cmd = ['# generate HRF:',
               'waver -dt '+str(tr_length)+' -GAM -inline 1@1 > GammaHR.1D']
        return cmd
    
    
    def tfitter(self, penalty='012', weight='0', comment=True):
        
        if comment:
            cmd = ['# deconvolve the VOI timecourse:']
        else:
            cmd = []
            
        pieces = ['3dTfitter','-RHS',self.mask_name+'_'+self.suffix+'_ts.1D',
                  '-FALTUNG','GammaHR.1D',self.mask_name+'_'+self.suffix+'_Neur',
                  penalty, str(weight)]
        cmd.append(' '.join(pieces))
        return cmd
    
    
    def create_interaction(self, vector_name, comment=True):
        
        if comment:
            cmd = ['# create interaction terms:']
        else:
            cmd = []
            
        pieces = ['1deval','-expr','\'a*b\'','-a','_'.join([self.mask_name, self.suffix, 'Neur.1D\\\'']),
                  '-b', vector_name+'.1D','>','_'.join([self.mask_name,self.suffix,'inter.1D'])]
        cmd.append(' '.join(pieces))
        return cmd
    
    
    def convolve_interaction(self, numout, tr_length=2, peak=1, comment=True):
        
        if comment:
            cmd = ['# convolve interaction term with HRF:']
        else:
            cmd = []
            
        pieces = ['waver','-GAM','-peak',str(peak),'-TR',str(tr_length),
                  '-input','_'.join([self.mask_name,self.suffix,'inter.1D']),
                  '-numout',str(numout),'>','_'.join([self.mask_name,self.suffix,'inter','ts.1D'])]
        cmd.append(' '.join(pieces))
        return cmd
    
    
    def deconvolve(self, motionfile, vector_name, other_regressors, jobs=8,
                   goforit=2, polort=2):
        
        cmd = ['# Run 3ddeconvolve:']
        
        num_stimts = 9+len(other_regressors)
        
        head_pieces = ['3dDeconvolve','-GOFORIT',str(goforit),'-float','-jobs',
                       str(jobs),'-input',self.functional+'_warp+tlrc',
                       '-nfirst','0','-num_stimts',str(num_stimts),'-polort',
                       str(polort),'\\']
        cmd.append(' '.join(head_pieces))
        
        
        motionlabels = ['roll','pitch','yaw','SI','LR','PA']
        subcmd = []
        
        for i in range(1,7,1):
            section = ['-stim_file',str(i),'\''+motionfile+'['+str(i)+']\'',
                       '-stim_label',str(i),motionlabels[i-1],'\\']
            subcmd.append(' '.join(section))
            
        for i in range(7,7+len(other_regressors),1):
            section = ['-stim_file',str(i),'\''+other_regressors[i-7]+'c.1D\'',
                       '-stim_label',str(i),'\''+other_regressors[i-7]+'\'','\\']
            subcmd.append(' '.join(section))
            
        regressors = [vector_name+'c.1D', '_'.join([self.mask_name,self.suffix,'ts.1D']),
                      '_'.join([self.mask_name, self.suffix, 'inter','ts.1D'])]
        reglabels = [vector_name,'_'.join([self.mask_name,self.suffix,'ts']),
                     '_'.join([self.mask_name,self.suffix,'PPI'])]
        
        for j,i in enumerate(range(7+len(other_regressors),7+len(other_regressors)+3,1)):
            section = ['-stim_file',str(i),'\''+regressors[j]+'\'',
                       '-stim_label',str(i), '\''+reglabels[j]+'\'','\\']
            subcmd.append(' '.join(section))
            
        footer = ['-nobout','-tout','-rout','-bucket','_'.join(['PPI',self.mask_name,self.suffix])]
        subcmd.append(' '.join(footer))
        
        cmd.append(subcmd)
        return cmd
    
    
    def zscore(self):
        
        cmd = ['# convert to z-scores:']
        pieces = ['3dmerge','-doall','-1zscore','-prefix','_'.join(['zPPI',self.mask_name,self.suffix]),
                  '_'.join(['PPI',self.mask_name,self.suffix])+'+tlrc']
        cmd.append(' '.join(pieces))
        return cmd
    
    
    def determine_R(self, betaind, R2ind):
        
        cmd = ['# determine R from R^2:']
        
        pieces = ['3dcalc','-a','_'.join(['PPI',self.mask_name,self.suffix])+'+tlrc\'['+str(R2ind)+']\'',
                  '-b','_'.join(['PPI',self.mask_name,self.suffix])+'+tlrc\'['+str(betaind)+']\'',
                  '-expr','\'ispositive(b)*sqrt(a)-isnegative(b)*sqrt(a)\'',
                  '-prefix','_'.join(['PPI',self.mask_name,self.suffix,'R'])]
        
        cmd.append(' '.join(pieces))
        return cmd
    
    
    def zscore_manual(self):
        
        cmd = ['# convert to z-scores:']
        
        pieces = ['3dcalc','-a','_'.join(['PPI',self.mask_name,self.suffix,'R'])+'+tlrc',
                  '-expr','\'log((1+a)/(1-a))/2\'','-prefix','_'.join(['zPPI',self.mask_name,self.suffix,'R'])]
        cmd.append(' '.join(pieces))
        return cmd
    
    
    def recursive_flatten(self, container, tabs):
        out = ''
        for item in container:
            if type(item) in (list, tuple):
                out = out+self.recursive_flatten(item, tabs+1)
            else:
                out = out+('\t'*tabs)+item+'\n'
        return out
    
    
    def write_out(self, master, filename):
        fid = open(filename,'w')
        script = self.recursive_flatten(master, 0)
        fid.write(script)
        fid.close()
        os.system('chmod 775 '+filename)
    
    
    def find_beta_r2(self, other_regressors):
        rlen = len(other_regressors)
        startind = 20
        betaind = startind + 3*rlen + 6
        R2ind = betaind+2
        return betaind, R2ind
    
    
    def writePPI(self, subjects, anatomical, functional, mask_name, dxyz, total_trs,
                 motionfile, contrast_vector, suffix='b',mrange=[1,2], other_regressors=[],
                 script_name='PPI_auto', mask_dir='scripts'):
        
        self.anatomical = anatomical
        self.functional = functional
        self.mask_name = mask_name
        self.dxyz = dxyz
        self.suffix = suffix
        self.mrange = mrange
        self.script_name = script_name
        self.mask_dir = mask_dir
        
        internal = []
        
        betaind, R2ind = self.find_beta_r2(other_regressors)
        
        internal.extend(self.clean(self.functional+'_warp+tlrc.BRIK',
                                   self.functional+'_warp')+['\n'])
        internal.extend(self.warp_functional()+['\n'])
        internal.extend(self.clean(mask_name+'r+tlrc.BRIK',mask_name+'r+tlrc')+['\n'])
        internal.extend(self.clean(mask_name+'r+orig.BRIK',mask_name+'r+orig')+['\n'])
        internal.extend(self.resample()+['\n'])
        internal.extend(self.maskave()+['\n'])
        internal.extend(self.detrend()+['\n'])
        internal.extend(self.transpose()+['\n'])
        internal.extend(self.generate_hrf()+['\n'])
        internal.extend(self.tfitter()+['\n'])
        internal.extend(self.create_interaction(contrast_vector)+['\n'])
        internal.extend(self.convolve_interaction(total_trs)+['\n'])
        internal.extend(self.clean('_'.join(['PPI',mask_name,suffix])+'+tlrc.BRIK',
                                   '_'.join(['PPI',mask_name,suffix]))+['\n'])
        internal.extend(self.deconvolve(motionfile, contrast_vector, other_regressors)+['\n'])
        internal.extend(self.clean('_'.join(['PPI',mask_name,suffix,'R'])+'+tlrc.BRIK',
                                   '_'.join(['PPI',mask_name,suffix,'R']))+['\n'])
        internal.extend(self.determine_R(betaind, R2ind)+['\n'])
        internal.extend(self.clean('_'.join(['zPPI',mask_name,suffix,'R'])+'+tlrc.BRIK',
                                   '_'.join(['zPPI',mask_name,suffix,'R']))+['\n'])
        internal.extend(self.zscore_manual()+['\n'])
            
        subj_loop = self.subject_loop(subjects, internal)
        master = self.header()+subj_loop
        
        self.write_out(master, self.script_name)
        



def write_template_script():
    
    fid = open('PPI_scriptwriter.py','w')
    lines = ['\n\n\n\n']
    lines.append('import os, glob, sys, subprocess')
    lines.append('from kktools.afni.PPIWriter import PPIWriter')
    lines.append('\n')
    lines.append('if __name__ == \"__main__\":\n')
    lines.append('\tsubjects = #[\'xx12\',\'yy12\']')
    lines.append('\tanatomical_name = #\'anat\'')
    lines.append('\tfunctional_name = #\'dataset\'')
    lines.append('\tmask_name = #\'nacc8mm\'')
    lines.append('\tmask_directory = #\'scripts\'')
    lines.append('\tregressor_of_interest_name = #\'anticipation_period\'')
    lines.append('\tfunctional_dxyz = #3.75')
    lines.append('\tfunctional_trs = #400')
    lines.append('\tmask_suffix = #\'b\'')
    lines.append('\tmask_mrange = #[1,2]')
    lines.append('\tscript_output_name = #\'PPI_auto_script\'')
    lines.append('\textra_regressors_names = #[\'csf\',\'wm\']')
    lines.append('\tmotionfile = #\'3dmotion.1D\'\n\n')
    lines.append('\tppi = PPIWriter()')
    lines.append('\tppi.writePPI(subjects, anatomical_name, functional_name, mask_name, functional_dxyz, functional_trs, motionfile, regressor_of_interest_name, suffix=mask_suffix, mrange=mask_mrange, other_regressors=extra_regressors_names, script_name=script_output_name, mask_dir=mask_directory)')
    lines.append('\n\n\n\n')
    
    for line in lines:
        fid.write(line+'\n')
        
    fid.close()
    
    



if __name__ == '__main__':
    
    prompt = raw_input('Write a template PPIwriter script to current directory?  ')
    if prompt.lower().startswith('y'):
        write_template_script()
    
    
    
        
        
        
        
        
        
        
        
    
        