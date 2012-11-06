
import os, sys
import glob
import subprocess
import shutil
from ..base.process import Process





class AfniFunction(object):
    
    
    def __init__(self):
        super(AfniFunction, self).__init__()
        
    
    def _clean_remove(self, files):
        for file in files:
            try:
                os.remove(file)
            except:
                pass
            
    
    def _clean(self, glob_prefix, clean_clean_type='standard', path=None):
        if path:
            prefix = os.path.join(path, glob_prefix)
        else:
            prefix = glob_prefix
            
        if clean_type is 'standard':
            self._clean_remove(glob.glob(prefix))
            
        elif clean_type is 'orig':
            self._clean_remove(glob.glob(prefix+'+orig*'))

        elif clean_type is 'tlrc':
            self._clean_remove(glob.glob(prefix+'+tlrc*'))
            
        elif clean_type is 'afni':
            self._clean_remove(glob.glob(prefix+'+orig*'))
            self._clean_remove(glob.glob(prefix+'+tlrc*'))
    
    
class Copy3d(AfniFunction):
    
    def __init__(self):
        super(Copy3d, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix):
        
        self._clean(output_path_prefix, clean_type='orig')
        cmd = ['3dcopy', input_path, output_path_prefix]
        subprocess.call(cmd)
        
        
        
class TcatBuffer(AfniFunction):
    
    def __init__(self):
        super(TcatBuffer, self).__init__()
        
        
    def __call__(self, functional_path, output_path_prefix,
            abs_leadin, abs_leadout):
        
        self._clean(output_path_prefix, clean_type='orig')
        cut_dset = functional_path+'['+str(abs_leadin)+'..'+str(abs_leadout)+']'
        cmd = ['3dTcat', '-prefix', output_path_prefix, cut_dset]
        subprocess.call(cmd)
        
        
        
class Refit(AfniFunction):
    
    def __init__(self):
        super(Refit, self).__init__()
        
        
    def __call__(self, input_path, tr_length):
        
        cmd = ['3dRefit', '-TR', str(tr_length), input_path]
        subprocess.call(cmd)
        
        
        
class Tshift(AfniFunction):
    
    def __init__(self):
        super(Tshift, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix, tshift_slice, tpattern):
        
        self._clean(output_path_prefix, clean_type='orig')
        cmd = ['3dTshift', '-slice', str(tshift_slice), '-tpattern',
               tpattern, '-prefix', output_path_prefix, input_path]
        subprocess.call(cmd)
        
        
        
class TcatDatasets(AfniFunction):
    
    def __init__(self):
        super(TcatDatasets, self).__init__()
        
        
    def __call__(self, input_paths, output_path_prefix, cleanup=True):
        
        self._clean(output_path_prefix, type='orig')
        cmd = ['3dTcat', '-prefix', output_path_prefix].extend(input_paths)
        subprocess.call(cmd)
        
        if cleanup:
            for ip in input_paths:
                self._clean(ip, clean_type='orig')
                

class Volreg(AfniFunction):
    
    def __init__(self):
        super(Volreg, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix, motionfile, volreg_base):
        
        self._clean(output_path_prefix, clean_type='orig')
        self._clean(motionfile)
        cmd = ['3dvolreg','-Fourier','-twopass','-prefix', output_path_prefix,
               '-base', str(volreg_base), '-dfile', motionfile, input_path]
        subprocess.call(cmd)
        
        

class Smooth(AfniFunction):
    
    def __init__(self):
        super(Smooth, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix, blur_kernel):
        
        self._clean(ouput_path_prefix, clean_type='orig')
        cmd = ['3dmerge', '-prefix', output_path_prefix, '-1blur_fwhm',
               str(blur_kernel), '-doall', input_path]
        subprocess.call(cmd)
        
        
        

class NormalizePSC(AfniFunction):
    
    def __init__(self):
        super(NormalizePSC, self).__init__()
        
        
    def __call__(self, input_path, ave_path_prefix, output_path_prefix, trs,
            expression='((a-b)/b)*100', cleanup=True):
        
        self._clean(output_path_prefix, clean_type='orig')
        self._clean(ave_path_prefix, clean_type='orig')
        
        ave_cmd = ['3dTstat', '-prefix', ave_path_prefix,
                   input_path+'[0..'+str(trs)+']']
        refit_cmd = ['3drefit', '-abuc', ave_path_prefix+'+orig']
        calc_cmd = ['3dcalc', '-datum', 'float', '-a',
                    input+path+'[0..'+str(trs)+']', '-b',
                    ave_path_prefix+'+orig', expression, '-prefix',
                    output_path_prefix]
        
        subprocess.call(ave_cmd)
        subprocess.call(refit_cmd)
        subprocess.call(calc_cmd)
        
        if cleanup:
            self._clean(ave_path_prefix, clean_type='orig')
            
            
            
            
class HighpassFilter(AfniFunction):
    
    def __init__(self):
        super(HighpassFilter, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix, highpass_value):
        
        self._clean(output_path_prefix, clean_type='orig')
        cmd = ['3dFourier', '-prefix', output_path_prefix, '-highpass',
               str(highpass_value), input_path]
        subprocess.call(cmd)
        
        
        
class TalairachWarp(AfniFunction):
    
    def __init__(self):
        super(TalairachWarp, self).__init__()
        
        
    def __call__(self, functional_path, output_path_prefix, template_path):
        
        self._clean(output_path_prefix, clean_type='tlrc')
        cmd = ['@auto_tlrc', '-warp_orig_vol', '-suffix', 'NONE',
               '-base', template_path, '-input', output_path_prefix]
        refit_cmd = ['3drefit', '-apar', output_path_prefix+'+tlrc',
                     functional_path]
        
        subprocess.call(cmd)
        subprocess.call(refit_cmd)

    
    
    
class MaskAve(AfniFunction):
    
    def __init__(self):
        super(MaskAve, self).__init__()
        
        
    def __call__(self, mask_path, dataset_path, mask_area_strs=['l','r','b'],
            mask_area_codes=[[1,1],[2,2],[1,2]], tmp_tc_dir='raw_tc'):
        
        subject_path = os.path.split(dataset_path)[0]
        subject_name = os.path.split(subject_path)[1]
        
        mask_name = os.path.split(mask_path)[1].split('+')[0]
        
        print 'maskave', subject_name, mask_name
        
        tmpdir = os.path.join(subject_path, tmp_tc_dir)
        try:
            os.makedirs(tmpdir)
        except:
            pass
        
        for area, codes in zip(mask_area_strs, mask_area_codes):
            
            raw_tc = '_'.join([subject_name, area, mask_name, 'raw.tc'])
            raw_tc = os.path.join(tmpdir, raw_tc)
            
            self._clean(raw_tc)
            
            cmd = ['3dmaskave', '-mask', mask_path, '-quiet', '-mrange',
                   str(codes[0]), str(codes[1]), dataset_path]
            
            fcontent = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            fcontent.wait()
            fcontent = fcontent.communicate()[0]
            
            fid = open(raw_tc, 'w')
            fid.write(fcontent)
            fid.close()
            
    
    
    
class FractionizeMask(AfniFunction):
    
    def __init__(self):
        super(FractionizeMask, self).__init__()
        
        
    def __call__(self, mask_path, dataset_path, anat_path, subject_mask_suffix='r'):
        
        subject_path = os.path.split(dataset_path)[0]
        mask_name = (os.path.split(mask_path)[1]).split('+')[0]
        
        subject_mask = os.path.join(subject_path, mask_name+subject_mask_suffix+'+orig')
        self._clean(subject_mask+'*')
        
        cmd = ['3dfractionize', '-template', dataset_path, '-input', mask_path,
           '-warp', anat_path, '-clip', '0.1', '-preserve', '-prefix',
           subject_mask]
        
        subprocess.call(cmd)
        return subject_mask
        
        
        
class MaskDump(AfniFunction):
    
    def __init__(self):
        super(MaskDump, self).__init__()
        self.fractionize = FractionizeMask()
        self.maskave = MaskAve()
        
        
    def run(self, dataset_path, anat_path, mask_paths, mask_area_strs=['l','r','b'],
            mask_area_codes=[[1,1],[2,2],[1,2]]):
        
        subject_path = os.path.split(dataset_path)[0]
        
        if clean_type(mask_paths) in (list, tuple):
            for mask in mask_paths:
                subject_mask = self.fractionize.run(mask, dataset_path, anat_path)
                self.maskave.run(subject_mask, dataset_path, mask_area_strs=mask_area_strs,
                                 mask_area_codes=mask_area_codes)
                
        else:
            subject_mask = self.fractionize.run(mask_paths, dataset_path, anat_path)
            self.maskave.run(subject_mask, dataset_path, mask_area_strs=mask_area_strs,
                             mask_area_codes=mask_area_codes)
        
        
    def run_over_subjects(self, subject_dirs=None, functional_name=None, anatomical_name=None,
                          mask_names=None, mask_dir=None, mask_area_strs=['l','r','b'],
                          mask_area_codes=[[1,1],[2,2],[1,2]]):
        
        
        for i, mask in enumerate(self.mask_names):
            if not mask.endswith('+tlrc'):
                self.mask_names[i] = mask+'+tlrc'
                
        mask_paths = [os.path.join(self.mask_dir, name) for name in self.mask_names]

        for dir in self.subject_dirs:
            dataset_path = os.path.join(dir, self.functional_name)
            anat_path = os.path.join(dir, self.anatomical_name)
            
            self.run(dataset_path, anat_path, mask_paths, mask_area_strs=mask_area_strs,
                     mask_area_codes=mask_area_codes)
        
        
    
class AfniWrapper(Process):
    
    def __init__(self, variable_dict=None):
        super(AfniWrapper, self).__init__()
        self.maskave = MaskAve()
        self.fractionize = FractionizeMask()
        self.maskdump = MaskDump()
        self.talairachwarp = TalairachWarp()
        self.highpassfilter = HighpassFilter()
        self.normalize = NormalizePSC()
        self.smooth = Smooth()
        self.volreg = Volreg()
        self.tcatdsets = TcatDatasets()
        self.tcatbuffer = TcatBuffer()
        self.tshift = Tshift()
        self.refit = Refit()
        self.copy3d = Copy3d()
        
        
        
    