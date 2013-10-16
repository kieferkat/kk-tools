import os
import glob
import subprocess

#---------------------------------------------------------------------------------#
#		MANDATORY USER VARIABLES:
#---------------------------------------------------------------------------------#

SCRIPT_FILENAME = 'preprocessor2'
SUBJECTS = ['sub1', 'sub2', 'sub3','alex','kiefer']
DATASET_NAME = 'hersheldata'
RAW_FUNCTIONALS = ['rawfunctional.nii.gz', 'rawfunctional2.nii.gz', 'raw_functional3.nii.gz']
RAW_ANATOMICAL = 'rawanatomical.nii.gz'
LEADINS = [6,6,6]
LEADOUTS = [400,410,370]


#---------------------------------------------------------------------------------#
#		OPTIONAL USER VARIABLES:
#---------------------------------------------------------------------------------#

SCRIPT_TITLE = 'Preprocessing Script'
ANATOMICAL_PREFIX = 'anat'
TEMPLATE_BRAIN_PATH = '/Users/span/abin/TT_N27+tlrc.'
VOLREG_BASE = 3
TSHIFT_SLICE = 0
TSHIFT_PATTERN = 'altplus'
MOTIONFILE = '3dmotion.1D'
BLUR_FWHM = 4
NORMALIZE_EXPRESSION = '\"((a-b)/b)*100\"'
FOURIER_HIGHPASS_VALUE = 0.011
AVERAGE_DATASET_PREFIX = 'average'



#---------------------------------------------------------------------------------#
#		DO NOT CHANGE:
#---------------------------------------------------------------------------------#


class ScriptWriter(object):

	def __init__(self):
		super(ScriptWriter, self).__init__()
		self.script_components = []
		self.breaker = '#----------------------------------------------------------------------#'
		self.break_in = '#------------\t'
		self.break_out = '\t------------#'

	def header(self, title):

		header = ['#! /bin/csh',
		'',
		self.breaker,
		'#\t\t'+title,
		'#\t\tAuto-written by ScriptWriter',
		self.breaker,
		'']

		return header


	def document(self, documentation):
		return ['\n', self.break_in, '# '+documentation, '#', '']


	def loop_over_subjects(self, subjects, components):

		subject_loop_begin = 'foreach subject ( '+' '.join(subjects)+' )'
		cd_to_subject = 'cd ../${subject}*'
		subject_loop_end = 'end'
		return [subject_loop_begin, [cd_to_subject], components, '', subject_loop_end]


	def flatten_script(self, pieces, indent=0):
		output = ''
		for piece in pieces:
			if type(piece) in [type([]), type(())]:
				output += self.flatten_script(piece, indent=indent+1)
			else:
				output += '\t'*indent + piece + '\n'
		return output


	def write_to_file(self, script_pieces, filename):
		fid = open(filename,'w')
		flattened_script = self.flatten_script(script_pieces)
		fid.write(flattened_script)
		fid.close()



class PreprocessWriter(ScriptWriter):

	def __init__(self):
		super(PreprocessWriter, self).__init__()
		self.title = SCRIPT_TITLE
		self.subjects = SUBJECTS
		self.dataset_name = DATASET_NAME
		self.current_dataset_prefix = '${current_dataset}'
		self.next_dataset_prefix = '${next_dataset_prefix}'
		self.average_dataset_prefix = AVERAGE_DATASET_PREFIX
		self.raw_anatomical_filename = RAW_ANATOMICAL
		self.raw_functional_filenames = RAW_FUNCTIONALS
		self.raw_functional_filename = '${raw_functional}'
		self.anatomical_prefix = ANATOMICAL_PREFIX
		self.functional_leadins = [str(x) for x in LEADINS]
		self.functional_leadouts = [str(x) for x in LEADOUTS]
		self.functional_leadin = '${leadin}'
		self.functional_leadout = '${leadout}'
		self.template_ttn27 = TEMPLATE_BRAIN_PATH
		self.volreg_base = str(VOLREG_BASE)
		self.tshift_slice = str(TSHIFT_SLICE)
		self.tshift_tpattern = TSHIFT_PATTERN
		self.motionfile_filename = MOTIONFILE
		self.blur_fwhm = str(BLUR_FWHM)
		self.normalize_expression = NORMALIZE_EXPRESSION
		self.fourier_highpass_value = str(FOURIER_HIGHPASS_VALUE)


	@property
	def copy_anatomical_command(self):
		return ' '.join(['3dcopy', '-overwrite', self.raw_anatomical_filename,
			self.anatomical_prefix])

	@property
	def warp_anatomical_command(self):
		return ' '.join(['@auto_tlrc', '-warp_orig_vol', '-suffix', 'NONE',
			'-base', self.template_ttn27, '-input', self.anatomical_prefix+'+orig'])

	@property
	def tcat_buffers_command(self):
		return ' '.join(['3dTcat', '-overwrite', '-prefix', self.next_dataset_prefix, 
			'\''+self.raw_functional_filename+'['+self.functional_leadin+'..'+self.functional_leadout+']\''])

	@property 
	def tshift_command(self):
		return ' '.join(['3dTshift', '-overwrite', '-slice', self.tshift_slice,
			'-tpattern', self.tshift_tpattern, '-prefix', self.next_dataset_prefix,
			self.current_dataset_prefix+'+orig'])

	@property 
	def tcat_datasets_command(self):
		return ' '.join(['3dTcat','-overwrite','-prefix',self.next_dataset_prefix]+['epi'+str(i)+'_ts+orig' for i in range(len(self.raw_functional_filenames))])

	@property 
	def clean_epi_command(self):
		return ' '.join(['rm','-rf','epi*orig*'])

	@property
	def volreg_command(self):
		return ' '.join(['3dvolreg','-Fourier','-twopass','-overwrite',
			'-prefix',self.next_dataset_prefix,'-base',self.volreg_base,
			'-dfile',self.motionfile_filename, self.current_dataset_prefix+'+orig'])

	@property
	def blur_command(self):
		return ' '.join(['3dmerge', '-overwrite', '-1blur_fwhm', self.blur_fwhm,
			'-doall', '-prefix', self.next_dataset_prefix, self.current_dataset_prefix+'+orig'])

	@property
	def average_command(self):
		return ' '.join(['3dTstat', '-overwrite', '-prefix', self.average_dataset_prefix, 
			self.current_dataset_prefix+'+orig'])

	@property 
	def normalize_command(self):
		return ' '.join(['3dcalc', '-overwrite', '-datum', 'float', '-a',
			self.current_dataset_prefix+'+orig', '-b', self.average_dataset_prefix+'+orig',
			'-expr', self.normalize_expression, '-prefix', self.next_dataset_prefix])

	@property 
	def clean_fourier_command(self):
		return ' '.join(['rm','-rf',self.next_dataset_prefix+'+orig*'])

	@property 
	def fourier_command(self):
		return ' '.join(['3dFourier', '-highpass', self.fourier_highpass_value, '-prefix',
			self.next_dataset_prefix, self.current_dataset_prefix+'+orig'])

	@property 
	def refit_command(self):
		return ' '.join(['3drefit', '-apar', self.anatomical_prefix+'+orig', 
			self.current_dataset_prefix+'+orig'])


	def update_datasets(self, suffix):
		self.current_dataset_prefix = self.next_dataset_prefix
		self.next_dataset_prefix = self.current_dataset_prefix+suffix



	def assemble(self):
		self.commands = []
		
		self.commands.extend(self.document('copy anatomical:'))
		self.commands.append(self.copy_anatomical_command)

		self.commands.extend(self.document('warp anatomical:'))
		self.commands.append(self.warp_anatomical_command)

		self.commands.extend(self.document('cut off leadin/leadout and tshift datasets:'))
		for i, (raw_func, leadin, leadout) in enumerate(zip(self.raw_functional_filenames, self.functional_leadins, self.functional_leadouts)):
			self.raw_functional_filename = raw_func
			self.functional_leadin = leadin
			self.functional_leadout = leadout
			self.next_dataset_prefix = 'epi'+str(i)
			self.commands.append(self.tcat_buffers_command)
			self.current_dataset_prefix = self.next_dataset_prefix
			self.next_dataset_prefix = 'epi'+str(i)+'_ts'
			self.commands.append(self.tshift_command)

		self.current_dataset_prefix = self.dataset_name
		self.next_dataset_prefix = self.current_dataset_prefix
		self.commands.extend(self.document('tcat functional datasets together:'))
		self.commands.append(self.tcat_datasets_command)

		self.commands.extend(self.document('cleanup epi datasets:'))
		self.commands.append(self.clean_epi_command)

		self.update_datasets('_m')
		self.commands.extend(self.document('motion correct:'))
		self.commands.append(self.volreg_command)

		self.update_datasets('b')
		self.commands.extend(self.document('gaussian blur:'))
		self.commands.append(self.blur_command)

		self.update_datasets('n')
		self.commands.extend(self.document('normalize to percent signal change:'))
		self.commands.append(self.average_command)
		self.commands.append(self.normalize_command)

		self.update_datasets('f')
		self.commands.extend(self.document('fourier highpass filter:'))
		self.commands.append(self.clean_fourier_command)
		self.commands.append(self.fourier_command)

		self.update_datasets('')
		self.commands.extend(self.document('refit functional to anatmomical:'))
		self.commands.append(self.refit_command)

		self.script = self.header(self.title)
		self.script.extend(self.loop_over_subjects(self.subjects, self.commands))



if __name__ == "__main__":


	PW = PreprocessWriter()
	PW.assemble()
	PW.write_to_file(PW.script, SCRIPT_FILENAME)



