import os
import glob

#---------------------------------------------------------------------------------#
#		MANDATORY USER VARIABLES:
#---------------------------------------------------------------------------------#

SCRIPT_FILENAME = 'regression'
SUBJECTS = ['sub1', 'sub2', 'sub3']
INPUT_DATASET = 'mydataset+orig'
MOTIONFILE = '3dmotion.1D'
REGRESSION_NAME = 'myregression'
MODELFILE = 'actreg_streak.txt'
NONWAVER_VECTORS = ['whitematter.1D','csf.1D']


#---------------------------------------------------------------------------------#
#		OPTIONAL USER VARIABLES:
#---------------------------------------------------------------------------------#

SCRIPT_TITLE = 'Preprocessing Script'
TR_LENGTH = 2.0
DECONVOLVE_JOBS = 8
DECONVOLVE_NFIRST = 0
DECONVOLVE_POLORT = 2
MOTION_LABELS = ['roll','pitch','yaw','dS','dL','dP']
MAKEVEC_PATH = '/usr/local/bin/makeVec.py'





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




class RegressionWriter(ScriptWriter):

	def __init__(self):
		super(RegressionWriter, self).__init__()
		self.title = SCRIPT_TITLE
		self.subjects = SUBJECTS
		self.input_dataset = INPUT_DATASET
		self.modelfile = MODELFILE
		self.modelfile_path = os.path.join(os.getcwd(),self.modelfile)
		self.extra_vectors = NONWAVER_VECTORS
		self.motionfile = MOTIONFILE
		self.regressor_files = []
		self.regressor_labels = []
		self.makevec_path = MAKEVEC_PATH
		self.tr_length = str(TR_LENGTH)
		self.current_vector = '${current_vector}'
		self.deconvolve_jobs = str(DECONVOLVE_JOBS)
		self.deconvolve_nfirst = str(DECONVOLVE_NFIRST)
		self.polort = str(DECONVOLVE_POLORT)
		self.motion_labels = MOTION_LABELS
		self.regression_name = REGRESSION_NAME



	def _read_modelfile(self):
		fid = open(self.modelfile,'r')
		li = fid.readlines()
		if len(li) == 1:
			li = li[0].split('\r')
		li = [x.strip('\n') for x in li if x.startswith('OUTPUT:')]
		li = [x.split(' ')[1].strip('\"')[:-3] for x in li]
		self.regressor_labels = li
		self.regressor_files = [x+'c.1D' for x in li]
		return li

	def _add_extra_vectors(self):
		for ev in self.extra_vectors:
			self.regressor_labels.append(ev.rstrip('.1D'))
			self.regressor_files.append(ev)


	@property 
	def afni_type(self):
		return '+'+self.input_dataset.split('+')[1]

	@property 
	def num_stimts(self):
		return str(6+len(self.regressor_files))

	@property 
	def makevec_command(self):
		return ' '.join([self.makevec_path, self.modelfile_path])

	@property
	def waver_command(self):
		return ' '.join(['waver','-dt',self.tr_length,'-input','\''+self.current_vector+'.1D\'',
			'>',self.current_vector+'c.1D'])

	@property 
	def deconvolve_command(self):
		cmd = [' '.join(['3dDeconvolve','-jobs',self.deconvolve_jobs,'-overwrite','-float',
			'-input',self.input_dataset,'-nfirst',self.deconvolve_nfirst,'-num_stimts',
			self.num_stimts,'-polort',self.polort,'\\'])]
		for i in range(6):
			cmd.append([' '.join(['-stim_file',str(i+1),'\''+self.motionfile+'['+str(i+1)+']\'',
				'-stim_label',str(i+1),'\''+self.motion_labels[i]+'\'','\\'])])
		for i in range(len(self.regressor_files)):
			cmd.append([' '.join(['-stim_file',str(i+7),'\''+self.regressor_files[i]+'\'',
				'-stim_label',str(i+7),'\''+self.regressor_labels[i]+'\'','\\'])])
		cmd.append([' '.join(['-tout','-fout','-bucket',self.regression_name])])
		return cmd


	@property 
	def zscore_command(self):
		return ' '.join(['3dmerge','-overwrite','-doall','-1zscore','-prefix','z'+self.regression_name,
			self.regression_name+self.afni_type])

	@property 
	def rm_command(self):
		return ' '.join(['rm','-rf', self.regression_name+self.afni_type+'*'])


	def assemble(self):
		self.commands = []

		self._read_modelfile()
		
		self.commands.extend(self.document('run makeVec.py on vector model file:'))
		self.commands.append(self.makevec_command)

		self.commands.extend(self.document('run waver on vectors:'))
		for model_label in self.regressor_labels:
			self.current_vector = model_label
			self.commands.append(self.waver_command)

		self._add_extra_vectors()

		self.commands.extend(self.document('run 3dDeconvolve:'))
		self.commands.extend(self.deconvolve_command)

		self.commands.extend(self.document('zscore with 3dmerge and clean up:'))
		self.commands.append(self.zscore_command)
		self.commands.append(self.rm_command)

		self.script = self.header(self.title)
		self.script.extend(self.loop_over_subjects(self.subjects, self.commands))


if __name__ == "__main__":

	RW = RegressionWriter()
	RW.assemble()
	RW.write_to_file(RW.script, SCRIPT_FILENAME)



		















