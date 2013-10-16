import os
import glob

#---------------------------------------------------------------------------------#
#		MANDATORY USER VARIABLES:
#---------------------------------------------------------------------------------#

SCRIPT_FILENAME = 'ttestplus'
REGRESSION_FILE = 'regression'
SUBJECTS_A = ['sub1','sub2','sub3']
SUBJECTS_B = []
MODELFILE = 'actreg_streak.txt'
ANATOMICAL_PARENT = 'anat'
EXTRA_VECTOR_LABELS = ['whitematter','csf']
RESAMPLE_DIMENSION = 2.9
OUTPUT_DIR = '../ttestplus/myttest'


#---------------------------------------------------------------------------------#
#		OPTIONAL USER VARIABLES:
#---------------------------------------------------------------------------------#

SCRIPT_TITLE = 'TtestPlus Script'
SCRIPTS_DIR = '.'


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


	def loop_over_variable(self, variable_name, variables, components):

		variable_loop_begin = 'foreach '+variable_name+' ( '+' '.join(variables)+' )'
		variable_loop_end = 'end'
		return ['', variable_loop_begin, '', components, '', variable_loop_end]


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




class TtestWriter(ScriptWriter):

	def __init__(self):
		super(TtestWriter, self).__init__()
		self.title = SCRIPT_TITLE
		self.scripts_dir = SCRIPTS_DIR
		self.script_filename = SCRIPT_FILENAME
		self.regression_file = REGRESSION_FILE
		self.subjectsA = SUBJECTS_A
		self.subjectsB = SUBJECTS_B
		self.modelfile = MODELFILE
		self.anat_parent = ANATOMICAL_PARENT
		self.extra_vector_labels = EXTRA_VECTOR_LABELS
		self.resampdim = str(RESAMPLE_DIMENSION)
		self.output_dir = OUTPUT_DIR


	def _read_modelfile(self):
		fid = open(self.modelfile,'r')
		li = fid.readlines()
		if len(li) == 1:
			li = li[0].split('\r')
		li = [x.strip('\n') for x in li if x.startswith('OUTPUT:')]
		li = [x.split(' ')[1].strip('\"')[:-3] for x in li]
		self.regressor_labels = li
		return li

	def _add_extra_vectors(self):
		for ev in self.extra_vector_labels:
			self.regressor_labels.append(ev)


	@property 
	def stats(self):
		return [str(19+(i*3)) for i in range(len(self.regressor_labels))]

	@property 
	def AminusB(self):
		if any(self.subjectsB):
			return '-AminusB'
		else:
			return ''

	@property 
	def set_stats(self):
		return ' '.join(['set','stats','=','( ']+self.stats+[' )'])

	@property 
	def adwarp_command(self):
		return ' '.join(['adwarp','-apar',self.anat_parent+'+tlrc.','-dpar',self.regression_file+'+orig.',
			'-dxyz',self.resampdim,'-prefix',self.regression_file])

	@property 
	def cd_scripts_command(self):
		return ' '.join(['cd',os.path.abspath(self.scripts_dir)])

	@property 
	def count_command(self):
		return ' '.join(['@','count','=','1'])

	@property 
	def mkdir_command(self):
		return ' '.join(['mkdir',os.path.abspath(self.output_dir)])

	@property 
	def setstat_command(self):
		return ' '.join(['set','stat','=','$stats[${count}]'])

	@property 
	def increment_count_command(self):
		return ' '.join(['@','count','=','$count','+','1'])


	@property
	def ttestplus_command(self):
		cmd = ['\n']+[' '.join(['3dttest++','-overwrite','-prefix','./${outfile}z','-toz',self.AminusB,'\\'])]
		subA = ['-setA']
		for subject in self.subjectsA:
			subA.append('\"../'+subject+'/'+self.regression_file+'+tlrc[${stat}]\" \\')
		cmd.append(subA)
		if any(self.subjectsB):
			subB = ['-setB']
			for subject in self.subjectsB:
				subB.append('\"../'+subject+'/'+self.regression_file+'+tlrc[${stat}]\" \\')
			cmd.append(subB)
		return cmd+['\n']

	@property 
	def mv_command(self):
		return ' '.join(['mv', '${outfile}z*',os.path.abspath(self.output_dir)])


	def assemble(self):
		self.commands = []

		self._read_modelfile()
		self._add_extra_vectors()

		self.commands.extend(self.document('adwarp regresions to parent dataset:'))
		adwarp_commands = [self.adwarp_command]
		self.commands.extend(self.loop_over_subjects(self.subjectsA+self.subjectsB, adwarp_commands))

		self.commands.extend(self.document('prepare files and variables for the ttest:'))
		self.commands.append(self.cd_scripts_command)
		self.commands.append(self.count_command)
		self.commands.append(self.mkdir_command)
		self.commands.extend(self.document('set the stats variable to iterate alongside outfiles:'))
		self.commands.append(self.set_stats)

		ttest_commands = []
		ttest_commands.append(self.setstat_command)
		ttest_commands.append(self.increment_count_command)
		ttest_commands.extend(self.ttestplus_command)
		ttest_commands.append(self.mv_command)
		self.commands.extend(self.document('preform the ttests:'))
		self.commands.extend(self.loop_over_variable('outfile',self.regressor_labels, ttest_commands))

		self.script = self.header(self.title)
		self.script.extend(self.commands)


if __name__ == "__main__":

	TW = TtestWriter()
	TW.assemble()
	TW.write_to_file(TW.script, SCRIPT_FILENAME)














