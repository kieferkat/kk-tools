import os
import glob
import subprocess
import cmd



class ScriptWriter(object):

	def __init__(self):
		super(ScriptWriter, self).__init__()
		self.script_components = []
		self.breaker = '#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#'
		self.break_in = '#~~~~~~~~~~\t'
		self.break_out = '\t~~~~~~~~~~#'

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



