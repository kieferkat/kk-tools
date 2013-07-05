

import os
from sharpblur import *


if __name__ == "__main__":

	sharpblur = SharpBlur()

	for s in [6]:
		os.chdir('subj'+str(s))
		for r in range(1,13):
			sharpblur.run('prun'+str(r)+'.nii','mask_brain.nii','rr_byvox_pos_prun'+str(r)+'.nii', zscore_in=False)
		os.chdir('..')



