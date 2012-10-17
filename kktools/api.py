

from data.crossvalidation import Crossvalidation
from data.datamanager import DataManager
from data.nifti import NiftiTools
from base.variables import Variables

from stats.linearsvm import ScikitsSVM

from utilities.cleaners import glob_remove
import utilities.parsers as parsers
import utilities.vector as vector

#import stats.regression as regression