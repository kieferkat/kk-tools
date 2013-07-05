
import glob
import os
import math
import sys
import pylab
import numpy as np
import scipy.stats as stats
import sys
import pylab
from vector import VectorTools
#from ..base.process import Process

# yeah, i do know there is a csv module. who cares?


def numpy_converter(func):
    def wrapper(self, matrix):
        matrix = np.array(matrix)
        return func(self, matrix)
    return wrapper


# class with collection of csv functions.

class CsvTools(object):
    
    def __init__(self):
        super(CsvTools, self).__init__()
        self.vector = VectorTools()
        
    
    def import_as_recarray(self, csv):
        return pylab.csv2rec(csv)
        
    
    def read(self, csv, delimiter=',', newline='\n'):
        fid = open(csv,'r')
        lines = fid.readlines()
        fid.close()
        if len(lines) == 1 and (lines[0].find('\r') != -1):
            lines = lines[0].split('\r')
        lines = [l.strip(newline).split(delimiter) for l in lines]
        return lines
    
    
    def write(self, lines, filename, delimiter=',', newline='\n'):
        fid = open(filename,'w')
        for line in lines:
            fid.write(delimiter.join([str(l) for l in line])+newline)
        fid.close()
        
    
    def load_csvs(self, subject_dirs, csv_name, verbose=True):
        csv_dict = {}
        for dir in subject_dirs:
            id = os.path.split(dir)[1]
            csv_path = os.path.join(dir, csv_name)
            
            try:
                if verbose:
                    print 'Attempting to load csv for subject:', id
                csv_lines = self.read(csv_path)
                csv_dict[id] = csv_lines
            except:
                if verbose:
                    print 'Error loading csv for subject:', id
        
        return csv_dict
    
    
    def split_header_data(self, csv):
        return csv[0], csv[1:]
    
    
    def csv_to_coldict(self, csv, verbose=True):
        coldict = {}
        header, data = self.split_header_data(csv)
        header = [x.lower() for x in header]
        for i, head in enumerate(header):
            if verbose:
                if head in coldict:
                    print 'Duplicate header, only last column used.', head
            coldict[head] = [row[i] for row in data]
        return coldict
    
    
    def subjectcsv_to_subjectdict(self, csv):

        header, data = self.split_header_data(csv)
        header = [x.lower() for x in header]
        subject_index = header.index('subject')
        
        subject_dict = {}
        for row in data:
            csub = row[subject_index]
            if csub not in subject_dict:
                subject_dict[csub] = {}
            for i, value in enumerate(row):
                if value is not csub:
                    col_title = header[i]
                    if col_title not in subject_dict[csub]:
                        subject_dict[csub][col_title] = []
                    subject_dict[csub][col_title].append(value)
                    
        return subject_dict
        
    
    def merge_csv_dicts(self, basedict, add_dicts, keylevel=0, verbose=True):
        
        if type(add_dicts) not in (list, tuple):
            add_dicts = [add_dicts]
        
        for adict in add_dicts:
            for akey, avals in adict.items():
                if keylevel == 0:
                    if akey in basedict:
                        if verbose:
                            print 'Duplicate key, replacing values.'
                    basedict[akey] = avals
                elif keylevel == 1:
                    for subkey, subvals in avals.items():
                        if not akey in basedict:
                            basedict[akey] = {subkey:subvals}
                        else:
                            if subkey in basedict[akey]:
                                if verbose:
                                    print 'Duplicate sub-key, replacing values.'
                            basedict[akey][subkey] = subvals
                            
        return basedict
        
        
    def _coldict_linehelper(self, coldict, header):
        lines = []
        for row in range(max([len(x) for x in coldict.values()])):
            line = []
            for headeritem in header:
                value_list = coldict[headeritem]
                if len(value_list) > row:
                    line.append(value_list[row])
                else:
                    line.append('')
            lines.append(line)
        return lines
    
        
    def coldict_tolines(self, coldict):
        lines = [coldict.keys()]
        datalines = self._coldict_linehelper(coldict, lines[0])
        lines.extend(datalines)
        return lines
    
    
    def subject_csvdicts_tolines(self, subjectdict):
        # all subjects must have same coldict keys for now
        lines = []
        for subject, coldict in subjectdict.items():
            print subject, coldict.keys()
            if lines == []:
                header = ['subject']+sorted(coldict.keys())
                lines.append(header)
                print header
            subjectlines = self._coldict_linehelper(coldict, lines[0][1:])
            lines.extend([[subject]+l for l in subjectlines])
            
        return lines
            
        
        
        
    def tofloat(self, matrix):
        return [[float(y) for y in x] for x in matrix]
        
        
    def tostring(self, matrix):
        return [[str(y) for y in x] for x in matrix]
        
    
    def transpose(self, matrix):
        return [[row[col] for row in matrix] for col in range(np.shape(matrix)[1])]
        
        
    @numpy_converter
    def mean_rows(self, X):
        return X.mean(axis=1)
        
    @numpy_converter
    def mean_cols(self, X):
        return X.mean(axis=0)
        
    @numpy_converter
    def variance_rows(self, X):
        return X.var(axis=1)
        
    @numpy_converter
    def variance_cols(self, X):
        return X.var(axis=0)
        
    @numpy_converter
    def stddev_rows(self, X):
        return X.std(axis=1)
        
    @numpy_converter
    def stddev_cols(self, X):
        return X.std(axis=0)
    
    @numpy_converter
    def stderr_rows(self, X):
        return stats.sem(X, axis=1)
        
    @numpy_converter
    def stderr_cols(self, X):
        return stats.sem(X, axis=0)
        
    
    # appends row and std error to a timecourses csv:
    def append_row_stderr(self, csvfile, delimiter=',', newline='\n'):
        l = read(csvfile, delimiter=delimiter, newline=newline)
        means = mean_cols(tofloat([x[1:] for x in l]))
        stderr = stderr_cols(tofloat([x[1:] for x in l]))
        l.append(['mean'].extend(tostring(means)))
        l.append(['stderr'].extend(tostring(stderr)))
        write(l, csvfile, delimiter=delimiter, newline=newline)
        
        
    def timecourse_to_datagraph(self, csv_dir, csv_prefix, outfile=None,
                                delimiter=',', newline='\n'):
        if not outfile:
            outfile = os.path.join(csv_dir,'dgraph_'+csv_prefix+'.csv')
        csvs = glob.glob(os.path.join(csv_dir, csv_prefix+'*'))
        names = [os.path.split(c)[1].split('.')[0] for c in csvs]
        matrices = [self.tofloat([x[1:] for x in self.read(csv)]) for csv in csvs]
        means = [self.mean_cols(x) for x in matrices]
        stderrs = [self.stderr_cols(x) for x in matrices]
        header, rows = [], []
        for i, name in enumerate(names):
            header.append(name+'_mean')
            header.append(name+'_stderr')
        rows.append(header)
        for row in range(len(means[0])):
            nrow = []
            for col in range(len(names)):
                nrow.append(means[col][row])
                nrow.append(stderrs[col][row])
            rows.append(nrow)
        self.write(rows, outfile)
    
    
    
    

    
    
    
