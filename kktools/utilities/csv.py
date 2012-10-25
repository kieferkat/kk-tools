
import glob
import os
import math
import sys
import pylab
import numpy as np
import scipy.stats as stats
import sys
import pylab

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
        
    
    def import_as_recarray(self, csv):
        return pylab.csv2rec(csv)
        
    
    def read(self, csv, delimiter=',', newline='\n'):
        fid = open(csv,'r')
        lines = fid.readlines()
        fid.close()
        lines = [l.strip(newline).split(delimiter) for l in lines]
        return lines
    
    
    def write(self, lines, filename, delimiter=',', newline='\n'):
        fid = open(filename,'w')
        for line in lines:
            fid.write(delimiter.join(line)+newline)
        fid.close()
        
        
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
        
    @numpy_decorator
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
    
    
    
    

    
    
    
