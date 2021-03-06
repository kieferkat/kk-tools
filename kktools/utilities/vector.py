
import os
import numpy as np
import glob as glob
import subprocess
from ..utilities.cleaners import glob_remove



class VectorTools(object):
    
    '''
    Class of functions that preforms common operations on vector files and
    vectors. "Vectors" in this case implies a 1D array of values. In file form
    these values are separated by newlines."
    '''
    
    def __init__(self):
        super(VectorTools, self).__init__()
        
        
    def read(self, file_path, usefloat=True):
        fid = open(file_path,'r')
        lines = fid.readlines()
        fid.close()
        if not usefloat:
            lines = [l.strip('\n') for l in lines]
        else:
            lines = [float(l.strip('\n')) for l in lines]
        return lines
    
    
    
    def write(self, vector, file_path):
        fid = open(file_path, 'w')
        [fid.write(str(x)+'\n') for x in vector]
        fid.close()
    
    


    def vectordir_todict(self, vector_dir, glob_prefix='*.tc', filename_split_keyinds=None,
                         filename_split='_', verbose=False):
        
        vector_dict = {}
        vectors = glob.glob(os.path.join(vector_dir, glob_prefix))
        for vector in vectors:
            
            vector_name = os.path.split(vector)[1].rstrip('.tc')
            
            if verbose:
                print 'Attempting to load vector:', vector_name
            vsplit = vector_name.split(filename_split)
            if filename_split_keyinds:
                id = ''.join([k for i,k in enumerate(vsplit) if i in filename_split_keyinds])
            else:
                id = vector_name
            id = id.lower()
            vec_data = self.read(vector)
            vector_dict[id] = vec_data
            
            #print vec_data[0:4]
    
                    
        return vector_dict


    def subject_vector_dict(self, subject_dirs, vector_dir, glob_prefix='*.tc',
                            filename_split_keyinds=[1,2], filename_split='_',
                            verbose=False):
        subject_dict = {}
        for dir in subject_dirs:
            id = os.path.split(dir)[1]
            vecdir = os.path.join(dir, vector_dir)
            
            if verbose:
                print 'Attempting to load vectors for subject:', id
            vector_dict = self.vectordir_todict(vecdir, glob_prefix=glob_prefix,
                                                filename_split_keyinds=filename_split_keyinds,
                                                filename_split=filename_split, verbose=verbose)
            subject_dict[id] = vector_dict
    
        
        return subject_dict
        
    


    def replace(self, vector, oldvalue, newvalue):
        return [x if not x == oldvalue else newvalue for x in vector]
    

    def zero_forward(self, vector, forward_amount):
        new_vec = []
        count = 0
        for val in vector:
            if count == 0:
                if float(val) != 0:
                    count = forward_amount
                    new_vec.append(val)
                else:
                    new_vec.append(val)
            else:
                new_vec.append('0')
                count -= count
        return new_vec
    
    
    
    def replace_at_interval(self, vector, newvalue, interval, start=0):
        rewrite_inds = [x-1 for x in range(start, len(vector)+1, interval)]
        return [newvalue if (i in rewrite_inds) else x for i,x in enumerate(vector)]
    
    
    # simply call stephanie's old makeVec.py for now.
    # eventually create a new makevec class because stephanie's is limited...
    def makevecs(self, dirs, vector_model_path,
                 makevec_path='/usr/local/bin/makeVec.py'):
        for dir in dirs:
            os.chdir(dir)
            subprocess.call([makevec_path], vector_model_path)
        
        
    

    def offset(self, vector, offset):
        if offset < 0:
            return [x for x in vector[abs(offset):]] + [0]*abs(offset)
        elif offset > 0:
            return [0]*offset + [x for x in vector[:-1*offset]]
        else:
            return vector



    def combine(self, vectors, operation_on_overlap='zero', usefloat=False,
                warning_on_overlap=True):
        
        if operation_on_overlap not in ['multiply','add','sum','subtract','zero']:
            print 'invalid operation specified'
            return None
        
        single_vector = []
        
        for i, item_set in enumerate(zip(*vectors)):
            
            if len(np.nonzero(item_set)[0]) == 0:
                if usefloat:
                    single_vector.append(0.)
                else:
                    single_vector.append(0)
                    
            elif len(np.nonzero(item_set)[0]) == 1:
                single_vector.append(item_set[np.nonzero(item_set)[0]])
                
            else:
                if warning_on_overlap:
                    print 'overlap at index: ', i
                    
                if operation_on_overlap == 'multiply':
                    val = reduce(lambda x, y: x*y, item_set)
                    single_vector.append(val)
                    
                elif operation_on_overlap == 'sum' or operation_on_overlap == 'add':
                    single_vector.append(sum(item_set))
                    
                elif operation_on_overlap == 'subtract':
                    val = reduce(lambda x, y: x-y, item_set)
                    single_vector.append(val)
                    
                elif operation_on_overlap == 'zero':
                    if usefloat:
                        single_vector.append(0.)
                    else:
                        single_vector.append(0)
                        
        return single_vector
            
            
            
    def parse_vector_modelfile(self, model_filepath):
        '''
        This function provides the backwards-compatible reading of model.txt files
        that makeVec.py used to use to specify 1D vectors.
        
        The functionr returns a dictionary in the following format, which *should*
        contain all the relevant information from the model file.
        
        Specification:
        
        vector model =
        {
            regressor_name =
            {
                input = subject behavioral csv string
                output = vector 1D output string
                marks =
                {   conditions dict with marks and conditions
                    1 =
                    {
                        mark_with = string value of mark
                        conditions =
                        [
                            ['TR', ['1','2']],
                            ['Ethnicity', ['1']]
                        ]
                    }
                    2 =
                    {
                        mark_with = -1
                        conditions =
                        [
                            ['TR', ['1,'2']],
                            ['Ethnicity',['2']]
                        ]
                    }
                }
            }
        }
        
        This (albeit clunky) dict will be converted into the clearer "layered-class"
        vector specification style in another function.
        
        '''
        
        vector_model = {}
        
        # read model file lines:
        mfid = open(model_filepath,'r')
        mlis = mfid.read()
        mfid.close()
        
        sections = mlis.split('BEGIN_VEC')
        for section in sections:
            submodel = {}
            conditions = {}
            name = ''
            lis = section.split('\n')
            marks = section.split('MARK')[1:]
            marks = [m.split('END_VEC')[0] for m in marks]
            marks = [m.strip(' \n') for m in marks]
            mark_with = [m.split('WITH')[1].strip() for m in marks]
            conds = [m.split('WITH')[0].split('AND') for m in m3]
            conds = [[x.strip() for x in cx] for cx in conds]
            conds = [[[z.strip() for z in y.split('=')] for y in x] for x in conds]
            
            for i, (mw, c) in enumerate(zip(mark_with, conds)):
                state_conds = []
                for cpair in c:
                    state_conds.append([cpair[0], cpair[1].split(',')])
                conditions[i] = {'mark_with':mw, 'conditions':state_conds}
            submodel['marks'] = conditions
                    
            for li in lis:
                if li.startswith('INPUT'):
                    submodel['input'] = li.split(':')[1].strip('\n\" ')
                elif li.startswith('OUTPUT'):
                    submodel['output'] = li.split(':')[1].strip('\n\" ')
                    name = li.split(':')[1].strip('\n\" ')[:-3]
            
            vector_model[name] = submodel
            
        return vector_model
    
    
    
    
    
    
class VectorRegressorLogical(object):
    
    def __init__(self, variable=None, logical=None, value=None, valuetype='raw'):
        super(VectorRegressorLogical, self).__init__()
        self.statement = [variable, logical, value, valuetype]
        
    def specify(self, variable, logical, value, valuetype='raw'):
        self.statement = [variable, logical, value, valuetype]
        
    def determine(self, csv_coldict, row_ind):
        if all(self.statement):
            self.statement = [str(x) for x in self.statement]
            check_var = csv_coldict[self.statement[0]][row_ind]
            logical = self.statement[1]
            
            if self.statement[3] == 'raw':
                check_val = self.statement[2]
            elif self.statement[3] == 'variable':
                check_val = csv_coldict[self.statement[3]][row_ind]
            
            if logical in ('=','==','is','equals'):
                return check_var == check_val
            elif logical in ('~=','!=','isnot','not','is not'):
                return check_var != check_val
            elif logical in ('<','lessthan','less than','is less than'):
                return float(check_var) < float(check_val)
            elif logical in ('>','greaterthan','greater than','is greater than'):
                return float(check_var) > float(check_val)
            elif logical in ('<=','less than or equal to', 'is less than or equal to',
                             'less than or equals', 'is less than or equals'):
                return float(check_var) <= float(check_val)
            elif logical in ('>=','greater than or equal to', 'is greater than or equal to',
                             'greater than or equals', 'is greater than or equals'):
                return float(check_var) >= float(check_val)
            
    
    
class VectorRegressorCondition(object):
    
    def __init__(self):
        super(VectorRegressorCondition, self).__init__()
        self.logicals = []        
        
    def state_raw(self, statement):
        spl = statement.split()
        
        
        
    def chain_and(self, logicals):
        chain = []
        for logical in logicals:
            chain.extend([logical, 'and'])
        self.logicals.append(chain[:-1])
        
    def chain_or(self, logicals):
        chain = []
        for logical in logicals:
            chain.extend([logical, 'or'])
        self.logicals.append(chain[:-1])
        
    def chain_xor(self, logicals):
        chain = []
        for logical in logicals:
            chain.extend([logical, 'xor'])
        self.logicals.append(chain[:-1])
            
            
    def _recurse_logical(self, logical, csv_coldict, row_ind):
        logic_op = 'and'
        truth_asessments = []
        for l in logical:
            if type(l) in (list, tuple):
                truth_assessments.append(self._recurse_logical(l, csv_coldict, row_ind))
            elif l in ('and', 'or', 'xor'):
                logic_op = l
            else:
                current_truth = logical.determine(csv_coldict, row_ind)
                if len(truth_assessments) > 0:
                    if logic_op == 'and':
                        if not truth_assessments[-1] == current_truth:
                            return False
                    elif logic_op == 'or':
                        if not truth_assessments[-1] and not current_truth:
                            return False
                    elif logic_op == 'xor':
                        if truth_assessments[-1] and current_truth:
                            return False
                truth_assessments.append(current_truth)
        
        if len(truth_assessments) == 1:
            return truth_assessments[0]
        else:
            return True
            
            
    def deterimine(self, csv_coldict, row_ind, iftrue, iffalse=0,
                   truetype='raw', falsetype='raw'):
        if truetype == 'variable':
            iftrue = csv_coldict[iftrue][row_ind]
        if falsetype == 'variable':
            iffalse = csv_coldict[iffalse][row_ind]
            
        truth = self._recurse_logical(self.logicals, csv_coldict, row_ind)
        
        if truth:
            return iftrue
        else:
            return iffalse
        




class VectorRegressor(object):
    
    def __init__(self):
        super(VectorModelRegressor, self).__init__()
        self.input_csv = None
        self.output_1D = None
            


class VectorModels(object):
    
    '''
    The intention of VectorModel is to replace/overhaul the limited system for
    vector specification that makeVec.py provides, while still providing
    backwards compatibility with "model.txt" style files that everyone is used to.
    '''
    
    def __init__(self):
        super(VectorModel, self).__init__()
        self.vector = VectorTools()
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        


            
            
            
            
            
            
            
            
            
            
            