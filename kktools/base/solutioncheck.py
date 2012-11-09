
import os, sys
import numpy as np










class SolutionChecker(object):
    
    def __init__(self, data_obj=None):
        super(SolutionChecker, self).__init__()
        self.data = data_obj
        
        
    def logan_solution_checker(self, coef_solution_filepath, log_file='logansol_log.txt'):
        
        log = open(log_file, 'a+')
        
        preloaded_solution = np.load(coef_solution_filepath)
        dense_solution = preloaded_solution.tolist().todense()
        print np.shape(dense_solution)
        
        log.write('File: '+coef_solution_filepath+'\n')
        
        for subsol in range(np.shape(dense_solution)[0]):
            log.write('Solution number: '+str(subsol+1)+'\n')
            if np.shape(dense_solution)[1] > np.shape(self.data.X)[1]:
                # handle intercept:
                rsol = np.zeros((np.shape(dense_solution)[1]-1))
                rsol[:] = dense_solution[subsol, 1:]
                print np.shape(self.data.X)
                Xbeta = np.dot(self.data.X, rsol)
                
            else:
                rsol = np.zeros((np.shape(dense_solution)[1]))
                rsol[:] = dense_solution[subsol, :]
                Xbeta = np.dot(self.data.X, rsol)
                
            labels = np.sign(Xbeta)
            accuracy = 0.
            
            correct = Xbeta
            correct_sum = correct.sum()
            
            print len(Xbeta)
            
            predictions = float(np.sign(Xbeta).sum())
            prediction_balance = predictions / len(Xbeta)
            print prediction_balance
            
            Y_signs = np.array(self.data.Y)
            accuracy = (np.sign(Xbeta) == Y_signs).sum() * 1. / labels.shape[0]
            
            print 'accuracy', accuracy
            log.write(str(accuracy)+'\n\n')
            
        log.write('\n\n')
        log.close()
        
        
        
    
