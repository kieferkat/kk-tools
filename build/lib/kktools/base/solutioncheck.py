
import os, sys
import numpy as np
import pylab as pl
from sklearn import preprocessing


class SolutionChecker(object):
    
    def __init__(self, data_obj=None):
        super(SolutionChecker, self).__init__()
        self.data = data_obj
        

        
    def compare_matrices(self, oldX, newX):
                
        print np.shape(oldX), np.shape(newX.T)
        diagonals = np.diag(np.dot(oldX, newX.T))/len(oldX)
        pl.hist(diagonals)
        pl.show()
        
        mean_squared_error = np.mean((np.ones(len(diagonals))-diagonals)**2.)
        print 'Mean squared error for columns comparison is:', mean_squared_error
        
        
    def Y_sign_check(self, Xbeta):
        
        labels = np.sign(Xbeta)
        accuracy = 0.
        
        Y_signs = np.array(np.sign(self.data.Y))
        accuracy = (labels == Y_signs).sum() * 1. / labels.shape[0]
        
        print 'accuracy', accuracy
        return accuracy
        
        
    def Y_strict_check(self, Xbeta):
        
        labels = np.sign(Xbeta)
        accuracy = 0.
        
        Yscaled = preprocessing.scale(self.data.Y)
        Y_signs = np.array(np.sign(Yscaled))
        accuracy = (labels == Y_signs).sum() * 1. / labels.shape[0]
        
        print 'accuracy', accuracy
        return accuracy
        
    
    def swaptrs(self, X):
        
        newX = X.copy()
        newX.shape = (X.shape[0], 7, 26630)
        ind1 = newX[:,0:2,:].copy()
        ind2 = newX[:,2:4,:].copy()
        
        print 'ind1 shape', ind1.shape
        print 'ind2 shape', ind2.shape
        
        newX[:,0:2,:] = ind2
        newX[:,2:4,:] = ind1
        
        newX.shape = (X.shape[0], 7*26630)
        print 'newX shape', newX.shape
        return newX
        
        
    def logan_solution_checker(self, coef_solution_filepath, log_file='logansol_log.txt',
                               median=False, scaleY=False, add_intercept=True, swap=False):
        
        log = open(log_file, 'a+')
        
        preloaded_solution = np.load(coef_solution_filepath)
        dense_solution = preloaded_solution.tolist().todense()
        print np.shape(dense_solution)
        
        log.write('File: '+coef_solution_filepath+'\n')
        
        if median:
            dense_solution = np.median(dense_solution, axis=0)
            print 'median shape:', np.shape(dense_solution)
        
        if swap:
            self.data.X = self.swaptrs(self.data.X)

        for subsol in range(np.shape(dense_solution)[0]):
            log.write('Solution number: '+str(subsol+1)+'\n')
            if np.shape(dense_solution)[1] > np.shape(self.data.X)[1]:
                # handle intercept:
                if not add_intercept:
                    rsol = np.zeros((np.shape(dense_solution)[1]-1))
                    rsol[:] = dense_solution[subsol, 1:]
                    print np.shape(self.data.X)
                    Xbeta = np.dot(self.data.X, rsol)
                else:
                    newX = np.hstack((np.ones((len(self.data.X),1)), self.data.X))
                    rsol = np.zeros((np.shape(dense_solution)[1]))
                    rsol[:] = dense_solution[subsol,:]
                    print np.shape(newX), np.shape(rsol)
                    Xbeta = np.dot(newX, rsol)
                
            else:
                rsol = np.zeros((np.shape(dense_solution)[1]))
                rsol[:] = dense_solution[subsol, :]
                Xbeta = np.dot(self.data.X, rsol)
            
            if scaleY:
                accuracy = self.Y_strict_check(Xbeta)
            else:
                accuracy = self.Y_sign_check(Xbeta)
        
            log.write(str(accuracy)+'\n\n')
                
            
        log.write('\n\n')
        log.close()
        
        
        
    
