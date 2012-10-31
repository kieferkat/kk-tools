

import sys, os, time
import numpy as np
import scipy.optimize
from nose.tools import *
import time
import h5py
import matplotlib
matplotlib.use('agg')
import pylab as pl
pl.ion()
import rpy2
from rpy2 import robjects as rpy

from optimization.cwpath import cwpath, strategy
from optimization.cwpath.cwpath import inner1d
from optimization.graphs.graph_laplacian import construct_adjacency_list
# NEED TO ADD CYTHON & GRAPHS IMPORTS

path_to_graphnetC_packages = os.path.abspath('/Users/span/kk_scripts/neuroparser/optimization/cwpath/.')
sys.path.append(path_to_graphnetC_packages)
import graphnet

# local imports:
from ..data.crossvalidation import CVObject
from ..data.nifti import NiftiTools



class GraphnetInterface(CVObject):


    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(GraphnetInterface, self).__init__(variable_dict=variable_dict, data_obj=data_obj)
        self.set_folds(folds)
        self.niftitools = NiftiTools()
        
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        if not self.Y is None:
            self.replace_Y_negative_ones()
        self.indices_dict = getattr(self.data, 'subject_trial_indices', None)
        
        
    def huber(self, r, delta):
        r = np.fabs(r)
        t = np.greater(r, delta)
        return (1-t)*r**2 + t*(2*delta*r - delta**2)
        
        
    def huber_svm(self, r, delta):
        t1 = np.greater(r, delta)
        t2 = np.greater(r, 0)
        return t1*(r - delta/2) + (1-t1)*t2*(r**2/(2*delta))
        
        
    def huber_svm_error(self, beta, Y, Xp2, delta):
        r = 1-Y*np.dot*(Xp2, beta)
        return self.huber(r, delta)
        
        
    def get_lambda_max(self, X, y):
        """ 
        Find the value of lambda at which all coefficients are set to zero
        by finding the minimum value such that 0 is in the subdifferential
        and the coefficients are all zero.
        """
        subgrads = np.fabs(inner1d(X.T, y))
        return np.max(subgrads)
        
        
    def adj_array_as_list(self, adj):
        v = []
        for a in adj:
            v.append(a[np.greater(a, -1)])
        return v
    
    
    def gen_adj(self, p):
        print 'generating adjacency matrix'
        Afull = np.zeros((p, p), dtype=int)
        A = -np.ones((p, p), dtype=int)
        counts = np.zeros(p)
        for i in range(p):
            for j in range(p):
                if np.random.uniform(0, 1) < 0.3:
                    if i != j:
                        if Afull[i,j] == 0:
                            Afull[i,j] = -1
                            Afull[j,i] = -1
                            Afull[i,i] += 1
                            Afull[j,j] += 1
                            A[i, counts[i]] = j
                            A[j, counts[j]] = i
                            counts[i] += 1
                            counts[j] += 1
        return self.adj_array_as_list(A), Afull
        
    
    def regression_type_selector(self, l1, l2, l3, delta, svmdelta):
        if l1 and l2 and l3 and delta and svmdelta:
            return 'RobustGraphNet'
        elif l1 and l2 and l3 and delta:
            return 'HuberSVMGraphNet'
        elif l1 and l2 and l3:
            return 'NaiveGraphNet'
        elif l1 and l2:
            return 'NaiveENet'
        elif l1:
            return 'Lasso'
        else:
            return None
        
    
    
    def test_graphnet(self, X, Y, G=None, l1=None, l2=None, l3=None, delta=None,
                      svmdelta=None, initial=None, adaptive=False, svm=False,
                      scipy_compare=True, tol=1e-5):
        
        tic = time.clock()
        
        problemkey = self.regression_type_selector(*[bool(x) for x in [l1, l2, l3, delta, svmdelta]])
        
        
        if problemkey in ('HuberSVMGraphNet', 'RobustGraphNet', 'NaiveGraphNet'):
            if G is None:
                #nx = 60
                #ny = 60
                #A, Afull = construct_adjacency_list(nx, ny, 1, return_full=True)
                A, Afull = self.gen_adj(X.shape[1])
            else:
                A = G.copy()
        
        if problemkey is 'RobustGraphNet':
            problemtype = graphnet.RobustGraphNet
            print 'Robust GraphNet with penalties (l1, l2, l3, delta): ', l1, l2, l3, delta
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, delta=delta)
        
        elif problemkey is 'HuberSVMGraphNet':
            problemtype = graphnet.GraphSVM
            print 'HuberSVM GraphNet with penalties (l1, l2, l3, delta): ', l1, l2, l3, delta
            Y = 2*np.round(np.random.uniform(0, 1, len(Y)))-1
            l = cwpath.CoordWise((X, Y, A), problemtype)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, delta=delta)
            
        elif problemkey is 'NaiveGraphNet':
            problemtype = graphnet.NaiveGraphNet
            print 'Testing GraphNet with penalties (l1, l2, l3): ', l1, l2, l3
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3)
            
        elif problemkey is 'NaiveENet':
            problemtype = graphnet.NaiveENet
            print 'Testing ENET with penalties (l1, l2): ', l1, l2
            l = cwpath.CoordWise((X, Y), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2)
            
        elif problemkey is 'Lasso':
            problemtype = graphnet.Lasso
            print 'Testing LASSO with penalty (l1): ', l1
            l = cwpath.CoordWise((X, Y), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1)
            
        else:
            print 'Incorrect parameters set (no problem key).'
            return False
        
        # Solve the problem:
        print 'Solving the problem...'
        coefficients, residuals = l.fit(tol=tol, initial=initial)
        
        print '\t---> Fitting problem with coordinate decesnt took: ', time.clock()-tic, 'seconds.'
        
        if adaptive:
            tic = time.clock()
            l1weights = 1./beta
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(l1=l1, l2=l2, l3=l3, delta=delta, l1weights=l1weights, newl1=l1)
            adaptive_coefficients, adaptive_residuals = l.fit(tol=tol, initial=initial)
            print '\t---> Fitting Adaptive GraphNet problem with coordinate descent took: ', time.clock()-tic, 'seconds.'
        
        
        if scipy_compare:
            
            l1 = l1[-1]
            beta = coefficients[-1]
        
            print '\t---> Fitting with scipy for comparison...'
            
            tic = time.clock()
            
            if problemkey is 'RobustGraphNet':
                def f(beta):
                    huber_sum = self.huber(Y - np.dot(X, beta), delta).sum()/2
                    beta_l1 = l1*np.dot(np.fabs(beta), l1weights)
                    beta_l2 = l2*np.linalg.norm(beta)**2/2
                    beta_l3 = l3*np.dot(beta, np.dot(Afull, beta))/2
                    return huber_sum + beta_l1 + beta_l2 + beta_l3
                
            elif problemkey is 'HuberSVMGraphNet':
                Xp2 = np.hstack([np.ones(X.shape[0])[:,np.newaxis], X])
                def f(beta):
                    ind = range(1, len(beta))
                    huber_err_sum = self.huber_svm_error(beta, Y, Xp2, delta).sum()
                    beta_l1 = np.fabs(beta[ind]).sum()*l1
                    beta_l2 = l2*(np.linalg.norm(beta[ind])**2/2)
                    beta_l3 = l3*(np.dot(beta[ind], np.dot(Afull, beta[ind])))/2
                    return huber_error_sum + beta_l1 + beta_l2 + beta_l3
                
            elif problemkey is 'NaiveGraphNet':
                def f(beta):
                    beta_XY = np.linalg.norm(Y - np.dot(X, beta))**2/2
                    beta_l1 = l1*np.fabs(beta).sum()
                    beta_l2 = l2*np.linalg.norm(beta)**2/2
                    beta_l3 = l3*np.dot(beta, np.dot(Afull, beta))/2
                    return beta_XY + beta_l1 + beta_l2 + beta_l3
                
            elif problemkey is 'NaiveENet':
                def f(beta):
                    beta_XY = np.linalg.norm(Y - np.dot(X, beta))**2/2
                    beta_l1 = l1*np.fabs(beta).sum()
                    beta_l2 = np.linalg.norm(beta)**2/2
                    
            elif problemkey is 'Lasso':
                def f(beta):
                    beta_XY = np.linalg.norm(Y - np.dot(X, beta))**2/2
                    beta_l1 = l1*np.fabs(beta).sum()
                    
            if problemkey is 'HuberSVMGraphNet':
                v = scipy.optimize.fmin_powell(f, np.zeros(Xp2.shape[1]), ftol=1.0e-14, xtol=1.0e-14, maxfun=100000)
            else:
                v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=100000)
                
            v = np.asarray(v)
            
            print '\t---> Fitting GraphNet with scipy took: ', time.clock()-tic, 'seconds.'
            
            assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < tol)
            if np.linalg.norm(v) > 1e-8:
                assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < tol)
            else:
                assert_true(np.linalg.norm(beta) < 1e-8)
                
            print '\t---> Coordinate-wise and Scipy optimization agree.'
            
        return (coefficients, residuals), problemkey
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            