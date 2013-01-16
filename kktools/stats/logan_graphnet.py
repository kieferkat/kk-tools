

import sys, os, time
import numpy as np
import scipy.optimize
from nose.tools import *
import time
import h5py
import simplejson
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

from graphnet_mask import adj_from_nii, convert_to_array, prepare_adj

path_to_graphnetC_packages = os.path.abspath('/Users/span/kk_scripts/neuroparser/optimization/cwpath/.')
sys.path.append(path_to_graphnetC_packages)
import graphnet

# local imports:
from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from normalize import simple_normalize



class GraphnetInterface(CVObject):


    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(GraphnetInterface, self).__init__(variable_dict=variable_dict, data_obj=data_obj)
        self.set_folds(folds)
        self.niftitools = NiftiTools()
        
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        self.trial_mask = getattr(self.data, 'trial_mask', None)
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
        
    '''
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
    '''
    
    def regression_type_selector(self, l1, l2, l3, delta, svmdelta):
        if l1 and l2 and l3 and delta and svmdelta:
            return 'HuberSVMGraphNet'
        elif l1 and l2 and l3 and delta:
            return 'RobustGraphNet'
        elif l1 and l2 and l3:
            return 'NaiveGraphNet'
        elif l1 and l2:
            return 'NaiveENet'
        elif l1:
            return 'Lasso'
        else:
            return None
        
        
    def setup_crossvalidation(self, folds=None, subject_indices=None):
        if subject_indices:
            self.subject_indices = subject_indices
        if getattr(self, 'subject_indices', None):
            self.prepare_folds(folds=folds, indices_dict=self.subject_indices)
        else:
            print 'no subject indices set, cant setup cv folds'
                        
            
    def crossvalidate(self, train_kwargs_dict, use_memmap=False):
        
        trainresults, testresults = self.traintest_crossvalidator(self.train_graphnet,
                                                                  self.test_graphnet,
                                                                  self.trainX, self.trainY,
                                                                  self.testX, self.testY,
                                                                  train_kwargs_dict=train_kwargs_dict,
                                                                  use_memmap=use_memmap)
        
        self.accuracies = testresults
        self.average_accuracy = sum(self.accuracies)/len(self.accuracies)
        print 'Average accuracy: ', self.average_accuracy
        
        return self.accuracies, self.average_accuracy
                
        
        
    def test_graphnet(self, X, Y, coefs):
        
        X = simple_normalize(X)
        
        correct = []
        print 'Checking accuracy for test group'
        
        if self.problemkey == 'RobustGraphNet':
            coefs = coefs[:-self.trainX_shape[0]]
        
        for trial, outcome in zip(X, Y):
            predict = trial*coefs
            print np.sum(predict)
            Ypredsign = np.sign(np.sum(predict))
            if Ypredsign < 0.:
                Ypredsign = 0.
            else:
                Ypredsign = 1.
            print Ypredsign, outcome, (Ypredsign == outcome)
            correct.append(Ypredsign == outcome)
            
        fold_accuracy = np.sum(correct) * 1. / len(correct)
        
        print 'fold accuracy: ', fold_accuracy
        return fold_accuracy
    
    
    def train_graphnet(self, X, Y, trial_mask=None, G=None, l1=None, l2=None, l3=None, delta=None,
                      svmdelta=None, initial=None, adaptive=False, svm=False,
                      scipy_compare=False, tol=1e-5):
        
        X = simple_normalize(X)
        
        tic = time.clock()
        
        problemkey = self.regression_type_selector(*[bool(x) for x in [l1, l2, l3, delta, svmdelta]])
        
        self.problemkey = problemkey
        self.trainX_shape = X.shape
        
        if problemkey in ('HuberSVMGraphNet', 'RobustGraphNet', 'NaiveGraphNet'):
            if G is None:
                #nx = 60
                #ny = 60
                #A, Afull = construct_adjacency_list(nx, ny, 1, return_full=True)
                #A, Afull = self.gen_adj(X.shape[1])
                A = prepare_adj(trial_mask, numt=1)
            else:
                A = G.copy()
        
        if problemkey is 'RobustGraphNet':
            problemtype = graphnet.RobustGraphNet
            print 'Robust GraphNet with penalties (l1, l2, l3, delta): ', l1, l2, l3, delta
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=[l1], l2=l2, l3=l3, delta=delta)
        
        elif problemkey is 'HuberSVMGraphNet':
            problemtype = graphnet.GraphSVM
            print 'HuberSVM GraphNet with penalties (l1, l2, l3, delta): ', l1, l2, l3, delta
            Y = 2*np.round(np.random.uniform(0, 1, len(Y)))-1
            l = cwpath.CoordWise((X, Y, A), problemtype)
            l.problem.assign_penalty(path_key='l1', l1=[l1], l2=l2, l3=l3, delta=delta)
            
        elif problemkey is 'NaiveGraphNet':
            problemtype = graphnet.NaiveGraphNet
            print 'Testing GraphNet with penalties (l1, l2, l3): ', l1, l2, l3
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=[l1], l2=l2, l3=l3)
            
        elif problemkey is 'NaiveENet':
            problemtype = graphnet.NaiveENet
            print 'Testing ENET with penalties (l1, l2): ', l1, l2
            l = cwpath.CoordWise((X, Y), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=[l1], l2=l2)
            
        elif problemkey is 'Lasso':
            problemtype = graphnet.Lasso
            print 'Testing LASSO with penalty (l1): ', l1
            l = cwpath.CoordWise((X, Y), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=[l1])
            
        else:
            print 'Incorrect parameters set (no problem key).'
            return False
        
        # Solve the problem:
        print 'Solving the problem...'
        coefficients, residuals = l.fit(tol=tol, initial=initial)
        
        self.coefficients = coefficients
        self.residuals = residuals
        
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
            
        return coefficients[0]
                
        
        
        
    class Gridsearch(object):
        
        def __init__(self):
            super(Gridsearch, self).__init__(savedir=os.getcwd())
            self.verbose = True
            self.savedir = savedir
            self.search_depth = 3
            self.depth_stepsizes = [10, 4, 1]
            self.grid_shrink = 0.4
            
            self.l1_range = [0,100]
            self.l1_granularity = 0.1
                        
            self.l2 = 100.
            self.l3 = 1000.
            
            self.folds = 5
            
            self.searches = []
            
            st = time.localtime()
            timestr = str(st.tm_mon)+'_'+str(st.tm_mday)+'_'+str(st.tm_hour)+'_'+str(st.tm_min)
            
            self.logfile_name = 'fgrid_'+timestr+'.json'
            
            self.records = {}
            
            
        def generate_l1_values(self, l1_lower, l1_upper, granularity, round_to_int=True,
                               inclusive_max=True):
            
            distance = float(l1_upper)-float(l1_lower)
            
            step = distance*granularity
            if round_to_int:
                step = round(step)
            
            if inclusive_max:
                l1_values = [l1_lower+(x*step) for x in range(int(round(1.*granularity))+1)]
            else:
                l1_values = [l1_lower+(x*step) for x in range(int(round(1.*granularity)))]
            
            if self.verbose:
                print 'l1_range:', l1_lower, l1_upper
                print 'distance:', distance
                print 'granularity:', granularity
                print 'step size:', step
                print 'l1 values:', l1_values
                
            return l1_values, step
        
        
        def simple_generate_l1_range(self, l1min, l1max, stepsize):
            
            l1_range = [l1min]
            while l1_range[-1]+stepsize < l1max:
                l1_range.append(l1_range[-1]+stepsize)
            l1_range.append(l1max)
            
            return l1_range
            
        
        
        def log_progress(self):
            
            jsonpath = os.path.join(self.savedir, self.logfile_name)
            jfid = open(jsonpath,'w')
            simplejson.dump(self.records, jfid)
            jfid.close()
            
            
        def run_naive_gnet(self, csearch):
            
            cparams = csearch['parameters']
            
            train_kwargs = {'trial_mask':self.gnet.trial_mask, 'l1':cparams['l1'],
                            'l2':cparams['l2'], 'l3':cparams['l3']}
            
            self.gnet.setup_crossvalidation(subject_indices=self.gnet.subject_indices, folds=self.folds)
            accuracies, average_accuracy = self.gnet.crossvalidate(train_kwargs, use_memmap=True)
            
            csearch['accuracies'] = accuracies
            csearch['average_accuracy'] = average_accuracy
            
            return csearch
        
            
        def fractal_l1_search(self, gnet, trial_mask, indices):
            
            self.records['l1_start_range'] = self.l1_range
            self.records['l1_current_range'] = self.l1_range
            self.records['l2'] = self.l2
            self.records['l3'] = self.l3
            self.records['l1_granularity'] = self.l1_granularity
            self.records['depth_step_sizes'] = self.depth_stepsizes
            self.records['grid_shrink'] = self.grid_shrink
            self.records['search_depth'] = self.search_depth
            self.records['folds'] = self.folds
            self.records['current_iter'] = 0
            self.records['current_depth'] = 0
            self.records['searches'] = self.searches
            
            
            search_count = 0
            l1min = self.l1_range[0]
            l1max = self.l1_range[2]
            best_acc = 0.
            best_l1 = 0
            cur_distance = l1max-l1min
            
            for depth, stepsize in zip(range(self.search_depth), self.depth_stepsizes):
                
                #cur_l1_range, stepsize = self.generate_l1_values(l1min, l1max, self.l1_granularity)
                
                cur_l1_range = self.simple_generate_l1_range(l1min, l1max, stepsize)
                self.records['current_depth'] = depth
                
                for l1 in cur_l1_range:
                    
                    cur_params = {'l1':l1, 'l2':self.l2, 'l3':self.l3}
                    
                    #check if parameters have already been calculated:
                    do_search = True
                    for search in self.searches:
                        old_params = search['parameters']
                        if old_params == cur_params:
                            do_search = False
                            if self.verbose:
                                print 'Already completed this search...'
                                print 'old values:', old_params
                                print 'new values:', cur_params
                                
                    if do_search:
                        if self.verbose:
                            print 'Parameters for this search:', cur_params      
                    
                        csearch = {}
                        
                        csearch['search_iter'] = search_count
                        self.records['current_iter'] = search_count
                        search_count += 1
                        
                        csearch['parameters'] = cur_params
                        
                        if self.verbose:
                            print '\nPREFORMING NEXT GRAPHNET\n'
                            print 'search number:', search_count
                            print 'depth:', depth
                            print 'l1 range:', cur_l1_range
                            print 'current l1:', l1
                            
                        csearch = self.run_naive_gnet(csearch)
                        
                        self.searches.append(csearch)
                        self.records['searches'] = self.searches
                        
                        for srec in self.searches:
                            cacc = srec['average_accuracy']
                            if cacc > best_acc:
                                best_acc = cacc
                                best_l1 = srec['parameters']['l1']
                        
                        self.records['best_acc'] = best_acc
                        self.records['best_l1'] = best_l1
                        
                        self.log_progress()
                        
                        
                # find best accuracy, redefine l1max, l1min:

                new_distance = float(cur_distance)*self.grid_shrink
                l1min = int(round(float(best_l1)-new_distance/2.))
                l1max = int(round(float(best_l1)+new_distance/2.))
                
                cur_distance = new_distance
                self.records['l1_current_range'] = [l1min, l1max]
                        
                if self.verbose:
                    print 'best acccuracy for depth:', best_acc
                    print 'l1 for best accuracy:', best_l1
                    print 'new l1min:', l1min
                    print 'new l1max:', l1max
                    
                self.log_progress()
                    
                
                
            
            
            
            
            
        
        
            
            
        
            
            
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            