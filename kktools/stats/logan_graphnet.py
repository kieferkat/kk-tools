

import sys, os
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
import copy
import random
from pprint import pprint

from optimization.cwpath import cwpath, strategy
from optimization.cwpath.cwpath import inner1d
from optimization.graphs.graph_laplacian import construct_adjacency_list
# NEED TO ADD CYTHON & GRAPHS IMPORTS

from graphnet_mask import adj_from_nii, convert_to_array, prepare_adj

path_to_graphnetC_packages = os.path.abspath('/Users/span/kk_scripts/neuroparser/optimization/cwpath/.')
sys.path.append(path_to_graphnetC_packages)
#import graphnet
import optimization.cwpath.graphnet as graphnet

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
        print l1, l2, l3, delta, svmdelta
        l1b = all(l1)
        if (l1b != False) and (l2 != None) and (l3 != None) and (delta != None) and (svmdelta != None):
            return 'HuberSVMGraphNet'
        elif (l1b != False) and (l2 != None) and (l3 != None) and (delta != None):
            return 'RobustGraphNet'
        elif (l1b != False) and (l2 != None) and (l3 != None):
            return 'NaiveGraphNet'
        elif (l1b != False) and (l2 != None):
            return 'NaiveENet'
        elif (l1b != False):
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
        self.average_accuracies = []
        for i in range(len(self.accuracies[0])):
            accs = []
            for j in range(len(self.accuracies)):
                accs.append(self.accuracies[j][i])
            self.average_accuracies.append(sum(accs)/len(accs))
                
        #self.average_accuracies = [sum(x)/len(x) for x in self.accuracies]
        print 'Average accuracies: ', self.average_accuracies
        
        # trainresults: list of coefficients for each l1 by fold
        # AS OF NOW JUST TAKING THE COEFS FOR ONE OF THE FOLDS:
        sub_tresults = trainresults[0]
        self.non_zero_coefs = [len([x for x in tr if x != 0.]) for tr in sub_tresults]
        
        return self.accuracies, self.average_accuracies, self.non_zero_coefs
                
        
        
    def test_graphnet(self, X, Y, coefs):
        
        X = simple_normalize(X)
        accuracies = []
        
        for i, coefset in enumerate(coefs):
            
            correct = []
            print 'Checking accuracy for test group'
            
            if self.problemkey == 'RobustGraphNet':
                coefset = coefset[:-self.trainX_shape[0]]
            
            for trial, outcome in zip(X, Y):
                predict = trial*coefset
                print np.sum(predict)
                Ypredsign = np.sign(np.sum(predict))
                if Ypredsign < 0.:
                    Ypredsign = 0.
                else:
                    Ypredsign = 1.
                print Ypredsign, outcome, (Ypredsign == outcome)
                correct.append(Ypredsign == outcome)
                
            fold_accuracy = np.sum(correct) * 1. / len(correct)
            
            print 'coef number:', i
            print 'fold accuracy: ', fold_accuracy
            accuracies.append(fold_accuracy)
            
            
        return accuracies
    
    
    def train_graphnet(self, X, Y, trial_mask=None, G=None, l1=None, l2=None, l3=None, delta=None,
                      svmdelta=None, initial=None, adaptive=False, svm=False,
                      scipy_compare=False, tol=1e-5, greymatter_mask=None, initial_l1weights=None,
                      use_adj_time=True):
                
        if not type(l1) in [list, tuple]:
            l1 = [l1]
                
        X = simple_normalize(X)
        
        tic = time.clock()
        
        #problemkey = self.regression_type_selector(*[bool(x) for x in [l1, l2, l3, delta, svmdelta]])
        
        problemkey = self.regression_type_selector(l1, l2, l3, delta, svmdelta)
        
        self.problemkey = problemkey
        self.trainX_shape = X.shape
        
        if problemkey in ('HuberSVMGraphNet', 'RobustGraphNet', 'NaiveGraphNet'):
            if G is None:
                #nx = 60
                #ny = 60
                #A, Afull = construct_adjacency_list(nx, ny, 1, return_full=True)
                #A, Afull = self.gen_adj(X.shape[1])
                #if greymatter_mask is not None:
                #    A, GMA = prepare_adj(trial_mask, numt=1, gm_mask=greymatter_mask)
                #else:
                #    A = prepare_adj(trial_mask, numt=1)
                #    GMA = None
                if use_adj_time:
                    A = prepare_adj(trial_mask, numt=1, gm_mask=greymatter_mask)
                else:
                    A = prepare_adj(trial_mask, numt=0, gm_mask=greymatter_mask)
                
            else:
                A = G.copy()
                
        if initial_l1weights is not None:
            newl1 = l1
        else:
            newl1 = None
        
        if problemkey is 'RobustGraphNet':
            problemtype = graphnet.RobustGraphNet
            print 'Robust GraphNet with penalties (l1, l2, l3, delta): ', l1, l2, l3, delta
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)#, gma=GMA)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, delta=delta, l1weights=initial_l1weights,
                                     newl1=newl1)
        
        elif problemkey is 'HuberSVMGraphNet':
            problemtype = graphnet.GraphSVM
            print 'HuberSVM GraphNet with penalties (l1, l2, l3, delta): ', l1, l2, l3, delta
            Y = 2*np.round(np.random.uniform(0, 1, len(Y)))-1
            l = cwpath.CoordWise((X, Y, A), problemtype)#, gma=GMA)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, delta=delta, l1weights=initial_l1weights,
                                     newl1=newl1)
            
        elif problemkey is 'NaiveGraphNet':
            problemtype = graphnet.NaiveGraphNet
            print 'Testing GraphNet with penalties (l1, l2, l3): ', l1, l2, l3
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)#, gma=GMA)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, l1weights=initial_l1weights,
                                     newl1=newl1)
            
        elif problemkey is 'NaiveENet':
            problemtype = graphnet.NaiveENet
            print 'Testing ENET with penalties (l1, l2): ', l1, l2
            l = cwpath.CoordWise((X, Y), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l1weights=initial_l1weights,
                                     newl1=newl1)
            
        elif problemkey is 'Lasso':
            problemtype = graphnet.Lasso
            print 'Testing LASSO with penalty (l1): ', l1
            l = cwpath.CoordWise((X, Y), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1, l1weights=initial_l1weights, newl1=newl1)
            
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
            safety = 1e-5
            l1weights = 1./(self.coefficients[-1]+safety)
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, delta=delta, l1weights=l1weights, newl1=l1)
            adaptive_coefficients, adaptive_residuals = l.fit(tol=tol, initial=initial)
            print '\t---> Fitting Adaptive GraphNet problem with coordinate descent took: ', time.clock()-tic, 'seconds.'
            
            self.firstpass_coefficients = self.coefficients
            self.firstpass_residuals = self.residuals
            self.coefficients = adaptive_coefficients
            self.residuals = adaptive_residuals
        
        
        if scipy_compare:
            
            l1 = l1[-1]
            beta = self.coefficients[-1]
        
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
            
        return self.coefficients
                
        
        
        
class Gridsearch(object):
    
    def __init__(self, savedir=os.getcwd()):
        super(Gridsearch, self).__init__()
        self.verbose = True
        self.savedir = savedir
        self.search_depth = 3
        self.depth_stepsizes = [5, 2.5, 0.5]
        self.grid_shrink = 0.4
        
        self.l1_range = range(8,70,1)
        #self.l1_granularity = 0.1
                    
        self.l2_range = [100000.]
        self.l3_range = [0.]
        
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
    
    
    def simple_generate_l1_range(self, l1min, l1max, stepsize, no_zero=True):
        
        l1min, l1max, stepsize = float(l1min), float(l1max), float(stepsize)
        
        l1_range = [l1min]
        while l1_range[-1]+stepsize < l1max:
            l1_range.append(l1_range[-1]+stepsize)
        l1_range.append(l1max)
        
        if no_zero:
            l1_range = [x for x in l1_range if x != 0]
        
        return l1_range
        
    
    
    def log_progress(self):
        
        jsonpath = os.path.join(self.savedir, self.logfile_name)
        jfid = open(jsonpath,'w')
        simplejson.dump(self.records, jfid)
        jfid.close()
        
        
    def run_naive_gnet(self, csearch, l1_list=None, use_memmap=False, greymatter_mask=None):
        
        cparams = csearch['parameters']
        
        print cparams
        
        if l1_list:
            print 'l1s:',l1_list
            train_kwargs = {'trial_mask':self.gnet.trial_mask, 'l1':l1_list,
                            'l2':cparams['l2'], 'l3':cparams['l3'], 'greymatter_mask':greymatter_mask}
        else:
            train_kwargs = {'trial_mask':self.gnet.trial_mask, 'l1':cparams['l1'],
                            'l2':cparams['l2'], 'l3':cparams['l3'],'greymatter_mask':greymatter_mask}
            
        
        self.gnet.setup_crossvalidation(subject_indices=self.gnet.subject_indices, folds=self.folds)
        
        # REAL:
        accuracies, average_accuracies, nz_coefs = self.gnet.crossvalidate(train_kwargs, use_memmap=use_memmap)
        
        # TEST:
        #accuracies = [[random.random() for x in range(len(l1_list))] for x in range(5)]
        #average_accuracies = []
        #for i in range(len(accuracies[0])):
        #    accs = []
        #    for j in range(len(accuracies)):
        #        accs.append(accuracies[j][i])
        #    average_accuracies.append(sum(accs)/len(accs))
        #nz_coefs = [random.randint(0,1000) for x in range(len(l1_list))]
        
        
        self.accuracies = accuracies
        self.average_accuracies = average_accuracies
        self.non_zero_coefs = nz_coefs
        
        if l1_list:
            self.csearches = []
            for ind, l1 in enumerate(l1_list):
                nsearch = {}
                nsearch['parameters'] = {'l1':l1, 'l2':cparams['l2'], 'l3':cparams['l3']}
                nsearch['parameters']['l1'] = l1
                group_accuracies = []
                for i in range(len(self.accuracies)):
                    group_accuracies.append(self.accuracies[i][ind])
                nsearch['accuracies'] = group_accuracies
                nsearch['average_accuracy'] = average_accuracies[ind]
                nsearch['non_zero_coefs'] = nz_coefs[ind]
                nsearch['search_iter'] = csearch['search_iter'] + ind
                
                #pprint(nsearch)
                
                self.csearches.append(nsearch)
            return self.csearches
        else:  
            csearch['accuracies'] = accuracies[0]
            csearch['average_accuracy'] = average_accuracies[0]
            csearch['non_zero_coefs'] = nz_coefs[0]
            return csearch
        
        
        
    
        
    def fractal_l1_search(self, gnet, graphnet_l1_multisearch=True, reverse_range=True,
                          name='', adaptive=True, use_memmap=False, greymatter_mask=None):
        
        self.gnet = gnet
        self.records['title'] = name
        if name:
            st = time.localtime()
            timestr = str(st.tm_mon)+'_'+str(st.tm_mday)+'_'+str(st.tm_hour)+'_'+str(st.tm_min)
            self.logfile_name = name+'_'+timestr+'.json'
        #self.records['l1_start_range'] = self.l1_range
        #self.records['l1_current_range'] = self.l1_range
        self.records['l1_range'] = self.l1_range
        self.records['l2_range'] = self.l2_range
        self.records['l3_range'] = self.l3_range
        #self.records['depth_step_sizes'] = self.depth_stepsizes
        #self.records['grid_shrink'] = self.grid_shrink
        #self.records['search_depth'] = self.search_depth
        self.records['folds'] = self.folds
        self.records['current_iter'] = 0
        #self.records['current_depth'] = 0
        self.records['searches'] = self.searches
        
        
        search_count = 0
        l1min = self.l1_range[0]
        l1max = self.l1_range[-1]
        best_acc = 0.
        best_l1 = -1
        best_l2 = -1
        best_l3 = -1
        cur_distance = l1max-l1min
        
        for l3 in self.l3_range:
            for l2 in self.l2_range:
                
                cur_l1_range = self.l1_range[:]
                                                  
                if reverse_range:
                    cur_l1_range.reverse()
                
                #self.records['current_depth'] = depth
                
                if graphnet_l1_multisearch:
                    
                    cur_params = {'l1':[], 'l2':l2, 'l3':l3}
                    
                    csearch = {}
                        
                    csearch['search_iter'] = search_count
                    self.records['current_iter'] = search_count
                    
                    csearch['parameters'] = cur_params
                    
                    if self.verbose:
                        print '\nPREFORMING NEXT MULTI-SEARCH GRAPHNET\n'
                        print 'l1 range:', cur_l1_range
                        print 'l2', l2
                        print 'l3', l3
                        
                    csearches = self.run_naive_gnet(csearch, l1_list=cur_l1_range,
                                                    use_memmap=use_memmap, greymatter_mask=greymatter_mask)
                    
                    for cs in csearches:
                        self.searches.append(cs)
                        search_count += 1
                    self.records['current_iter'] = search_count
                    self.records['searches'] = self.searches
                    
                    for srec in self.searches:
                        cacc = srec['average_accuracy']
                        if cacc > best_acc:
                            best_acc = cacc
                            best_parameters = srec['parameters']
                    
                    self.records['best_acc'] = best_acc
                    self.records['best_parameters'] = best_parameters
                    
                    self.log_progress()
                    
                
                else:
                    
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
                            
                            csearch['parameters'] = cur_params
                            
                            if self.verbose:
                                print '\nPREFORMING NEXT GRAPHNET\n'
                                print 'search number:', search_count
                                print 'depth:', depth
                                print 'l1 range:', cur_l1_range
                                print 'current l1:', l1
                                print 'best acccuracy:', best_acc
                                print 'l1 for best accuracy:', best_l1
                                
                            csearch = self.run_naive_gnet(csearch, use_memmap=use_memmap,
                                                          greymatter_mask=greymatter_mask)
                            
                            self.searches.append(csearch)
                            self.records['searches'] = self.searches
                            
                            for srec in self.searches:
                                cacc = srec['average_accuracy']
                                if cacc > best_acc:
                                    best_acc = cacc
                                    best_l1 = srec['parameters']['l1']
                            
                            self.records['best_acc'] = best_acc
                            self.records['best_l1'] = best_l1
                            
                            search_count += 1
                            self.log_progress()
                        
                        
                        
                # find best accuracy, redefine l1max, l1min:
    
                '''
                new_distance = float(cur_distance)*self.grid_shrink
                l1min = int(round(float(best_l1)-new_distance/2.))
                l1min = max([0,l1min])
                l1max = int(round(float(best_l1)+new_distance/2.))
                
                cur_distance = new_distance
                self.records['l1_current_range'] = [l1min, l1max]
                        
                if self.verbose:
                    print 'best acccuracy:', best_acc
                    print 'l1 for best accuracy:', best_l1
                    print 'new l1min:', l1min
                    print 'new l1max:', l1max
                    
                self.log_progress()
                '''    
                
                
                
                
                
                
                
            
            
                
            
        
            
            
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            