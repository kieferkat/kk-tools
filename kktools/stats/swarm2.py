
import numpy as np
import os
import random
import itertools
import time
import json
import nibabel as nib
import scipy.stats as stats
from pprint import pprint
from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from linearsvm import ScikitsSVM
from normalize import simple_normalize
from threshold import threshold_by_pvalue



class SwarmData(object):
    
    def __init__(self):
        super(SwarmData, self).__init__()
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.coefs = None



class SwarmDataManager(object):
    
    def __init__(self, data_obj):
        super(SwarmDataManager, self).__init__()
        self.data = data_obj
        self.X = self.data.X
        self.Y = self.data.Y
        self.dimensions = self.X.shape[1]
        
        
    def normalize_x(self, Xset):
        Xnorm = simple_normalize(Xset)
        return Xnorm
    
    
    def subselect_x(self, Xset, inds):
        inds = [int(x) for x in inds]
        return Xset[inds,:]
        
        
    def subselect_y(self, Yset, inds):
        inds = [int(x) for x in inds]
        return Yset[inds]
        
        
    def prepare_data_folds(self, folds=4, do_svm=False, save_trainXY=True):
        self.data_folds = {}
        
        cvo = CVObject(data_obj=self.data)
        cvo.prepare_folds(folds=folds)
        
        for i in range(folds):
            sd = SwarmData()
            
            print len(cvo.trainX[i])
            print len(cvo.trainY[i])
            
            #sd.trainX = self.normalize_x(self.subselect_x(self.X, cvo.trainX[i]))
            sd.trainX = self.subselect_x(self.X, cvo.trainX[i])
            sd.trainY = self.subselect_y(self.Y, cvo.trainY[i])
            
            sd.testX = self.normalize_x(self.subselect_x(self.X, cvo.testX[i]))
            #sd.testX = self.subselect_x(self.X, cvo.testX[i])
            sd.testY = self.subselect_y(self.Y, cvo.testY[i])
            
            if do_svm:
                svm = ScikitsSVM()
                sd.coefs = svm.fit_linearsvc(sd.trainX, sd.trainY).coef_[0]
                
                #fake:
                #sd.coefs = np.random.random_sample(sd.trainX.shape[1])
                
                
            if not save_trainXY:
                sd.trainX = []
                sd.trainY = []
            
            self.data_folds[i] = sd
                
                
        



class Particle(object):
    
    def __init__(self):
        super(Particle, self).__init__()
        
        self.id = 0
        self.neighbors = []
        
        self.current_accuracy = 0.
        self.prior_accuracy = 0.
        self.best_accuracy = 0.
        self.accuracies = []
        
        self.current_fitness = 0.
        self.prior_fitness = 0.
        self.best_fitness = 0.
        self.fitnesses = []
        
        self.coefs = []
        self.best_coefs = []
        
        self.velocity = []
        self.abs_velocities = []
        

    
    def initialize_coefs(self, dimensions, particle_type='weight', multiplier=1.):
        
        self.dimensions = dimensions
        
        if particle_type == 'weight':
            random_range = 2.*np.random.random_sample(size=dimensions)-1.
            self.coefs = multiplier*random_range
            
        elif particle_type == 'binary':
            self.coefs = np.random.randint(2, size=dimensions)
            
        elif particle_type == 'probability':
            self.coefs = np.random.random_sample(size=dimensions)
            
        self.best_coefs = self.coefs.copy()
            
            
    def initialize_velocity(self, dimensions, velocity_start='zeros',
                            multiplier=1.):
    
        if velocity_start == 'zeros':
            self.velocity = np.zeros(dimensions)
            
        elif velocity_start == 'probability':
            self.velocity = 2.*(np.random.random_sample(size=dimensions)-1.)*multiplier
            
            
            
            
class FitnessFunction(object):
    
    def __init__(self, fitness_type='logarithmic', score_correct_one=True):
        super(FitnessFunction, self).__init__()
        
        self.fitness_type = fitness_type
        self.score_correct_one = score_correct_one
        
        
    def __call__(self, outcome, prediction):
        if self.fitness_type == 'logarithmic':
            return self.logarithmic(outcome, prediction)
        
        
    def logarithmic(self, outcome, prediction):
        
        abs_prediction = np.absolute(prediction)
        
        if np.sign(outcome) == np.sign(prediction):
            correct = 1.
        else:
            correct = -1.

        justified_prediction = correct * abs_prediction
        
        score = 1. / (1. + np.exp(-1. * justified_prediction))
        score = (score - .5) * 2.
        
        if self.score_correct_one and correct == 1.:
            return 1.
        else:
            return score
        
        
        
        
class VelocityFunction(object):

    def __init__(self, use_inertia=True, inertia_constant=0.7, acceleration_sum=4.2,
                 use_constriction=True, constriction_k=1.0, vmax_range=None,
                 vmin_range=None):
        super(VelocityFunction, self).__init__()
        self.verbose = True
        
        self.acceleration_sum = acceleration_sum
        
        self.use_inertia = use_inertia
        self.inertia_constant = inertia_constant
        
        self.use_constriction = use_constriction
        self.constriction_k = constriction_k
        if self.use_constriction:
            self.calculate_constriction()
        
        self.vmax_range = vmax_range
        self.vmin_range = vmin_range
        
        
        
        
    def __call__(self, particles, particle, neighborhood_best):
        return self.local_velocity(particles, particle, neighborhood_best)
        
        
    def calculate_constriction(self):
        a_ = (2. * self.constriction_k)
        b_ = 2. - self.acceleration_sum
        c_ = self.acceleration_sum**2. - 4.*self.acceleration_sum
        d_ = abs(b_ - np.sqrt(c_))
        self.constriction = a_ / d_
        
        if self.verbose:
            print 'constriction coefficient = ', self.constriction
        
    
    def stochastic_acceleration_update(self):
        self.acceleration_a = self.acceleration_sum * np.random.random()
        self.acceleration_b = self.acceleration_sum - self.acceleration_a
    
    def inertia_update(self, particle):
        past_v = particle.velocity.copy()
        past_v *= self.inertia_constant
        return past_v
        
        
    def apply_restrictions(self, particle, next_v):
        
        if self.use_inertia:
            next_v += self.inertia_update(particle)
            
        if self.use_constriction:
            next_v *= self.constriction
            
        if self.vmax_range is not None and self.vmin_range is not None:
            next_v = np.maximum(self.vmin_range, next_v)
            next_v = np.minimum(self.vmax_range, next_v)
            
        return next_v
            
        
    def local_velocity(self, particles, p, nb):
        
        self.stochastic_acceleration_update()
        
        if p == nb:
            nb_d = np.zeros(particles[p].dimensions)
        else:
            nb_d = particles[nb].best_coefs - particles[p].coefs
            
        ind_d = particles[p].best_coefs - particles[p].coefs
        
        nb_d *= self.acceleration_a
        ind_d *= self.acceleration_b
        
        next_v = nb_d + ind_d
        
        next_v = self.apply_restrictions(particles[p], next_v)
        
        return next_v

    
    def global_velocity(self, particle, global_best_coefs):
        
        self.stochastic_acceleration_update()
        
        gb_d = global_best_coefs - particle.coefs
        ind_d = particle.best_coefs - particle.coefs
        
        gb_d *= self.acceleration_a
        ind_d *= self.acceleration_b
        
        next_v = gb_d + ind_d
        
        next_v = self.apply_restrictions(particle, next_v)
        
        
        
        



class Swarm(object):
    
    def __init__(self, swarmdata, id=0, vrange=None, xrange=None, particle_type='probability',
                 particle_multiplier=1., neighborhood_type='circle',
                 population_size=10, fitness_function=FitnessFunction(),
                 velocity_function=VelocityFunction(), max_iters=1000):
        
        super(Swarm, self).__init__()
        
        self.id = id
        self.vrange = vrange
        self.xrange = xrange
        self.particle_type = particle_type
        self.particle_multiplier = particle_multiplier
        self.neighborhood_type = neighborhood_type
        self.population_size = population_size
        self.particles = {}
        
        self.fitness_function = fitness_function
        self.velocity_function = velocity_function
        
        self.swarmdata = swarmdata
        self.max_iters = max_iters
        
        self.verbose = True
        
        
        
        
    def initialize_particle_limits(self, particle_dimensions):
        
        if not self.vrange is None:
            if self.verbose:
                print 'setting vmax'
            if self.vrange == 'auto':
                self.vmax_range = np.array([(1./3.)*particle_dimensions for x in range(particle_dimensions)])
                self.vmin_range = -1.*self.vmax_range
            else:
                self.vmax_range = np.array([self.vrange[1] for x in range(particle_dimensions)])
                self.vmin_range = np.array([self.vrange[0] for x in range(particle_dimensions)])
            self.velocity_function.vmax_range = self.vmax_range
            self.velocity_function.vmin_range = self.vmin_range
        
        if not self.xrange is None:
            if self.verbose:
                print 'setting xmax'
            self.xmax_range = np.array([self.xrange[1] for x in range(particle_dimensions)])
            self.xmin_range = np.array([self.xrange[0] for x in range(particle_dimensions)])
            self.use_xmax = True
    
        
    def initialize_global_records(self, particle_dimensions):
        
        if self.verbose: print 'initializing global records'
        
        self.global_best_coefs = np.zeros(particle_dimensions)
        self.global_best_accuracy = None
        self.global_best_fitness = None
        
        
    def initialize_particles(self, particle_dimensions):
        
        if self.verbose:
            print 'initializing particles, dimensions:', particle_dimensions
            print 'population type:', self.neighborhood_type
            
        self.initialize_particle_limits(particle_dimensions)
        self.initialize_global_records(particle_dimensions)
        
        for i in range(self.population_size):
            particle = Particle()
            particle.id = i
            
            if self.neighborhood_type == 'circle':
                if i == 0:
                    particle.neighbors = [self.population_size-1, i+1]
                elif i == self.population_size-1:
                    particle.neighbors = [i-1, 0]
                else:
                    particle.neighbors = [i-1, i+1]
                    
            elif self.neighborhood_type == 'wheel':
                if i == 0:
                    particle.neighbors = [self.population_size-2, i+1, self.population_size-1]
                elif i == self.population_size-2:
                    particle.neighbors = [i-1, 0, self.population_size-1]
                elif i == self.population_size-1:
                    particle.neighbors = range(self.population_size-1)
                else:
                    particle.neighbors = [i-1, i+1, self.population_size-1]
                    
            elif self.neighborhood_type == 'axle':
                if i == self.population_size-1:
                    particle.neighbors = range(self.population_size-1)
                else:
                    particle.neighbors = [self.population_size-1]
                    
            particle.initialize_coefs(particle_dimensions, particle_type=self.particle_type,
                                      multiplier=self.particle_multiplier)
            
            particle.initialize_velocity(particle_dimensions)
                                
            self.particles[i] = particle
                

                
    def beta_fitness(self, particle, X, Y):
        
        coefs = particle.coefs
        
        correct = []
        fitnesses = []
        
        for iter, (tX, tY) in enumerate(zip(X, Y)):
            predictors = coefs*tX
            prediction = np.sum(predictors)
            
            if np.sign(prediction) == np.sign(tY):
                correct.append(1.)
            else:
                correct.append(0.)
                
            fitnesses.append(self.fitness_function(float(tY), prediction))
            
        avg_correct = sum(correct)/len(correct)
        avg_fitness = sum(fitnesses)/len(fitnesses)
        
        return avg_correct, avg_fitness
    
    
    
    def binary_fitness(self, particle, X, Y, test_coefs):
        
        correct = []
        fitnesses = []
        
        coefs = test_coefs.copy() * particle.coefs
        
        for tX, tY in zip(X, Y):
            predictors = coefs*tX
            prediction = np.sum(predictors)
            
            if np.sign(prediction) == np.sign(tY):
                correct.append(1.)
            else:
                correct.append(-1.)
                
            fitnesses.append(self.fitness_function(float(tY), prediction))
            
        avg_correct = sum(correct)/len(correct)
        avg_fitness = sum(fitnesses)/len(fitnesses)
        
        return avg_correct, avg_fitness
        
    
    
    
    def multi_asynchronous_update(self, Xs, Ys, test_coefs):
            
        if self.verbose:
            print 'updating particle swarm...'
        
        for p in self.particles:
            particle = self.particles[p]
            
            # Currently this section only works for binary/probability swarms
            p_accuracy, p_fitness = [], []
            for c_X, c_Y, c_coefs in zip(Xs, Ys, test_coefs):
                c_accuracy, c_fitness = self.binary_fitness(particle, c_X, c_Y, c_coefs)
                p_accuracy.append(c_accuracy)
                p_fitness.append(c_fitness)
                
            p_accuracy = sum(p_accuracy)/len(p_accuracy)
            p_fitness = sum(p_fitness)/len(p_fitness)
            
            
            if self.global_best_fitness is None:
                self.global_best_fitness = p_fitness
                self.global_best_coefs = particle.coefs.copy()
            if self.global_best_accuracy is None:
                self.global_best_accuracy = p_accuracy
            
            
            particle.prior_fitness = particle.current_fitness
            particle.current_fitness = p_fitness
            particle.fitnesses.append(p_fitness)
            
            particle.prior_accuracy = particle.current_accuracy
            particle.current_accuracy = p_accuracy
            particle.accuracies.append(p_accuracy)
            
            if p_fitness > particle.best_fitness:
                particle.best_coefs = particle.coefs.copy()
                particle.best_fitness = p_fitness
                
            if p_fitness > self.global_best_fitness:
                self.global_best_coefs = particle.coefs.copy()
                self.global_best_fitness = p_fitness
            
            if p_accuracy > particle.best_accuracy:
                particle.best_accuracy = p_accuracy
                
            if p_accuracy > self.global_best_accuracy:
                self.global_best_accuracy = p_accuracy
                
            
            neighborhood_best = p
            for nbr in particle.neighbors:
                if self.particles[nbr].best_fitness > p_fitness:
                    neighborhood_best = nbr
            
            particle.abs_velocities.append(np.sum(np.absolute(particle.velocity)))
            particle.velocity = self.velocity_function(self.particles, p, neighborhood_best)
            
        
            if self.particle_type == 'weight': 
                particle.coefs += particle.velocity

                if self.use_xmax:
                    particle.coefs = np.maximum(self.xmax_range, particle.coefs)
                    particle.coefs = np.minimum(self.xmin_range, particle.coefs)
            
            elif self.particle_type == 'probability':
                # sigmoid chance selector
                chance_vector = np.random.random_sample(size=particle.dimensions)
                sigmoid_velocity = self._sigmoid(particle.velocity)
                particle.coefs = np.array((sigmoid_velocity > chance_vector), dtype=np.int)
                
            # re-assign just in case:
            self.particles[p] = particle


    def _sigmoid(self, velocity):
        return np.array(1. / (1. + np.exp(-1. * velocity)))
        

    def reporter(self, iter):
        items = [[p.id, p.current_fitness, p.current_accuracy, p.coefs] for p in self.particles.values()]
        items = sorted(items, key=lambda k: k[1])
        items.reverse()
        print '\nITERATION:', iter
        print 'PID:\t\t\tCUR FIT:\t\t\tCUR ACC:\t\t\tCOEF SUM:'
        for id, fit, acc, coef in items:
            print id, '\t\t\t', fit, '\t\t\t', acc, '\t\t\t', np.sum(coef)
            
        print 'GLOBAL BEST FIT:\t\t\tGLOBAL BEST ACC:'
        print self.global_best_fitness, '\t\t\t', self.global_best_accuracy
        


    def run(self):
        
        self.initialize_particles(self.swarmdata.dimensions)
        
        testXs = []
        testYs = []
        test_coefs = []
        
        
        if self.verbose:
            print 'sorting X, Y, coef from data folds...'
            
            
        for fold in sorted(self.swarmdata.data_folds.keys()):
            fobj = self.swarmdata.data_folds[fold]
            testXs.append(fobj.testX)
            testYs.append(fobj.testY)
            test_coefs.append(fobj.coefs)
            
            
        if self.verbose:
            print 'beginning update cycle. iterations:', self.max_iters
           
           
        for iter in range(self.max_iters):
                
            self.multi_asynchronous_update(testXs, testYs, test_coefs)
            self.reporter(iter)
            
        
















