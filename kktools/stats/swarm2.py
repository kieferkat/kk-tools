
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
            self.coefs = 1.*np.random.randint(2, size=dimensions)
            
        elif particle_type == 'probability':
            self.coefs = np.random.random_sample(size=dimensions)
            
        self.best_coefs = self.coefs.copy()
            
            
    def initialize_velocity(self, dimensions, velocity_start='zeros'):
    
        if velocity_start == 'zeros':
            self.velocity = np.zeros(dimensions)
            
            
            
            
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
                 use_constriction=True, constriction_k=1.0, vmax_range=None):
        super(VelocityFunction, self).__init__()
        
        self.acceleration_sum = acceleration_sum
        
        self.use_inertia = use_inertia
        self.inertia_constant = inertia_constant
        
        self.use_constriction = use_constriction
        self.constriction_k = constriction_k
        if self.use_constriction:
            self.calculate_constriction()
        
        self.vmax_range = vmax_range
        
        self.verbose = True
        
        
    def __call__(self, particles, particle, neighborhood_best):
        return self.velocity(particles, particle, neighborhood_best)
        
        
    def calculate_constriction(self):
        a_ = (2. * self.constriction_k)
        b_ = 2. - self.acceleration_sum
        c_ = self.acceleration_sum**2. - 4.*self.acceleration_sum
        d_ = abs(b_ - np.sqrt(c_))
        self.constriction = a_ / d_
        
        if self.verbose:
            print 'constriction coefficient = ', self.constriction
        
        
    def velocity(self, particles, p, nb):
        
        acceleration_a = self.acceleration_sum * np.random.random()
        acceleration_b = self.acceleration_sum - acceleration_a
        
        if p == nb:
            nb_d = np.zeros(particles[p].dimensions)
        else:
            nb_d = particles[nb].best_coefs - particles[p].coefs
            
        ind_d = particles[p].best_coefs - particles[p].coefs
        
        nb_d *= acceleration_a
        ind_d *= acceleration_b
        
        next_v = nb_d + ind_d
        
        if self.use_inertia:
            past_v = particles[p].velocity.copy()
            past_v *= self.inertia_constant
            next_v += past_v
            
        if self.use_constriction:
            next_v *= self.constriction
            
        if self.vmax_range is not None:
            next_v = np.maximum(-1. * self.vmax_range, next_v)
            next_v = np.minimum(self.vmax_range, next_v)
            
        return next_v



class Swarm(object):
    
    def __init__(self, id=0, vmax=None, xmax=None, particle_type='weight',
                 particle_multiplier=1., neighborhood_type='circle',
                 population_size=5, fitness_function=FitnessFunction(),
                 velocity_function=VelocityFunction()):
        
        super(Swarm, self).__init__()
        
        self.id = id
        self.vmax = vmax
        self.xmax = xmax
        self.particle_type = particle_type
        self.particle_multiplier = particle_multiplier
        self.neighborhood_type = neighborhood_type
        self.population_size = population_size
        self.particles = {}
        
        self.fitness_function = fitness_function
        self.velocity_function = velocity_function
        
        
    def initialize_particle_limits(self, particle_dimensions):
        
        if not self.vmax is None:
            self.vmax_range = np.array([self.vmax for x in range(particle_dimensions)])
            self.velocity_function.vmax_range = self.vmax_range
        
        if not self.xmax is None:
            self.xmax_range = np.array([self.xmax for x in range(particle_dimensions)])
            self.use_xmax = True
    
        
    def initialize_global_records(self, particle_dimensions):
        
        self.global_best_coefs = np.zeros(particle_dimensions)
        self.global_best_accuracy = None
        self.global_best_fitness = None
        
        
    def initialize_particles(self, particle_dimensions):
        
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
                                      particle_multiplier=self.particle_multiplier)
            
            particle.initialize_velocity(particle_dimensions)
                                
            self.particles[i] = particle
                

                
    def beta_fitness(self, particle, testX, testY):
        
        coefs = particle.coefs
        
        correct = []
        fitnesses = []
        
        for iter, (tX, tY) in enumerate(zip(testX, testY)):
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
    
    
    
    def asynchronous_update(self, X, Y):
        
        assess_particle_fitness = self.beta_fitness
        
        
        for p in self.particles:
            particle = self.particles[p]
            
            p_accuracy, p_fitness = assess_particle_fitness(particle, X, Y)
            
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
            for nbr in particles.neighbors:
                if self.particles.best_fitness > p_fitness:
                    neighborhood_best = nbr
            
            particle.abs_velocities.append(np.sum(np.absolute(particle.velocity)))
            particle.velocity = self.velocity_function(self.particles, p, neighborhood_best)
            
            
            particle.coefs += particle.velocity

            if self.use_xmax:
                particle.coefs = np.maximum(-1.*self.xmax_range, particle.coefs)
                particle.coefs = np.minimum(self.xmax_range, particle.coefs)
                
            # re-assign just in case:
            self.particles[p] = particle




















